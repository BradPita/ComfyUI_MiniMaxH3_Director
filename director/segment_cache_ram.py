"""In-process RAM staging for the Director segment cache (opt-in).

The on-disk cache lives in ``output/minimax_seg_cache/<cache_key>/`` and is
implemented in :mod:`..director.segment_cache`. This module provides a *RAM
backed directory* with exactly the same file names, so every piece of caching
policy (fingerprints, stale rejection, prune / clear / inspect, the
confirm-first-pass hold) keeps living in one place: ``segment_cache`` only asks
for the cache root and never needs to know which store answered.

Why it exists
-------------
The pixel-frame payload dominates staging cost: 124 frames at 864x480 as uint8
is ~150 MB per segment, and the disk path pays an FFV1 encode (or a torch.save)
on write plus a decode on read. On a RAM store the frame tensor and the
AV-latent / audio dicts are kept as live Python objects — no serialization, no
disk I/O. That removes ~0.2-1.0 s per segment on typical hardware; see
``docs/ram-cache.md`` for the measured breakdown and the trade-offs.

What it costs
-------------
* RAM: ~0.16 GB per 5 s segment at 0.4 MP (0.4 MP x 3 ch x 124 f, uint8), plus
  a few MB of latents/audio.
* Durability: the store dies with the ComfyUI process. A restart re-samples,
  exactly like an empty disk cache. This is why the node that enables it is
  labelled ``(64GB RAM and above)``.
* Budget is enforced with LRU eviction per *slot* (``seg_XXXX``), never leaving
  half a slot behind.

Key namespace
-------------
RAM cache keys are ``"ram:" + <normal disk cache key>``. The prefix contains
``:``, a character :func:`segment_cache.normalize_cache_name` can never emit and
:func:`segment_cache._sanitize_dir_token` rejects, so a RAM key can never be
mistaken for a disk directory (and vice versa). Nothing here is global state
that needs cleanup: routing is a pure function of the key string.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.ram")

RAM_KEY_PREFIX = "ram:"
DEFAULT_BUDGET_GB = 8.0
GIB = 1024 * 1024 * 1024
_BUDGET_ENV = "MINIMAX_DIRECTOR_RAM_CACHE_GB"
_MIN_BUDGET_GB = 1.0
_MAX_BUDGET_GB = 512.0
# Slots that may sit unwritten in the spill queue (≈ 2-3 segments of payloads).
_SPILL_MAX_PENDING = 6

_LOCK = threading.RLock()
_STORES: dict[str, "_Store"] = {}
# (key, name) -> None, oldest first. Touched on read and write.
_LRU: "OrderedDict[tuple[str, str], None]" = OrderedDict()
_TOTAL_BYTES = 0
_BUDGET_BYTES = int(DEFAULT_BUDGET_GB * GIB)
# Monotonic stand-in for st_mtime_ns so ``.pre`` file signatures change on write.
_CLOCK = 0
_EPOCH_NS = time.time_ns()


# --------------------------------------------------------------------------- #
# key helpers
# --------------------------------------------------------------------------- #
def is_ram_key(key: Any) -> bool:
    """True when ``key`` addresses the RAM store instead of ``output/``."""
    try:
        return str(key or "").startswith(RAM_KEY_PREFIX)
    except Exception:  # pragma: no cover - str() on a broken object
        return False


def make_ram_key(base: Any) -> str:
    """Namespaced RAM key for a plain cache key. Empty base stays empty."""
    text = str(base or "").strip()
    if not text:
        return ""
    if text.startswith(RAM_KEY_PREFIX):
        return text
    return f"{RAM_KEY_PREFIX}{text}"


def base_key(key: Any) -> str:
    """Strip the RAM prefix (for reports)."""
    text = str(key or "")
    return text[len(RAM_KEY_PREFIX):] if text.startswith(RAM_KEY_PREFIX) else text


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def _coerce_gb(value: Any) -> float | None:
    try:
        gb = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if gb != gb or gb <= 0:  # NaN / non-positive
        return None
    return gb


def configure(budget_gb: Any = None) -> int:
    """Set the process-wide RAM budget in GiB. ``None``/0 falls back to env/default.

    The value is clamped to [1, 512] GiB so a typo (or a 0) can never starve the
    rest of ComfyUI. Returns the applied budget in bytes.
    """
    global _BUDGET_BYTES
    gb = _coerce_gb(budget_gb)
    if gb is None:
        gb = _coerce_gb(os.environ.get(_BUDGET_ENV))
    if gb is None:
        gb = DEFAULT_BUDGET_GB
    gb = min(_MAX_BUDGET_GB, max(_MIN_BUDGET_GB, gb))
    with _LOCK:
        _BUDGET_BYTES = int(gb * GIB)
        _evict_locked()
    return _BUDGET_BYTES


def budget_bytes() -> int:
    with _LOCK:
        return _BUDGET_BYTES


def set_budget_bytes(value: int) -> None:
    """Test / tooling hook: set an exact budget (no clamping) and evict."""
    global _BUDGET_BYTES
    with _LOCK:
        _BUDGET_BYTES = max(0, int(value))
        _evict_locked()


def stats() -> dict[str, Any]:
    """Snapshot for the run report / diagnostics."""
    with _LOCK:
        return {
            "keys": len(_STORES),
            "entries": sum(len(store.entries) for store in _STORES.values()),
            "bytes": _TOTAL_BYTES,
            "budget_bytes": _BUDGET_BYTES,
            "budget_gb": round(_BUDGET_BYTES / GIB, 2),
        }


def reset() -> None:
    """Drop every cached slot (tests, or a hard「清空缓存」)."""
    global _TOTAL_BYTES
    with _LOCK:
        _STORES.clear()
        _LRU.clear()
        _TOTAL_BYTES = 0


# --------------------------------------------------------------------------- #
# optional write-behind to disk
# --------------------------------------------------------------------------- #
# When enabled, every successful RAM write is also handed to a background worker
# that persists it under the *base* key (``<name>_<nodeid>``, no ``ram:`` prefix)
# with the on-disk file layout. Reads always come from RAM, so the spill never
# slows a run; it only buys cross-restart reuse. Best effort: a full queue or a
# write error drops the spill, never the run.
_SPILL_SINK: Callable[[str, str, Any, bool], None] | None = None
_SPILL_QUEUE: "queue.Queue | None" = None
_SPILL_THREAD: threading.Thread | None = None
_SPILL_DROPPED = 0


def _spill_worker(sink: Callable[[str, str, Any, bool], None], work: "queue.Queue") -> None:
    while True:
        item = work.get()
        try:
            if item is None:
                return
            base, name, payload, is_text = item
            try:
                sink(base, name, payload, is_text)
            except Exception as exc:  # pragma: no cover - depends on the sink
                log.debug("RAM spill skipped for %s (%s).", name, exc)
        finally:
            work.task_done()


def enable_spill(sink: Callable[[str, str, Any, bool], None], *, max_pending: int = _SPILL_MAX_PENDING) -> None:
    """Mirror every RAM write to ``sink(base_key, name, payload, is_text)``.

    ``sink`` runs on a daemon worker thread; it must be safe to lose writes.
    """
    global _SPILL_SINK, _SPILL_QUEUE, _SPILL_THREAD, _SPILL_DROPPED
    with _LOCK:
        if _SPILL_SINK is not None and _SPILL_QUEUE is not None:
            _SPILL_SINK = sink
            return
        work: "queue.Queue" = queue.Queue(maxsize=max(1, int(max_pending)))
        thread = threading.Thread(
            target=_spill_worker, args=(sink, work), name="mmx-ram-spill", daemon=True
        )
        _SPILL_SINK = sink
        _SPILL_QUEUE = work
        _SPILL_THREAD = thread
        _SPILL_DROPPED = 0
        thread.start()


def disable_spill() -> None:
    """Stop mirroring (in-flight writes still finish; queued ones are dropped)."""
    global _SPILL_SINK, _SPILL_QUEUE, _SPILL_THREAD
    with _LOCK:
        work = _SPILL_QUEUE
        _SPILL_SINK = None
        _SPILL_QUEUE = None
        _SPILL_THREAD = None
    if work is not None:
        try:
            work.put_nowait(None)  # worker exits after the current item
        except queue.Full:
            pass


def spill_enabled() -> bool:
    return _SPILL_SINK is not None


def spill_stats() -> dict[str, Any]:
    work = _SPILL_QUEUE
    return {
        "enabled": _SPILL_SINK is not None,
        "pending": work.qsize() if work is not None else 0,
        "dropped": _SPILL_DROPPED,
    }


def _spill_enqueue(key: str, name: str, payload: Any, is_text: bool) -> None:
    global _SPILL_DROPPED
    sink = _SPILL_SINK
    work = _SPILL_QUEUE
    if sink is None or work is None:
        return
    try:
        work.put_nowait((base_key(key), name, payload, is_text))
    except queue.Full:
        _SPILL_DROPPED += 1
        if _SPILL_DROPPED == 1:
            log.warning(
                "RAM spill queue full; further segment writes go to RAM only "
                "(disk mirror may be incomplete)."
            )


# --------------------------------------------------------------------------- #
# byte accounting
# --------------------------------------------------------------------------- #
def _nbytes(value: Any) -> int:
    """Rough resident size of a cached payload (tensors dominate)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8", "ignore")) + 64
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            element = getattr(value, "element_size", None)
            itemsize = int(element() if callable(element) else 4)
            return int(numel()) * itemsize + 128
        except Exception:
            return 128
    unbind = getattr(value, "unbind", None)  # comfy NestedTensor (AV latents)
    if callable(unbind) and not isinstance(value, (dict, list, tuple)):
        try:
            return 64 + sum(_nbytes(part) for part in unbind())
        except Exception:
            return 128
    if isinstance(value, dict):
        return 128 + sum(_nbytes(k) + _nbytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 64 + sum(_nbytes(item) for item in value)
    return int(sys.getsizeof(value))


def _clone_payload(value: Any) -> Any:
    """Hand out a private copy so callers cannot mutate the staged original.

    Sampling / trim code does rewrite context latents in place; the disk store is
    immune to that by construction (every read deserializes fresh), so the RAM
    store has to defend itself too.
    """
    clone = getattr(value, "clone", None)
    if callable(clone) and hasattr(value, "detach"):
        try:
            return clone()
        except Exception:
            return value
    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        try:
            parts = tuple(_clone_payload(part) for part in unbind())
            return type(value)(parts)
        except Exception:
            return value
    if isinstance(value, dict):
        return {key: _clone_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_payload(item) for item in value)
    return value


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def _slot_group(name: str) -> str:
    """``seg_0000.pre.av.pt`` -> ``seg_0000.pre``; other names map to themselves."""
    text = str(name).lstrip(".")
    parts = text.split(".")
    if not parts:
        return str(name)
    group = parts[0]
    if len(parts) > 1 and parts[1] == "pre":
        group = f"{group}.pre"
    return group


class _Entry:
    __slots__ = ("value", "nbytes", "is_text", "mtime_ns")

    def __init__(self, value: Any, *, is_text: bool, mtime_ns: int) -> None:
        self.value = value
        self.nbytes = _nbytes(value)
        self.is_text = is_text
        self.mtime_ns = mtime_ns


class _Store:
    __slots__ = ("key", "entries")

    def __init__(self, key: str) -> None:
        self.key = key
        self.entries: dict[str, _Entry] = {}


class _RamStat:
    __slots__ = ("st_size", "st_mtime_ns", "st_mtime")

    def __init__(self, size: int, mtime_ns: int) -> None:
        self.st_size = int(size)
        self.st_mtime_ns = int(mtime_ns)
        self.st_mtime = self.st_mtime_ns / 1_000_000_000


def _next_mtime_ns() -> int:
    """Strictly increasing stand-in for st_mtime_ns (cheap, sortable)."""
    global _CLOCK
    _CLOCK += 1
    return _EPOCH_NS + _CLOCK


def _store_for(key: str) -> _Store:
    store = _STORES.get(key)
    if store is None:
        store = _Store(key)
        _STORES[key] = store
    return store


def _remove_locked(key: str, name: str) -> bool:
    global _TOTAL_BYTES
    store = _STORES.get(key)
    entry = store.entries.pop(name, None) if store is not None else None
    _LRU.pop((key, name), None)
    if entry is None:
        return False
    _TOTAL_BYTES -= entry.nbytes
    return True


def _evict_locked(protect: tuple[str, str] | None = None) -> None:
    """LRU-evict whole slots until the budget is respected.

    ``protect`` is the slot that was just written. A single segment can be larger
    than the whole budget (a 1088p / 243f segment is ~1.4 GB); dropping it the
    instant it lands would make the cache useless exactly where staging is most
    expensive, so the newest slot is always kept — the budget is a target, not a
    hard cap, and overshoot is bounded by one segment.
    """
    while _TOTAL_BYTES > _BUDGET_BYTES and _LRU:
        key, name = next(iter(_LRU))
        if protect is not None and (key, name) == protect:
            return
        store = _STORES.get(key)
        group = _slot_group(name)
        victims = (
            [slot for slot in store.entries if _slot_group(slot) == group]
            if store is not None
            else []
        )
        if not victims:
            victims = [name]
        for victim in victims:
            _remove_locked(key, victim)


def _put_text(key: str, name: str, text: str) -> None:
    global _TOTAL_BYTES
    with _LOCK:
        _remove_locked(key, name)
        store = _store_for(key)
        entry = _Entry(text, is_text=True, mtime_ns=_next_mtime_ns())
        store.entries[name] = entry
        _TOTAL_BYTES += entry.nbytes
        _LRU[(key, name)] = None
        _LRU.move_to_end((key, name))
        _evict_locked(protect=(key, name))
    _spill_enqueue(key, name, text, True)


def _put_object(key: str, name: str, value: Any) -> None:
    global _TOTAL_BYTES
    with _LOCK:
        _remove_locked(key, name)
        store = _store_for(key)
        entry = _Entry(value, is_text=False, mtime_ns=_next_mtime_ns())
        store.entries[name] = entry
        _TOTAL_BYTES += entry.nbytes
        _LRU[(key, name)] = None
        _LRU.move_to_end((key, name))
        _evict_locked(protect=(key, name))
    _spill_enqueue(key, name, value, False)


def _touch_locked(key: str, name: str) -> None:
    if (key, name) in _LRU:
        _LRU.move_to_end((key, name))


def _get_entry(key: str, name: str, *, touch: bool = True) -> _Entry | None:
    with _LOCK:
        store = _STORES.get(key)
        entry = store.entries.get(name) if store is not None else None
        if entry is not None and touch:
            _touch_locked(key, name)
        return entry


def _names(key: str) -> list[str]:
    with _LOCK:
        store = _STORES.get(key)
        return sorted(store.entries) if store is not None else []


# --------------------------------------------------------------------------- #
# Path-like handles
# --------------------------------------------------------------------------- #
class RamFile:
    """Minimal ``pathlib.Path`` stand-in for one cached file inside the store.

    Only the surface ``segment_cache`` actually uses is implemented:
    ``is_file`` / ``is_symlink`` / ``is_dir`` / ``unlink`` / ``read_text`` /
    ``write_text`` / ``stat`` / ``name``. ``torch.save`` / ``torch.load`` are
    routed through ``segment_cache._save_torch`` / ``_load_torch`` so payloads
    stay live objects here (no pickle round trip).
    """

    __slots__ = ("_root", "name")

    def __init__(self, root: "RamRoot", name: str) -> None:
        self._root = root
        self.name = str(name)

    # -- path protocol ------------------------------------------------------
    def is_file(self) -> bool:
        return self._root.has(self.name)

    def exists(self) -> bool:
        return self.is_file()

    def is_symlink(self) -> bool:
        return False

    def is_dir(self) -> bool:
        return False

    def unlink(self, missing_ok: bool = False) -> None:
        if not self._root.remove(self.name) and not missing_ok:
            raise FileNotFoundError(self.name)

    def stat(self) -> _RamStat:
        entry = _get_entry(self._root.key, self.name)
        if entry is None:
            raise FileNotFoundError(self.name)
        return _RamStat(entry.nbytes, entry.mtime_ns)

    # -- text payloads (meta / handoff json) --------------------------------
    def read_text(self, encoding: str = "utf-8", errors: str | None = None) -> str:
        entry = _get_entry(self._root.key, self.name)
        if entry is None:
            raise FileNotFoundError(self.name)
        if not entry.is_text:
            raise IsADirectoryError(f"{self!s} holds a binary payload")
        return entry.value

    def write_text(self, data: Any, encoding: str = "utf-8", errors: str | None = None) -> int:
        text = data if isinstance(data, str) else str(data)
        _put_text(self._root.key, self.name, text)
        return len(text)

    # -- live object payloads (frames / latents / audio) --------------------
    def set_object(self, value: Any) -> None:
        _put_object(self._root.key, self.name, value)

    def get_object(self, clone: bool = True) -> Any:
        """Return the staged payload. ``clone=False`` only for read-only callers
        that immediately produce a fresh tensor (frame decode does)."""
        entry = _get_entry(self._root.key, self.name)
        if entry is None:
            raise FileNotFoundError(self.name)
        if entry.is_text:
            return entry.value
        return _clone_payload(entry.value) if clone else entry.value

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ram://{self._root.key}/{self.name}"

    __repr__ = __str__

    def __lt__(self, other: Any) -> bool:
        # ``segment_cache.first_pass_cache_disk_signature`` does
        # ``sorted(root.glob(...))``; pathlib sorts by string parts.
        if isinstance(other, RamFile):
            return (self._root.key, self.name) < (other._root.key, other.name)
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RamFile):
            return (self._root.key, self.name) == (other._root.key, other.name)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._root.key, self.name))


class RamRoot:
    """Minimal ``pathlib.Path`` stand-in for one cache key's directory."""

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = str(key)

    # -- path protocol ------------------------------------------------------
    def __truediv__(self, name: Any) -> RamFile:
        return RamFile(self, str(name))

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        with _LOCK:
            _store_for(self.key)

    def is_dir(self) -> bool:
        return True

    def exists(self) -> bool:
        return True

    def iterdir(self) -> list[RamFile]:
        return [RamFile(self, name) for name in _names(self.key)]

    def glob(self, pattern: str) -> list[RamFile]:
        return [
            RamFile(self, name)
            for name in _names(self.key)
            if fnmatch.fnmatchcase(name, pattern)
        ]

    # -- store ops used by RamFile -----------------------------------------
    def has(self, name: str) -> bool:
        return _get_entry(self.key, name) is not None

    def remove(self, name: str) -> bool:
        with _LOCK:
            return _remove_locked(self.key, name)

    def bytes(self) -> int:
        with _LOCK:
            store = _STORES.get(self.key)
            return sum(entry.nbytes for entry in store.entries.values()) if store else 0

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ram://{self.key}/"

    __repr__ = __str__


def root_for(key: Any) -> RamRoot:
    """Cache root for a RAM key; registers the (empty) store on first use."""
    token = str(key or "").strip()
    with _LOCK:
        _store_for(token)
    return RamRoot(token)
