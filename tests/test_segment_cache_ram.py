"""Tests for the opt-in RAM staging store (``director/segment_cache_ram.py``).

Two layers:

* ``RamStoreTests`` — the store itself (Path-like handles, LRU budget, slot
  grouping). Pure stdlib, no ComfyUI, no torch.
* ``SegmentCacheRamPlumbingTests`` — the wiring inside ``director/segment_cache.py``
  (key namespace, ``_cache_root`` routing, frame slot round trip, torch payload
  round trip) using a tiny torch / folder_paths stub, so it also runs on a
  machine without ComfyUI. It exercises plumbing only: fingerprint comparison
  needs the real plan/selflift/refine modules and is covered by the code review.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from director import segment_cache_ram as ram  # noqa: E402  (stdlib-only module)


class FakeTensor:
    """Minimal stand-in for torch.Tensor: shape/dtype arithmetic, no data."""

    def __init__(self, shape=(1,), dtype="float32", scale=1.0):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.scale = float(scale)

    # -- shape / size ------------------------------------------------------
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def element_size(self) -> int:
        return 1 if self.dtype == "uint8" else 4

    # -- no-op conversions used by the cache -------------------------------
    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def float(self):
        return FakeTensor(self.shape, "float32", self.scale)

    def div(self, value):
        return FakeTensor(self.shape, "float32", self.scale / float(value))

    def mul(self, value):
        return FakeTensor(self.shape, self.dtype, self.scale * float(value))

    def clamp(self, low=None, high=None):
        return self

    def round(self):
        return self

    def to(self, dtype):
        return FakeTensor(self.shape, str(dtype), self.scale)

    def clone(self):
        return FakeTensor(self.shape, self.dtype, self.scale)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeTensor(shape={self.shape}, dtype={self.dtype})"


class RamKeyTests(unittest.TestCase):
    def setUp(self):
        ram.reset()

    def test_namespace_round_trip(self):
        self.assertTrue(ram.is_ram_key("ram:name_57"))
        self.assertFalse(ram.is_ram_key("name_57"))
        self.assertFalse(ram.is_ram_key(None))
        self.assertEqual(ram.make_ram_key("name_57"), "ram:name_57")
        self.assertEqual(ram.make_ram_key(""), "")
        self.assertEqual(ram.make_ram_key("ram:name_57"), "ram:name_57")
        self.assertEqual(ram.base_key("ram:name_57"), "name_57")
        self.assertEqual(ram.base_key("name_57"), "name_57")


class RamStoreTests(unittest.TestCase):
    def setUp(self):
        ram.reset()
        ram.disable_spill()
        ram.set_budget_bytes(1024 * 1024)
        self.root = ram.root_for("ram:57")
        self.addCleanup(ram.reset)
        self.addCleanup(ram.set_budget_bytes, int(ram.DEFAULT_BUDGET_GB * ram.GIB))

    # -- Path-like surface -------------------------------------------------
    def test_text_round_trip_and_stat(self):
        slot = self.root / "seg_0000.meta.json"
        self.assertFalse(slot.is_file())
        self.assertFalse(slot.is_symlink())
        self.assertEqual(list(self.root.iterdir()), [])

        slot.write_text('{"seed": 1}', encoding="utf-8")
        self.assertTrue(slot.is_file())
        self.assertTrue(slot.exists())
        self.assertEqual(slot.read_text(encoding="utf-8"), '{"seed": 1}')
        self.assertEqual(slot.name, "seg_0000.meta.json")
        self.assertGreater(slot.stat().st_size, 0)
        self.assertGreater(slot.stat().st_mtime_ns, 0)
        self.assertEqual([f.name for f in self.root.iterdir()], ["seg_0000.meta.json"])
        self.assertEqual([f.name for f in self.root.glob("seg_*.meta.json")], ["seg_0000.meta.json"])
        self.assertEqual(list(self.root.glob("seg_*.pre.*")), [])

        slot.unlink()
        self.assertFalse(slot.is_file())
        with self.assertRaises(FileNotFoundError):
            slot.read_text(encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            slot.stat()

    def test_glob_matches_pre_and_raw_names(self):
        for name in ("seg_0000.pre.meta.json", "seg_0000.pre.av.pt", "seg_0000.pt"):
            (self.root / name).write_text("x", encoding="utf-8")
        pre = sorted(f.name for f in self.root.glob("seg_*.pre.*"))
        self.assertEqual(pre, ["seg_0000.pre.av.pt", "seg_0000.pre.meta.json"])
        raw = sorted(f.name for f in self.root.glob("seg_*.pt"))
        self.assertEqual(raw, ["seg_0000.pre.av.pt", "seg_0000.pt"])

    def test_mtime_increases_per_write(self):
        first = self.root / "seg_0000.pt"
        second = self.root / "seg_0000.meta.json"
        first.write_text("a", encoding="utf-8")
        second.write_text("b", encoding="utf-8")
        self.assertLess(first.stat().st_mtime_ns, second.stat().st_mtime_ns)

    def test_object_payload_is_cloned_on_read(self):
        ram.set_budget_bytes(64 * 1024 * 1024)  # 124x48x86x3 uint8 ≈ 1.5 MB
        frames = FakeTensor((124, 48, 86, 3), "uint8")
        (self.root / "seg_0000.pt").set_object({"frames": frames})
        loaded = (self.root / "seg_0000.pt").get_object()
        self.assertIsNot(loaded["frames"], frames)
        self.assertEqual(loaded["frames"].shape, frames.shape)
        # Mutating the copy must not leak back into the store.
        loaded["frames"].shape = (1,)
        again = (self.root / "seg_0000.pt").get_object()
        self.assertEqual(again["frames"].shape, (124, 48, 86, 3))

    def test_text_slot_refuses_object_payload(self):
        slot = self.root / "seg_0000.meta.json"
        slot.set_object(FakeTensor((2,), "uint8"))
        with self.assertRaises(IsADirectoryError):
            slot.read_text(encoding="utf-8")

    # -- budget / eviction -------------------------------------------------
    def test_budget_evicts_entire_slot_not_half(self):
        ram.set_budget_bytes(4096)
        for idx in range(4):
            (self.root / f"seg_{idx:04d}.pt").set_object(FakeTensor((1000,), "uint8"))
            (self.root / f"seg_{idx:04d}.meta.json").write_text("{}", encoding="utf-8")
        names = [f.name for f in self.root.iterdir()]
        self.assertTrue(names, "budget must not evict everything")
        groups = {ram._slot_group(name) for name in names}
        for group in groups:
            self.assertIn(f"{group}.pt", names, f"slot {group} left half behind")
            self.assertIn(f"{group}.meta.json", names, f"slot {group} left half behind")
        self.assertLessEqual(ram.stats()["bytes"], 4096)
        self.assertIsNotNone((self.root / "seg_0003.pt").is_file())

    def test_read_touches_lru(self):
        slot_bytes = 1000 + 128  # FakeTensor numel * itemsize + overhead
        ram.set_budget_bytes(slot_bytes * 2)
        (self.root / "seg_0000.pt").set_object(FakeTensor((1000,), "uint8"))
        (self.root / "seg_0001.pt").set_object(FakeTensor((1000,), "uint8"))
        (self.root / "seg_0000.pt").get_object()  # touch → most recently used
        (self.root / "seg_0002.pt").set_object(FakeTensor((1000,), "uint8"))
        self.assertTrue((self.root / "seg_0000.pt").is_file())
        self.assertFalse((self.root / "seg_0001.pt").is_file())
        self.assertTrue((self.root / "seg_0002.pt").is_file())

    def test_single_oversized_slot_is_kept(self):
        # 1088p / 243f is ~1.4 GB: a smaller budget must not silently drop the
        # slot that was just written (it would disable staging where it matters).
        ram.set_budget_bytes(1024)
        (self.root / "seg_0000.pt").set_object(FakeTensor((4096,), "uint8"))
        self.assertTrue((self.root / "seg_0000.pt").is_file())
        # Older slots still go first, so the overshoot stays at one segment.
        ram.set_budget_bytes(2 * 4096)
        (self.root / "seg_0001.pt").set_object(FakeTensor((4096,), "uint8"))
        self.assertTrue((self.root / "seg_0001.pt").is_file())
        self.assertFalse((self.root / "seg_0000.pt").is_file())

    def test_unlink_and_reset_clear_accounting(self):
        (self.root / "seg_0000.pt").set_object(FakeTensor((10,), "uint8"))
        self.assertGreater(ram.stats()["bytes"], 0)
        (self.root / "seg_0000.pt").unlink()
        self.assertEqual(ram.stats()["bytes"], 0)
        (self.root / "seg_0001.pt").set_object(FakeTensor((10,), "uint8"))
        ram.reset()
        self.assertEqual(ram.stats(), {
            "keys": 0,
            "entries": 0,
            "bytes": 0,
            "budget_bytes": ram.budget_bytes(),
            "budget_gb": round(ram.budget_bytes() / ram.GIB, 2),
        })

    def test_configure_clamps(self):
        self.assertEqual(ram.configure(0), int(8 * ram.GIB))
        self.assertEqual(ram.configure(0.1), int(1 * ram.GIB))
        self.assertEqual(ram.configure(9999), int(512 * ram.GIB))
        self.assertEqual(ram.configure(12), int(12 * ram.GIB))

    def test_stores_are_isolated_per_key(self):
        other = ram.root_for("ram:other")
        (self.root / "seg_0000.pt").set_object(FakeTensor((4,), "uint8"))
        self.assertTrue((self.root / "seg_0000.pt").is_file())
        self.assertFalse((other / "seg_0000.pt").is_file())

    def test_slot_group(self):
        self.assertEqual(ram._slot_group("seg_0000.pt"), "seg_0000")
        self.assertEqual(ram._slot_group("seg_0000.pre.av.pt"), "seg_0000.pre")
        self.assertEqual(ram._slot_group(".seg_0000.abc.tmp"), "seg_0000")
        self.assertEqual(ram._slot_group("whatever"), "whatever")


# --------------------------------------------------------------------------- #
# segment_cache plumbing (torch / folder_paths stubbed)
# --------------------------------------------------------------------------- #
def _build_stub_modules(output_dir: str) -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.uint8 = "uint8"

    def _save(payload, path, *_args, **_kwargs):
        # Only the spill sink writes to disk here; a marker is enough to prove it.
        Path(path).write_bytes(b"STUB" + repr(payload).encode("utf-8", "ignore"))

    def _unsupported(*_args, **_kwargs):
        raise AssertionError("disk torch load is not exercised by this test")

    torch.save = _save
    torch.load = _unsupported

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: output_dir

    frames = types.ModuleType("mmxplugin.lib.frames_ffv1")
    frames.FRAMES_SUFFIX = ".frames.mkv"

    def _encode(*_args, **_kwargs):
        raise AssertionError("RAM staging must never call the FFV1 encoder")

    frames.encode_frames_ffv1 = _encode
    frames.decode_frames_ffv1 = _encode

    latency = types.ModuleType("mmxplugin.director.h3_latent_continue")
    latency.CONTINUE_PIPELINE_ID = "continue"
    latency.clamp_seam_min_mask = lambda value: value

    motion = types.ModuleType("mmxplugin.director.h3_motion_context")
    motion.CONTINUITY_PIPELINE_ID = "continuity"
    motion.trim_context_prefix = lambda *_a, **_k: None
    motion.trim_export_tail = lambda frames, audio, *_a, **_k: (frames, audio)

    plan = types.ModuleType("mmxplugin.director.plan")

    class _Plan:
        pass

    class _Seg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    plan.DirectorPlan = _Plan
    plan.SegmentPlan = _Seg
    plan.resolve_ref_image_size = lambda *_a, **_k: 0

    return {
        "torch": torch,
        "folder_paths": folder_paths,
        "mmxplugin.lib.frames_ffv1": frames,
        "mmxplugin.director.h3_latent_continue": latency,
        "mmxplugin.director.h3_motion_context": motion,
        "mmxplugin.director.plan": plan,
    }


class SegmentCacheRamPlumbingTests(unittest.TestCase):
    """Imports the real segment_cache with ComfyUI's modules stubbed out."""

    @classmethod
    def setUpClass(cls):
        saved = {name: sys.modules.get(name) for name in (
            "torch", "folder_paths", "mmxplugin", "mmxplugin.lib",
            "mmxplugin.director", "mmxplugin.director.segment_cache",
            "mmxplugin.director.segment_cache_ram",
        )}
        cls._saved = saved
        for name, path in (
            ("mmxplugin", REPO_ROOT),
            ("mmxplugin.lib", REPO_ROOT / "lib"),
            ("mmxplugin.director", REPO_ROOT / "director"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        cls._outdir = tempfile.mkdtemp(prefix="mmx-cache-out-")
        sys.modules.update(_build_stub_modules(cls._outdir))

        spec = importlib.util.spec_from_file_location(
            "mmxplugin.director.segment_cache",
            REPO_ROOT / "director" / "segment_cache.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.sc = module

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name.startswith("mmxplugin"):
                sys.modules.pop(name, None)
        for name, value in cls._saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def setUp(self):
        # segment_cache imports its own module instance (package-relative), so the
        # store under test here is *its* module, not the top-level import above.
        self.ram = self.sc._ramcache
        self.ram.reset()
        self.ram.disable_spill()
        # Real payloads (154 MB frames / 13 MB latents) must not be evicted here.
        self.ram.set_budget_bytes(4 * 1024 ** 3)

    def tearDown(self):
        self.ram.reset()

    # -- key namespace -----------------------------------------------------
    def test_resolve_key_defaults_to_disk(self):
        self.assertEqual(self.sc.resolve_segment_cache_key("", "57"), "57")
        self.assertEqual(self.sc.resolve_segment_cache_key("hero", "57"), "hero_57")

    def test_resolve_key_ram_uses_prefix(self):
        self.assertEqual(
            self.sc.resolve_segment_cache_key("", "57", backend="ram"), "ram:57"
        )
        self.assertEqual(
            self.sc.resolve_segment_cache_key("hero", 57, backend="RAM"), "ram:hero_57"
        )
        self.assertEqual(self.sc.resolve_segment_cache_key("hero", "", backend="ram"), "ram:hero")

    def test_ram_key_can_never_become_a_disk_path(self):
        with self.assertRaises(ValueError):
            self.sc._cache_dir("ram:hero_57")

    def test_cache_root_routes_by_key(self):
        disk_root = self.sc._cache_dir_or_none("57")
        self.assertIsInstance(disk_root, Path)
        ram_root = self.sc._cache_root("ram:57")
        self.assertIsInstance(ram_root, self.ram.RamRoot)
        self.assertIsInstance(self.sc._cache_dir_or_none("ram:57"), self.ram.RamRoot)
        # Unusable keys still degrade to None instead of raising.
        self.assertIsNone(self.sc._cache_dir_or_none("../escape"))

    # -- frame slots -------------------------------------------------------
    def test_frame_slot_round_trip_stays_in_ram(self):
        plan = types.SimpleNamespace(cache_frames_codec="ffv1", frame_rate=24)
        frames = FakeTensor((124, 480, 864, 3), "uint8")
        root = self.sc._cache_root("ram:57")

        # ffv1 is requested but RAM must store the raw tensor (no encoder call).
        self.sc._store_segment_frames(root, 3, frames, first_pass=False, plan=plan)
        self.assertTrue(self.sc._frames_exist(root, 3, first_pass=False))
        self.assertFalse(self.sc._frames_exist(root, 3, first_pass=True))
        self.assertTrue((root / "seg_0003.pt").is_file())
        self.assertFalse((root / "seg_0003.frames.mkv").is_file())

        loaded = self.sc._load_frames(root, 3, first_pass=False)
        self.assertEqual(loaded.shape, frames.shape)
        self.assertEqual(loaded.dtype, "float32")
        self.assertEqual(loaded.scale, 1.0 / 255.0)

        # Rewriting with raw codec must keep the same slot instead of doubling it.
        plan_raw = types.SimpleNamespace(cache_frames_codec="raw", frame_rate=24)
        self.sc._store_segment_frames(root, 3, frames, first_pass=True, plan=plan_raw)
        self.assertTrue(self.sc._frames_exist(root, 3, first_pass=True))
        self.assertTrue(self.sc._frames_exist(root, 3, first_pass=False))

    # -- generic payload slots --------------------------------------------
    def test_write_via_temp_and_torch_helpers(self):
        root = self.sc._cache_root("ram:57")
        meta = root / "seg_0000.meta.json"
        self.sc._write_via_temp(meta, lambda p: p.write_text('{"seed": 7}', encoding="utf-8"))
        self.assertEqual(meta.read_text(encoding="utf-8"), '{"seed": 7}')

        latent = root / "seg_0000.av.pt"
        payload = {"samples": FakeTensor((16, 31, 6, 108), "float32")}
        self.sc._save_torch(latent, payload)
        self.assertTrue(latent.is_file())
        loaded = self.sc._load_torch(latent, weights_only=False)
        self.assertEqual(loaded["samples"].shape, (16, 31, 6, 108))
        self.assertIsNot(loaded, payload)

    def test_prune_and_clear_see_ram_slots(self):
        root = self.sc._cache_root("ram:57")
        for idx in (0, 1, 5):
            (root / f"seg_{idx:04d}.pt").write_text("x", encoding="utf-8")
            (root / f"seg_{idx:04d}.meta.json").write_text("{}", encoding="utf-8")
        # Mirrors prune_segment_cache's contract: drop indices off the timeline.
        valid = {0, 1}
        for path in list(root.iterdir()):
            match = self.sc._SEG_CACHE_FILE_RE.match(path.name)
            if match and int(match.group(1)) not in valid:
                self.assertTrue(self.sc._safe_unlink(path))
        self.assertEqual(
            sorted(f.name for f in root.iterdir()),
            ["seg_0000.meta.json", "seg_0000.pt", "seg_0001.meta.json", "seg_0001.pt"],
        )
        removed = self.sc.clear_segment_cache("ram:57", kind="final")
        self.assertEqual(removed, 4)
        self.assertEqual(list(root.iterdir()), [])

    def test_clear_on_a_ram_key_also_drops_the_disk_mirror(self):
        key = self.sc.resolve_segment_cache_key("", "57", backend="ram")
        mirror = Path(self._outdir) / "minimax_seg_cache" / "57"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "seg_0000.pt").write_text("spilled", encoding="utf-8")
        root = self.sc._cache_root(key)
        (root / "seg_0000.pt").set_object(FakeTensor((4,), "uint8"))

        removed = self.sc.clear_segment_cache(key, kind="final")

        self.assertEqual(removed, 2)
        self.assertFalse((mirror / "seg_0000.pt").exists())
        self.assertEqual(list(root.iterdir()), [])

    def test_spill_writes_real_disk_files_under_the_base_key(self):
        key = self.sc.resolve_segment_cache_key("", "57", backend="ram")
        self.sc.set_ram_spill(True)
        self.addCleanup(self.sc.set_ram_spill, False)
        root = self.sc._cache_root(key)
        (root / "seg_0000.meta.json").write_text('{"seed": 1}', encoding="utf-8")
        (root / "seg_0000.av.pt").set_object({"samples": FakeTensor((2,), "float32")})

        mirror = Path(self._outdir) / "minimax_seg_cache" / "57"
        deadline = time.time() + 5
        while time.time() < deadline:
            if (mirror / "seg_0000.meta.json").exists() and (
                mirror / "seg_0000.av.pt"
            ).exists():
                break
            time.sleep(0.02)
        self.assertTrue((mirror / "seg_0000.meta.json").is_file())
        self.assertEqual(
            (mirror / "seg_0000.meta.json").read_text(encoding="utf-8"), '{"seed": 1}'
        )
        self.assertTrue((mirror / "seg_0000.av.pt").is_file())

    def test_first_pass_signature_changes_after_write(self):
        key = self.sc.resolve_segment_cache_key("", "57", backend="ram")
        before = self.sc.first_pass_cache_disk_signature(key)
        root = self.sc._cache_root(key)
        (root / "seg_0001.pre.meta.json").write_text("{}", encoding="utf-8")
        (root / "seg_0000.pre.meta.json").write_text("{}", encoding="utf-8")
        after = self.sc.first_pass_cache_disk_signature(key)
        self.assertEqual(before, "")
        self.assertIn("seg_0000.pre.meta.json", after)
        self.assertIn("seg_0001.pre.meta.json", after)
        self.assertNotEqual(before, after)
        # sorted() over slots must work (pathlib sorts by parts).
        self.assertLess(after.index("seg_0000"), after.index("seg_0001"))


class RamSpillTests(unittest.TestCase):
    """Write-behind: RAM reads, background disk mirror."""

    def setUp(self):
        ram.reset()
        ram.set_budget_bytes(64 * 1024 * 1024)
        ram.disable_spill()
        self.root = ram.root_for("ram:hero_57")
        self.addCleanup(ram.reset)
        self.addCleanup(ram.disable_spill)

    def test_spill_is_off_until_enabled(self):
        self.assertFalse(ram.spill_enabled())
        (self.root / "seg_0000.pt").set_object(FakeTensor((4,), "uint8"))
        self.assertEqual(ram.spill_stats()["pending"], 0)

    def test_spill_mirrors_writes_under_the_base_key(self):
        written = []
        done = threading.Event()

        def sink(base, name, payload, is_text):
            written.append((base, name, is_text))
            if len(written) >= 3:
                done.set()

        ram.enable_spill(sink, max_pending=8)
        (self.root / "seg_0000.meta.json").write_text("{}", encoding="utf-8")
        (self.root / "seg_0000.pt").set_object(FakeTensor((4,), "uint8"))
        (self.root / "seg_0000.av.pt").set_object({"samples": FakeTensor((2,), "uint8")})
        self.assertTrue(done.wait(5), "spill worker did not drain the queue")
        self.assertEqual(
            set(written),
            {
                ("hero_57", "seg_0000.meta.json", True),
                ("hero_57", "seg_0000.pt", False),
                ("hero_57", "seg_0000.av.pt", False),
            },
        )

    def test_full_queue_drops_instead_of_blocking_generation(self):
        gate = threading.Event()

        def sink(base, name, payload, is_text):
            gate.wait(5)

        ram.enable_spill(sink, max_pending=1)
        for idx in range(6):
            (self.root / f"seg_{idx:04d}.pt").set_object(FakeTensor((4,), "uint8"))
        self.assertGreaterEqual(ram.spill_stats()["dropped"], 1)
        gate.set()
        ram.disable_spill()

    def test_spill_does_not_block_reads_or_writes(self):
        gate = threading.Event()

        def sink(base, name, payload, is_text):
            gate.wait(5)

        ram.enable_spill(sink, max_pending=2)
        (self.root / "seg_0000.pt").set_object(FakeTensor((4,), "uint8"))
        # RAM read must still answer while the worker is stuck in the sink.
        self.assertTrue((self.root / "seg_0000.pt").is_file())
        self.assertIsNotNone((self.root / "seg_0000.pt").get_object())
        gate.set()
        ram.disable_spill()


if __name__ == "__main__":
    unittest.main()
