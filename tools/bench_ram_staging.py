"""Measure the real cost of segment staging: disk (raw / FFV1) vs RAM.

The Director writes one frame payload + one AV latent + one audio payload per
segment. On the default ``MiniMaxH3Director`` node all three go to
``output/minimax_seg_cache/``; the ``(64GB RAM and above)`` variant keeps them as
live objects in :mod:`director.segment_cache_ram`. This script times both so you
can decide on *your* hardware instead of trusting a table.

Run it with the Python that runs ComfyUI (needs torch + numpy + ffmpeg):

    cd ComfyUI/custom_nodes/ComfyUI_MiniMaxH3_Director
    python tools/bench_ram_staging.py --segments 5

Nothing is written outside a temp dir, and no ComfyUI node is executed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402  (ComfyUI's python has it)

from director import segment_cache_ram as ram  # noqa: E402
from lib.frames_ffv1 import decode_frames_ffv1, encode_frames_ffv1  # noqa: E402


def _timed(fn, repeats: int) -> float:
    """Best-of-N wall time in milliseconds (best reflects the steady state)."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def _mb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 * 1024):.0f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=124, help="frames per segment")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--segments", type=int, default=5, help="segments to average over")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per payload")
    parser.add_argument("--no-ffv1", action="store_true", help="skip the FFV1 rows")
    args = parser.parse_args()

    frames = torch.randint(
        0, 256, (args.frames, args.height, args.width, 3), dtype=torch.uint8
    )
    latent_frames = max(1, (args.frames - 1) // 4 + 1)
    latent = {
        "samples": torch.randn(16, latent_frames, args.height // 8, args.width // 8),
        "audio_samples": torch.randn(1, 8, 128),
    }
    audio = {"waveform": torch.randn(1, 2, 32000 * 5), "sample_rate": 32000}

    frames_bytes = frames.numel()
    latent_bytes = sum(t.numel() * t.element_size() for t in latent.values())
    audio_bytes = sum(t.numel() * t.element_size() for t in audio.values() if torch.is_tensor(t))
    total_mb = (frames_bytes + latent_bytes + audio_bytes) / (1024 * 1024)

    print(
        f"segment: {args.frames}f {args.width}x{args.height} | "
        f"frames {_mb(frames_bytes)} + latent {_mb(latent_bytes)} + audio {_mb(audio_bytes)}"
        f" = {total_mb:.0f} MB resident"
    )
    print(f"segments averaged: {args.segments} | repeats: {args.repeats}\n")
    print(f"{'backend':<26}{'write ms':>10}{'read ms':>10}{'round trip ms':>16}")
    print("-" * 62)

    ram.configure(64)
    ram.reset()
    root = ram.root_for("ram:bench")

    with tempfile.TemporaryDirectory(prefix="mmx-bench-") as tmp:
        tmp_dir = Path(tmp)

        def disk_raw_write(idx: int) -> None:
            torch.save(frames, tmp_dir / f"seg_{idx}.pt")
            torch.save(latent, tmp_dir / f"seg_{idx}.av.pt")
            torch.save(audio, tmp_dir / f"seg_{idx}.audio.pt")

        def disk_raw_read(idx: int) -> None:
            torch.load(tmp_dir / f"seg_{idx}.pt", map_location="cpu", weights_only=True)
            torch.load(tmp_dir / f"seg_{idx}.av.pt", map_location="cpu", weights_only=False)
            torch.load(tmp_dir / f"seg_{idx}.audio.pt", map_location="cpu", weights_only=False)

        def ram_write(idx: int) -> None:
            (root / f"seg_{idx:04d}.pt").set_object(frames)
            (root / f"seg_{idx:04d}.av.pt").set_object(latent)
            (root / f"seg_{idx:04d}.audio.pt").set_object(audio)

        def ram_read(idx: int) -> None:
            (root / f"seg_{idx:04d}.pt").get_object()
            (root / f"seg_{idx:04d}.av.pt").get_object()
            (root / f"seg_{idx:04d}.audio.pt").get_object()

        rows = [
            ("disk / raw uint8 (.pt)", disk_raw_write, disk_raw_read),
            ("RAM (live tensors)", ram_write, ram_read),
        ]

        if not args.no_ffv1:
            def ffv1_write(idx: int) -> None:
                encode_frames_ffv1(tmp_dir / f"seg_{idx}.frames.mkv", frames, fps=24.0)
                torch.save(latent, tmp_dir / f"seg_{idx}.av.pt")
                torch.save(audio, tmp_dir / f"seg_{idx}.audio.pt")

            def ffv1_read(idx: int) -> None:
                decode_frames_ffv1(tmp_dir / f"seg_{idx}.frames.mkv")
                torch.load(tmp_dir / f"seg_{idx}.av.pt", map_location="cpu", weights_only=False)
                torch.load(tmp_dir / f"seg_{idx}.audio.pt", map_location="cpu", weights_only=False)

            rows.insert(1, ("disk / FFV1 frames", ffv1_write, ffv1_read))

        for label, writer, reader in rows:
            # Warm-up + a separate slot per repeat so nothing is served from cache.
            writer(1000)
            reader(1000)
            write_ms = _timed(lambda: [writer(i) for i in range(args.segments)], args.repeats)
            write_ms /= args.segments
            read_ms = _timed(lambda: [reader(i) for i in range(args.segments)], args.repeats)
            read_ms /= args.segments
            print(f"{label:<26}{write_ms:>10.1f}{read_ms:>10.1f}{write_ms + read_ms:>16.1f}")

    print(
        "\nround trip = one staging write + one read-back of the segment (what a"
        "\nre-run inside the same session pays per cached segment)."
        "\nA sampling pass for one segment is typically 30-180 s, so staging is a"
        "\nlow single-digit percentage of a run; RAM removes it entirely, at"
        f"\n~{total_mb / 1024:.2f} GB of RAM per segment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
