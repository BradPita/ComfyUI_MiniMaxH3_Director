"""Regression tests for spatial tiling × model-level denoise mask.

Reported crash (ComfyUI 0.42.0, SelfLift tiling + 段间引导 continue):

    RuntimeError: The size of tensor a (26) must match the size of tensor b (80)
    at non-singleton dimension 3
    ... comfy/ldm/minimax/model.py: out[0] = out[0] * denoise_mask

Root cause: ``process_conds`` bakes the per-step mask into the conditioning once
(``MiniMaxH3._denoise_mask_conds`` → a ``(1, 1, T, H, W)`` video mask on the full
canvas) and H3's forward multiplies the *tile* latent by it. Cropping only the
mask handed to the inpaint wrapper (``_tiled_model_call(denoise_mask=...)``) left
the cond mask at canvas size → 26-row tile × 80-row mask.

These tests drive ``_tiled_model_call`` with a fake model that reproduces the
model-side multiply, and assert the invariant that crashed: for every tile the
cond mask must be cropped exactly like the latent.

Run: python -m unittest tests.test_spatial_tile_mask -v
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

try:  # real ComfyUI when the test runs inside a ComfyUI install
    import comfy.utils  # noqa: F401
except Exception:  # pragma: no cover - standalone checkout
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    def pack_latents(latents):
        shapes = [tuple(t.shape) for t in latents]
        return torch.cat([t.reshape(t.shape[0], 1, -1) for t in latents], dim=-1), shapes

    def unpack_latents(combined, shapes):
        out = []
        for shape in shapes:
            cut = math.prod(shape[1:])
            chunk = combined[:, :, :cut]
            combined = combined[:, :, cut:]
            out.append(chunk.reshape([chunk.shape[0]] + list(shape)[1:]))
        return out

    utils.pack_latents = pack_latents
    utils.unpack_latents = unpack_latents
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils

from director import spatial_tiled_sampling as tiles  # noqa: E402


class _Cond:
    """Stand-in for comfy.conds.CONDConstant / CONDRegular."""

    def __init__(self, cond):
        self.cond = cond


class _FakeModel:
    """Minimal KSamplerX0Inpaint: records every forward and its mask shapes."""

    def __init__(self, guider):
        self.inner_model = guider
        self.latent_image = None
        self.noise = None
        self.calls = []  # (video H, video W, cond mask H, cond mask W, arg mask H)

    def __call__(self, x, sigma, denoise_mask=None, **_kwargs):
        base = self.inner_model.inner_model
        shapes = base.latent_shapes
        video, audio = comfy.utils.unpack_latents(x, shapes)
        cond_mask = _cond_mask(self.inner_model)
        arg_h = None
        if torch.is_tensor(denoise_mask):
            # the wrapper mask arrives packed (video + audio), like ComfyUI's
            arg_h = int(comfy.utils.unpack_latents(denoise_mask, shapes)[0].shape[-2])
        self.calls.append(
            (
                int(video.shape[-2]),
                int(video.shape[-1]),
                int(cond_mask.shape[-2]) if cond_mask is not None else None,
                int(cond_mask.shape[-1]) if cond_mask is not None else None,
                arg_h,
            )
        )
        return comfy.utils.pack_latents([video * 0 + 1.0, audio])[0]


def _cond_mask(guider):
    for group in (guider.conds or {}).values():
        for cond in group:
            entry = cond[1] if isinstance(cond, (list, tuple)) else cond
            wrapper = (entry.get("model_conds") or {}).get("denoise_mask")
            if wrapper is not None:
                return wrapper.cond
    return None


def _guider(video_shape, audio_shape, mask=None):
    model_conds = {"latent_shapes": _Cond([video_shape, audio_shape])}
    if mask is not None:
        model_conds["denoise_mask"] = _Cond(mask)
    base = types.SimpleNamespace(latent_shapes=[video_shape, audio_shape])
    return types.SimpleNamespace(
        conds={"positive": [[torch.zeros(1, 4, 8), {"model_conds": model_conds}]]},
        inner_model=base,
    )


class CropVideoMaskTests(unittest.TestCase):
    def test_crops_5d_canvas_mask(self):
        mask = torch.ones(1, 1, 4, 80, 46)
        cropped, did = tiles._crop_video_mask(mask, "H", 0, 26, full_h=80, full_w=46)
        self.assertTrue(did)
        self.assertEqual(tuple(cropped.shape), (1, 1, 4, 26, 46))

    def test_crops_4d_canvas_mask(self):
        mask = torch.ones(1, 4, 80, 46)
        cropped, did = tiles._crop_video_mask(mask, "H", 22, 48, full_h=80, full_w=46)
        self.assertTrue(did)
        self.assertEqual(tuple(cropped.shape), (1, 4, 26, 46))

    def test_crops_along_width(self):
        mask = torch.ones(1, 1, 4, 46, 80)
        cropped, did = tiles._crop_video_mask(mask, "W", 52, 78, full_h=46, full_w=80)
        self.assertTrue(did)
        self.assertEqual(tuple(cropped.shape), (1, 1, 4, 46, 26))

    def test_pads_odd_tiles_like_the_latent(self):
        mask = torch.ones(1, 1, 4, 80, 46)
        cropped, did = tiles._crop_video_mask(mask, "H", 0, 25, full_h=80, full_w=46)
        self.assertTrue(did)
        self.assertEqual(int(cropped.shape[-2]), 26)  # 25 → padded to even

    def test_rejects_foreign_shape_instead_of_guessing(self):
        mask = torch.ones(1, 1, 4, 50, 46)
        _, did = tiles._crop_video_mask(mask, "H", 0, 26, full_h=80, full_w=46)
        self.assertFalse(did)

        nested_like = object()
        _, did = tiles._crop_video_mask(nested_like, "H", 0, 26, full_h=80, full_w=46)
        self.assertFalse(did)


class TiledMaskIntegrationTests(unittest.TestCase):
    """Drives the real tiler with a fake model that checks mask alignment."""

    def _run(self, mask, *, n_tiles=4, overlap=2):
        video_shape = (1, 24, 4, 80, 46)
        audio_shape = (1, 8, 128)
        guider = _guider(video_shape, audio_shape, mask)
        model = _FakeModel(guider)
        packed_x, shapes = comfy.utils.pack_latents(
            [torch.ones(*video_shape), torch.ones(*audio_shape)]
        )
        if mask is not None:
            # The sampler always hands the tiler a canvas-sized wrapper mask:
            # prepare_mask expands it to the latent channel count
            # (utils.reshape_mask) and pack_latents joins [video, audio].
            canvas_mask = torch.ones(1, 1, 4, 80, 46).expand(-1, video_shape[1], -1, -1, -1)
            arg_mask, _ = comfy.utils.pack_latents(
                [canvas_mask, torch.ones(*audio_shape)]
            )
        else:
            arg_mask = None
        out = tiles._tiled_model_call(
            model,
            packed_x,
            torch.tensor(0.5),
            denoise_mask=arg_mask,
            n_tiles=n_tiles,
            overlap_pixels=overlap * tiles.VAE_SPATIAL,
        )
        return model, guider, out

    def test_every_tile_sees_a_cond_mask_cropped_like_its_latent(self):
        mask = torch.ones(1, 1, 4, 80, 46)
        model, guider, _out = self._run(mask)
        self.assertGreater(len(model.calls), 1, "tiling must actually happen")
        for h, w, mask_h, mask_w, arg_h in model.calls:
            self.assertEqual(mask_h, h, f"cond mask H {mask_h} != tile H {h}")
            self.assertEqual(mask_w, w, f"cond mask W {mask_w} != tile W {w}")
            self.assertEqual(arg_h, h, f"call mask H {arg_h} != tile H {h}")
        # The cond is restored to the full canvas after the step.
        self.assertEqual(
            tuple(_cond_mask(guider).shape), (1, 1, 4, 80, 46)
        )

    def test_two_tiles_also_stay_aligned(self):
        mask = torch.ones(1, 1, 4, 80, 46)
        model, _guider_obj, _out = self._run(mask, n_tiles=2)
        self.assertGreater(len(model.calls), 1)
        for h, _w, mask_h, _mask_w, arg_h in model.calls:
            self.assertEqual(mask_h, h)
            self.assertEqual(arg_h, h)

    def test_unusable_mask_falls_back_to_a_full_frame_forward(self):
        # e.g. a mask on a different canvas (canvas change mid-run) — the pass
        # must degrade to one full-frame forward and never multiply 26 by 80.
        mask = torch.ones(1, 1, 4, 50, 46)
        model, _guider_obj, _out = self._run(mask)
        self.assertEqual(len(model.calls), 1, "fallback must run exactly one full frame")
        h, _w, mask_h, _mask_w, arg_h = model.calls[0]
        self.assertEqual(h, 80)       # full canvas, not a tile
        self.assertEqual(mask_h, 50)  # cond mask untouched
        self.assertEqual(arg_h, 80)   # wrapper mask stays on the canvas too
        self.assertTrue(model._director_spatial_plan.get("disabled"))

    def test_no_mask_tiles_as_before(self):
        model, _guider_obj, _out = self._run(None)
        self.assertGreater(len(model.calls), 1)
        for h, _w, mask_h, _mask_w, arg_h in model.calls:
            self.assertLess(h, 80)  # a real tile
            self.assertIsNone(mask_h)
            self.assertIsNone(arg_h)


if __name__ == "__main__":
    unittest.main()
