"""MiniMax H3 Director — RAM staging variant（64GB RAM and above）.

Identical timeline / sampling / refine behaviour as :class:`MiniMaxH3Director`,
but the per-segment staging cache lives in **process memory** instead of
``output/minimax_seg_cache/``. See ``director/segment_cache_ram.py`` for the
store and ``docs/ram-cache.md`` for the measurements and trade-offs.

Why a separate node type instead of a widget on the main node:

* the two staging backends need different cache keys (``ram:`` prefix), and a
  widget would silently move an existing workflow between stores;
* a distinct type id / display name lets both nodes sit in the same install so
  the same workflow can be timed with and without RAM staging
  (registering the same type id twice is not possible in ComfyUI anyway).
"""

from __future__ import annotations

from .director import MiniMaxH3Director

_RAM_WIDGET = "ram_cache_budget_gb"
_SPILL_WIDGET = "ram_cache_spill_to_disk"


class MiniMaxH3DirectorRAM(MiniMaxH3Director):
    """Director with in-process RAM segment staging (see docs/ram-cache.md)."""

    CACHE_BACKEND = "ram"
    RAM_CACHE_BUDGET_GB = 8
    RAM_CACHE_SPILL = False

    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        required = data["required"]

        codec_type, codec_meta = required["cache_frames_codec"]
        codec_meta = dict(codec_meta)
        codec_meta["tooltip"] = (
            "内存暂存不使用：RAM 暂存直接保存 uint8 帧张量，没有编解码，"
            "本节点里这个下拉框只为了控件布局与主节点一致，选什么都一样。"
        )
        required["cache_frames_codec"] = (codec_type, codec_meta)

        # Appended last on purpose: the frontend restores widgets_values by
        # position, so a new node type must never insert in the middle.
        required[_RAM_WIDGET] = (
            "INT",
            {
                "default": cls.RAM_CACHE_BUDGET_GB,
                "min": 1,
                "max": 512,
                "step": 1,
                "tooltip": (
                    "内存暂存预算（GB，进程内全局，LRU 按段淘汰）。"
                    "0.4MP / 124 帧的一段约 0.16 GB（uint8 帧）+ 十几 MB latent/音频，"
                    "8 GB ≈ 45 段。设得比物理内存大没有用，只会挤掉模型换页。"
                    "也可用环境变量 MINIMAX_DIRECTOR_RAM_CACHE_GB 设默认值。"
                ),
            },
        )
        required[_SPILL_WIDGET] = (
            "BOOLEAN",
            {
                "default": cls.RAM_CACHE_SPILL,
                "tooltip": (
                    "顺便落盘（后台线程）：读永远是内存命中，写入同时由后台线程补一份到 "
                    "output/minimax_seg_cache/（和主节点同一个目录）。"
                    "好处是关掉 ComfyUI 再开还能续跑；代价是多一遍磁盘写入"
                    "（不挡采样，但会占 CPU/IO，慢盘上可能拖慢一点）。"
                    "关掉 = 纯内存，进程退出即失效。"
                ),
            },
        )
        return data

    def execute(self, *args, **kwargs):
        # ComfyUI passes inputs by keyword; these are node-level knobs that the
        # base class reads off the instance (CACHE_BACKEND is set on the class).
        budget = kwargs.pop(_RAM_WIDGET, self.RAM_CACHE_BUDGET_GB)
        try:
            self.RAM_CACHE_BUDGET_GB = max(0, int(budget))
        except (TypeError, ValueError):
            self.RAM_CACHE_BUDGET_GB = 8
        self.RAM_CACHE_SPILL = bool(kwargs.pop(_SPILL_WIDGET, self.RAM_CACHE_SPILL))
        return super().execute(*args, **kwargs)

    DESCRIPTION = (
        "MiniMax H3 Director with in-process RAM segment staging."
        " Same timeline / sampling / refine chain as MiniMaxH3Director, but the"
        " per-segment staging cache (decoded frames + AV latents + audio) lives in"
        " memory instead of output/minimax_seg_cache/, so a re-run inside the same"
        " ComfyUI session needs no disk read, no FFV1 decode and no re-encode."
        " Cost: ~0.16 GB RAM per 5s 0.4MP segment (LRU budget, default 8 GB)."
        " Reads always come from RAM; ram_cache_spill_to_disk can mirror writes to"
        " disk in the background so a restart resumes instead of re-sampling."
        " Intended for 64GB RAM and above."
    )
