# 分块（spatial tiling） × 模型级 denoise mask 的崩溃

> 现象：`RuntimeError: The size of tensor a (26) must match the size of tensor b (80) at non-singleton dimension 3`
> 位置：`comfy/ldm/minimax/model.py` → `out[0] = out[0] * denoise_mask`
> 触发组合：**SelfLift 开「空间分块」+ 段间引导 continue（keep-mask / 原生 inpaint）**

## 1. 这是谁的问题

不是暂存后端的问题（磁盘/内存暂存都一样），也不是画布或模型的问题，是
**分块采样器与 ComfyUI 0.42 的模型级 mask 机制不兼容**。原来的分块实现只裁了
"传给 inpaint 包装器的那份 mask"，没裁"编进 cond 的那份 mask"。

## 2. 形状是怎么对不上的

一次 tile 前向里有**两份** mask，来源完全不同：

| mask | 来源 | 形状 | 谁消费 |
| --- | --- | --- | --- |
| 采样器参数 `denoise_mask` | `KSampler.sample()`：`prepare_mask` 扩到 latent 通道数后 `pack_latents([video_mask, audio_mask])` | packed `(1,1,N)`，N 含 24 通道 | `KSamplerX0Inpaint.__call__`，用于 `x` / `out` 的混合 |
| **模型级 mask（cond）** | `process_conds` → `MiniMaxH3.extra_conds` → `_denoise_mask_conds`，在采样开始前算好一次 | `(1, 1, T, H, W)`，**整张画布** | `MiniMaxH3Model.forward`：`out[0] * denoise_mask` + `mask_row_values(...)` 推每行 timestep |

原来的 `_tiled_model_call` 只裁第一份：

```python
if denoise_mask is not None:
    tile_mask, _, _ = _crop_packed(denoise_mask, ...)   # 只影响 x/out 混合
```

而第二份留在 cond 里、尺寸是整张画布。于是 tile 的 `out[0]` 是
`(1, 24, T, 26, 46)`（H=26 的 tile），乘上 `(1, 1, T, 80, 46)`（H=80 的画布）
→ 报错里的 `a (26)` / `b (80)`。

日志里的形状完全对得上：HQ latent 是 `(24, 117, 80, 46)`，分块沿 H 切成
`[(0,26), (2,28), (12,38), …]`，所以 tile H=26、画布 H=80。

**为什么只有第 3 段炸**：keep-mask 只在 `prefix_steps > 0`（真的从前一段接到前缀）
时才装。第 2 段当时日志是 `Segment 2 continuity resolve failed ... will pin from AV
cache if present` → 没有前缀 → 没装 mask → 分块正常；第 3 段拿到了前一段结果 →
装上了 keep-mask → 撞车。

## 3. 修法（本分支已修）

`director/spatial_tiled_sampling.py`：

1. **跟着 latent 一起裁 cond mask**：`_install_tile_payloads` 现在会把
   `model_conds["denoise_mask"]` 按同一 tile 区间裁开（4D `(B,T,H,W)` / 5D
   `(B,1,T,H,W)` 都支持），并补成偶数尺寸（与 latent 的 `_pad_even_hw` 一致），
   调用结束后精确还原。音频 mask 不裁（音频不分块）。
2. **不能裁就拒绝分块，而不是猜**：形状对不上（换画布、未知布局）时抛
   `_TileMaskMisaligned`，该次采样退化为**整帧前向**并打印明确告警，
   同一阶段后续步不再重试分块。宁可慢，也不再把 26 和 80 乘在一起。
3. 采样器那份 mask（packed）裁剪失败时同样退化整帧，不再直接抛 reshape 错误。

回归测试：`tests/test_spatial_tile_mask.py`
（用真 torch 驱动 `_tiled_model_call`，假模型复现 `out[0] * denoise_mask`，
断言**每个 tile 的 cond mask 都必须与 tile latent 同高同宽**；另测外来形状走整帧回退）。

```bash
python -m unittest tests.test_spatial_tile_mask -v
```

## 4. 打补丁之前/之后的临时规避

不改代码时，任选其一即可绕开（都会掉到整帧前向或去掉 mask）：

* SelfLift 上**关掉「空间分块」**（`enable_tiling`）——最省事，但 736×1280 这种画布
  的显存/速度会明显变化；
* 关掉「段间引导」的 continue / 原生 inpaint 前缀（即不要 keep-mask）；
* 或只把该段单独跑（`选择运行`），让它没有可衔接的前一段。

修复后这三条都不再必要：分块和 keep-mask 可以同时用。
