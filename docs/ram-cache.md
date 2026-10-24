# 内存暂存（RAM staging）— 可行性与设计

> 对应节点：**`MiniMaxH3Director (64GB RAM and above)`**（type id `MiniMaxH3DirectorRAM`）
> 分支：`feat/ram-segment-cache` · 默认行为完全不变（主节点仍写 `output/minimax_seg_cache/`）

## 1. 结论：可行，但要认清它省的是什么

原实现的分段暂存（`director/segment_cache.py`）每段会在
`output/minimax_seg_cache/<key>/` 下写四类文件：

| 文件 | 内容 | 0.4MP / 124 帧一段的体量 |
| --- | --- | --- |
| `seg_XXXX.frames.mkv` 或 `seg_XXXX.pt` | 解码后的像素帧（uint8） | **~154 MB**（FFV1 约 1/3） |
| `seg_XXXX.av.pt` | AV latent（CPU, fp32） | ~13 MB |
| `seg_XXXX.audio.pt` | 导出音频波形 | ~1.3 MB |
| `seg_XXXX.meta.json` / `.handoff.json` | 指纹与衔接元数据 | < 10 KB |

一段的总写入 + 回读（`torch.save`/`torch.load` 或 FFV1 编解码 + 磁盘 I/O）
在 NVMe 上大约 **0.3–1.0 s**；命中缓存时省掉的是"采样"（单段通常 30–180 s），
所以**暂存本身只占单段总时长的 1–3%**。

因此，把暂存搬到内存的真实收益是：

1. 每段一次写入 + 同段二次写入（相位对齐 trim 会重写整段）不再有 FFV1/torch.save 编解码；
2. 同一次会话内重跑（改一段提示词再 Queue）时，所有命中段零解码、零磁盘读；
3. 段间衔接 / 二采在同一进程内回读 `.pre` 帧时不再往返磁盘；
4. `output/` 在网络盘、机械盘、云挂载上时收益明显放大。

代价：

* 内存 **~0.17 GB / 段**（0.4MP、5 s）；30 段长片 ≈ 5 GB，与模型同驻内存；
* **进程退出即失效**——重启 ComfyUI 后等同空缓存，必须重新采样；
* 因此默认预算 8 GB（≈45 段）并带 LRU 淘汰，节点名标注 `(64GB RAM and above)`。

一句话：**它不是"让采样变快"，而是"把暂存的开销从磁盘和编解码里拿掉"。**
想要 5–30% 级别的整体提速，该动的是步数 / 分辨率 / 二采策略，不是暂存后端。
本分支的价值在于"同一张工作流可以立刻 A/B 对比"，以及慢盘环境下的稳定收益。

## 2. 实现方式（为什么改动这么小）

`segment_cache.py` 里所有缓存策略（指纹比对、stale 拒绝、prune / clear /
inspect、先确认一采的 `.pre` 挂起）都只有**一份**，没有为内存复制一套：

1. **key 前缀路由**：`resolve_segment_cache_key(..., backend="ram")` 返回
   `ram:<name>_<node_id>`。`:` 是 `normalize_cache_name()` 永远不会产生的字符，
   也会被 `_sanitize_dir_token()` 拒绝，所以 RAM key 不可能变成一个磁盘目录，
   反之亦然——路由是 key 的纯函数，没有全局开关、没有泄漏的注册表。
2. **Path 适配层**：`director/segment_cache_ram.py` 提供 `RamRoot` / `RamFile`，
   只实现 `segment_cache` 真正用到的那点 `pathlib` 接口
   （`is_file` / `is_dir` / `iterdir` / `glob` / `read_text` / `write_text` /
   `unlink` / `stat`）。`_cache_root()` / `_cache_dir_or_none()` 一处分支即可。
3. **零序列化**：`torch.save` / `torch.load` 收敛成 `_save_torch` / `_load_torch`
   两个薄封装——磁盘走原来的路径，内存直接保存 / 返回活对象（帧就是 uint8 张量，
   latent / 音频是 dict）。读出时做一次私有拷贝，防止采样侧原地改写污染暂存
   （磁盘路径天然免疫这一点，内存必须自己防）。
4. **LRU 按"段"淘汰**：预算按 GiB 计（默认 8，`MINIMAX_DIRECTOR_RAM_CACHE_GB`
   或节点控件可改，夹在 1–512 GiB），淘汰以 `seg_XXXX` / `seg_XXXX.pre` 为组，
   不会出现"帧在、meta 没了"的半段状态。预算对"单段就超预算"的情况是软上限：
   刚写入的那一段永远保留（否则 1088p 长片在低预算下等于没有缓存），超出量最多一段。

改到 `segment_cache.py` 的只有：key 解析、`_cache_root` / `_cache_dir_or_none`
路由、`_write_via_temp` 的内存分支、两个 torch 读写的薄封装、帧存储的 FFV1 跳过。
指纹 / stale / prune / inspect 逻辑一行没动，所以后续跟上游合并冲突面很小。

## 3. 用法

1. 同一个自定义节点目录里拉到本分支（或直接复制 `director/segment_cache_ram.py`
   + 相关改动），重启 ComfyUI。
2. 右侧节点列表 `MiniMaxH3 → MiniMaxH3Director (64GB RAM and above)`。
   控件和主节点完全一致，多一个 **`ram_cache_budget_gb`**（性能组最下方）。
3. 时间轴、Refine、SelfLift、Semantic Bridge、导演包导入导出都照常接——
   前端把该类型当作 Director 处理。

### A/B 对比同一张工作流

* 复制一份工作流，把导演台节点换成 `(64GB RAM and above)` 版本即可；
  两者**不共享缓存**（磁盘 key 是 `hero_57`，内存 key 是 `ram:hero_57`），
  所以不会互相"加热"结果。
* 想量化暂存本身的耗时（不需要跑采样）：

  ```bash
  cd ComfyUI/custom_nodes/ComfyUI_MiniMaxH3_Director
  python tools/bench_ram_staging.py --segments 5
  ```

  输出 disk/raw、disk/FFV1、RAM 三种后端的"写入 / 回读 / 往返"毫秒数。

### 预算换算

| 画布 / 帧数 | 每段内存 | 8 GB 可留 |
| --- | --- | --- |
| 864×480 / 124（默认） | ~0.17 GB | ~45 段 |
| 1280×720 / 124 | ~0.37 GB | ~21 段 |
| 1920×1088 / 243（放大后） | ~1.4 GB | ~5 段 |

`ram_cache_budget_gb` 是**进程级全局**预算（同进程所有 RAM 导演台共享），
设得比可用内存大没有意义：ComfyUI 还需要内存放模型（Qwen3-VL 32B + UNET + VAE）。

## 4. 边界与已知行为

* **进程退出即失效**：RAM 暂存不落盘。重启后第一次 Queue 等同空缓存。
  需要跨重启复用就继续用主节点（磁盘暂存）。
* **先确认一采（confirm_first_pass）**：第一次 Queue 只出 `images_pre_refine`，
  第二次 Queue 命中内存里的 `.pre` 槽继续二采——两次必须在**同一个 ComfyUI 进程**内。
  `IS_CHANGED` 的签名也按 RAM key 取值，所以第二次 Queue 会正常重跑。
* **`cache_frames_codec` 无效**：内存暂存直接保存张量，没有编解码；控件保留只为
  和主节点控件布局一致。
* **缓存状态面板 / 清空缓存**：Refine 的「先确认一采」卡片与「清空分段缓存」
  会带 `backend: "ram"` 请求，因此看到的是内存暂存；ComfyUI 重启后两者都显示为空。
* **显存无关**：这是 CPU 内存，不是 VRAM。`clear_vram_between_segments` 等开关不受影响。
* **多进程**：ComfyUI 若以多 worker 方式部署，各进程内存互不可见；
  此时请用主节点（磁盘暂存）。

## 5. 测试

```bash
python3 -m unittest tests.test_segment_cache_ram -v
```

* `RamStoreTests`：Path 适配层、预算 / 按段淘汰、LRU 触碰、拷贝隔离、统计与清零；
* `SegmentCacheRamPlumbingTests`：用 torch / folder_paths 的桩件导入真实的
  `segment_cache`，验证 key 前缀、`_cache_root` 路由、帧槽往返（并断言绝不调用
  FFV1）、`_write_via_temp` / `_save_torch` 内存路径、`prune` / `clear` /
  `.pre` 签名。指纹比对本身需要真实的 plan/refine/selflift 模块，不在桩件测试范围内。
