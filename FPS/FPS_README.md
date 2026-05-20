# FPS 实验结果总览

更新时间：2026-05-18

本文件汇总当前 LiteViLNet 仓库中的 FPS 实验结果。核心原则是：不同设备、系统、后端、精度、测试范围不能混成同一列。论文表格中必须注明 device、OS、backend、precision、resolution、batch size，以及是否包含 preprocessing。

## 统一口径

| 项目 | 设置 |
|---|---|
| 输入分辨率 | `384x1248` |
| Batch size | `1` |
| 默认精度 | FP16 或 mixed FP16 |
| 默认测试范围 | model-only random tensors |
| 默认不包含 | resize、depth/normal/LiDAR/ADI 预处理、CPU 到 GPU 拷贝、可视化、mask 保存、视频生成 |
| RTX 4060 Ti PyTorch runs | 3 runs |

## 论文主表推荐

如果论文中保留两列 FPS，建议写成：

```text
FPS-1: RTX 4060 Ti, Linux, PyTorch FP16, 384x1248, batch size 1, model-only, preprocessing excluded
FPS-2: Jetson Orin NX, FP16/mixed FP16, 384x1248, batch size 1, model-only, preprocessing excluded
```

当前推荐主表使用下面这张表。注意：RTX 4060 Ti 与 Jetson 列均已更新为 2026-05-18 的 checkpoint-loaded model-only 结果。

| Model | RTX 4060 Ti FPS-1 | Jetson FPS-2 | Params | RTX checkpoint 状态 | Jetson 状态 | 主表建议 |
|---|---:|---:|---:|---|---|---|
| LiteViLNetRGBDepth | 163.79 | 22.18 ± 0.21 | 14.04M | robot checkpoint | robot checkpoint | 可作为本文主模型速度 |
| LiteViLNet-Paper | 159.54 | 22.19 ± 0.12 | 14.04M | paper checkpoint compatible，aux heads ignored | paper checkpoint | 可作为论文 preset 速度 |
| USNet | 104.43 | 6.26 ± 0.00 | 30.74M | author checkpoint loaded_strict | author checkpoint loaded | 可作为主要对比模型 |
| PLiDAR | 17.10 | 3.52 ± 0.00 | 76.93M | checkpoint loaded_strict | checkpoint loaded_strict | 可作为 checkpoint FPS |
| SNE-RoadSeg | 13.33 | 2.70 ± 0.01 | 132.06M | official checkpoint compatible wrapper loaded_strict | official checkpoint compatible wrapper loaded_strict | 可作为 checkpoint FPS；保留 wrapper 说明 |
| LRDNet | 9.28 | 1.73 ± 0.01 | 28.57M | `LRDNet+.hdf5` loaded | `LRDNet+.hdf5` loaded | 可作为 checkpoint FPS |

说明：

- RTX 4060 Ti 的 LiteViLNet、USNet、PLARD/PLiDAR、SNE-RoadSeg 来自 2026-05-18 Linux PyTorch FP16 checkpoint-loaded 测试。
- RTX 4060 Ti 的 LRDNet 来自 2026-05-18 Linux TensorFlow mixed FP16，已加载 `LRDNet+.hdf5`。
- LiteViLNet 两个模型在全模型测试后又单独复测，主表采用单独复测值。
- SNE-RoadSeg 官方 fresh 源码和官方权重不完全一致；本次用 compatible wrapper 匹配权重 key 后 strict load。该权重实际对应 ResNet-50 block pattern 和 `final_0` 到 `final_4` 输出头，参数量为 `132.06M`。
- PLARD 即论文表中的 PLiDAR/PLARD 实现，当前 RTX 4060 Ti 结果已使用 `plard_kitti_road.pth` strict load。
- Jetson 上 PLiDAR、SNE-RoadSeg、LRDNet 已于 2026-05-18 复测为 checkpoint-loaded model-only 结果，可替换旧的 architecture-only / load-failed 记录。

## 部署加速结果

TensorRT 是部署加速后端，不能和 PyTorch FPS 混写到同一列。建议单独列出或在正文单独说明。

| Model | Device | Backend | FPS | Throughput | Mean Latency | Engine / Params | Status |
|---|---|---|---:|---:|---:|---:|---|
| LiteViLNetRGBDepth | RTX 4060 Ti Linux | TensorRT FP16 | 358.03 | 437.28 qps | 2.793 ms | 39.43 MB engine | engine built successfully |
| LiteViLNetRGBDepth | Jetson Orin NX | TensorRT FP16 | 68.73 ± 0.06 | - | 14.55 ms | 30.6 MB engine | robot checkpoint |

建议写法：

```text
On RTX 4060 Ti, LiteViLNetRGBDepth reaches 163.79 FPS with PyTorch FP16 model-only benchmarking and 437.28 qps with TensorRT FP16. On Jetson Orin NX, it reaches 22.18 FPS with PyTorch FP16 and 68.73 FPS with TensorRT FP16.
```

## RTX 4060 Ti Linux 详情

### PyTorch FP16 checkpoint-final

环境：RTX 4060 Ti 16GB，Linux，PyTorch `2.1.0+cu121`，CUDA `12.1`，cuDNN `8902`，FP16，batch size 1，`384x1248`，3 runs，model-only，preprocessing excluded。

| Model | FPS | Trimmed FPS | Mean latency | P50 latency | P95 latency | Params | Checkpoint / Status |
|---|---:|---:|---:|---:|---:|---:|---|
| LiteViLNetRGBDepth | 163.79 | 164.33 | 6.107 ms | 6.054 ms | 6.560 ms | 14.04M | robot checkpoint |
| LiteViLNet-Paper | 159.54 | 160.19 | 6.270 ms | 6.203 ms | 6.839 ms | 14.04M | loaded compatible，aux heads ignored |
| USNet | 104.43 | 104.84 | 9.592 ms | 9.430 ms | 10.771 ms | 30.74M | loaded_strict |
| PLARD / PLiDAR | 17.10 | 17.14 | 58.554 ms | 57.683 ms | 63.292 ms | 76.93M | loaded_strict |
| SNE-RoadSeg | 13.33 | 13.35 | 75.141 ms | 74.333 ms | 80.745 ms | 132.06M | loaded_strict via compatible wrapper |

结果来源：

```text
F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_final_4060ti_20260518\pytorch_model_only\
F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_litevilnet_retest_20260518_2models\
```

### TensorFlow / LRDNet

环境：RTX 4060 Ti 16GB，Linux，TensorFlow `2.15.1`，Python virtualenv `/home/peter/.venvs/lrdnet_tf`，mixed FP16，batch size 1，`384x1248`，3 runs，warmup / iters = `10 / 30`，model-only，preprocessing excluded。

| Model | FPS | Trimmed FPS | Mean latency | P50 latency | P95 latency | Params | Checkpoint / Status |
|---|---:|---:|---:|---:|---:|---:|---|
| LRDNet | 9.28 | 9.29 | 107.78 ms | 106.01 ms | 113.34 ms | 28.57M | `LRDNet+.hdf5` loaded |

说明：TensorFlow 已识别 Linux 下的 RTX 4060 Ti GPU，3 次 run 均为 `status=ok`。本结果为 checkpoint-loaded model-only 前向速度，不包含 ADI/LiDAR 预处理。

结果来源：

```text
F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_lrdnet_tf_4060ti\
```

### TensorRT

环境：RTX 4060 Ti 16GB，Linux，TensorRT `8.6.1`，FP16 engine，`trtexec`，batch size 1，`384x1248`，model-only，preprocessing excluded。本轮 `trtexec` 不在 PATH，沿用 2026-05-17 TensorRT 结果。

| Model | Conservative FPS | trtexec Throughput | Mean Host Latency | GPU Compute Mean | Engine | Status |
|---|---:|---:|---:|---:|---:|---|
| LiteViLNetRGBDepth | 358.03 | 437.28 qps | 2.793 ms | 2.285 ms | 39.43 MB | engine built successfully |

说明：`358.03 FPS` 来自 `1000 / mean host latency`，比 `437.28 qps` 更保守。若论文报告 TensorRT，建议使用保守 FPS，同时在备注中给出 throughput。

## Jetson 详情

环境：Jetson Orin NX，`MAXN_SUPER` 性能模式，FP16 或 mixed FP16，batch size 1，`384x1248`，model-only，preprocessing excluded。

| Model / Config | Backend | Input | FPS | Mean latency | P95 latency | Params / Engine | Status |
|---|---|---|---:|---:|---:|---:|---|
| LiteViLNetRGBDepth robot checkpoint | Jetson TensorRT FP16 | RGB + Depth3 | 68.73 ± 0.06 | 14.55 ms | 14.74 ms | 30.6 MB engine | robot checkpoint |
| LiteViLNetRGBDepth robot checkpoint | Jetson PyTorch FP16 | RGB + Depth3 | 22.18 ± 0.21 | 45.09 ms | 46.36 ms | 14.04M | robot checkpoint |
| LiteViLNet-Paper | Jetson PyTorch FP16 | RGB + LiDAR/ADI | 22.19 ± 0.12 | 45.07 ms | 46.65 ms | 14.04M | paper preset checkpoint |
| USNet | Jetson PyTorch FP16 | RGB + Depth | 6.26 ± 0.00 | 159.68 ms | 159.90 ms | 30.74M | author checkpoint loaded |
| PLiDAR | Jetson PyTorch FP16 | RGB + LiDAR/ADI | 3.52 ± 0.00 | 283.90 ms | 284.61 ms | 76.93M | checkpoint loaded_strict |
| SNE-RoadSeg | Jetson PyTorch FP16 | RGB + Normal | 2.70 ± 0.01 | 370.94 ms | 373.48 ms | 132.06M | official checkpoint compatible wrapper loaded_strict |
| LRDNet | Jetson TensorFlow mixed FP16 | RGB + LiDAR/ADI | 1.73 ± 0.01 | 579.63 ms | 593.24 ms | 28.57M | `LRDNet+.hdf5` loaded |

结果来源：

```text
tools/fps_jetson/results/
F:\LiteViLNet_fps_4060ti\jetson_checkpoint_retest_20260518_161335\
```

## Windows / WSL 历史结果

这些结果保留用于追溯，不建议和 Linux PyTorch 主表混用。Windows 与 Linux 差距较大，大概率不是同一测试口径。

| Model / Config | Backend | FPS | P50 latency | Params | 说明 |
|---|---|---:|---:|---:|---|
| USNet | Windows PyTorch FP16 | 67.78 ± 4.97 | 14.46 ms | 30.74M | model-only |
| USNet + GPU-fast SNE | Windows PyTorch FP16 | 66.83 ± 2.10 | 14.64 ms | 30.74M | end-to-end，GPU 向量化 SNE |
| USNet + original CPU SNE | Windows PyTorch FP16 | 9.15 ± 0.44 | 108.77 ms | 30.74M | end-to-end，更接近原始 USNet 预处理口径 |
| LiteViLNetRGBDepth | Windows PyTorch FP16 | 52.85 ± 3.20 | 18.30 ms | 14.04M | model-only，旧口径待核 |
| LiteViLNetRGBDepth + depth3 encode | Windows PyTorch FP16 | 48.21 ± 6.77 | 18.71 ms | 14.04M | end-to-end，旧口径待核 |
| LiteViLNet-Paper | Windows PyTorch FP16 | 44.60 ± 4.18 | 22.09 ms | 14.04M | model-only，旧口径待核 |
| LiteViLNet-Paper + depth3 encode | Windows PyTorch FP16 | 54.20 ± 3.14 | 17.88 ms | 14.04M | end-to-end，旧口径待核 |
| PLARD | Windows PyTorch FP16 | 16.31 ± 0.84 | 60.46 ms | 76.93M | architecture-only 旧结果 |
| SNE-RoadSeg | Windows PyTorch FP16 | 11.14 ± 0.18 | 88.78 ms | 201.32M | checkpoint load failed，architecture-only 旧结果 |
| LRDNet | WSL TensorFlow FP16 | 4.26 ± 0.07 | 228.54 ms | 28.57M | architecture-only 旧记录 |

## 原始结果文件

| 类别 | 文件 |
|---|---|
| 2026-05-18 PyTorch checkpoint 总报告 | `F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_final_4060ti_20260518\pytorch_model_only\summary_report.md` |
| 2026-05-18 PyTorch checkpoint CSV | `F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_final_4060ti_20260518\pytorch_model_only\aggregate_fps_summary.csv` |
| 2026-05-18 Lite 两模型单独复测报告 | `F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_litevilnet_retest_20260518_2models\summary_report.md` |
| LRDNet TensorFlow checkpoint 报告 | `F:\LiteViLNet_fps_4060ti\tools\fps_4060ti\results\linux_lrdnet_tf_4060ti\summary_report.md` |
| Jetson checkpoint-loaded 复测报告 | `F:\LiteViLNet_fps_4060ti\jetson_checkpoint_retest_20260518_161335\JETSON_CHECKPOINT_SUMMARY.md` |
| LiteViLNet TensorRT JSON | `F:\LiteViLNet_fps_4060ti\results\litevilnet_rgbdepth_tensorrt_fp16_linux.json` |

## 使用注意

- 主论文表优先使用“论文主表推荐”中的同口径结果。
- TensorRT 结果只能说明部署加速效果，不能和 PyTorch model-only FPS 放在同一列直接比较。
- Windows PyTorch、Linux PyTorch、Linux TensorRT、Jetson TensorRT 不要混写。
- RTX 4060 Ti 和 Jetson 上 PLiDAR、SNE-RoadSeg、LRDNet 均已有 checkpoint-loaded model-only 结果；SNE-RoadSeg 需注明 compatible wrapper。
- PLiDAR 与 PLARD 按同一实现处理，论文命名建议使用 `PLiDAR`。

## 结论

| 结论项 | 数值 |
|---|---:|
| LiteViLNetRGBDepth Linux PyTorch FP16 model-only | 163.79 FPS |
| LiteViLNet-Paper Linux PyTorch FP16 model-only | 159.54 FPS |
| USNet Linux PyTorch FP16 checkpoint-final model-only | 104.43 FPS |
| PLARD / PLiDAR Linux PyTorch FP16 checkpoint-final model-only | 17.10 FPS |
| SNE-RoadSeg Linux PyTorch FP16 official compatible checkpoint model-only | 13.33 FPS |
| LRDNet Linux TensorFlow FP16 checkpoint model-only | 9.28 FPS |
| PLiDAR Jetson PyTorch FP16 checkpoint model-only | 3.52 FPS |
| SNE-RoadSeg Jetson PyTorch FP16 official compatible checkpoint model-only | 2.70 FPS |
| LRDNet Jetson TensorFlow FP16 checkpoint model-only | 1.73 FPS |
| LiteViLNetRGBDepth Linux TensorRT FP16 mean latency | 2.793 ms |
| LiteViLNetRGBDepth Linux TensorRT FP16 latency-FPS | 358.03 FPS |
| LiteViLNetRGBDepth Linux TensorRT FP16 throughput | 437.28 qps |

论文或报告中不要混用 Windows、Linux、PyTorch、TensorRT、model-only、end-to-end 的 FPS。每张表最好只放同一平台、同一后端、同一精度、同一测试范围的数据。
