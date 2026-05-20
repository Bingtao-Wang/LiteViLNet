# LiteViLNet IEEE RAL 实验计划

## 结论

LiteViLNet 应该采用双主线实验策略：

1. 精度评价必须对齐 KITTI Road 风格指标：`MaxF`、`AP`、`PRE`、`REC`、`FPR`、`FNR`。
2. 部署价值必须在 Jetson 上被证明：参数量、FLOPs、模型大小、TensorRT 延迟、FPS、显存、功耗、单帧能耗。

当前最优先的工作不是继续改模型结构，而是先稳定评价协议。评价链路不稳定，后续模型优化结果就无法成为可靠的论文证据。

## 阶段 1：评价指标统一

目标：仓库里所有 `MaxF` 都必须表示 threshold-swept maximum F1，即扫描多个阈值后取得的最大 F1，并与 KITTI Road 指标习惯保持一致。

任务：

- 重构 `BinarySegmentationMeter`。
- 输出 `MaxF`、`AP`、`PRE`、`REC`、`FPR`、`FNR`、`BestThreshold`。
- 从主实验链路中移除 `F1@0.5`。
- 训练和验证阶段都使用 swept `MaxF` 选择 best checkpoint。
- 更新以下脚本：
  - `tools/train.py`
  - `tools/train_distill_edge.py`
  - `tools/train_ablation.py`
  - `tools/evaluate.py`
- 确保 CSV 和 JSON 输出字段使用 KITTI-style 命名。
- 明确说明本地验证是 image-space validation，不等同于 KITTI 官方 BEV 评价。

目标日志格式：

```text
MaxF / AP / PRE / REC / FPR / FNR / BestThreshold
```

## 阶段 2：KITTI 官方评估链路

目标：生成可以用于 KITTI Road 官方服务器或兼容 devkit 评价的预测结果。

任务：

- 增加预测结果导出工具：

```text
tools/export_kitti_predictions.py
```

- 将预测 mask 输出到：

```text
runs/kitti_submission/<preset>/
```

- 增加评价协议文档：

```text
docs/ral/EVALUATION_PROTOCOL.md
```

- 在文档中明确区分：
  - 本地验证指标
  - KITTI 官方/devkit 指标
  - 论文最终表格指标

论文规则：

```text
KITTI Road 官方结果应作为最终论文级精度结果。
```

## 阶段 3：Jetson 部署链路

目标：证明 LiteViLNet 的轻量化部署价值，而不是只证明参数量更小。

必须记录的部署指标：

- 参数量
- FLOPs
- PyTorch checkpoint 大小
- ONNX 模型大小
- TensorRT engine 大小
- 平均延迟
- P95 延迟
- FPS
- GPU 显存
- 功耗
- 单帧能耗

需要标准化的部署工具：

- `tools/export_onnx.py`
- `tools/build_tensorrt.py`
- `tools/benchmark_tensorrt.py`
- `tools/benchmark_pytorch.py`
- `tools/collect_jetson_power.py`

建议的汇总输出：

```text
runs/summary/accuracy.csv
runs/summary/deployment.csv
runs/summary/ral_table.csv
```

部署对比对象：

- `litevilnet_paper`
- `litevilnet_edge`
- `litevilnet_baseline`
- LiteViLNet variants

## 阶段 4：轻量化模型路线

目标：让轻量化贡献变得明确、可测量、可发表。

基础对比模型：

- `litevilnet_paper`：精度参考。
- `litevilnet_edge`：继承自 VLLiNet 的轻量参考。
- `litevilnet_baseline`：第一个 LiteViLNet baseline。

候选优化方向：

- 轻量 RGB backbone。
- 轻量 LiDAR/ADI encoder。
- 高效融合模块。
- 适合导出的 decoder。
- attention 模块替换或简化。
- TensorRT 友好的算子集合。

每个 variant 必须报告：

- `MaxF` 变化。
- `AP` 变化。
- 参数量下降。
- FLOPs 下降。
- 延迟下降。
- FPS 提升。
- Jetson 上功耗或单帧能耗改善。

主对比逻辑应该是 accuracy-efficiency tradeoff，而不是只比较精度。

## 阶段 5：论文实验资产

目标：让实验输出可以直接服务 IEEE RAL 写作。

必须准备的实验资产：

- 固定评价协议。
- 固定表格字段。
- 精度表。
- 部署表。
- 消融表。
- MaxF-latency tradeoff 图。
- MaxF-parameter tradeoff 图。
- FPS-power 或 FPS-energy 图。
- 定性分割可视化样例。
- 实验配置和 checkpoint metadata。

建议的图表输出：

```text
runs/summary/ral_accuracy_table.csv
runs/summary/ral_deployment_table.csv
runs/summary/ral_ablation_table.csv
runs/figures/maxf_latency.png
runs/figures/maxf_params.png
runs/figures/fps_power.png
```

## 执行顺序

推荐顺序：

```text
阶段 1 -> 阶段 2 -> 阶段 3 -> 阶段 4 -> 阶段 5
```

不要在评价指标和部署链路稳定之前优先改模型结构。

原因：

```text
如果评价链路不稳定，模型改进无法成为可靠的论文证据。
```

## 当前优先任务

下一步工程任务应该是阶段 1：

```text
统一训练、验证、评估、CSV 输出和 README 文档中的 MaxF/AP/PRE/REC/FPR/FNR。
```
