# LiteViLNet 本机 Agent 工作指南

## 项目定位

- 本仓库根目录：`Z:\Database\Research04-LiteViLNet\LiteViLNet`
- 当前任务主线：用 LiteViLNet 的 RGB+Depth 分支处理机器人可通行区域分割。
- 当前数据状态：本机已经采集了新的 RGB+Depth session，但新数据尚未完成标注；在训练前必须先准备二值 mask 和 split 文件。
- 任务语义：预测机器人在当前场景中可以安全通行的地面 / 路径区域，不要把它泛化成普通 road、lane 或 KITTI road segmentation。
- 命名约定：
  - 模型 / 展示名使用 `LiteViLNet`
  - 路径、实验名、模块名使用 `litevilnet`
  - 不再新建错误拼写 `litevillinet`，已有历史目录只在读取旧文件时兼容。

## 本机目录约定

- 源码：`litevilnet/`
- 工具入口：`tools/`
- 机器人数据：`data/robot_road/`
- 实验输出：`output/<experiment_name>/`
- 历史 `runs/` 目录只作为旧实验或通用脚本输出参考；机器人 RGB+Depth 新实验优先写入 `output/`。
- 本指南只记录本机采集、标注、训练、预测和视频生成流程；额外环境或交付流程等用户明确提出时再单独处理。

## 当前代码链路

- 模型：`litevilnet/models/litevilnet_rgbdepth.py`
  - `LiteViLNetRGBDepth`
  - 输入为 RGB + `depth3`
  - RGB backbone 复用 MobileNetV3，depth 分支使用 `LiDAREncoder(in_channels=3)`，再经多尺度融合、bridge 和 decoder 输出二分类 mask。
- 数据集：`litevilnet/data/robot_road_dataset.py`
  - 训练类：`RobotRoadRGBDepthDataset`
  - 预测类：`RobotRoadPredictionDataset`
  - 兼容两种 session 结构：
    - `data/robot_road/raw/<session>/rgb` 和 `depth`
    - `data/robot_road/<session>/rgb` 和 `depth`
  - 当前本机数据实际采用第二种直接 session 结构。
- 训练入口：`tools/train_robot_rgbdepth.py`
  - 当前是 train-fit / smoke / refine 训练逻辑，没有严格独立验证集。
  - `MaxF` 来自训练帧拟合指标，不能当作泛化性能或论文正式结果。
- 预测入口：`tools/predict_robot_rgbdepth.py`
  - 输出 mask、prob、2x2 overlay、四类单图 panel 和 manifest。
- 视频入口：`tools/make_robot_video.py`
  - 从预测 overlay 和 mask 序列生成 `overlay.mp4` 与 `mask.mp4`。

## 当前本机可见数据

`data/robot_road/` 下已有多个 session，RGB 与 depth 数量必须一一对应。当前核对到：

- `session_20260509_172746`：369 帧
- `session_20260511_142931`：1766 帧
- `session_20260511_153551`：1229 帧
- `session_20260511_154515`：651 帧
- `session_20260511_160433`：303 帧
- `session_20260511_170017`：0 帧，不要用于训练或预测
- `session_20260511_170058`：300 帧
- `session_20260511_170628`：344 帧
- `session_20260511_171041`：310 帧

每个有效 session 通常包含：

```text
data/robot_road/<session>/
  rgb/*.png
  depth/*.png
  preview/
  camera_info.json
  manifest.csv
  metadata.yaml
```

## Depth 编码约定

模型第二路输入不是 ADI，也不要命名为 ADI。它是由 aligned depth 转成的 `depth3`：

```text
depth_norm = clip(depth_mm, 0, 12000) / 12000
valid_mask = depth_mm > 0 and depth_mm < 12000
inverse_depth = valid_mask * (1 - depth_norm)
depth3 = [depth_norm, valid_mask, inverse_depth]
depth3 = (depth3 - 0.5) / 0.5
```

要求：

- depth 优先为毫米单位 `uint16 png`。
- depth 必须已经与 RGB 对齐。
- RGB 与 depth 文件名必须一一对应，例如 `000001.png`。
- 如果发现 depth 未对齐、尺寸异常或大量无效值，先向用户说明风险，不要直接训练。

## 标注规范

新采集数据训练前必须先标注 RGB 图像，depth 只作为输入模态和辅助参考，不作为独立标注目标。

- CVAT 本机入口：
  - 安装目录：`E:\_SOFTWARE\cvat`
  - 访问地址：`http://localhost:8080`
  - 管理员账号：`admin`
  - 管理员密码：`admin123456`
  - 启动：在 `E:\_SOFTWARE\cvat` 中运行 `docker compose up -d`
  - 停止：在 `E:\_SOFTWARE\cvat` 中运行 `docker compose down`
- CVAT label 名称：`walkable_path`
- mask 目标：
  - 单通道 PNG
  - 背景为 `0`
  - 可通行区域为 `255`
  - 文件名必须与 frame id 对齐，例如 `000001.png`

当前训练代码默认读取：

```text
data/robot_road/annotations/manual_masks/<frame>.png
```

如果要同时维护多个 session 的 mask，更推荐先检查并改造 `litevilnet/data/robot_road_dataset.py`，增加 `--mask_dir` 或 session-aware mask 路径；不要盲目把不同 session 的同名 `000001.png` mask 混在同一个目录里。

## Split 文件规范

训练必须有 split 文件，建议放在：

```text
data/robot_road/splits/<session>_labeled.txt
```

内容为一行一个 frame id，不带扩展名：

```text
000001
000014
000060
```

训练前检查：

- split 非空。
- split 中每个 frame 都有 RGB。
- split 中每个 frame 都有 depth。
- split 中每个 frame 都有 mask。
- mask 尺寸与 RGB 一致，或确认 dataset 中会按最近邻 resize。
- mask 像素值只包含 `0/255` 或至少能稳定二值化为背景 / 可通行区域。

## 实验输出规则

采用“一次训练 = 一个完整实验目录”的规则。不要覆盖已有实验，也不要把日志、权重、预测和视频散落在不同目录。

推荐命名：

```text
output/<version>_litevilnet_rgbdepth_robot_path_<data_or_session>_<purpose>/
```

示例：

```text
output/3.1_litevilnet_rgbdepth_robot_path_seed40_refine/
output/3.2_litevilnet_rgbdepth_robot_path_session_20260511_142931_smoke/
output/3.3_litevilnet_rgbdepth_robot_path_multi_session_refine/
```

每个训练实验目录至少应包含：

```text
README.md
config.json
train.log
best_model.pth
latest_model.pth
train_metrics.json
```

预测和视频应放在同一个实验目录下：

```text
predict_all/
predict_subset/
video/
```

## 本机 Conda 环境

当前已验证可运行 LiteViLNet RGB+Depth 主线的 conda 环境：

```text
VLLiNet
```

验证结果：

- Python：`3.10.19`
- PyTorch：`2.5.1+cu121`
- CUDA：可用
- GPU：`NVIDIA GeForce RTX 4060 Ti`
- `LiteViLNetRGBDepth` 可以导入并完成 CUDA 前向测试。
- `tools.train_robot_rgbdepth`、`tools.predict_robot_rgbdepth`、`tools.make_robot_video` 的命令行入口可以正常启动。
- 机器人 RGB+Depth 训练、预测、视频生成主线所需核心依赖可用：`torch`、`torchvision`、`numpy`、`PIL`、`tqdm`、`cv2`、`albumentations`。

推荐命令前缀：

```powershell
conda run -n VLLiNet python -m <module>
```

示例：

```powershell
conda run -n VLLiNet python -m tools.train_robot_rgbdepth ...
```

注意：

- 当前 `VLLiNet` 环境缺少 `pandas` 和 `matplotlib`，但不影响机器人 RGB+Depth 的训练、预测和视频生成。
- 若要运行绘图、统计汇总、论文结果表格等脚本，再补装 `pandas matplotlib`。

## 本机训练命令模板

PowerShell 示例：

```powershell
$env:PYTHONPATH = "."
$EXP_DIR = "output/3.2_litevilnet_rgbdepth_robot_path_session_20260511_142931_smoke"

python -m tools.train_robot_rgbdepth `
  --data_root data/robot_road `
  --session session_20260511_142931 `
  --split_file data/robot_road/splits/session_20260511_142931_labeled.txt `
  --img_h 384 `
  --img_w 608 `
  --epochs 80 `
  --batch_size 2 `
  --lr 2e-4 `
  --weight_decay 1e-4 `
  --amp `
  --num_workers 2 `
  --save_dir $EXP_DIR `
  --log_dir $EXP_DIR
```

如果使用 conda 环境，优先使用已验证的 `VLLiNet`：

```powershell
conda run -n VLLiNet python -m tools.train_robot_rgbdepth ...
```

注意：

- 未标注的新 session 不能直接训练。
- `session_20260511_170017` 当前没有 RGB/Depth 帧，不要作为训练 session。
- RTX 4060 Ti 16GB 本机训练建议从 `batch_size 2` 开始；显存不足时先降 `batch_size`，再考虑降低输入分辨率。
- `--amp` 只有在 CUDA 可用时才会启用混合精度。

## 预测命令模板

预测整个 session：

```powershell
$env:PYTHONPATH = "."
$EXP_DIR = "output/3.2_litevilnet_rgbdepth_robot_path_session_20260511_142931_smoke"

python -m tools.predict_robot_rgbdepth `
  --data_root data/robot_road `
  --session session_20260511_142931 `
  --checkpoint "$EXP_DIR/best_model.pth" `
  --metrics "$EXP_DIR/train_metrics.json" `
  --img_h 384 `
  --img_w 608 `
  --batch_size 2 `
  --num_workers 2 `
  --output_dir "$EXP_DIR/predict_all"
```

只预测部分帧：

```powershell
python -m tools.predict_robot_rgbdepth `
  --data_root data/robot_road `
  --session session_20260511_142931 `
  --checkpoint "$EXP_DIR/best_model.pth" `
  --metrics "$EXP_DIR/train_metrics.json" `
  --img_h 384 `
  --img_w 608 `
  --batch_size 2 `
  --num_workers 2 `
  --frames_file data/robot_road/splits/session_20260511_142931_preview.txt `
  --output_dir "$EXP_DIR/predict_subset"
```

预测输出应包含：

```text
predict_all/mask/*.png
predict_all/prob/*.png
predict_all/overlay/*.jpg
predict_all/panels/rgb/*.jpg
predict_all/panels/depth_color/*.jpg
predict_all/panels/prediction_overlay/*.jpg
predict_all/panels/probability/*.jpg
predict_all/manifest.json
```

`overlay/*.jpg` 是 2x2 布局：

- 左上：RGB
- 右上：彩色 depth 可视化
- 左下：预测叠加图
- 右下：可通行概率热力图

## 视频生成命令模板

```powershell
$EXP_DIR = "output/3.2_litevilnet_rgbdepth_robot_path_session_20260511_142931_smoke"

python -m tools.make_robot_video `
  --frames_dir "$EXP_DIR/predict_all/overlay" `
  --masks_dir "$EXP_DIR/predict_all/mask" `
  --fps 15 `
  --output_dir "$EXP_DIR/video"
```

输出：

```text
video/overlay.mp4
video/mask.mp4
```

完成后必须检查：

- `predict_all/mask` 数量是否等于待预测 RGB 数量。
- `predict_all/overlay` 数量是否等于待预测 RGB 数量。
- `predict_all/panels/*` 四个子目录数量是否一致。
- `video/overlay.mp4` 能打开且帧数合理。
- `video/mask.mp4` 能打开且帧数合理。

## Agent 工作方式

- 修改代码前先读当前实现，确认真实参数、路径和输出格式；不要凭旧文档猜。
- 本项目当前重点是本机新采集数据的标注、训练、预测和视频生成，优先围绕这条链路工作。
- 对数据路径、session、split、mask、checkpoint 做显式核对后再训练。
- 新实验必须创建新的 `output/` 子目录，不覆盖历史结果。
- 遇到未标注数据时，先组织标注输入 / 导出规则 / split，而不是直接启动训练。
- 遇到指标时要说明性质：当前训练脚本的 `MaxF` 是训练拟合指标，不是独立验证或论文级泛化结果。
- 保持沟通直接、简洁、可执行：先给结论，再给必要理由和命令。
- 主动指出会影响实验可信度的问题，例如 frame 泄漏、同名 mask 冲突、未对齐 depth、session 混用、输出目录覆盖、阈值来源不清。
