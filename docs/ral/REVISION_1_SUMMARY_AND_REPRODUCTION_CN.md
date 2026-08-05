# LiteViLNet RA-L 第一轮修改、证据审计与复现说明

> **内部文档，请勿作为双盲附件上传。** 本文为作者核验保留本机数据/结果路径和完整审计说明。投稿请只使用 `tools/package_ral_reproduction.sh` 生成并通过自动身份扫描的匿名压缩包。

> 状态：原修订实验与 Table I 的五个官方 baseline 同协议三种子重训练已完成；为赶 rebuttal，本轮 ORFD 快照采用 USNet seed-40 单次重训和 OFF-Net 发布 checkpoint 的独立评测，SNE/RoadFormer 及 OFF-Net 三 seed 重训命令保留但不纳入当前 Table III。作者原图保持不变；Fig. 1、3、4、5 的手工绘制素材和逐图说明已准备，但投稿前仍需作者手工覆盖最终图片文件。

> 正文表述原则：论文正文面向正式读者，重点呈现 LiteViLNet 的创新设计、精度—效率优势、跨城市/越野场景验证和边缘部署价值。为回应评审而补充的精确 split seed、逐类别数量、manifest 路径、SHA-256、训练 seed、梯度累积和单步训练显存等审计信息，不再堆放于 Abstract 或 Section IV-A；它们集中保留在 Response 与本复现文档中。正文中仅保留理解实验所必需的数据集划分类型、评价协议、主要训练设置和三次独立运行统计。

> 颜色约定：`cyan` 表示可直接对应 Editor/Reviewer 意见的修改；`blue` 表示作者在完成逐条回复之外主动加入的表述优化、实现一致性补充或信息层级整理。具体而言，评审明确询问的 ADI 窗口/零分母/归一化公式与 CMA 线性复杂度公式保持 `cyan`；源码审计额外发现的 ECA/CA 前处理、门控输入和残差顺序、Large-Kernel Bridge 前向公式、深监督推理行为及 loss 配置统一为 `blue`。颜色只表示修改来源，不表示内容重要性或证据等级。

## 1. 这轮修改的核心结论

这轮工作不是简单润色，而是对论文、代码、数据划分、指标、公式、速度口径和机器人实验做了一次证据审计。最重要的发现是：

1. 原论文把本地 KITTI perspective-view 验证和官方 KITTI BEV test-server 结果放在同一张表里直接排名，这是不成立的。LiteViLNet 没有官方 test-server submission。
2. 历史代码按文件名排序后直接做 80/20 切分，导致 58 张验证图全部属于 `UU`。修订实验改用公开种子 `20260723` 的固定分层随机 231/58 split；历史 split 和结果只用于追溯。
3. 原 CMA 公式不是实际实现。代码只生成一个全局 RGB query，并对 LiDAR 的 `N=H_lW_l` 个空间 key/value 做 `1×N` 注意力；不存在 `N×N` 矩阵。
4. 原 ADI、loss、geometry encoder 和 RGB-D 部署描述均与代码或官方参考实现不一致，现已按真实实现重写。
5. 机器人分支使用 aligned depth 生成 `depth3`，不是 ADI，也不是 KITTI 模型 zero-shot transfer。历史导航没有保留成功率、碰撞、干预、横向误差或功耗日志，因此只能作为定性系统演示。
6. MSFM 增加约 10.35M 参数，必须与“简单逐尺度相加 + 同一 Bridge + 同一深监督”控制一起看。修正后的分层 split 结果见第 7 节。
7. ORFD 用作第二数据集。除 8,392/1,245 train/validation 外，本轮进一步发现下载包的 2,193-frame testing partition 也包含完整 GT；训练和 validation 都与 test 无同名帧。论文现以 held-out test 作为主结果。
8. 按官方 OFF-Net commit `50e63d2` 的固定 argmax、原始 `1280×720` GT confusion-matrix 口径，full 为 `96.74 ± 0.09% F-score / 93.68 ± 0.18% IoU`。它相对已独立评测的 OFF-Net 发布 checkpoint 高 `3.94/7.11 pp`，相对发布表格值高 `6.44/11.38 pp`；相对 USNet seed-40 高 `1.12/2.07 pp`。Full 的 AP 为 `98.31 ± 0.37%`，对 compact 的原有三 seed 优势是 `+0.71 ± 0.67 pp`。
9. Table I 现已改为完全同协议的本地核心比较。USNet 为 `97.88±0.07%` MaxF，SNE-RoadSeg 为 `97.23±0.21%`，PLARD 为 `95.25±0.19%`，OFF-Net 为 `95.36±0.66%`，RoadFormer 为 `97.28±0.05%`，LiteViLNet 为 `97.23±0.15%`。LiteViLNet 与 SNE-RoadSeg 仅差 `0.01 pp`，参数少 `93.0%`；相对 SNE-RoadSeg、PLARD、RoadFormer 的 RTX 4090 D FPS-1 分别为 `11.21×`、`8.03×`、`12.40×`。ORFD 快照另记录 USNet seed-40 与 OFF-Net 发布 checkpoint，分别为 `95.62/91.61` 和 `92.80/86.57` 的 fixed F-score/IoU。

## 2. 审稿意见—修改证据矩阵

| 审稿关注 | 修改位置 | 证据/实验 |
|---|---|---|
| 只有 KITTI，数据太小 | 论文 Section IV-A、IV-D、Table III | ORFD 官方 train/val/test，8,392/1,245/2,193 帧；held-out test F-score `96.74±0.09%` |
| KITTI 比较不公平、数字错误 | Section IV-A、IV-B、Table I | 从作者官方源码同协议重训 USNet/SNE-RoadSeg/PLARD/OFF-Net/RoadFormer；六种方法统一 split、尺寸、预算、评测与三种子统计 |
| 缺少多种子 | Section IV-A、IV-C、Tables I–II；精确 seed 见 Response/本文档 | 三次独立运行，mean ± sample std；复现 seed 为 40/41/42 |
| MSFM 单独下降、缺简单融合控制 | Section III-C、IV-C、Table II | `optimal`：simple addition + Bridge + DeepSup |
| 缺 FLOPs/显存/延迟 | Section IV-A、IV-C、Table II | fvcore GMAC-equivalent、CUDA allocation、FP16 latency distribution |
| CMA 的二次复杂度与速度冲突 | Section III-C、Eq. (3)、Fig. 3 | 真实 tensor shape；`1×N_l` score memory |
| ADI 缺窗口、洞、归一化细节 | Section III-A、Eqs. (1)–(2) | 官方 PLARD MATLAB 流程的 Python port 与单元测试 |
| Fig. 1 与文字矛盾 | Fig. 1、Section III-B | RGB 为 MobileNetV3；geometry 为 conv stem + 4 DSConv |
| RGB-D-compatible ADI 不成立 | Sections III-A、IV-E | KITTI ADI 与机器人 `depth3` 完全分开 |
| 机器人只有定性图 | Section IV-E、Figs. 5–6 | 明确降级为 qualitative demonstration；不编造导航指标 |
| 贡献创新性夸大 | Abstract、Sections I–III、Conclusion | 改为 system-oriented integration 与可复现 profiling |
| TwinLiteNet+/KGD 等未进表 | Sections II、IV-B、Response | 解释任务/数据/标签/指标不一致；另做本地 KD strategy control |

## 3. 目录、环境与硬件

工作区：

```text
/home/aihub/daojie/LiteViLNet_ws/
  LiteViLNet/                 # 代码
  LiteViLNetPaperRAL/         # 论文和 Response
```

本轮环境：

```text
conda env: litevilnet_ral
Python: 3.10.20
PyTorch: 2.7.1+cu128
torchvision: 0.22.1+cu128
cuDNN: 90701
NumPy: 1.26.4
Albumentations: 1.4.24
OpenCV headless: 4.10.0
fvcore: 0.1.5.post20221221
GPU: 2 × NVIDIA GeForce RTX 4090 D, 48 GB
CPU: AMD Ryzen 9 9950X 16-Core Processor (32 logical CPUs)
```

可复现环境文件：

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda activate litevilnet_ral
```

注意：早期诊断 profiling 时两张 GPU 上还有不属于本任务的 `DriveOccWorld` 进程，各占约 15 GB 且有计算活动。它不改变确定性训练的数据/初始化，但会污染 wall-clock latency。带完整 `nvidia-smi` process snapshot 的受干扰 profiling 被保留作诊断；论文采用的是 2026-07-24 在目标 GPU 无其他 compute process 时重新得到的 clean 三次测量。

## 4. KITTI 数据协议

数据根目录：

```text
/data/Database/Research04-LiteViLNet/LiteViLNet/data/kitti_road
```

### 4.1 历史 split：只用于追溯

```text
configs/splits/kitti_road/train.txt
configs/splits/kitti_road/val.txt
configs/splits/kitti_road/manifest_metadata.json
```

类别计数：

```text
train: UM=95, UMM=96, UU=40
val:   UM=0,  UMM=0,  UU=58
```

这解释了为什么历史表中的很多数值更接近官方表的 `UU` 子类别，而不是 overall。它不能被描述成类别覆盖完整的 KITTI 验证。

### 4.2 修订主 split：固定分层随机

```text
configs/splits/kitti_road/stratified_seed20260723/train.txt
configs/splits/kitti_road/stratified_seed20260723/val.txt
configs/splits/kitti_road/stratified_seed20260723/manifest_metadata.json
```

类别计数：

```text
train: UM=76, UMM=77, UU=78  (231)
val:   UM=19, UMM=19, UU=20 (58)
```

哈希：

```text
train.txt SHA-256 = 93a8b849a531e9bd938c65120816f5ad4bd62f563e7f0d68ac6c0e6046425867
val.txt   SHA-256 = 69b10e5ff641d5cea81d2f0832ada2c31ee5f3b3f8ced9e4e962f889a722976f
```

重新生成和验证：

```bash
python -m tools.create_kitti_split_manifests \
  --data_root /data/Database/Research04-LiteViLNet/LiteViLNet/data/kitti_road \
  --strategy stratified-random \
  --seed 20260723 \
  --output_dir configs/splits/kitti_road/stratified_seed20260723
```

## 5. 指标与训练设置

本地二值 mask 使用 101 个阈值 `[0, 0.01, …, 1]`，报告：

- `MaxF`：阈值扫描最大 F1；
- `PRE`、`REC`、`FPR`、`FNR`、`IoU`：在 MaxF 阈值处；
- `AP`：同一离散 precision-recall curve 的面积；
- `BestThreshold`：产生 MaxF 的阈值。

直方图实现与逐阈值 full-mask 参考实现精确对齐，单元测试见：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

KITTI 主训练参数：

```text
resolution: 384×1248
seeds: 40, 41, 42
optimizer: AdamW
lr: 2e-4
weight decay: 5e-4
scheduler: cosine annealing
epochs: at most 150
early-stop patience: 40
physical batch: 2
gradient accumulation: 8
nominal effective batch: 16
AMP: enabled
cuDNN deterministic: true
RGB initialization: ImageNet MobileNetV3
```

训练集 231 张中最后一张被 `drop_last` 丢弃，因此每个 epoch 有 115 个 physical batches；115 不能整除 8，最后一次 optimizer update 实际只累计 3 个 physical batches（6 张图），但代码仍按 8 缩放 loss。该行为在所有 KITTI 修订 run 中保持一致，复现时不要把它误改成每一步都严格 16 张。

示例：

```bash
LITEVILNET_KITTI_TRAIN_SPLIT=configs/splits/kitti_road/stratified_seed20260723/train.txt \
LITEVILNET_KITTI_VAL_SPLIT=configs/splits/kitti_road/stratified_seed20260723/val.txt \
LITEVILNET_REVISION_OUTPUT=/data/Database/Research04-LiteViLNet/revision_1_runs/kitti_ablation_stratified_seed20260723 \
bash tools/run_revision_ablation_queue.sh 0 full:42 full:40 full:41
```

若要从头复现 Table II，先在两个 shell 中设置同一组公共变量：

```bash
export LITEVILNET_KITTI_TRAIN_SPLIT=configs/splits/kitti_road/stratified_seed20260723/train.txt
export LITEVILNET_KITTI_VAL_SPLIT=configs/splits/kitti_road/stratified_seed20260723/val.txt
export LITEVILNET_REVISION_OUTPUT=/data/Database/Research04-LiteViLNet/revision_1_runs/kitti_ablation_stratified_seed20260723
```

上面是本轮证据目录。重新执行时建议把 `LITEVILNET_REVISION_OUTPUT` 改成带日期的新目录，先保留本轮 checkpoint/result；KD 重跑同理可设置新的 `LITEVILNET_KITTI_DISTILL_OUTPUT`。

然后分别运行（配置之间写入不同目录，可安全双卡并行）：

```bash
# shell 1 / GPU 0
bash tools/run_revision_ablation_queue.sh 0 \
  baseline:40 baseline:41 baseline:42 \
  add_lidar:40 add_lidar:41 add_lidar:42 \
  optimal:40 optimal:41 optimal:42

# shell 2 / GPU 1
bash tools/run_revision_ablation_queue.sh 1 \
  add_fusion:40 add_fusion:41 add_fusion:42 \
  add_bridge:40 add_bridge:41 add_bridge:42 \
  full:40 full:41 full:42 \
  transformer_bridge:40 transformer_bridge:41 transformer_bridge:42
```

待 `full/seed_42` teacher 完成后，KD control 使用：

```bash
bash tools/run_kitti_distill_queue.sh 0 40 41 42
```

每个 run 都写入：

```text
CONFIG/seed_N/result.json
CONFIG/seed_N/best_model.pth
logs/CONFIG/seed_N/train.log
```

`result.json` 包含 split、样本数、seed、epoch、超参数、系统版本和最佳验证指标。

## 6. 模型实现纠错

### 6.1 CMA / MSFM

实际实现：

```text
RGB feature -> GAP -> one query Q: B×1×d
LiDAR feature -> spatial K: B×N×d
LiDAR feature -> spatial V: B×d×N
attention alpha: B×1×N
```

对 `384×1248` 输入的第一层：

```text
N = 192×624 = 119,808
FP16 score memory/sample ≈ 119,808×2 bytes = 0.23 MiB
```

所以 score memory 是 `O(BN)`，不是 `O(BN²)`。原论文的“bidirectional standard spatial cross-attention”公式属于文档错误。

### 6.2 Geometry encoder

RGB 分支是 MobileNetV3-Large。Geometry 分支是：

```text
two-convolution stem + four DSConv stages
output channels: [16, 24, 40, 112, 960]
output scales: [1/2, 1/4, 1/8, 1/16, 1/32]
parameters: about 0.12M
```

两路只匹配输出尺寸和通道，不是相同 backbone。

### 6.3 Loss

真实训练 loss：

```text
BCE + Dice + 0.1 × Boundary
auxiliary weights = [0.4, 0.3, 0.2]
```

原稿的 Lovász/Focal 和 `[0.5,0.3,0.2]` 不符合代码。辅助 head 推理时 bypass；其 115 个 stored parameters 仍计入参数量。

## 7. KITTI 修订实验结果

### 7.1 Table I 官方源码同协议基线

Table I 的核心精度行不再混用 KITTI official BEV test-server 与本地 perspective-view 指标。USNet、SNE-RoadSeg、PLARD、OFF-Net、RoadFormer 和 LiteViLNet 现在统一使用：

```text
split: category-stratified 231 train / 58 validation
input: 384×1248
budget: at most 150 epochs
seeds: 40, 41, 42
evaluator: all validation pixels accumulated, then 101 thresholds swept
statistics: mean ± sample SD
```

官方来源与固定版本：

| 方法 | 作者官方仓库 | 固定 commit | 正式训练所用架构 |
|---|---|---|---|
| USNet | `https://github.com/morancyc/USNet.git` | `d761158ad42df7dcb62fa257dd02ce11c85f94a5` | 官方 ResNet-18 双分支 USNet |
| SNE-RoadSeg | `https://github.com/hlwang1124/SNE-RoadSeg.git` | `5e7900bfd59887634ced687ffe85a73018a38659` | 官方双 ResNet-152 RoadSeg + 官方 SNE normal |
| PLARD | `https://github.com/zhechen/PLARD.git` | `44485803092e729661c696ab6c03f6f2fabc8701` | 官方 RGB--ADI PLARD + 三路监督 |
| OFF-Net | `https://github.com/chaytonmin/Off-Road-Freespace-Detection` | `50e63d24836198e8fb5af707e521f414104b4876` | 官方 MiT-B2 RGB--normal fusion + 官方 SNE |
| RoadFormer | `https://github.com/LiJiahang617/Road-Former.git` | `f675a3467cb168ebc727648390c304279bbcb079` | 官方 TwinConvNeXt-B + RoadFormer decoder |

本地适配器没有重写基线网络和 loss。它直接导入官方定义，保留 USNet 的 evidential objective、SNE-RoadSeg 的 dual-ResNet/SNE、PLARD 的 RGB--ADI 图与三路监督、OFF-Net 的 MiT-B2 fusion/loss/SNE，以及 RoadFormer 的 TwinConvNeXt-B、Hungarian matching losses 和 decoder；新增部分仅为统一 manifest、预算、seed、normal 编码/缓存、评测与 provenance 输出。五个仓库的 remote、完整 commit、语义 diff 和关键文件 SHA-256 均由 `source_provenance.json` 与汇总器逐项复核。

逐 seed 正式结果（百分数）：

| Method | Seed | Best epoch | MaxF | PRE | REC | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| USNet | 40 | 135 | 97.8094 | 98.0025 | 97.6171 | `09e0b77829b05d52c1ab945d9ac3fe48ea870718cddf683a7a767855950734c7` |
| USNet | 41 | 125 | 97.8853 | 98.0283 | 97.7428 | `b4a55ab275be5ac15a520fe6e6a0018ef18451ad3bb80a0c5ad9c749bbdac065` |
| USNet | 42 | 105 | 97.9493 | 98.0707 | 97.8282 | `273d490b7228e3f1fdf3233deb91f18aff7d854c207f37d44e975b7109483e2b` |
| SNE-RoadSeg | 40 | 35 | 96.9893 | 97.0651 | 96.9136 | `7cd1edf45f7e8b7350a602a4308a8dbac115ba39dd68718b708db26d1a609d70` |
| SNE-RoadSeg | 41 | 40 | 97.3079 | 97.4405 | 97.1757 | `df44300104f2468b5e55d6272d443c364fb60cb1fa86ac08b0a8ba1cd8758eb7` |
| SNE-RoadSeg | 42 | 45 | 97.3805 | 97.6629 | 97.0998 | `8eafe3bbfd8e2fa14377d631da6fd406e1d940a41f86c5060c89b5f1eeb1f9e0` |
| OFF-Net | 40 | 100 | 95.9813 | 95.9080 | 96.0548 | `f8193fb055a94b2290216f38b697e014a037cc946599851e07570b2c23a3aa78` |
| OFF-Net | 41 | 145 | 95.4243 | 94.6025 | 96.2607 | `3b0642f66d13fad53bb21a40e92ed4fd76965cff4d071f9fa6bbef3899f1df61` |
| OFF-Net | 42 | 120 | 94.6617 | 94.1409 | 95.1882 | `04ef8174615d98d8ddb3fdc3a1838046965d4bd60eb5e96b35a291183666f495` |

Table I 汇总：

| Method | n | MaxF | PRE | REC | Params |
|---|---:|---:|---:|---:|---:|
| USNet | 3 | 97.88 ± 0.07 | 98.03 ± 0.03 | 97.73 ± 0.11 | 30.74M |
| SNE-RoadSeg | 3 | 97.23 ± 0.21 | 97.39 ± 0.30 | 97.06 ± 0.13 | 201.32M |
| PLARD | 3 | 95.25 ± 0.19 | 95.46 ± 0.29 | 95.03 ± 0.09 | 76.93M |
| OFF-Net | 3 | 95.36 ± 0.66 | 94.88 ± 0.92 | 95.83 ± 0.57 | 25.21M |
| RoadFormer | 3 | 97.28 ± 0.05 | 97.96 ± 0.09 | 96.61 ± 0.16 | 206.86M |
| LiteViLNet | 3 | 97.23 ± 0.15 | 97.31 ± 0.59 | 97.16 ± 0.30 | 14.04M |

专业解读：LiteViLNet 的 MaxF 比 SNE-RoadSeg 高 `0.0085 pp`（表中四舍五入后均为 97.23），且 MaxF sample SD 更低，同时参数量少 `93.03%`。相对 USNet，LiteViLNet 参数量少 `54.34%`，MaxF 差 `0.6469 pp`；相对 SNE-RoadSeg、PLARD、RoadFormer 的 RTX 4090 D model-only FPS-1 分别为 `11.21×`、`8.03×`、`12.40×`。因此正文把 LiteViLNet 定位为精度—参数—目标设备速度的 Pareto operating point，不声称所有精度指标绝对第一。

USNet 的参数量较大但 FPS-1 略高，是算子效率而非计时错误。参数量只统计权重；统一输入下 fvcore 可识别的 USNet 计算量约为 `39.13 GMAC-eq`，虽高于 LiteViLNet 的 `10.17 GMAC-eq`，但 USNet 以 4090 D 上高度优化的大块标准卷积为主。LiteViLNet 的深度卷积、逐尺度融合/attention、插值及较多小 kernel 降低了参数和 MAC，却可能带来更低算术强度和更多调度开销。原始三次 repeat 的 USNet 均约 `4.170 ms`，LiteViLNet 为 `4.617±0.012 ms`；两者都排除了 normal/SNE 或 ADI 预处理。因此论文强调 LiteViLNet 相对 SNE-RoadSeg、PLARD 和 RoadFormer 的准确率—参数—速度优势，不把“参数更少”误写成“必然比所有网络更快”。

统一 RTX 4090 D FPS-1 快照（`384×1248`、batch 1、PyTorch FP32、100 次 warmup、300 次计时、三次 repeat、仅 model forward）如下：

| Method | FPS-1 | 参数量 |
|---|---:|---:|
| USNet | 239.81 | 30.74M |
| SNE-RoadSeg | 19.32 | 201.32M |
| PLARD | 26.97 | 76.93M |
| RoadFormer | 17.46 | 206.86M |
| LiteViLNet | 216.61 | 14.04M |

正式汇总与匿名逐 seed 证据：

```text
docs/ral/table1_matched_baselines/results/summary.json
docs/ral/table1_matched_baselines/results/summary.csv
docs/ral/table1_matched_baselines/results/seeds/*.json
```

完整复现命令、官方数据链接、源码边界与哈希规则见：

```text
docs/ral/table1_matched_baselines/README.md
docs/ral/table1_matched_baselines/README_CN.md
docs/ral/table1_matched_baselines/source_provenance.json
```

### 7.2 主分层 split

三种子最终结果（百分数，mean ± sample SD）：

| Variant | MaxF | AP | PRE | REC | IoU |
|---|---:|---:|---:|---:|---:|
| RGB-only baseline | 96.73 ± 0.08 | 98.82 ± 0.24 | 96.86 ± 0.08 | 96.61 ± 0.08 | 93.67 ± 0.14 |
| + geometry, simple addition | 96.93 ± 0.10 | 99.13 ± 0.14 | 97.07 ± 0.33 | 96.79 ± 0.18 | 94.04 ± 0.19 |
| + MSFM | 97.05 ± 0.23 | 98.89 ± 0.41 | 97.24 ± 0.54 | 96.85 ± 0.08 | 94.27 ± 0.44 |
| + large-kernel bridge, no DS | 97.24 ± 0.16 | 98.84 ± 0.40 | 97.46 ± 0.28 | 97.02 ± 0.11 | 94.63 ± 0.29 |
| Full: MSFM + bridge + DS | 97.23 ± 0.15 | 98.95 ± 0.13 | 97.31 ± 0.59 | 97.16 ± 0.30 | 94.62 ± 0.28 |
| Matched simple + bridge + DS | 97.13 ± 0.02 | 99.05 ± 0.15 | 97.46 ± 0.02 | 96.80 ± 0.03 | 94.42 ± 0.05 |
| Standard Transformer bridge control | 97.10 ± 0.10 | 98.47 ± 0.32 | 97.47 ± 0.55 | 96.74 ± 0.37 | 94.37 ± 0.19 |
| KD student control | 96.87 ± 0.06 | 98.97 ± 0.36 | 97.15 ± 0.28 | 96.59 ± 0.27 | 93.93 ± 0.11 |

MaxF 的 seed-matched 增量：

| 左配置 − 右配置 | paired mean ± sample SD（percentage points） | 方向 |
|---|---:|---|
| geometry − RGB-only | +0.20 ± 0.11 | 3/3 为正 |
| MSFM − simple geometry | +0.12 ± 0.17 | 2/3 为正 |
| bridge − MSFM | +0.19 ± 0.25 | 2/3 为正 |
| full DS − bridge/no-DS | −0.005 ± 0.178 | 1/3 为正 |
| full MSFM − matched simple | +0.10 ± 0.15 | 2/3 为正 |
| full LKB − Transformer bridge | +0.13 ± 0.14 | 3/3 为正 |
| KD student − 同结构非 KD student | −0.06 ± 0.08 | 1/3 为正 |

所以论文必须诚实表达：geometry encoder 是最稳定的低成本增益；MSFM 和 bridge 的均值增益在这个小 split 上会随 seed 改变方向；deep supervision 没有产生可重复的均值提升；matched simple 配置是稳定的低成本 Pareto 方案。大核 bridge 相比标准 MHSA–FFN control 在三种子上方向一致，同时参数/MAC 更低。固定 full seed-42 teacher 的 logit/boundary KD control 没有改善同结构 student，因此它只能作为负的本地策略对照，不能被描述成 KGD 的复现。

KD 汇总证据：

```text
runs/revision_1/kitti_distillation_summary.json
```

生成命令：

```bash
python -m tools.summarize_distillation_control \
  --distill-root /data/Database/Research04-LiteViLNet/revision_1_runs/kitti_distill_edge_stratified_seed20260723 \
  --baseline-root /data/Database/Research04-LiteViLNet/revision_1_runs/kitti_ablation_stratified_seed20260723 \
  --output runs/revision_1/kitti_distillation_summary.json
```

KD 编排期间曾误并发启动两个 seed-40 进程，可能同时覆盖同一路径；该输出已终止并隔离为 `seed_40_collision_20260724_0144`，不进入任何论文数字。最终 clean seed-40 重新独占运行，`summarize_distillation_control.py` 只接受目录名完整匹配 `seed_<整数>` 的结果。三个 clean best checkpoint 均已成功加载，seed 40/41/42 的 SHA-256 分别为：

```text
46c717f2823fc3bcaa26cdfa6d8cae40c9cbfaf446d503147b314598e6468396
720a71ebdfcec967be03a80717d0b6e5991e43f39c15969aeae885b2bb148d3d
7d901a949e9ed184ac96213ba1dc988d2f03a0f97a47956fed94edfa1b52d4bc
```

最终汇总命令：

```bash
python -m tools.summarize_revision_experiments \
  /data/Database/Research04-LiteViLNet/revision_1_runs/kitti_ablation_stratified_seed20260723 \
  --pair add_lidar:baseline \
  --pair add_fusion:add_lidar \
  --pair add_bridge:add_fusion \
  --pair full:add_bridge \
  --pair full:optimal \
  --pair full:transformer_bridge \
  --output runs/revision_1/kitti_stratified_summary.json \
  --csv runs/revision_1/kitti_stratified_summary.csv
```

`paired_comparisons` 按相同 seed 计算左侧配置减右侧配置的逐 seed 差值，再报告 mean/sample-SD；论文中的模块增益必须读取这里，不能用两个独立均值的标准差代替成对差值标准差。

### 7.3 历史 UU-only split

历史三种子可追溯结果：

| Variant | MaxF mean ± sample std |
|---|---:|
| RGB-only baseline | 95.56 ± 0.17 |
| + LiDAR, simple addition | 95.82 ± 0.12 |
| Simple addition + Bridge + DeepSup | 95.98 ± 0.10 |
| Full MSFM + Bridge + DeepSup | 96.16 ± 0.08 |

这些值不能替代分层 split 主结果，也不能与 KITTI official BEV server 排名。

### 7.4 结构成本

fvcore 约定：一个 fused multiply-add 记为一个 MAC-equivalent。分析器不支持的 element-wise/activation op 保留在 JSON 中，因此不能称为 exact FLOPs。

| Config | Params (M) | GMAC-equivalent |
|---|---:|---:|
| baseline | 3.305 | 3.928 |
| add_lidar | 3.426 | 4.353 |
| add_fusion | 13.781 | 10.054 |
| add_bridge/full | 14.035 | 10.174 |
| optimal: simple + Bridge + DeepSup | 3.680 | 4.472 |
| transformer_bridge | 24.852 | 15.655 |

原始证据：

```text
runs/revision_1/ablation_cost_384x1248.json
```

### 7.5 延迟与显存

统一口径：RTX 4090 D、`384×1248`、batch 1、FP16、随机常驻 GPU 的双模态 tensor、model-only、50 次 warm-up + 200 次计时；表中延迟是三次独立 profiler invocation 的 mean ± sample SD。`peak/+fwd` 分别是 CUDA peak allocated 与 warm-up 后一次 forward 的 incremental peak，单位 MiB。

| Config | CUDA peak/+fwd (MiB) | mean latency (ms) | 三次 mean 的 sample SD (ms) |
|---|---:|---:|---:|
| baseline | 45.99 / 34.06 | 1.655 | 0.003 |
| add_lidar | 60.77 / 48.57 | 1.893 | 0.005 |
| add_fusion | 87.92 / 47.77 | 4.313 | 0.009 |
| add_bridge | 89.85 / 49.21 | 4.335 | 0.005 |
| full | 88.57 / 47.93 | 4.350 | 0.007 |
| optimal / matched simple | 68.58 / 47.77 | 1.970 | 0.016 |
| transformer_bridge | 110.43 / 48.05 | 4.502 | 0.014 |

原始 clean 证据：

```text
runs/revision_1/ablation_profile_4090d_clean_repeat_1.json
runs/revision_1/ablation_profile_4090d_clean_repeat_2.json
runs/revision_1/ablation_profile_4090d_clean_repeat_3.json
runs/revision_1/ablation_profile_4090d_clean_summary.json
```

ORFD `704×1280`、physical batch 8、full 配置的一次真实 FP16 forward + BCE/Dice/boundary loss + backward + AdamW update 已完成：peak allocated `6446.78 MiB`，peak reserved `6966 MiB`，输入形状为每路 `8×3×704×1280`，loss 为 `2.4933`。证据文件：

```text
runs/revision_1/orfd_full_batch8_training_smoke_complete.json
```

受 `DriveOccWorld` 共驻进程影响的诊断文件：

```text
runs/revision_1/ablation_profile_4090d_repeat_1.json
runs/revision_1/ablation_profile_4090d_repeat_2.json
runs/revision_1/ablation_profile_4090d_repeat_3.json
runs/revision_1/ablation_profile_4090d_long.json
```

这些 JSON 自带 GPU process snapshot，不进入论文延迟结论。

以下命令已用于生成论文最终性能证据，也可在一个无其他 compute process 的 GPU 上复跑：

```bash
LITEVILNET_ORFD_ROOT=/dev/shm/litevilnet_orfd/Final_Dataset \
  tools/run_post_revision_benchmarks.sh 0
```

脚本会先检查目标 GPU；发现任何已有 compute process 时直接拒绝运行。随后依次执行三次独立 model-only profile、完整 8,392-sample ORFD loader 的 batch-8 训练步 smoke、KITTI raw LiDAR→ADI→mask 流水线和机器人 800×1280 depth3→mask 流水线。

## 8. ADI 与端到端 KITTI 流水线

KITTI 网络读取的是已有的预计算 ADI PNG。为复现 ADI 生成，Python port 对齐 PLARD released MATLAB reference：

1. `P2 R0_rect Tr_velo_to_cam` 标定投影；
2. 丢弃 LiDAR `x < 5 m`；
3. 重复像素保留欧氏距离最近点；
4. 21×21 逆距离高度插值；
5. 无邻居保持 invalid/zero；
6. 7×7 altitude-gradient magnitude 平均；
7. 每图正值减正最小值、乘 20、开方；
8. Gaussian `sigma=2`；
9. clip 到 `[0,1]`，复制三通道。

测试：

- duplicate projection nearest-point；
- 21×21 vectorized interpolation 对齐 direct loop；
- 7×7 vectorized gradient 对齐 direct loop。

7×7 实现按 49 个相对邻位使用 NumPy 切片视图累加，避免每个邻位构造整幅 shifted frame；这只是代数等价的内存/速度优化，direct-loop 单元测试仍约束其数值行为。

重要限制：工作区已有 ADI PNG 与参考流程重新生成的 ADI 差异明显。Python port 的合成单元测试通过，并不等于证明历史 PNG 的 provenance。论文必须写成“网络消费 precomputed ADI；可复现 regeneration follows released PLARD reference”，不能声称现有 PNG 已由此 port 精确重建。

进一步核验官方 `zhechen/PLARD` commit `44485803092e729661c696ab6c03f6f2fabc8701` 后确认：其 MATLAB 文件明确写明原始 ADI 代码已丢失，2020 版本是重实现，细节可能不同；README 也说明重新调过参数。因此本轮 raw-LiDAR pipeline 是“公开 reference regeneration path 的计时”，不是历史 stored ADI 的逐像素复原。

真实流水线命令：

```bash
python -m tools.benchmark_kitti_adi_pipeline \
  --data_root /data/Database/Research04-LiteViLNet/LiteViLNet/data/kitti_road \
  --velodyne_root /data/Database/Research04-LiteViLNet/datasets/KITTI_Road/velodyne_extracted \
  --split_file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --checkpoint /path/to/full/seed_42/best_model.pth \
  --img_h 384 --img_w 1248 --precision fp16 \
  --warmup 5 --iters 58 \
  --output runs/revision_1/kitti_adi_end_to_end_4090d_clean.json
```

在 58 张固定 validation 图上的 RTX 4090 D 结果：

| Stage | mean (ms) | sample SD (ms) | P95 (ms) |
|---|---:|---:|---:|
| PNG/BIN/calibration load | 60.01 | 11.84 | 80.39 |
| calibrated projection | 3.09 | 0.36 | 3.59 |
| 21×21 interpolation | 7.65 | 0.30 | 8.02 |
| 7×7 gradient + normalization | 70.85 | 3.40 | 78.23 |
| tensor preparation | 7.31 | 0.34 | 7.95 |
| CPU preprocessing total | 148.91 | 12.49 | 170.31 |
| pinned host→device | 0.52 | 0.02 | 0.53 |
| model | 5.43 | 0.29 | 5.98 |
| sigmoid/resize/threshold/device→host | 0.15 | 0.02 | 0.16 |
| **total** | **155.16** | **12.45** | **176.44** |

reference-regenerated 与 stored ADI 的 MAE 为 `0.3751 ± 0.0298`（58 张，P95 `0.4269`）。这项差异随 JSON 一起保留；它证明不能把 reference-path timing 误写成“已验证与训练输入相同的端到端精度链路”。全部 Table I–III 精度仍使用 stored ADI。

证据：

```text
runs/revision_1/kitti_adi_end_to_end_4090d_clean.json
checkpoint SHA-256 = 49f07b83fdad95dc7330c0d568e38532dcb2f748ef117e89a95ac824d957fa79
```

## 9. ORFD 第二数据集

Archive：

```text
/data/Database/Research04-LiteViLNet/datasets/ORFD/ORFD_merged.zip
```

正式训练使用的本机 tmpfs staging 根目录：

```text
/dev/shm/litevilnet_orfd/Final_Dataset
```

原因不是改变数据，而是首次向 CIFS 网络盘解压时发生过静默位翻转：例如
`training/dense_depth/1620330543635.png` 在网络盘副本的 SHA-256 是
`33b6854e...`，PIL/OpenCV 都报告 PNG IDAT CRC 错误；ZIP archive 中同一条目通过
CRC，重新解压后的 SHA-256 为
`f446ac019c8f7daa097f6335d651a6a4f4204f75dcbe0a5707240d0d9238993b`，
且两种解码器均可完整读取。正式结果只使用由 `unzip` exit 0 校验过的 tmpfs
副本。机器重启后可重新 staging：

```bash
mkdir -p /dev/shm/litevilnet_orfd
unzip -q -o \
  /data/Database/Research04-LiteViLNet/datasets/ORFD/ORFD_merged.zip \
  -d /dev/shm/litevilnet_orfd
```

实际计数：

```text
training:   8,392
validation: 1,245
testing:    2,193
total:     11,830
```

为新增的本地 ORFD baseline 复现，又从完整官方 ZIP 单独解压了一份可长期保留的本机副本：

```text
runs/revision_1/matched_orfd/local_data/Final_Dataset
ZIP SHA-256 = 02359e4b569b12766e317097d84d45d8b9609d8eccea63a9e6c0024e9a2dc92b
unzip -tq   = No errors detected
```

该副本的 RGB、dense depth、GT 和 calib 在 training/validation/testing 中均分别为
8,392/1,245/2,193，所有 11,830 个 dense-depth 时间戳都有同名 calib。正式 normal
cache 强制核验 `released_calibration_files=11830`、`exact_sample_matches=11830`、
`nearest_timestamp_matches=0` 和 `maximum_nearest_timestamp_gap=0`。两套 cache 分别
保存 SNE-RoadSeg 与 OFF-Net 作者源码生成的 float32 normal；前者含 USNet 所需的
8,392 份 pre-SNE flipped training cache，共 20,222 个数组，后者共 11,830 个数组。
每套 cache 生成后又在不加 `--force` 的情况下完整扫描一次 shape/dtype，全部条目均
被成功复用。对应元数据随匿名附件提交，但数百 GB 的可重建数组本身不打包。

tmpfs 副本的全部 ZIP 条目通过 CRC；training/validation/testing 的 `image_data`、`dense_depth`、`gt_image` 分别逐一核验为 8,392/1,245/2,193 个同名配对。协议核验参照 ORFD 官方仓库 `chaytonmin/Off-Road-Freespace-Detection` 的 commit `50e63d24836198e8fb5af707e521f414104b4876`。官方 loader 的标签规则是 RGB 顺序 channel 2 > 200；本次 archive 的 fill-color GT 实际以单通道 PNG 存储，读取成 RGB 后三个通道相同，因此白色区域为 positive。

ORFD 第二路输入不是 ADI：

```text
depth_norm = cv2.resize(float32(uint16_depth) / 65535, 1280×704, INTER_LINEAR)
valid = depth_norm > 0
inverse = valid × (1 - depth_norm)
depth3 = [depth_norm, valid, inverse]
depth3 -> [-1, 1]
```

archive 原图为 `720×1280`；官方 loader 用 OpenCV resize 到 `704×1280`，不是裁掉 16 行。适配器严格先执行官方 raw-depth 路径的 `float32 depth/65535`，再用 `cv2.INTER_LINEAR` resize；然后为了满足 LiteViLNet 三通道 geometry 接口，把 resized depth 扩展为上述 `depth3`。标签用最近邻 resize，并在 LiteViLNet 的全分辨率输出上计算指标。训练 physical batch 为 8、无梯度累积。早期 `runs/revision_1/orfd_full_batch8_training_smoke.json` 是数据未完整 staging 时的诊断，不能用于论文；最终证据是完整 8,392-sample loader 上的 `orfd_full_batch8_training_smoke_complete.json`。

适配器对每张 PNG 执行“完整读取字节 → 严格 decode/load”，遇到瞬时 I/O 错误最多指数退避重试 5 次。它没有开启 PIL 的 `LOAD_TRUNCATED_IMAGES`：永久损坏文件会带完整路径失败，而不会以残缺像素继续训练。对应行为由 `tests/test_orfd_dataset.py` 约束。

正式复现实验可在两张卡上拆分为：

```bash
LITEVILNET_ORFD_ROOT=/dev/shm/litevilnet_orfd/Final_Dataset \
LITEVILNET_ORFD_OUTPUT=/data/Database/Research04-LiteViLNet/revision_1_runs/orfd_ablation \
bash tools/run_orfd_revision_queue.sh 0 full:42 full:40 full:41 add_lidar:42

LITEVILNET_ORFD_ROOT=/dev/shm/litevilnet_orfd/Final_Dataset \
LITEVILNET_ORFD_OUTPUT=/data/Database/Research04-LiteViLNet/revision_1_runs/orfd_ablation \
bash tools/run_orfd_revision_queue.sh 1 optimal:42 optimal:40 optimal:41 add_fusion:42
```

其中 `full` 与 `optimal` 报告三种子 mean ± sample-SD；`add_lidar`、`add_fusion` 是明确标为 `n=1` 的结构诊断，不能伪装成多种子结论。每个 ORFD run 最多 30 epochs、early-stop patience 10、physical batch 8、无梯度累积；`drop_last` 虽启用，但 8,392 可被 8 整除，因此不会丢弃训练帧。结果中的 `epochs_completed` 可能小于 30，这是预期的 early stopping，不是训练中断。

### 9.1 Checkpoint-selection validation 诊断

以下 validation 数字用于 early stopping 和 checkpoint selection，不再作为论文 Table III 的主结果。它们仍完整保留，便于复现选模过程（百分数；`full`/`optimal` 为 mean ± sample SD，诊断行为 `n=1`）：

| Variant | n | MaxF | AP | PRE | REC | IoU |
|---|---:|---:|---:|---:|---:|---:|
| full: MSFM + LKB + DS | 3 | 89.05 ± 1.54 | 90.69 ± 1.73 | 87.70 ± 2.80 | 90.47 ± 0.35 | 80.28 ± 2.48 |
| optimal: simple + LKB + DS | 3 | 90.40 ± 0.14 | 90.03 ± 0.44 | 90.87 ± 0.68 | 89.94 ± 0.90 | 82.48 ± 0.22 |
| add_lidar: simple only | 1 | 90.25 | 94.73 | 90.03 | 90.48 | 82.24 |
| add_fusion: MSFM only | 1 | 89.80 | 89.58 | 91.18 | 88.46 | 81.49 |

`full − optimal` 的成对结果（percentage points）：

```text
MaxF = −1.35 ± 1.40   (seed 40/41/42 全部为负)
AP   = +0.66 ± 1.82
PRE  = −3.17 ± 3.40
REC  = +0.53 ± 0.69
IoU  = −2.20 ± 2.26
```

这些是 checkpoint-selection partition 上的结构诊断，不能代替 held-out test 结论。`add_fusion − add_lidar = −0.45` MaxF point 只有一个 seed，只能视为诊断，不能讨论方差或显著性。

每 seed best epoch / early-stop 完成 epoch：

```text
full:    seed40 6/16, seed41 4/14, seed42 2/12
optimal: seed40 10/20, seed41 7/17, seed42 9/19
add_lidar seed42 1/11
add_fusion seed42 13/23
```

8 个 checkpoint 全部经过对应 config 的 `strict=True` state-dict load，checkpoint epoch/best metric 与 `result.json` 一致。SHA-256（依次为 full 40/41/42、optimal 40/41/42、add_lidar 42、add_fusion 42）：

```text
832ab22c90e429050bf4444c6d865e9cc1a6db981a9f73061fab53da44800ff2
bc3f86de7744afa4f27fbd2f3b2e27c797298cfd229f7a50c91678decdf1fd72
5fa2d064f4fa2b62fb1b41dfa632e945a044227c4fbcef5f141fc1355ea4bde2
69ee66aa9c605f3213a13b11087cd294c968efa08776f223ecb88e7c04e4b079
39109d92a1467127971031a8e80cc54e6f258a2bed7714771d2b437c1d7734af
2fca0a100ba9be520d0bbe3e08d97e9e6e4de3758b30dba08cbf5e7f2ec2510e
7a490778a0301eb372cb5282b620341515414ab50b60247018cf5260e8d6079d
eec4fe4098a0d4c6701d5a580830f859e1ddcafadb6fbb45155fb45cb6a0a434
```

汇总证据：

```text
runs/revision_1/orfd_summary.json
runs/revision_1/orfd_summary.csv
```

### 9.2 Held-out official testing 主结果

官方仓库 `chaytonmin/Off-Road-Freespace-Detection` commit
`50e63d24836198e8fb5af707e521f414104b4876` 的 `test.py` 使用二分类
`argmax` 预测，在恢复到原始图像尺寸后累计 confusion matrix；`util/util.py`
再由 foreground TP/FP/FN 计算 PRE、REC、F-score 和 IoU。为了与其 published
test row 可比，本轮 evaluator 对 LiteViLNet 使用等价口径：

1. 网络输入为 `704×1280`，与官方 ORFD 训练尺寸一致；
2. 单 logit 模型使用 `logit > 0`，即 sigmoid probability `> 0.5`；
3. 二值预测用 nearest-neighbor 恢复到原始 `720×1280` GT；
4. 2,193 帧只累计一个全局 confusion matrix；
5. checkpoint 只由 validation 选出，testing 不参与训练或选模；
6. training/test 和 validation/test 的文件名交集均为 0。

复现单个 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.evaluate_orfd \
  --config full \
  --checkpoint /data/Database/Research04-LiteViLNet/revision_1_runs/orfd_ablation/full/seed_42/best_model.pth \
  --data-root /data/Database/Research04-LiteViLNet/datasets/ORFD/extracted/Final_Dataset \
  --split test --img-h 704 --img-w 1280 \
  --batch-size 8 --num-workers 8 --precision fp16 --quiet \
  --output runs/revision_1/orfd_test/full_seed42.json
```

对 full/optimal 的 seeds 40/41/42 全部执行后汇总：

```bash
python -m tools.summarize_orfd_test \
  --input-dir runs/revision_1/orfd_test \
  --output-json runs/revision_1/orfd_test_summary.json \
  --output-csv runs/revision_1/orfd_test_summary.csv
```

Table III 使用的 official-protocol test 结果：

> 当前提交快照：USNet 使用已完成的 seed 40；OFF-Net 使用作者发布 checkpoint 的独立本地评测。SNE-RoadSeg、RoadFormer 和 OFF-Net 多 seed 重训命令仍保留，但不把未完成过程写成正式结果。

| Variant | n | F-score | AP | PRE | REC | IoU |
|---|---:|---:|---:|---:|---:|---:|
| USNet local seed 40 | 1 | 95.62 | 97.17 | 95.32 | 95.92 | 91.61 |
| OFF-Net released checkpoint, local test | 1 | 92.80 | 97.58 | 94.53 | 91.13 | 86.57 |
| OFF-Net published | -- | 90.30 | -- | 86.60 | 94.30 | 82.30 |
| full: MSFM + LKB + DS | 3 | **96.74 ± 0.09** | **98.31 ± 0.37** | 97.03 ± 0.44 | 96.45 ± 0.39 | 93.68 ± 0.18 |
| optimal/compact: simple + LKB + DS | 3 | 96.77 ± 0.27 | 97.60 ± 0.41 | 97.15 ± 0.74 | 96.39 ± 0.28 | 93.74 ± 0.52 |

其中 AP 是同一 test 上的 101-threshold sweep，用于补充衡量概率排序质量；其余
四项严格使用官方 fixed-argmax confusion-matrix 口径。Full 相对 compact：

```text
F-score = −0.03 ± 0.28 pp
AP      = +0.71 ± 0.67 pp  (seed 40/41/42 全部为正)
REC     = +0.06 ± 0.56 pp
IoU     = −0.06 ± 0.53 pp
```

因此论文的专业叙事是：full 与 compact 在 fixed operating point 的 F-score/IoU
基本等价；full 在三个 seeds 上均有更高 AP，且 F-score 跨 run 波动显著更小，体现
MSFM 的概率排序和稳定性价值；compact 则提供更低成本的部署选项。不得把这写成
full 在所有指标上都优于 compact。

原始/汇总证据：

```text
runs/revision_1/orfd_test/full_seed40.json
runs/revision_1/orfd_test/full_seed41.json
runs/revision_1/orfd_test/full_seed42.json
runs/revision_1/orfd_test/optimal_seed40.json
runs/revision_1/orfd_test/optimal_seed41.json
runs/revision_1/orfd_test/optimal_seed42.json
runs/revision_1/orfd_test_summary.json
runs/revision_1/orfd_test_summary.csv
```

## 10. 机器人 RGB-D 路径和修订图片复现

### 10.1 真实 depth3 定义

必须遵守仓库 `AGENTS.md`：

```text
depth_norm = clip(depth_mm, 0, 12000) / 12000
valid = 0 < depth_mm < 12000
inverse = valid × (1 - depth_norm)
depth3 = [depth_norm, valid, inverse]
depth3 -> [-1, 1]
```

机器人 checkpoint 是 session-specific fine-tuning，历史配置每个 session 只有 10–35 个手工标注帧。`train_metrics.json` 是训练拟合，不是独立泛化结果。

Jetson 已有可信的 model-only 记录：

```text
mode: MAXN_SUPER
input: 384×1248, batch 1, FP16
PyTorch RGB-D checkpoint: 22.18 ± 0.21 FPS
mean/P95: 45.09/46.36 ms
TensorRT FP16: 68.73 ± 0.06 FPS
mean/P95: 14.55/14.74 ms
scope: model-only; preprocessing and control excluded
power: not logged
```

机器人 native network input pipeline 的 clean RTX 4090 D 计时使用真实 400 对 PNG 中轮转的 200 帧，`800×1280`、batch 1、FP16、30 warm-up；它包括 PNG decode、RGB normalization、aligned depth→`depth3`、pinned transfer、模型和 mask postprocess，但不包括 controller：

| Stage | mean (ms) | sample SD (ms) | P95 (ms) |
|---|---:|---:|---:|
| decode + RGB normalize + depth3 | 85.02 | 7.91 | 99.40 |
| host→device | 2.31 | 0.13 | 2.52 |
| model | 7.69 | 0.24 | 8.04 |
| sigmoid/resize/threshold/device→host | 0.21 | 0.02 | 0.23 |
| **total** | **95.39** | **7.91** | **109.45** |

即该 Python/PNG pipeline 约 `10.48 FPS`，瓶颈是 CPU decode/depth3，不是 7.69-ms 模型。它不能替代 Jetson 或 controller 的端到端测量。

```text
evidence: runs/revision_1/robot_depth3_end_to_end_4090d_clean_800x1280.json
checkpoint SHA-256 = 4439c0fbd4646cbd931bd15d7a39271ef22b16602461d36d7759dc89c636de62
```

### 10.2 手工绘图参考素材的一键生成

脚本只向 `runs/revision_1/figure_materials/` 输出参考材料，不会覆盖 `LiteViLNetPaperRAL/figures/` 下的作者原图。架构图和 MSFM 图仅用于核对模块/箭头；最终投稿图由作者按 `LiteViLNetPaperRAL/FIGURE_MANUAL_REVISION_GUIDE_CN.md` 手工绘制：

```bash
NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
conda run -n litevilnet_ral python -m tools.generate_revision_figures \
  --only all \
  --device cuda:0 \
  --manifest runs/revision_1/revision_figure_manifest.json
```

脚本输出：

```text
LiteViLNet/runs/revision_1/figure_materials/fig_architecture2.png
LiteViLNet/runs/revision_1/figure_materials/fig_architecture.pdf
LiteViLNet/runs/revision_1/figure_materials/fig_msfm1.png
LiteViLNet/runs/revision_1/figure_materials/fig_msfm.pdf
LiteViLNet/runs/revision_1/figure_materials/real_experiment_all1.png
LiteViLNet/runs/revision_1/figure_materials/fig_qualitative.png
LiteViLNet/runs/revision_1/figure_materials/fig_qualitative.pdf
LiteViLNet/runs/revision_1/figure_materials/qualitative_panels/<sample_id>/*.png
LiteViLNet/runs/revision_1/revision_figure_manifest.json
```

脚本在输出目录与论文 `figures/` 目录相同时会直接报错，防止误覆盖原图。

### 10.3 Fig. 1 与 MSFM 图的手工修改要求

Fig. 1 手工修改后需要明确展示：

- KITTI 的 `LiDAR → calibrated projection → 21×21 height interpolation → 7×7 gradient → per-image normalization → 3-channel ADI` 位于网络外；
- RGB 五个 stage 使用 MobileNetV3；
- geometry 第一个 stage 是 two-convolution stem，后四个 stage 是 DSConv；
- 两路在五个尺度进入 MSFM，最深特征经过 Large-Kernel Bridge，再由 U-Net decoder 恢复分辨率；
- RGB-D 部署是独立的 `aligned depth → depth3` 路径，不再被画成 ADI。

Fig. 3 需要严格按 `attention_modules.py` 绘制：pooled RGB query、spatial ADI K/V、`B×1×N_l` attention、enhanced RGB residual、gate blend，以及最终 `3×3 Conv–BN–ReLU`。图中不能再存在双向 `N_l×N_l` attention 的暗示。

### 10.4 机器人组合图修正

参考素材中的 Unitree-G1 两个内容块已经交换；作者原图保持恢复状态，需按以下顺序手工交换，同时保留正确标签：

```text
RGB → aligned depth → segmentation overlay → confidence heatmap
```

脚本只在参考副本中通过 panel 黑色无效像素比例做检查。参考副本的 G1 Depth panel 黑色像素比例为 `0.02563`，Segmentation panel 为 `0`。单独的 Kuafu 四联图不需要修改。

### 10.5 KITTI 定性图证据

Fig. 4 的手工替换素材只使用固定分层 validation manifest 中的样本，并覆盖 UM、UMM、UU：

| Sample | Category | F1 | IoU |
|---|---|---:|---:|
| `um_000054` | UM | 0.994 | 0.988 |
| `umm_000017` | UMM | 0.991 | 0.982 |
| `umm_000050` | UMM | 0.980 | 0.960 |
| `uu_000059` | UU | 0.989 | 0.978 |

统一协议：

- checkpoint：`full/seed_42/best_model.pth`，best epoch 45；
- checkpoint SHA-256：`49f07b83fdad95dc7330c0d568e38532dcb2f748ef117e89a95ac824d957fa79`；
- 网络输入：`384×1248`；
- 所有样本使用 validation-global threshold `0.66`；
- logit 统一双线性恢复到原始样本尺寸后计算图中 F1/IoU；
- 每行依次显示对应的 RGB、stored ADI、prediction overlay 和 TP/FP/FN error map；
- 四个 ADI 的 SHA-256 均不相同，具体输入、逐面板素材和输出哈希记录于 `revision_figure_manifest.json`。

图中逐样本 F1/IoU 只用于解释定性样例，不参与 Table I/II 的全 validation 累计指标。

## 11. 论文和 Response 核验清单

提交前请逐项检查：

最终数字回填范围（只改这些审稿驱动位置，不扩大正文改动）：

- `root.tex` Abstract：将 published OFF-Net 的旧式单行比较替换为本地共同协议的
  ORFD 多方法结论；不写本机路径、manifest、hash 或 seed。
- Section IV-A：把 ORFD evaluator 表述改成四个 baseline 与 LiteViLNet 共用的
  fixed-argmax/original-GT protocol，并补一句作者官方源码、本地重训、validation-only
  选模；保留各方法自己的官方优化 recipe。
- Table I/Section IV-B：新增 OFF-Net 的三种子精度、25.21M 参数和最终 RTX 4090 D
  FPS-1，同时保留 USNet 高 FPS 的真实测量，不作选择性删除；USNet 的网络第二路来自
  官方 SNE 输出，Input 单元格应精准标为 `RGB+Normal`，而不是按其内部变量名写成
  `RGB+Depth`。
- Table III/Section IV-D：加入已完成的 USNet seed-40 本地结果与 OFF-Net 作者发布
  checkpoint 的独立核验，并增加 Params 列；SNE-RoadSeg/RoadFormer 的 ORFD 训练入口
  保留在补充材料中，PLARD 因没有官方 ORFD-compatible ADI 构造链而不做不成立的输入替换。
- Conclusion：只使用最终本地可比结果总结优势，不再用 published OFF-Net
  `90.30/82.30` 作为核心胜幅。
- `ral_response_1.tex`：同步更新 `\RevisedTableOne`、`\RevisedTableThree` 及所有引用
  旧 ORFD 胜幅的回复；礼貌说明基线集合按“作者官方、可本地重训和统一评测”原则调整，
  并在相关回复下重复完整新表，方便评审直接核验。
- 上述全部属于评审要求，正文使用精准 `cyan`；不在 Response 中讨论 self-audit-only
  的 `blue` 修改，也不修改任何论文图片。

最终 Table-I 输入标签核对表：

| 方法 | 网络实际第二路输入 | 正文 Input 单元格 |
|---|---|---|
| USNet | 官方 SNE surface normal | `RGB+Normal` |
| SNE-RoadSeg | 官方 SNE surface normal | `RGB+Normal` |
| PLARD | LiDAR-derived ADI | `RGB+ADI` |
| OFF-Net | OFF-Net 官方 SNE surface normal | `RGB+Normal` |
| RoadFormer | SNE-RoadSeg normal cache encoded for official loader | `RGB+Normal` |
| LiteViLNet | LiDAR-derived ADI | `RGB+ADI` |

- [x] Abstract、Table I、Table II、Conclusion 的 KITTI MaxF/参数/GMAC 数字来自同一个分层 split 汇总 JSON。
- [x] ORFD Table III 已先纳入 USNet seed-40 和 OFF-Net 发布 checkpoint 的共同 fixed-argmax evaluator 结果；LiteViLNet 现有数字来自 `orfd_test_summary.json`。SNE/RoadFormer 与 OFF-Net 多 seed 扩展命令保留待后续 GPU 空闲时运行。
- [x] Table I 的 USNet/SNE-RoadSeg/PLARD/OFF-Net/RoadFormer/LiteViLNet 六个精度行全部来自同一 231/58 split、尺寸、150-epoch budget、三种子和 evaluator，不再混合 official BEV 与 local PV 排名。
- [x] USNet/SNE-RoadSeg/PLARD/OFF-Net/RoadFormer 的官方仓库、固定 commit、源码哈希、逐 seed JSON 与 checkpoint SHA 均已整理到待打包结果树；最终匿名 Supplement 将在 ORFD/FPS 完成后重建。
- [ ] OFF-Net 的 RTX 4090 D FPS-1 尚待 GPU 独占后测量；当前 GPU 被外部任务占用，
  因此 Table I/Response 暂以 ``--'' 保持，不把未完成测速写成结果。
- [x] 论文没有 `best CNN`、`standard split`、`collision-free`、`RGB-D-compatible ADI`。
- [x] PyTorch 22.18/22.19 FPS 与 TensorRT 68.73 FPS 的 checkpoint/backend 已分开。
- [x] TensorRT 没有被用来声称 KITTI accuracy 等价。
- [x] 没有虚构机器人 success/intervention/collision/lateral-error/power。
- [x] cyan 已收缩到评审要求的协议、公式、关键数字和结论短语，不再整段着色。
- [x] Fig. 1、3、4、5 的手工绘制说明、参考副本和 Fig. 4 逐面板素材已准备，生成脚本不会覆盖作者原图。
- [ ] 最终 ORFD/Table-I 数字回填后重新编译论文与 Response，并再次检查 fatal、undefined reference、overfull 和页数；此前 8/24 页 PDF 只证明旧快照可编译。
- [ ] 作者按 `FIGURE_MANUAL_REVISION_GUIDE_CN.md` 手工覆盖 Fig. 1、3、4、5，并重新编译最终 PDF。
- [x] Response 每条都引用对应 Section/Table/Figure；图片宏继续引用论文原文件名，作者手工覆盖后会自动同步到 Response。

## 12. 证据限制

1. 没有 KITTI official test-server submission；本地 PV 数字不能建立官方排名。
2. 旧的预计算 ADI provenance 不完整，reference regeneration 与 stored PNG 不一致。
3. RTX 4060 Ti/Jetson 的历史 checkpoint FPS 汇总仅保存在 `FPS/FPS_README.md` 作为审计记录；当前 Table I 已改用可复核的 RTX 4090 D FPS-1，论文不再用历史 4060 Ti 数值作主结果。
4. 早期受共驻任务污染的 latency 只作诊断；论文 Table II 使用无其他 compute process 的 clean 三次 4090D profile。
5. ORFD 是在第二数据集上重新训练/验证，不是 KITTI→ORFD zero-shot generalization。
6. 机器人实验是 perception/control integration demonstration，不是受控导航 benchmark。
7. 没有功耗日志，不能声称 energy efficiency。
8. 机器人演示的主要未量化失效风险是少量 session-specific 标签导致的过拟合、无效/截断 depth（`M=0`），以及 false-positive 或碎片化 mask 对中心路径提取的影响；截图不足以恢复这些失效的发生率。

## 13. 主要代码改动索引

```text
litevilnet/data/dataset.py                 fixed manifest + deterministic worker seed
litevilnet/data/orfd_dataset.py            ORFD official-split RGB+dense-depth adapter + optional original-size test GT
litevilnet/data/adi.py                     PLARD reference ADI port
litevilnet/data/robot_road_dataset.py      truthful robot depth3 loader
litevilnet/metrics/deployment_metrics.py   exact histogram MaxF/AP acceleration
litevilnet/utils/common.py                 latency n/sample-std metadata
tools/train_ablation.py                    exact splits, seeds, determinism, ORFD, metadata
tools/train_distill_edge.py                split/seed/accumulation/reproducibility control
tools/create_kitti_split_manifests.py      historical + stratified manifest generator
tools/profile_ablation.py                  params/GMAC/memory/latency profiler
tools/benchmark_kitti_adi_pipeline.py      raw LiDAR-to-mask stage timing
tools/benchmark_robot_end_to_end.py        RGB-D depth3-to-mask stage timing
tools/smoke_orfd_training.py               ORFD real-batch forward/backward OOM smoke
tools/evaluate_orfd.py                     strict-checkpoint ORFD val/test evaluator + official fixed-argmax test metric
tools/summarize_orfd_test.py               held-out test multi-seed/paired summary JSON+CSV
tools/summarize_revision_experiments.py    mean ± sample std + seed-matched paired differences
tools/summarize_distillation_control.py    KD student 与同 seed 非 KD student 配对汇总
tools/summarize_profile_repeats.py         repeated profiler invocation mean ± sample std
tools/generate_revision_figures.py         只生成 Fig. 1/3/4/5 手工绘图参考素材与哈希清单，不覆盖论文原图
tools/fetch_matched_baseline_sources.sh    从作者官方仓库拉取并固定五个 baseline commit
tools/prepare_matched_kitti_baselines.py   构造同协议无泄漏软链接数据树
tools/prepare_matched_roadformer.py        为官方 RoadFormer loader 编码同源 normal 与 split tree
tools/cache_official_sne_normals.py        直接调用官方 SNE 并缓存确定性 float32 normal
tools/cache_official_orfd_normals.py       ORFD 完整标定核验与两套作者 SNE cache
tools/train_matched_kitti_baseline.py      USNet/SNE-RoadSeg/PLARD 官方模型与 loss 的同协议训练适配器
tools/train_matched_kitti_offnet.py        OFF-Net 官方图的 KITTI 同协议训练适配器
tools/train_matched_kitti_roadformer.py    RoadFormer 官方图/MMCV 算子的同协议训练适配器
tools/train_matched_orfd_baseline.py       USNet/SNE-RoadSeg/OFF-Net 的 ORFD 同协议训练适配器
tools/train_matched_orfd_roadformer.py     RoadFormer ORFD 官方图训练适配器
tools/summarize_matched_kitti_baselines.py 严格来源/协议/checkpoint 核验与 mean/sample-SD 汇总
tools/benchmark_matched_kitti_fps.py       六方法统一 RTX 4090 D FP32 FPS-1 benchmark
tools/summarize_matched_kitti_fps.py       FPS 协议/来源/参数量核验与 JSON/CSV 汇总
tools/package_table1_matched_baselines.sh  Table I 轻量匿名复现包
tools/package_ral_reproduction.sh          全部修订证据的双盲匿名复现包
tools/run_revision_ablation_queue.sh       reproducible KITTI queue
tools/run_kitti_distill_queue.sh           reproducible three-seed KD-control queue
tools/run_orfd_revision_queue.sh           reproducible ORFD queue
tools/run_post_revision_benchmarks.sh      GPU 独占后的 profile/smoke/两条流水线
tests/test_deployment_metrics.py           metric equivalence test
tests/test_adi.py                          ADI reference-operation tests
tests/test_orfd_dataset.py                 ORFD discovery/depth3/label/strict-image-retry tests
tests/test_robot_road_dataset.py           robot depth3 数值和 validity 边界测试
tests/test_revision_figure_manifest.py     素材输出目录保护、哈希、validation 样本、G1 顺序和图结构契约测试
```

## 14. 双盲附件交付

旧版 Table I 轻量复现包（已作废，仅供本机追溯，禁止投稿）：

```text
dist/LiteViLNet_RAL_TableI_Reproduction.tar.gz
SHA-256 = 0414aaffb67692137bf114011f0f8def05f8d834613c36cc94880363d2ae2ed8
```

旧版完整修订复现包（已作废，仅供本机追溯，禁止投稿）：

```text
dist/LiteViLNet_RAL_Anonymous_Reproduction.tar.gz
SHA-256 = bb1a8e9071e804d24ce6eb08657466360f9192899e05aa260e420e987b4b54ab
```

上述旧包已由当前快照替换。当前可交付包为：

```text
dist/LiteViLNet_RAL_Anonymous_Reproduction.tar.gz
SHA-256 = 9949bc3eae45d8ae154019df0a5081d64b9b7c35457fe4ed99ec67e4a5930a0c
dist/LiteViLNet_RAL_TableI_Reproduction.tar.gz
SHA-256 = e5e5e661dd26d806d4e79c69066ca17e5dbad12b44b1d5108daac17e0f51e989
```

两包均已重新执行 companion hash、tar owner/group、身份/绝对路径扫描；OFF-Net
KITTI FPS-1 在目标 GPU 被外部任务占用期间保留为论文中的 ``--''，不影响其余
精度、复现代码和匿名证据交付。

当前两包已完成以下检查：

- companion `.sha256` 校验通过；
- 包内 `ARTIFACT_MANIFEST.sha256` 的逐文件校验通过；
- tar owner/group 固定为数值 `0/0`，gzip 不记录原文件名/时间；
- 作者姓名、单位域名、本机账户、邮箱、SSH remote、Linux/Windows 绝对路径 deny-scan 通过；
- 不含数据、checkpoint、normal cache、第三方源码、Git 历史、论文源码或图片；
- 官方第三方代码由包内 fetch 脚本从作者仓库的固定 commit 重建。

注意：此处是内部作者文档，包含本机路径与审计细节，不能随双盲附件上传。若投稿系统要求上传 `root.tex` 而不只是 PDF，还必须真正删除作者栏注释中的实名/单位/邮箱；仅用 `%` 注释不会匿名化 TeX 源文件。
