# LiteViLNet RA-L 第一轮修改、证据审计与复现说明

> 状态：全部数值实验、checkpoint 审计、论文/Response 数值回填、四组修订图片和代码测试均已完成；本文不再含实验或图片占位符。

## 1. 这轮修改的核心结论

这轮工作不是简单润色，而是对论文、代码、数据划分、指标、公式、速度口径和机器人实验做了一次证据审计。最重要的发现是：

1. 原论文把本地 KITTI perspective-view 验证和官方 KITTI BEV test-server 结果放在同一张表里直接排名，这是不成立的。LiteViLNet 没有官方 test-server submission。
2. 历史代码按文件名排序后直接做 80/20 切分，导致 58 张验证图全部属于 `UU`。修订实验改用公开种子 `20260723` 的固定分层随机 231/58 split；历史 split 和结果只用于追溯。
3. 原 CMA 公式不是实际实现。代码只生成一个全局 RGB query，并对 LiDAR 的 `N=H_lW_l` 个空间 key/value 做 `1×N` 注意力；不存在 `N×N` 矩阵。
4. 原 ADI、loss、geometry encoder 和 RGB-D 部署描述均与代码或官方参考实现不一致，现已按真实实现重写。
5. 机器人分支使用 aligned depth 生成 `depth3`，不是 ADI，也不是 KITTI 模型 zero-shot transfer。历史导航没有保留成功率、碰撞、干预、横向误差或功耗日志，因此只能作为定性系统演示。
6. MSFM 增加约 10.35M 参数，必须与“简单逐尺度相加 + 同一 Bridge + 同一深监督”控制一起看。修正后的分层 split 结果见第 7 节。
7. ORFD 用作第二数据集。除 8,392/1,245 train/validation 外，本轮进一步发现下载包的 2,193-frame testing partition 也包含完整 GT；训练和 validation 都与 test 无同名帧。论文现以 held-out test 作为主结果。
8. 按官方 OFF-Net commit `50e63d2` 的固定 argmax、原始 `1280×720` GT confusion-matrix 口径，full 为 `96.74 ± 0.09% F-score / 93.68 ± 0.18% IoU`，相对 OFF-Net 公布值高 `6.44/11.38 pp`。Full 的 AP 为 `98.31 ± 0.37%`，对 compact 的 seed-matched 优势是 `+0.71 ± 0.67 pp`，三个 seeds 均为正；其 F-score 标准差也约为 compact 的三分之一。

## 2. 审稿意见—修改证据矩阵

| 审稿关注 | 修改位置 | 证据/实验 |
|---|---|---|
| 只有 KITTI，数据太小 | 论文 Section IV-A、IV-D、Table III | ORFD 官方 train/val/test，8,392/1,245/2,193 帧；held-out test F-score `96.74±0.09%` |
| KITTI 比较不公平、数字错误 | Section IV-A、IV-B、Table I | 官方 BEV 与本地 PV 分组；修正 USNet、PLARD、SNE-RoadSegV2、RoadFormer |
| 缺少多种子 | Section IV-A、IV-C、Tables I–II | seeds 40/41/42，mean ± sample std |
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

### 7.1 主分层 split

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

### 7.2 历史 UU-only split

历史三种子可追溯结果：

| Variant | MaxF mean ± sample std |
|---|---:|
| RGB-only baseline | 95.56 ± 0.17 |
| + LiDAR, simple addition | 95.82 ± 0.12 |
| Simple addition + Bridge + DeepSup | 95.98 ± 0.10 |
| Full MSFM + Bridge + DeepSup | 96.16 ± 0.08 |

这些值不能替代分层 split 主结果，也不能与 KITTI official BEV server 排名。

### 7.3 结构成本

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

### 7.4 延迟与显存

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

| Variant | n | F-score | AP | PRE | REC | IoU |
|---|---:|---:|---:|---:|---:|---:|
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

### 10.2 四组图片的一键生成

四组图片已由确定性脚本生成/修正，正文和 Response 引用的是同一组 PNG：

```bash
NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
conda run -n litevilnet_ral python -m tools.generate_revision_figures \
  --only all \
  --device cuda:0 \
  --manifest runs/revision_1/revision_figure_manifest.json
```

脚本输出：

```text
LiteViLNetPaperRAL/figures/fig_architecture2.png
LiteViLNetPaperRAL/figures/fig_architecture.pdf
LiteViLNetPaperRAL/figures/fig_msfm1.png
LiteViLNetPaperRAL/figures/fig_msfm.pdf
LiteViLNetPaperRAL/figures/real_experiment_all1.png
LiteViLNetPaperRAL/figures/fig_qualitative.png
LiteViLNetPaperRAL/figures/fig_qualitative.pdf
LiteViLNet/runs/revision_1/revision_figure_manifest.json
```

### 10.3 Fig. 1 与 MSFM 图的实现对齐

新版 Fig. 1 明确展示：

- KITTI 的 `LiDAR → calibrated projection → 21×21 height interpolation → 7×7 gradient → per-image normalization → 3-channel ADI` 位于网络外；
- RGB 五个 stage 使用 MobileNetV3；
- geometry 第一个 stage 是 two-convolution stem，后四个 stage 是 DSConv；
- 两路在五个尺度进入 MSFM，最深特征经过 Large-Kernel Bridge，再由 U-Net decoder 恢复分辨率；
- RGB-D 部署是独立的 `aligned depth → depth3` 路径，不再被画成 ADI。

新版 Fig. 3 严格按 `attention_modules.py` 绘制：pooled RGB query、spatial ADI K/V、`B×1×N_l` attention、enhanced RGB residual、gate blend，以及最终 `3×3 Conv–BN–ReLU`。图中不再存在双向 `N_l×N_l` attention 的暗示。

### 10.4 机器人组合图修正

Unitree-G1 的两个内容块已经确定性交换，同时保留正确标签：

```text
RGB → aligned depth → segmentation overlay → confidence heatmap
```

脚本通过 panel 黑色无效像素比例做幂等检查。最终 G1 Depth panel 的黑色像素比例为 `0.02563`，Segmentation panel 为 `0`；第二次运行会识别为 `already corrected`，不会再次交换。单独的 Kuafu 四联图保持原有正确顺序。

### 10.5 KITTI 定性图证据

旧图的四个 UU 样本和重复 ADI 已全部替换。新版 Fig. 4 只使用固定分层 validation manifest 中的样本，并覆盖 UM、UMM、UU：

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
- 四个 ADI 的 SHA-256 均不相同，具体输入和输出哈希记录于 `revision_figure_manifest.json`。

图中逐样本 F1/IoU 只用于解释定性样例，不参与 Table I/II 的全 validation 累计指标。

## 11. 论文和 Response 核验清单

提交前请逐项检查：

- [x] Abstract、Table I、Table II、Conclusion 的 KITTI MaxF/参数/GMAC 数字来自同一个分层 split 汇总 JSON。
- [x] Abstract、Table III、Conclusion 的 ORFD test 数字来自 `orfd_test_summary.json`，且 F/PRE/REC/IoU 使用官方 fixed-argmax 原始 GT 尺寸口径。
- [x] Table I 的 official BEV 和 local PV 分组没有直接排名文字。
- [x] USNet 为 `96.89/96.51/97.27`；single-scale PLARD 为 `96.83/96.79/96.86`。
- [x] 论文没有 `best CNN`、`standard split`、`collision-free`、`RGB-D-compatible ADI`。
- [x] PyTorch 22.18/22.19 FPS 与 TensorRT 68.73 FPS 的 checkpoint/backend 已分开。
- [x] TensorRT 没有被用来声称 KITTI accuracy 等价。
- [x] 没有虚构机器人 success/intervention/collision/lateral-error/power。
- [x] 相对原稿新增或重写的可见正文、公式、标题、图注与表格关键信息均以 cyan 标出；纯 LaTeX 结构命令不着色。
- [x] Fig. 1、MSFM 图、KITTI 定性图、Unitree-G1 两格已按第 10 节修改，并由 manifest 哈希验证。
- [x] `latexmk -pdf root.tex` 和 `latexmk -pdf ral_response_1.tex` 无 fatal/undefined reference/overfull；最终 PDF 分别为 8/22 页，论文末页参考文献已平衡。
- [x] Response 每条都引用对应 Section/Table/Figure，并在相关回复中直接复现 Tables I--III 与 Figs. 1、3、4、5--6。

## 12. 证据限制

1. 没有 KITTI official test-server submission；本地 PV 数字不能建立官方排名。
2. 旧的预计算 ADI provenance 不完整，reference regeneration 与 stored PNG 不一致。
3. RTX 4060 Ti/Jetson 的历史 checkpoint FPS 汇总保存在 `FPS/FPS_README.md`，但 README 指向的 Windows `F:` 原始结果目录不在当前 Linux 主机；论文使用这些数字时必须保留这一证据限制。
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
tools/generate_revision_figures.py         Fig. 1/Fig. 3/Fig. 4/机器人组合图确定性生成与哈希清单
tools/run_revision_ablation_queue.sh       reproducible KITTI queue
tools/run_kitti_distill_queue.sh           reproducible three-seed KD-control queue
tools/run_orfd_revision_queue.sh           reproducible ORFD queue
tools/run_post_revision_benchmarks.sh      GPU 独占后的 profile/smoke/两条流水线
tests/test_deployment_metrics.py           metric equivalence test
tests/test_adi.py                          ADI reference-operation tests
tests/test_orfd_dataset.py                 ORFD discovery/depth3/label/strict-image-retry tests
tests/test_robot_road_dataset.py           robot depth3 数值和 validity 边界测试
tests/test_revision_figure_manifest.py     图片输出哈希、validation 样本、G1 顺序和图结构契约测试
```
