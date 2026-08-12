# Table I 同协议基线复现手册（中文）

## 目的

这套材料用于复现修订稿 Table I 的核心精度比较。USNet、SNE-RoadSeg、PLARD、
RoadFormer、OFF-Net 和 LiteViLNet 全部使用同一组 KITTI 训练/验证样本、同一 `384 x 1248` 输入尺寸、
同一个像素累计的 101 阈值 MaxF/PRE/REC 脚本。表中不再把官方 BEV test-server
结果与本地 perspective-view 验证结果放在一起排名。

## 官方来源与本地代码边界

| 方法 | 作者官方仓库 | 固定 commit | 训练时直接导入的官方定义 |
|---|---|---|---|
| USNet | <https://github.com/morancyc/USNet> | `d761158ad42df7dcb62fa257dd02ce11c85f94a5` | `model/usnet.py`、`loss.py` |
| SNE-RoadSeg | <https://github.com/hlwang1124/SNE-RoadSeg> | `5e7900bfd59887634ced687ffe85a73018a38659` | `models/networks.py`、`models/roadseg_model.py`、`models/sne_model.py` |
| PLARD | <https://github.com/zhechen/PLARD> | `44485803092e729661c696ab6c03f6f2fabc8701` | `ptsemseg/models/plard.py`、`ptsemseg/loss.py`、KITTI loader |
| RoadFormer | <https://github.com/LiJiahang617/Road-Former> | `f675a3467cb168ebc727648390c304279bbcb079` | 官方 TwinConvNeXt、RoadFormer head/pixel decoder 与 KITTI config |
| OFF-Net | <https://github.com/chaytonmin/Off-Road-Freespace-Detection> | `50e63d24836198e8fb5af707e521f414104b4876` | 官方 MiT-B2 fusion、loss、SNE 与 ORFD loader |

Table I 的纳入规则以本地可复现性为核心：必须能从作者官方仓库取得完整可训练网络和
所需输入链路，并且能在同一协议下本地重训和测速该精确计算图。只能核对论文/server
数字，或只能找到不兼容第三方旧版 port 的方法仍保留在 Related Work，但不再与本地
perspective-view 结果直接排名。按此规则，USNet/SNE-RoadSeg/OFF-Net 覆盖 CNN 融合路线，
PLARD 代表经典 RGB--LiDAR CNN，RoadFormer 代表 Transformer RGB--normal 路线。

推荐直接运行随附脚本，从作者官方 GitHub 仓库拉取并切到上述 detached commit：

```bash
bash tools/fetch_matched_baseline_sources.sh third_party/matched_baselines
export USNET_SOURCE="$PWD/third_party/matched_baselines/USNet"
export SNE_SOURCE="$PWD/third_party/matched_baselines/SNE-RoadSeg"
export PLARD_SOURCE="$PWD/third_party/matched_baselines/PLARD"
export ROADFORMER_SOURCE="$PWD/third_party/matched_baselines/Road-Former"
export OFFNET_SOURCE="$PWD/third_party/matched_baselines/OFF-Net"
```

该脚本不会使用镜像仓库或第三方 fork；如果目标目录不是 Git 仓库或包含未提交修改，
会拒绝覆盖。训练适配器还会再次检查 commit，并用忽略 CRLF/LF 行尾差异的方式把
工作树与该 commit 比较；出现语义修改时默认中止。

- `depth_u16`：来自 SNE-RoadSeg 官方 README 给出的下载链接，压缩包 SHA-256 为
  `d32bf0052ec81f87996c0c7ca2e86952b9f780ad97b7347151e62def5f8efb92`。
- ImageNet 初始化权重沿用官方源码指定的 torchvision ResNet-18/ResNet-152 V1
  权重，其 SHA-256 分别为
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` 和
  `394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58`。
- RoadFormer 官方 KITTI config 内嵌的 ConvNeXt-Base ImageNet-21K 权重下载后
  SHA-256 为 `262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d`。
- 五个官方源码目录的 remote、commit、语义 diff 和关键文件哈希都会由脚本复核；
  当前正式实验使用的 PLARD/RoadFormer/OFF-Net 是 clean fresh clone，USNet/SNE-RoadSeg
  工作副本仅有 CRLF/LF 行尾差异，忽略行尾后的 git diff 为零。

我们没有重新实现基线网络。`train_matched_kitti_baseline.py` 直接导入官方模型和
loss，只补充 split 接口、seed、输出元数据与统一评测。每个 `result.json` 都会记录
官方 remote、commit、关键源码 SHA-256、checkpoint SHA-256、完整命令、软件版本和
GPU 型号。

本地训练循环不是冒充成官方脚本：它直接调用官方网络/loss，并逐项对齐固定版本中
记录 recipe 的文件。USNet 的依据是 `train.py`、`utils.py`、`dataset/kitti.py` 和
`dataset/custom_transforms.py`；SNE-RoadSeg 的依据是 `train.py`、
`data/kitti_dataset.py`、`options/*.py`、`models/networks.py` 与
`models/roadseg_model.py`；PLARD 的依据是官方 `train.py`、loader、model 和 loss；
OFF-Net 的依据是官方 `train.py`、ORFD loader、model、loss 与 SNE；RoadFormer 的
依据是官方 KITTI config 及其 vendored OpenMMLab 模块。这些文件的 Git blob 与本次
工作副本 SHA-256 均列在
`source_provenance.json`，所以可以直接审查哪些行为来自官方、哪些是统一协议适配。
每个文件还同时记录 fresh clone 的 LF SHA-256 和正式实验所用 CRLF 工作副本的
SHA-256，避免把纯行尾差异误判为代码改动。

更具体地说，适配器保留 USNet 的 ResNet-18 双分支、evidential objective、
AdamW/backbone 学习率比例、poly schedule 与官方增强；保留 SNE-RoadSeg 的双
ResNet-152、官方 SNE normal、cross-entropy、SGD 和分段衰减；保留 PLARD 的
RGB--ADI 网络、三路监督与 SGD；保留 OFF-Net 的 MiT-B2 fusion、cross-entropy、
SGD 和 SNE；保留 RoadFormer 的 TwinConvNeXt-B、Hungarian matching losses、AdamW
和 Transformer decoder。我们新增的部分仅为：

USNet 从 `depth_u16` 调用其 vendored 官方 SNE，再把三通道 surface normal 送入网络；
因此 Table I 的网络输入应写为 `RGB+Normal`，不能根据内部 `depth` 变量名误写成
`RGB+Depth`。

1. 让五种官方模型读取同一份无泄漏 231/58 manifest；
2. 把确定性的官方 SNE 输出缓存为 float32，以免每个 epoch 重复计算；
3. 将训练预算统一为 150 epochs，并显式设置三次独立运行的 seed；
4. 对所有方法调用同一个像素累计、101 阈值评测器并按 validation MaxF 选模；
5. 保存完整 provenance 和逐 seed 原始 JSON。

因此，补充材料中的五种 baseline 数值是“官方模型与官方训练 recipe + 本文统一
数据/预算/评测外壳”的本地重训练结果，不是对网络结构的第三方重写，也不是从论文
表格转录的官方 test-server 数字。机器可读来源见 `source_provenance.json`。

## 关键复现脚本分别做什么

1. `fetch_matched_baseline_sources.sh`
   - 只克隆作者官方 GitHub 仓库；
   - 固定并复核上述五个完整 commit；
   - 遇到非官方 remote、非 Git 目录或 dirty worktree 时拒绝覆盖。
2. `prepare_matched_kitti_baselines.py`
   - 读取版本化的 231/58 分层 manifest；
   - 检查重复 ID 和 train/val 泄漏；
   - 检查 RGB、GT、calib、ADI、depth 是否齐全；
   - 用软链接生成 USNet/SNE-RoadSeg/PLARD 都能读取的目录，不复制原始图片。
3. `cache_official_sne_normals.py`
   - 分别直接导入 SNE-RoadSeg 与 OFF-Net 的官方 `models/sne_model.py`；
   - 按 `depth_u16/1000` 得到米制深度；
   - 缓存原始 normal 和 USNet“先翻转 depth、再算 SNE”的训练变体；
   - 保存 float32，不改变数值算法。

USNet 仓库还随附了自己的 `model/sne_model.py`。与 SNE-RoadSeg 官方文件相比，
它只把固定的 top-crop 判断包装成默认开启的 `crop_top=True` 选项；USNet 官方
dataset 正是用该默认值。因此缓存器使用 SNE-RoadSeg 作者仓库中的原始实现，并为
USNet 保留“先翻转 depth 再调用同一 SNE”的官方次序；两份官方实现处于该默认设置
时执行相同数值路径。两份文件的固定 commit 和 SHA-256 都记录在
`source_provenance.json`。
4. `prepare_matched_roadformer.py`
   - 把同一 float32 SNE normal 确定性映射为 RoadFormer 官方 loader 所需的 uint16 PNG；
   - 生成独立的 231/58 目录并再次记录 manifest hash。
5. `train_matched_kitti_baseline.py`
   - USNet 保留官方 ResNet-18、evidential loss、AdamW 和 poly schedule；
   - SNE-RoadSeg 保留官方双 ResNet-152、cross entropy、SGD 和衰减策略；
   - PLARD 保留官方 RGB--ADI 图、三路 loss、SGD、冻结 BN 和渐进衰减；
   - 都按相同 validation MaxF 选 best checkpoint。
6. `train_matched_kitti_roadformer.py` / `train_matched_kitti_offnet.py`
   - 分别直接注册 RoadFormer 与 OFF-Net 官方 Python graph，不修改第三方源码；
   - 把官方 `mmcv_custom` 算子名映射到同 ABI 的 MMCV 1.7 CUDA 算子；
   - 使用官方 FP32 路径、TwinConvNeXt-B 初始化、loss、AdamW 和梯度裁剪。
7. `run_matched_kitti_baselines.sh` / `run_matched_kitti_offnet.sh`
   - 在两张 GPU 上顺序运行五种 baseline 的三个 seed；
   - 训练完成后自动调用汇总器，不把并发训练 wall-clock 当作推理速度证据。
8. `summarize_matched_kitti_baselines.py`
   - 先核对 seed、split hash、输入尺寸、epoch、AMP、官方 commit 和参数量；
   - 再生成逐 seed JSON、副本、CSV 以及 mean $\pm$ sample standard deviation。
9. `benchmark_matched_kitti_fps.py`
   - 六种模型统一使用 RTX 4090 D、`384 x 1248`、batch 1、PyTorch FP32；
   - 用 CUDA event 测量 resident inputs 的 model-only latency，执行相同 warmup/iteration/repeat。
10. `run_matched_kitti_fps.sh`
   - 在一张空闲 RTX 4090 D 上顺序调用六种方法的统一测速；
   - 自动切换 RoadFormer/OFF-Net 独立环境，并在全部完成后调用汇总器。
11. `summarize_matched_kitti_fps.py`
   - 拒绝设备、精度、shape、timing scope、参数量或官方 commit 不一致的测速结果；
   - 生成六方法 FPS-1 JSON/CSV 和匿名逐方法证据。
12. `sanitize_table1_supplement.py`
   - 只在临时打包副本中匿名化本地路径，不改写原始实验 JSON；
   - 扫描 home/data 路径、账户名、邮箱和 SSH remote，发现泄漏立即失败。
13. `package_table1_matched_baselines.sh`
   - 检查十五份逐 seed JSON、六份 FPS JSON 与汇总证据是否完整；
   - 生成可投稿的小型附件和 SHA-256，不携带数据、checkpoint、cache 或第三方代码。

## 一次完整复现

先创建环境：

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda activate litevilnet_ral
```

RoadFormer 和 OFF-Net 使用其官方代码兼容的独立旧版环境：

```bash
conda env create -f configs/environments/litevilnet_roadformer_ral.yml
```

该环境固定 Python 3.8.20、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117
和 MMCV-full 1.7.0。官方多尺度可变形注意力 CUDA 算子只支持 FP32，因此 RoadFormer
训练和六方法统一 FPS-1 都明确使用 FP32，不用隐式精度差异制造速度优势。

准备数据软链接树。`KITTI_ROOT` 指向 KITTI Road 的 RGB/GT/calib/ADI 根目录；
`DEPTH_ROOT` 指向从 SNE-RoadSeg 官方链接下载并解压后的目录：

```bash
export KITTI_ROOT=/path/to/kitti_road
export DEPTH_ROOT=/path/to/extracted_depth_archive
export MATCHED_ROOT=runs/revision_1/matched_baselines/kitti_road

python tools/prepare_matched_kitti_baselines.py \
  --data-root "$KITTI_ROOT" \
  --depth-root "$DEPTH_ROOT/depth_u16" \
  --train-file configs/splits/kitti_road/stratified_seed20260723/train.txt \
  --val-file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --output-root "$MATCHED_ROOT"
```

用固定版本的官方 SNE 实现生成一次 deterministic normal 缓存：

```bash
python tools/cache_official_sne_normals.py \
  --data-root "$MATCHED_ROOT" \
  --official-source "$SNE_SOURCE" \
  --output-root "$MATCHED_ROOT" \
  --workers 8
```

OFF-Net 使用其自身固定版本的 SNE，必须生成独立 cache：

```bash
export OFFNET_SOURCE="$PWD/third_party/matched_baselines/OFF-Net"
export KITTI_OFFNET_NORMAL_ROOT=runs/revision_1/matched_baselines/kitti_offnet_normals
conda run -n litevilnet_roadformer_ral env PYTHONPATH=. \
  python tools/cache_official_sne_normals.py \
  --data-root "$MATCHED_ROOT" --official-source "$OFFNET_SOURCE" \
  --output-root "$KITTI_OFFNET_NORMAL_ROOT" --profile offnet --workers 4
```

再把同一 normal cache 确定性编码为 RoadFormer 官方 loader 所需的 uint16 PNG：

```bash
export ROADFORMER_DATA_ROOT=runs/revision_1/matched_baselines/roadformer_kitti
python tools/prepare_matched_roadformer.py \
  --matched-root "$MATCHED_ROOT" \
  --train-file configs/splits/kitti_road/stratified_seed20260723/train.txt \
  --val-file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --output-root "$ROADFORMER_DATA_ROOT"
```

最后在两张 GPU 上分别顺序训练原有四种方法的三个 seed：

```bash
export OUTPUT_ROOT=runs/revision_1/matched_baselines/formal
export USNET_GPU=0
export SNE_GPU=1
export USNET_SOURCE="$PWD/third_party/matched_baselines/USNet"
export SNE_SOURCE="$PWD/third_party/matched_baselines/SNE-RoadSeg"
export PLARD_SOURCE="$PWD/third_party/matched_baselines/PLARD"
export ROADFORMER_SOURCE="$PWD/third_party/matched_baselines/Road-Former"
export MATCHED_ROOT=runs/revision_1/matched_baselines/kitti_road
export ROADFORMER_DATA_ROOT=runs/revision_1/matched_baselines/roadformer_kitti
export SEEDS="40 41 42"
export EPOCHS=150
bash tools/run_matched_kitti_baselines.sh
```

再用同一协议运行 OFF-Net 三个 seed；汇总器会在五种 baseline 全部齐全后出表：

```bash
export KITTI_OFFNET_NORMAL_ROOT
export OFFNET_SOURCE
bash tools/run_matched_kitti_offnet.sh
```

若只复现单个 seed，可直接照英文手册第 5 节调用
`train_matched_kitti_baseline.py`、`train_matched_kitti_roadformer.py` 或
`train_matched_kitti_offnet.py`。不要把训练进程的 wall-clock 用作 Table I
推理速度；论文 FPS-1 来自单模型、batch-1、FP32 的独立 RTX 4090 D benchmark。

最重要的输出如下：

```text
runs/revision_1/matched_baselines/formal/<method>_seed<seed>/
  best_model.pth
  train_metrics.jsonl
  result.json

docs/ral/table1_matched_baselines/results/
  summary.json
  summary.csv
  seeds/*.json
  fps_4090d_summary.json
  fps_4090d_summary.csv
  fps/*.json
```

`summary.json` 只有在三个 seed、split hash、输入尺寸、epoch budget、官方 commit、
参数量和各方法预期的 batch/AMP 设置全部一致时才会生成。论文中的 `mean +- sample std` 从这个文件
自动取得，不手工挑结果或转录。

写入 `docs/` 的逐 seed 副本默认通过 `--anonymous-seed-copies` 去除本机绝对路径；
`runs/` 中的原始 JSON 保持不变供本地审计。指标、源码/checkpoint hash、commit、GPU
和软件版本不会因匿名化而改变。

## 统一 FPS-1 复现

精度训练完成后，使用每种方法任一完成 seed 的 best checkpoint 测速。六种方法必须
统一为 RTX 4090 D、`384 x 1248`、batch 1、PyTorch FP32、100 次 warmup、每次
300 个 timed iterations、三次 repeat；输入 tensor 常驻 GPU，CUDA event 只包围
model forward。解码、预处理、H2D 和后处理均不计入 FPS-1。

六种方法可以在激活 `litevilnet_ral` 后用统一入口顺序测速；RoadFormer 和 OFF-Net 会由脚本自动
切到其独立环境，并在全部完成后自动汇总：

```bash
export LITEVILNET_CHECKPOINT=/path/to/litevilnet/full/best_model.pth
export USNET_CHECKPOINT=runs/revision_1/matched_baselines/formal/usnet_seed40/best_model.pth
export SNE_CHECKPOINT=runs/revision_1/matched_baselines/formal/sne_roadseg_seed40/best_model.pth
export PLARD_CHECKPOINT=runs/revision_1/matched_baselines/formal/plard_seed40/best_model.pth
export ROADFORMER_CHECKPOINT=runs/revision_1/matched_baselines/formal/roadformer_seed40/best_model.pth
export OFFNET_CHECKPOINT=runs/revision_1/matched_baselines/formal/offnet_seed40/best_model.pth
export FPS_GPU=0
bash tools/run_matched_kitti_fps.sh
```

若六份逐方法 JSON 已存在、只需重新汇总，则运行：

```bash
python tools/summarize_matched_kitti_fps.py \
  --input-root runs/revision_1/matched_baselines/fps_4090d \
  --output-json docs/ral/table1_matched_baselines/results/fps_4090d_summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/fps_4090d_summary.csv \
  --result-output-dir docs/ral/table1_matched_baselines/results/fps \
  --anonymous-result-copies
```

汇总器会核对六个 JSON 的 GPU 名、精度、输入、timing scope、重复次数、参数量、
checkpoint hash 和官方 commit，任一不一致就拒绝出表。

## 正式结果快照

正式三种子 perspective-view validation 结果如下（百分数，mean ± sample SD）：

| 方法 | Seeds | MaxF | PRE | REC | 参数量 |
|---|---|---:|---:|---:|---:|
| USNet | 40/41/42 | 97.88 ± 0.07 | 98.03 ± 0.03 | 97.73 ± 0.11 | 30.74M |
| SNE-RoadSeg | 40/41/42 | 97.23 ± 0.21 | 97.39 ± 0.30 | 97.06 ± 0.13 | 201.32M |
| PLARD | 40/41/42 | 95.25 ± 0.19 | 95.46 ± 0.29 | 95.03 ± 0.09 | 76.93M |
| OFF-Net | 40/41/42 | 95.36 ± 0.66 | 94.88 ± 0.92 | 95.83 ± 0.57 | 25.21M |
| RoadFormer | 40/41/42 | 97.28 ± 0.05 | 97.96 ± 0.09 | 96.61 ± 0.16 | 206.86M |
| LiteViLNet | 40/41/42 | 97.23 ± 0.15 | 97.31 ± 0.59 | 97.16 ± 0.30 | 14.04M |

未四舍五入的指标、best epoch、官方源码哈希和 baseline best checkpoint SHA-256 位于
`results/summary.json` 与 `results/seeds/*.json`；LiteViLNet 的对应证据位于完整匿名附件
的主训练队列。Table I 因此可以直接核对为同 split/尺寸/预算/evaluator 的六方法结果。

RTX 4090 D 上统一复测的 FPS-1 为：USNet `239.81`、SNE-RoadSeg `19.32`、PLARD
`26.97`、RoadFormer `17.46`、OFF-Net `65.04`、LiteViLNet `216.61`。协议是 `384 x 1248`、batch 1、
PyTorch FP32、输入常驻 GPU、100 次 warmup、300 次计时和三次独立 repeat，CUDA event
只包围 model forward。Jetson FPS-2 只在论文中保留已有的匹配 Orin NX 测量，不把未测
配置填成推断值。

USNet 的参数更多但 FPS-1 略高并不矛盾。参数量衡量权重数，不直接衡量高分辨率激活、
算子调度或 GPU kernel 效率。作为交叉检查，fvcore 在同一输入上可识别 USNet 约
`39.13 GMAC-eq`，高于 LiteViLNet 的 `10.17 GMAC-eq`；然而 USNet 主要由 RTX 上高度
优化的大块标准卷积组成，而 LiteViLNet 包含深度卷积、逐尺度融合/attention、插值和
更多小 kernel。后者虽显著降低参数和可识别 MAC，却可能具有较低算术强度和更多 kernel
调度开销。原始 CUDA-event 结果中 USNet 三次 repeat 均约为 `4.170 ms`，LiteViLNet
为 `4.617±0.012 ms`，因此该排序不是单次计时异常。两者都采用 model-only scope：
USNet 的 normal/SNE 构造和 LiteViLNet 的 ADI 构造均不计入 FPS-1。

## 如何理解指标

- `MaxF`：对 0.00--1.00 共 101 个阈值扫描后取得的最高 F1。
- `PRE/REC`：取得 MaxF 的同一个阈值下的 precision/recall。
- 所有验证图片的像素先累计，再统一计算，不是逐图 F1 后平均。
- 这些数值是同协议 KITTI perspective-view 本地验证结果，不冒充官方 BEV
  test-server 成绩。

正式实验结束后，本文档会在结果目录中附上逐 seed 原始 JSON、汇总表和 hash；
大型 checkpoint 与 normal cache 不随论文仓库提交，但可由上述命令完整重建。

结果齐全后，可生成不包含数据、checkpoint 和第三方源码的投稿附件：

```bash
bash tools/package_table1_matched_baselines.sh
```

输出为 `dist/LiteViLNet_RAL_TableI_Reproduction.tar.gz` 及其 `.sha256` 文件。打包器
只有在十五份逐 seed JSON、六份 FPS JSON 和汇总结果都存在时才会执行成功。它在临时目录中把绝对
路径替换为 `$USNET_SOURCE`、`$SNE_SOURCE`、`$PLARD_SOURCE`、`$ROADFORMER_SOURCE`、`$OFFNET_SOURCE`、`$KITTI_ROOT` 等可移植占位符，且不会
改写本机原始 JSON；随后扫描 home/data 路径、邮箱、SSH remote，以及通过
`LITEVILNET_DOUBLE_BLIND_TOKENS` 提供的逗号分隔身份关键词；当前运行用户名和
hostname 也会自动加入 deny-scan。打包器并把 tar
内的 owner/group 固定为数值 `0/0`。任何潜在双盲信息都会令打包失败。
压缩包内部还包含 `ARTIFACT_MANIFEST.sha256`，以仓库相对路径记录每个交付文件的
SHA-256，方便评审逐文件核验。
