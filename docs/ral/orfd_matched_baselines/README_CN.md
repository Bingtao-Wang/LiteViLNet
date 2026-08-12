# ORFD 与 OFF-Net 本地复现说明（中文）

本目录用于复现两组新增证据：一是在 ORFD 官方 train/validation/testing 上重训
USNet 等官方图并独立评测 OFF-Net 发布 checkpoint；二是在 Table I 的统一 KITTI 协议下
新增 OFF-Net。网络、loss 和优化 recipe 均直接从作者官方仓库的固定 commit 导入，
LiteViLNet 仓库只提供数据接口、seed、checkpoint 记录、统一评测和匿名结果导出。

PLARD 保留在 KITTI 的统一对比中，但不适配到 ORFD。其官方可训练入口依赖
LiDAR 生成的 ADI，而 ORFD 发布的是 registered dense depth，PLARD 官方仓库没有
对应的 ORFD ADI 构造链。直接把 dense depth 当作 ADI 会改变方法输入定义，因此
ORFD 只纳入具有官方图且存在兼容、可执行输入路径的方法。

## 1. 官方代码来源

| 方法 | 作者官方仓库 | 固定 commit |
|---|---|---|
| USNet | `https://github.com/morancyc/USNet` | `d761158ad42df7dcb62fa257dd02ce11c85f94a5` |
| SNE-RoadSeg | `https://github.com/hlwang1124/SNE-RoadSeg` | `5e7900bfd59887634ced687ffe85a73018a38659` |
| OFF-Net | `https://github.com/chaytonmin/Off-Road-Freespace-Detection` | `50e63d24836198e8fb5af707e521f414104b4876` |
| RoadFormer | `https://github.com/LiJiahang617/Road-Former` | `f675a3467cb168ebc727648390c304279bbcb079` |

`fetch_matched_baseline_sources.sh` 只允许上述官方 remote，并拒绝 dirty tree。
每个正式结果保存 official remote/commit、导入文件 SHA-256、checkpoint SHA-256、
命令、软件版本和 GPU；汇总器在出表前逐项复核。
初始化也保留官方图的选择：USNet 使用 torchvision ImageNet ResNet-18
（SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`），
SNE-RoadSeg 使用 torchvision ImageNet ResNet-152
（SHA-256 `394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58`），
RoadFormer 使用其官方 ORFD config 内置 URL 的 ConvNeXt-Base ImageNet-21K
（下载文件 SHA-256
`262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d`）。
OFF-Net 的可选本地重训保留官方 Kaiming 初始化；当前 Table III 使用作者发布
checkpoint 的独立本地评测，不将其表述为多种子本地重训结果。

从零复现时先在 LiteViLNet 仓库根目录创建两个固定环境并拉取官方源码：

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda env create -f configs/environments/litevilnet_roadformer_ral.yml
bash tools/fetch_matched_baseline_sources.sh
```

USNet/SNE-RoadSeg 使用 `litevilnet_ral`，OFF-Net/RoadFormer 使用兼容其
官方图的 `litevilnet_roadformer_ral`。fetch 脚本会先核对上表 remote 与 commit。

## 独立 seed 的续跑

正式队列保持各方法原有 recipe 不变。若任务中断，应使用原输出目录并以
`--resume` 续跑；训练器会根据 checkpoint 重建已完成的 physical/optimizer
step 计数。容量感知的续接入口如下（可安全重复执行）：

```bash
# 恢复 OFF-Net seeds 40/41/42
setsid -f bash tools/dispatch_orfd_offnet_resume.sh

# GPU1 与 SNE/RoadFormer 共用时的容量感知续接（seed40 由 GPU0 独立等待器负责）
GPU=1 SEEDS='41 42' setsid -f bash tools/dispatch_orfd_offnet_capacity_queue.sh

# SNE seed40 延后时，优先在 GPU0 恢复 OFF-Net seed40
GPU=0 SEEDS='40' setsid -f bash tools/dispatch_orfd_offnet_resume.sh

# 恢复 SNE-RoadSeg seeds 40/41/42（高显存单所有者队列）
setsid -f bash tools/dispatch_orfd_sne_capacity_queue.sh

# USNet seed 42（约 4 GB，可与 GPU1 上的任务共存）
setsid -f bash tools/dispatch_orfd_usnet42_resume.sh

# RoadFormer seed 40--42（GPU0）
setsid -f bash tools/dispatch_orfd_roadformer_after_sne.sh

# OFF-Net GPU1 优先队列完成后再恢复 SNE seed42
setsid -f bash tools/dispatch_orfd_sne42_after_offnet.sh

# GPU0 的 OFF-Net seed40 完成后恢复 SNE seed40
setsid -f bash tools/dispatch_orfd_sne40_after_offnet.sh

```

脚本只有在目标 `result.json` 不存在且没有同 seed 训练进程时才会 claim，且
会等待显存余量后再启动；默认 SNE 使用 GPU1，RoadFormer 使用 GPU0。脚本不
修改 batch size、epoch 数、AMP、验证间隔或输入分辨率。所有 seed 的正式
`result.json` 齐全后，才运行严格汇总器生成表格。长队列运行时也可以执行
`setsid -f bash tools/monitor_orfd_partial_summaries.sh`，它会在 USNet、
SNE-RoadSeg、OFF-Net、RoadFormer 的三个 seed 齐全后分别生成对应的
`summary_<method>.{json,csv}` 快照，且不会覆盖最终全方法的
`summary.json`。

RoadFormer 的正式 ORFD 命令仍保持 FP32、batch 4、50 epochs 和
`704x1280`。训练器只使用 activation checkpointing 及 PyTorch
`save_on_cpu` 将反向传播保存的临时 tensor 放到主存，以适配 48-GB
显存，并开启官方 TwinConvNeXt 已提供的 `with_cp` backbone 检查点；参数、
优化器更新、数据顺序和评估均未改变。`result.json` 会记录
这些显存保护措施及 allocator 配置，便于精确复现。默认配置为
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32`。

## 2. 为什么有两套 normal cache

OFF-Net 自带的 ORFD SNE 与 SNE-RoadSeg 的 SNE 并不完全相同：前者在 ORFD 路径中
不做 camera-horizon top crop，后者会做。为避免同名实现混用：

- USNet、SNE-RoadSeg、RoadFormer 使用 `profile=sne_roadseg`；
- OFF-Net 使用 `profile=offnet`；
- USNet 的训练翻转严格按“先翻 depth，再算 SNE”缓存；
- RoadFormer 再把 float32 normal 编码成官方 loader 所需的 uint16 PNG。

ORFD 完整发布包共有 8,392/1,245/2,193 张 train/validation/testing，且每个样本都有
同时间戳 calib。缓存器保留 nearest-timestamp 逻辑仅用于诊断不完整解压；正式命令
启用 `--require-exact-calibration`，并进一步强制核验 11,830 个 calib、11,830 个
精确匹配、0 个推断匹配和最大时间差 0。精确/推断数量、使用的内参数量和最大时间差
全部写进 `normal_cache_metadata.json`，因此不完整解压不会进入正式训练。
数据请从作者官方仓库 Dataset 部分维护的链接获取：
`https://github.com/chaytonmin/Off-Road-Freespace-Detection#dataset`，解压前先核对下述 hash。
正式复现所用官方 ZIP 的 SHA-256 为
`02359e4b569b12766e317097d84d45d8b9609d8eccea63a9e6c0024e9a2dc92b`，
`unzip -tq` 返回无错误；解压后 RGB、dense depth、GT、calib 均严格匹配
8,392/1,245/2,193，RGB 与 depth 没有 stem 差集。

## 3. ORFD 正式协议

当前论文 ORFD 表使用 USNet seeds 40/41/42 的三次重训统计，以及 OFF-Net 发布 checkpoint 的一次独立核验。它们共用官方 split、`704×1280` 输入、完整 2,193 张 test 和同一个 OFF-Net-style
固定 argmax confusion-matrix evaluator。直接比较的 F/PRE/REC/IoU 都把 prediction
最近邻恢复到原始 `1280×720`，再与未改动的原始 GT 累计一个 confusion matrix，
避免不同网络 output stride 导致不同 GT 量化；另统一报告 101 阈值 MaxF/AP。
其中 MaxF/AP 统一在 `704×1280` 输入网格上统计；RoadFormer 在其通用
metadata 原尺寸恢复之前取 logits，因此阈值扫描和固定 argmax 恢复顺序
与其他方法一致。
所有方法都按 validation 的 101-threshold MaxF 选择 best checkpoint，与 LiteViLNet
一致；test 只在选模后运行。作者 checkpoint 交叉核验
还单独记录官方 `test.py` 的字面路径：OFF-Net 自身的 1/4-scale loader GT 与 prediction
一起恢复；该诊断不混入本地统一排名。SNE-RoadSeg/OFF-Net/RoadFormer 的多 seed 本地重训仍使用下表和英文手册中的原始 recipe：

| 方法 | epochs | physical batch | 梯度累积 | effective batch | 精度 |
|---|---:|---:|---:|---:|---|
| USNet | 30 | 2 | 1 | 2 | AMP |
| SNE-RoadSeg | 30 | 2 | 1 | 2 | AMP |
| OFF-Net | 30 | 2 | 4 | 8 | FP32 |
| RoadFormer | 50 | 4 | 1 | 4 | FP32 |

当前论文不把 SNE-RoadSeg/RoadFormer 或 OFF-Net 的未完成重训当作结果；OFF-Net 行明确使用作者发布 checkpoint 的一次独立本地评测，而不是声称完成了三 seed 重训。

USNet 已完成 seeds 40/41/42，论文 ORFD 表使用以下三次结果的均值和样本标准差（百分比）：

| seed | F-score | AP | PRE | REC | IoU |
|---:|---:|---:|---:|---:|---:|
| 40 | 95.6188 | 97.1748 | 95.3155 | 95.9241 | 91.6054 |
| 41 | 95.5689 | 97.9032 | 94.4440 | 96.7208 | 91.5137 |
| 42 | 96.6522 | 98.3595 | 96.8555 | 96.4498 | 93.5213 |
| 均值 ± sample SD | 95.9466±0.6116 | 97.8125±0.5975 | 95.5383±1.2211 | 96.3649±0.4051 | 92.2135±1.1335 |

SNE-RoadSeg/OFF-Net/RoadFormer 的 ORFD 本地多 seed 重训仍未完成；不会从中途 checkpoint 或日志推断缺失的 test 数字。

OFF-Net 官方命令是四卡全局 batch 8，每个 DataParallel replica 看到 batch 2。
单卡复现保留 physical batch 2，每 4 个 physical batch 更新一次，因此 effective
batch 仍为 8，同时 BN 的每卡 batch 暴露不变。RoadFormer 保留官方 ORFD 配置中的
50 epochs 和 batch 4。运行命令、环境与数据准备见同目录英文 README，正式双卡入口为
`tools/run_matched_orfd_baselines.sh`。
8,392 张训练图可被这里使用的 physical batch 2 和 4 整除，因此 `drop_last=True`
不会丢弃任何训练样本。模型构造和 DataLoader 启动前统一设置 Python、NumPy、
PyTorch/CUDA、sampler 和 worker RNG；它们都由结果中记录的 run seed 派生。
CPU 单元测试
`test_offnet_four_microbatches_match_one_global_mean_loss_update` 会进一步核对
这种累积方式与直接 batch-8 mean-loss 的梯度和一次 SGD 更新一致。

正式 normal cache 使用英文 README 中的单进程 CUDA 命令生成，并在不加 `--force`
的情况下完整重跑一次；第二次会逐个核验数组 shape/dtype，并把 20,222/11,830 个条目
全部记录为 reused。CPU 也可用 `--device cpu --workers N` 重建，但可能存在极小的
后端浮点差异。

作者发布的 OFF-Net checkpoint 也用精确标定 cache 做了独立交叉核验。严格加载后，
统一原始 GT 评测得到 F/PRE/REC/IoU = 92.80/94.53/91.13/86.57，MaxF/AP =
92.92/97.58；作者 `test.py` 的 1/4-scale GT 字面往返路径得到
92.85/94.52/91.24/86.66。该结果只用于验证模型加载与评测实现，不混入三种子本地
重训均值；完整计数和未取整数值保存在
`runs/revision_1/matched_orfd/official_offnet_checkpoint_test_exact.json`。

## 4. KITTI OFF-Net

KITTI OFF-Net 使用 Table I 已固定的 231/58 分层 split、`384×1248`、150 epochs、
每卡 batch 2、seeds 40/41/42 和 101 阈值 MaxF。normal 单独由 OFF-Net 自带 SNE
生成，不复用 SNE-RoadSeg cache。训练入口为 `tools/run_matched_kitti_offnet.sh`；
训练后由 `run_matched_kitti_fps.sh` 在同一 RTX 4090 D FP32 FPS-1 协议下测速。

## 5. 结果与匿名交付

本地完整结果位于：

```text
runs/revision_1/matched_orfd/formal/<method>_seed<seed>/
runs/revision_1/matched_baselines/formal/offnet_seed<seed>/
```

投稿附件只包含匿名逐 seed JSON、汇总 CSV/JSON、代码和说明，不包含数据、normal
cache、checkpoint、第三方源码或 `.git`。打包器会替换本机绝对路径，扫描邮箱、
home/data 路径、SSH remote 和额外身份关键词，并把 tar owner/group 固定为 `0/0`。
