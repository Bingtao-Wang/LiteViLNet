#!/usr/bin/env python
"""
VLLiNet 消融实验训练脚本
系统性评估各个组件的贡献

消融实验配置:
1. baseline: 仅RGB + 基础解码器
2. add_lidar: + LiDAR编码器（简单融合）
3. add_fusion: + 多尺度融合模块
4. add_bridge: + LargeKernelBridge
5. add_deep_sup: + 深度监督
6. full: 完整模型（V3 Final）
"""

import argparse
import logging
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 导入消融实验模型
from litevilnet.models.vllinet_ablation import get_ablation_model, count_parameters
from litevilnet.models.losses import VLLiNetLoss

# 导入工具库
from litevilnet.data import (
    get_dataloader, RoadMetrics, AverageMeter,
    print_metrics, set_seed, get_lr, EarlyStopping,
    save_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description='VLLiNet Ablation Study')

    # --- 消融实验配置 ---
    parser.add_argument('--config', type=str, default='full',
                       choices=['baseline', 'add_lidar', 'add_fusion', 'add_bridge', 'add_deep_sup', 'full', 'optimal', 'transformer_bridge'],
                       help='Ablation configuration')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all ablation experiments sequentially')

    # --- 数据配置 ---
    parser.add_argument('--data_root', type=str, default='datasets/data/kitti_road')
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=1248)
    parser.add_argument('--use_synthetic', action='store_true')

    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=100,
                       help='Reduced epochs for ablation study')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=30,
                       help='Reduced patience for ablation study')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)

    # --- 路径配置 ---
    parser.add_argument('--save_dir', type=str, default='experiments/runs/ablation_results')
    parser.add_argument('--log_dir', type=str, default='experiments/runs/ablation_logs')

    return parser.parse_args()


def setup_logging(log_dir, config_name):
    """设置日志"""
    config_log_dir = os.path.join(log_dir, config_name)
    os.makedirs(config_log_dir, exist_ok=True)

    # 清除之前的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config_log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, args=None):
    """训练一个epoch"""
    model.train()
    loss_meter = AverageMeter()
    metrics = RoadMetrics()

    accumulate_grad_batches = args.accumulate_grad_batches if args else 1

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", mininterval=1.0)

    for batch_idx, batch in pbar:
        rgb = batch['rgb'].to(device, non_blocking=True)
        adi = batch['adi'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # 前向传播 (AMP)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(rgb, adi, return_aux=True)
                if isinstance(output, tuple):
                    out, aux_outputs = output
                    loss = criterion(out, aux_outputs, label)
                else:
                    loss = criterion(output, None, label)
                    out = output

            # Loss Scaling & Backward
            scaled_loss = loss / accumulate_grad_batches
            scaler.scale(scaled_loss).backward()

            # 优化器步进
            if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 非 AMP 模式
            output = model(rgb, adi, return_aux=True)
            if isinstance(output, tuple):
                out, aux_outputs = output
                loss = criterion(out, aux_outputs, label)
            else:
                loss = criterion(output, None, label)
                out = output

            (loss / accumulate_grad_batches).backward()

            if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        # 记录
        loss_val = loss.item()
        loss_meter.update(loss_val, rgb.size(0))
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        # 计算指标
        with torch.no_grad():
            if isinstance(out, tuple):
                out = out[0]
            out_prob = torch.sigmoid(out)
            if out_prob.shape[-2:] != label.shape[-2:]:
                out_resized = F.interpolate(out_prob, size=label.shape[-2:], mode='nearest')
            else:
                out_resized = out_prob
            metrics.update(out_resized, label)

    return loss_meter.avg, metrics.compute()


def validate(model, dataloader, device, desc="Validating"):
    """验证"""
    model.eval()
    loss_meter = AverageMeter()
    metrics = RoadMetrics()

    pbar = tqdm(dataloader, desc=desc, mininterval=1.0, leave=False)

    with torch.no_grad():
        for batch in pbar:
            rgb = batch['rgb'].to(device, non_blocking=True)
            adi = batch['adi'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            output = model(rgb, adi)
            if isinstance(output, tuple):
                output = output[0]

            if output.shape[-2:] != label.shape[-2:]:
                output_resized = F.interpolate(output, size=label.shape[-2:], mode='bilinear', align_corners=False)
            else:
                output_resized = output

            loss = F.binary_cross_entropy_with_logits(output_resized.squeeze(1), label.float())
            loss_meter.update(loss.item(), rgb.size(0))
            metrics.update(torch.sigmoid(output_resized), label)

    return loss_meter.avg, metrics.compute()


def train_single_config(config_name, args):
    """训练单个消融配置"""

    # 设置日志
    logger = setup_logging(args.log_dir, config_name)
    logger.info("=" * 80)
    logger.info(f"Ablation Study: {config_name}")
    logger.info("=" * 80)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型
    model = get_ablation_model(config_name, pretrained=True).to(device)
    params = count_parameters(model)
    logger.info(f"Model: {config_name}")
    logger.info(f"Parameters: {params/1e6:.2f}M")

    # 打印模型配置
    logger.info(f"Config:")
    logger.info(f"  - use_lidar: {model.use_lidar}")
    logger.info(f"  - use_multiscale_fusion: {model.use_multiscale_fusion}")
    logger.info(f"  - use_bridge: {model.use_bridge}")
    logger.info(f"  - use_deep_supervision: {model.use_deep_supervision}")

    # 2. 数据加载
    train_loader = get_dataloader(
        data_root=args.data_root, split='train', category=args.category,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_h=args.img_h, img_w=args.img_w, use_augmentation=True,
        use_synthetic=args.use_synthetic
    )
    val_loader = get_dataloader(
        data_root=args.data_root, split='val', category=args.category,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_h=args.img_h, img_w=args.img_w, use_augmentation=False,
        use_synthetic=args.use_synthetic
    )

    # 3. 损失与优化
    criterion = VLLiNetLoss(use_deep_supervision=model.use_deep_supervision).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    best_metric = 0.0
    best_epoch = 0

    # 保存目录
    config_save_dir = os.path.join(args.save_dir, config_name)
    os.makedirs(config_save_dir, exist_ok=True)

    # 4. 训练主循环
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*40} Epoch {epoch + 1}/{args.epochs} {'='*40}")

        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args=args)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, desc="Val")

        scheduler.step()

        # 日志输出
        logger.info(f"Train Loss: {train_loss:.4f} | MaxF: {train_metrics['MaxF']:.4f}")
        logger.info(f"Val   Loss: {val_loss:.4f} | MaxF: {val_metrics['MaxF']:.4f}")

        # 保存最佳模型
        current_metric = val_metrics['MaxF']
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch + 1
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_metric,
                os.path.join(config_save_dir, 'best_model.pth'),
                args
            )
            logger.info(f"🔥 New Best Model! MaxF: {best_metric:.4f}")

        # 早停
        if early_stopping(current_metric):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"\nTraining completed!")
    logger.info(f"Best MaxF: {best_metric:.4f} at epoch {best_epoch}")

    # 返回结果
    return {
        'config': config_name,
        'best_maxf': float(best_metric),
        'best_epoch': best_epoch,
        'params': params,
        'use_lidar': model.use_lidar,
        'use_multiscale_fusion': model.use_multiscale_fusion,
        'use_bridge': model.use_bridge,
        'use_deep_supervision': model.use_deep_supervision,
    }


def main():
    args = parse_args()

    if args.run_all:
        # 运行所有消融实验
        configs = ['baseline', 'add_lidar', 'add_fusion', 'add_bridge', 'add_deep_sup', 'full']
        results = []

        print("\n" + "=" * 80)
        print("Running All Ablation Experiments")
        print("=" * 80 + "\n")

        for config in configs:
            print(f"\n{'='*80}")
            print(f"Starting: {config}")
            print(f"{'='*80}\n")

            result = train_single_config(config, args)
            results.append(result)

            print(f"\n{'='*80}")
            print(f"Completed: {config} | Best MaxF: {result['best_maxf']:.4f}")
            print(f"{'='*80}\n")

        # 保存汇总结果
        summary_path = os.path.join(args.save_dir, 'ablation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 打印汇总表格
        print("\n" + "=" * 80)
        print("Ablation Study Results Summary")
        print("=" * 80)
        print(f"{'Config':<20} | {'MaxF':<8} | {'Params':<10} | {'Components'}")
        print("-" * 80)

        for r in results:
            components = []
            if r['use_lidar']: components.append('LiDAR')
            if r['use_multiscale_fusion']: components.append('Fusion')
            if r['use_bridge']: components.append('Bridge')
            if r['use_deep_supervision']: components.append('DeepSup')

            comp_str = '+'.join(components) if components else 'RGB Only'

            print(f"{r['config']:<20} | {r['best_maxf']:<8.4f} | {r['params']/1e6:<10.2f} | {comp_str}")

        print("=" * 80)
        print(f"\nResults saved to: {summary_path}")

    else:
        # 运行单个配置
        result = train_single_config(args.config, args)

        # 保存单个结果
        result_path = os.path.join(args.save_dir, args.config, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResult saved to: {result_path}")


if __name__ == '__main__':
    main()
