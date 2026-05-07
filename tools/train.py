#!/usr/bin/env python
"""
VLLiNet Training Script (Pure Version)
Target: MaxF > 97% via TTA (Test Time Augmentation) later
Features:
- LargeKernelBridge Architecture
- No EMA (removed due to Batch Size=2 instability)
- Strong Regularization
"""

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. 导入核心模型组件
from litevilnet.models import VLLiNet_Lite
from litevilnet.models.losses import VLLiNetLoss 

# 2. 导入工具库
from litevilnet.data import (
    get_dataloader, RoadMetrics, AverageMeter,
    print_metrics, set_seed, get_lr, EarlyStopping,
    save_checkpoint
)

def parse_args():
    parser = argparse.ArgumentParser(description='VLLiNet Training - Final Pure')
    
    # --- 数据配置 ---
    parser.add_argument('--data_root', type=str, default='data/kitti_road')
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=1248)
    parser.add_argument('--use_synthetic', action='store_true')
    
    # --- 模型配置 ---
    parser.add_argument('--variant', type=str, default='lite')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--use_deep_supervision', type=bool, default=True)
    parser.add_argument('--no_deep_supervision', action='store_true')
    
    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4) # 强正则化
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    
    # --- 路径配置 ---
    parser.add_argument('--save_dir', type=str, default='weights/litevillinet/baseline')
    parser.add_argument('--log_dir', type=str, default='runs/train/litevillinet_baseline')
    parser.add_argument('--resume', type=str, default='')
    
    return parser.parse_args()

def setup_logging(log_dir):
    """Configure logging with timestamp format"""
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter with milliseconds
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, args=None):
    """Train for one epoch - silent batch processing"""
    model.train()
    loss_meter = AverageMeter()
    metrics = RoadMetrics()

    accumulate_grad_batches = args.accumulate_grad_batches if args else 1

    # Silent batch loop - no progress bar
    for batch_idx, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device, non_blocking=True)
        adi = batch['adi'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # --- 前向传播 (AMP) ---
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

            # --- 优化器步进 ---
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

        # 记录 (silent)
        loss_val = loss.item()
        loss_meter.update(loss_val, rgb.size(0))

        # 计算指标 (silent)
        with torch.no_grad():
            if isinstance(out, tuple): out = out[0]
            out_prob = torch.sigmoid(out)
            if out_prob.shape[-2:] != label.shape[-2:]:
                out_resized = F.interpolate(out_prob, size=label.shape[-2:], mode='nearest')
            else:
                out_resized = out_prob
            metrics.update(out_resized, label)

    return loss_meter.avg, metrics.compute()

def validate(model, dataloader, device):
    """Validate for one epoch - silent batch processing"""
    model.eval()
    loss_meter = AverageMeter()
    metrics = RoadMetrics()

    # Silent batch loop - no progress bar
    with torch.no_grad():
        for batch in dataloader:
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

def main():
    args = parse_args()
    set_seed(42)
    logger = setup_logging(args.log_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================================
    # Startup Block (The Header)
    # ============================================================================
    logger.info(f"Arguments: {args}")
    logger.info(f"Command: {' '.join(sys.argv)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型
    use_ds = args.use_deep_supervision and not args.no_deep_supervision
    model = VLLiNet_Lite(pretrained=args.pretrained, use_deep_supervision=use_ds).to(device)
    logger.info("Model: VLLiNet Lite (LargeKernelBridge Version - NO EMA)")

    # 2. 数据加载
    train_loader = get_dataloader(data_root=args.data_root, split='train', category=args.category,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                img_h=args.img_h, img_w=args.img_w, use_augmentation=True, use_synthetic=args.use_synthetic)
    val_loader = get_dataloader(data_root=args.data_root, split='val', category=args.category,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              img_h=args.img_h, img_w=args.img_w, use_augmentation=False, use_synthetic=args.use_synthetic)

    # 3. 损失与优化
    criterion = VLLiNetLoss(use_deep_supervision=use_ds).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    best_metric = 0.0
    start_epoch = 0

    # 4. Resume 逻辑
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']

    # 5. 训练主循环
    logger.info("Starting training...")

    # ============================================================================
    # Epoch Loop (The Body)
    # ============================================================================
    for epoch in range(start_epoch, args.epochs):
        # Epoch Separator
        logger.info(f"\n{'='*40} Epoch {epoch + 1}/{args.epochs} {'='*40}")

        # --- Train (Silent) ---
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args=args)

        # --- Validate (Silent) ---
        val_loss, val_metrics = validate(model, val_loader, device)

        scheduler.step()

        # Metrics: Two lines
        logger.info(f"Train Loss : {train_loss:.4f} | MaxF: {train_metrics['MaxF']:.4f} | Precision: {train_metrics['Precision']:.4f} | Recall: {train_metrics['Recall']:.4f} | IoU: {train_metrics['IoU']:.4f}")
        logger.info(f"Val        : {val_loss:.4f} | MaxF: {val_metrics['MaxF']:.4f} | Precision: {val_metrics['Precision']:.4f} | Recall: {val_metrics['Recall']:.4f} | IoU: {val_metrics['IoU']:.4f}")

        # --- 保存逻辑 ---
        current_metric = val_metrics['MaxF']
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_metric,
                os.path.join(args.save_dir, 'best_model.pth'),
                args
            )
            # Best Model Alert (The Highlight)
            logger.info(f"🔥 New Best Model Saved! MaxF: {best_metric:.4f}")

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            best_metric,
            os.path.join(args.save_dir, 'latest_model.pth'),
            args
        )

        # 早停
        if early_stopping(current_metric):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(f"\nTraining completed! Best MaxF: {best_metric:.4f}")

if __name__ == '__main__':
    main()