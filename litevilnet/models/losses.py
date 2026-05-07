"""
VLLiNet 损失函数 (V3.3 维度修复版)
修复 Target 缺少通道维度导致的 ValueError
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 维度对齐
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        pred_flat = pred.view(-1)
        target_flat = target.float().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            
        # 确保 target 和 pred 都可以 view(-1) 展平
        pred = pred.view(-1)
        target = target.float().view(-1)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary Aware Loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        # [修复] 确保 Target 有通道维度 [B, 1, H, W]，否则 AvgPool 会报错或算错
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        pred = torch.sigmoid(pred)
        target = target.float()
        
        # 简化的边缘提取 (AvgPool差分)
        avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        pred_avg = avg_pool(pred)
        target_avg = avg_pool(target)
        
        pred_boundary = torch.abs(pred - pred_avg)
        target_boundary = torch.abs(target - target_avg)
        
        return F.mse_loss(pred_boundary, target_boundary)


class VLLiNetLoss(nn.Module):
    """
    VLLiNet 组合损失函数
    Loss = BCE + Dice + 0.1 * Boundary
    """
    def __init__(self, use_deep_supervision=True, aux_weights=[0.4, 0.3, 0.2, 0.1]):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        self.aux_weights = aux_weights
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        
    def _calc_single_loss(self, pred, target):
        # 1. 空间尺寸对齐
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. [核心修复] 通道维度对齐
        # 如果 target 是 [B, H, W]，强制变为 [B, 1, H, W] 以匹配 pred
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # 现在 pred 和 target 都是 [B, 1, H, W]，可以安全计算 BCE
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        loss_bound = self.boundary(pred, target)
        
        return loss_bce + loss_dice + 0.1 * loss_bound

    def forward(self, pred, aux_preds, target):
        loss = self._calc_single_loss(pred, target)
        
        if self.use_deep_supervision and aux_preds is not None:
            for i, aux_p in enumerate(aux_preds):
                if i < len(self.aux_weights):
                    loss += self.aux_weights[i] * self._calc_single_loss(aux_p, target)
                    
        return loss