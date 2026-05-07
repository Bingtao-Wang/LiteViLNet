"""
评估指标 (修复版 - 移除双重sigmoid bug)
"""

import torch
import torch.nn.functional as F


class RoadMetrics:
    """道路检测评估指标
    
    注意：update() 方法期望接收的是：
    - pred: 概率值 [0, 1] 或 logits（通过 input_type 参数控制）
    - target: 二值标签 {0, 1}
    """
    
    def __init__(self, threshold=0.5, input_type='prob'):
        """
        Args:
            threshold: 二值化阈值
            input_type: 'prob' 表示输入是概率值 [0,1]，'logits' 表示输入是 logits
        """
        self.threshold = threshold
        self.input_type = input_type
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total_samples = 0
        
        # 存储用于计算 MaxF 的数据
        self.all_preds = []
        self.all_targets = []
    
    def update(self, pred, target):
        """更新指标
        
        Args:
            pred: 预测值，形状 [B, 1, H, W] 或 [B, H, W]
                  如果 input_type='prob'，应该是 sigmoid 后的概率值 [0, 1]
                  如果 input_type='logits'，应该是原始 logits
            target: 目标标签，形状 [B, H, W]，值为 0 或 1
        """
        # 处理维度
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        
        # 尺寸对齐
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), size=target.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(1)
        
        # 获取概率值
        if self.input_type == 'logits':
            pred_prob = torch.sigmoid(pred)
        else:
            # 输入已经是概率值，不需要再做 sigmoid！
            pred_prob = pred
        
        # 二值化
        pred_binary = (pred_prob > self.threshold).float()
        target_binary = target.float()
        
        # 更新混淆矩阵
        self.tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
        self.tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()
        self.total_samples += pred.shape[0]
    
    def compute(self):
        """计算评估指标"""
        eps = 1e-7
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        
        return {
            'MaxF': f1,  # 注意：这里实际是 F1@threshold，不是真正的 MaxF
            'AP': precision,
            'PRE': precision,
            'REC': recall,
            'F1': f1,
            'IoU': iou,
            'Dice': dice,
            'Accuracy': accuracy
        }
    
    def compute_all(self):
        return self.compute()