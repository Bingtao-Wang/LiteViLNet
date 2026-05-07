"""
VLLiNet 融合模块
"""

import torch.nn as nn

from .attention_modules import FusionModule


class MultiScaleFusionModule(nn.Module):
    """多尺度融合模块 - 5个fusion_modules (All Attention, Slim版本)"""

    def __init__(self, channels_list=[16, 24, 40, 112, 960]):
        super().__init__()

        self.num_levels = len(channels_list)

        # All Attention: FusionModule for ALL 5 levels (Slim version with bottleneck)
        self.fusion_modules = nn.ModuleList([
            FusionModule(channels)
            for channels in channels_list
        ])
    
    def forward(self, rgb_features, lidar_features):
        fused_features = []
        
        for i, fusion in enumerate(self.fusion_modules):
            fused = fusion(rgb_features[i], lidar_features[i])
            fused_features.append(fused)
        
        return fused_features
