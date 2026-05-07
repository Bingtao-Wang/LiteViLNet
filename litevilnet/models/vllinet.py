"""
VLLiNet 主模型 (V2.2: Bottleneck Hybrid)
FPS 恢复版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV3Backbone, LiDAREncoder
from .fusion_module import MultiScaleFusionModule
from .decoder import VLLiNetDecoder
# 导入新增的 Bridge
from .attention_modules import LargeKernelBridge


class VLLiNet_Lite(nn.Module):
    """
    VLLiNet Lite V2.2
    结构: MobileNetV3 + Slim Fusion + Bottleneck Bridge + Decoder
    """
    
    def __init__(self, pretrained=True, use_deep_supervision=True, decoder_channels=None):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision

        # RGB骨干网络
        self.rgb_backbone = MobileNetV3Backbone(pretrained=pretrained)

        # LiDAR编码器
        self.lidar_encoder = LiDAREncoder(in_channels=3)

        # 通道数
        channels_list = [16, 24, 40, 112, 960]

        # 融合模块
        self.fusion_module = MultiScaleFusionModule(channels_list=channels_list)

        # --- 核心修复：轻量级 Transformer Bridge ---
        # 960 -> 128 -> Transformer -> 960
        # 这将大幅减少 FLOPs，恢复 FPS
        self.bridge = LargeKernelBridge(
            in_channels=960,
            reduced_channels=128,  # 关键参数
            num_heads=4,
            mlp_ratio=2.0
        )

        # 解码器 (支持自定义通道数)
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16, 16]  # 默认配置

        self.decoder = VLLiNetDecoder(
            encoder_channels=channels_list,
            decoder_channels=decoder_channels,
            num_classes=1,
            use_deep_supervision=use_deep_supervision
        )
    
    def forward(self, rgb, adi, return_aux=False):
        # 提取特征
        rgb_features = self.rgb_backbone(rgb)
        lidar_features = self.lidar_encoder(adi)
        
        # 多尺度融合
        fused_features = self.fusion_module(rgb_features, lidar_features)
        
        # --- 插入轻量级 Bridge ---
        # 增强最深层语义 (1/32 scale)
        fused_features[-1] = self.bridge(fused_features[-1])
        
        # 解码
        if return_aux and self.use_deep_supervision:
            out, aux_outputs = self.decoder(fused_features, return_aux=True)
            return out, aux_outputs
        else:
            out = self.decoder(fused_features, return_aux=False)
            return out


# 别名
VLLiNet = VLLiNet_Lite
VLLiNet_Pro = VLLiNet_Lite


def get_vllinet(variant='lite', pretrained=True, use_deep_supervision=True):
    return VLLiNet_Lite(pretrained=pretrained, use_deep_supervision=use_deep_supervision)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)