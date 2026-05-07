"""
VLLiNet 消融实验模型
用于系统性评估各个组件的贡献

消融实验配置:
1. Baseline: 仅 MobileNetV3 + 简单融合 + 基础解码器
2. +LiDAR: 添加 LiDAR 编码器
3. +MultiScaleFusion: 添加多尺度融合模块
4. +LargeKernelBridge: 添加大核桥接模块
5. +DeepSupervision: 添加深度监督
6. Full (V3 Final): 完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV3Backbone, LiDAREncoder
from .fusion_module import MultiScaleFusionModule
from .decoder import VLLiNetDecoder
from .attention_modules import LargeKernelBridge, StandardTransformerBridge


class SimpleFusionModule(nn.Module):
    """简单的逐元素相加融合（用于消融实验基线）"""
    def __init__(self, channels_list):
        super().__init__()
        # 不需要任何参数，直接相加

    def forward(self, rgb_features, lidar_features):
        """逐元素相加融合"""
        fused = []
        for rgb_feat, lidar_feat in zip(rgb_features, lidar_features):
            fused.append(rgb_feat + lidar_feat)
        return fused


class VLLiNet_Ablation(nn.Module):
    """
    VLLiNet 消融实验模型

    Args:
        use_lidar: 是否使用 LiDAR 分支
        use_multiscale_fusion: 是否使用多尺度融合模块（否则使用简单相加）
        use_bridge: 是否使用 LargeKernelBridge
        use_deep_supervision: 是否使用深度监督
        pretrained: 是否使用预训练的 MobileNetV3
    """

    def __init__(self,
                 use_lidar=True,
                 use_multiscale_fusion=True,
                 use_bridge=True,
                 use_deep_supervision=True,
                 bridge_type='lkb',
                 pretrained=True):
        super().__init__()

        self.use_lidar = use_lidar
        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_bridge = use_bridge
        self.use_deep_supervision = use_deep_supervision
        self.bridge_type = bridge_type

        # RGB骨干网络（始终存在）
        self.rgb_backbone = MobileNetV3Backbone(pretrained=pretrained)

        # LiDAR编码器（可选）
        if use_lidar:
            self.lidar_encoder = LiDAREncoder(in_channels=3)

        # 通道数
        channels_list = [16, 24, 40, 112, 960]

        # 融合模块（可选高级融合或简单相加）
        if use_lidar:
            if use_multiscale_fusion:
                self.fusion_module = MultiScaleFusionModule(channels_list=channels_list)
            else:
                self.fusion_module = SimpleFusionModule(channels_list=channels_list)

        # Bridge模块（可选）
        if use_bridge:
            if bridge_type == 'transformer':
                self.bridge = StandardTransformerBridge(
                    in_channels=960,
                    num_heads=8,
                    mlp_ratio=4.0,
                    drop=0.2
                )
            else:
                self.bridge = LargeKernelBridge(
                    in_channels=960,
                    reduced_channels=128,
                    num_heads=4,
                    mlp_ratio=2.0
                )

        # 解码器
        self.decoder = VLLiNetDecoder(
            encoder_channels=channels_list,
            decoder_channels=[128, 64, 32, 16, 16],
            num_classes=1,
            use_deep_supervision=use_deep_supervision
        )

    def forward(self, rgb, adi, return_aux=False):
        # 提取RGB特征
        rgb_features = self.rgb_backbone(rgb)

        # 根据配置决定是否使用LiDAR
        if self.use_lidar:
            lidar_features = self.lidar_encoder(adi)
            # 融合特征
            fused_features = self.fusion_module(rgb_features, lidar_features)
        else:
            # 仅使用RGB特征
            fused_features = rgb_features

        # 根据配置决定是否使用Bridge
        if self.use_bridge:
            fused_features[-1] = self.bridge(fused_features[-1])

        # 解码
        if return_aux and self.use_deep_supervision:
            out, aux_outputs = self.decoder(fused_features, return_aux=True)
            return out, aux_outputs
        else:
            out = self.decoder(fused_features, return_aux=False)
            return out


def get_ablation_model(config_name='full', pretrained=True):
    """
    获取消融实验模型

    Args:
        config_name: 配置名称
            - 'baseline': 仅RGB + 简单解码器
            - 'add_lidar': + LiDAR编码器（简单融合）
            - 'add_fusion': + 多尺度融合模块
            - 'add_bridge': + LargeKernelBridge
            - 'add_deep_sup': + 深度监督
            - 'full': 完整模型（V3 Final）
            - 'optimal': 最优配置（简单融合 + Bridge + DeepSup）
    """

    configs = {
        'baseline': {
            'use_lidar': False,
            'use_multiscale_fusion': False,
            'use_bridge': False,
            'use_deep_supervision': False,
        },
        'add_lidar': {
            'use_lidar': True,
            'use_multiscale_fusion': False,  # 简单相加融合
            'use_bridge': False,
            'use_deep_supervision': False,
        },
        'add_fusion': {
            'use_lidar': True,
            'use_multiscale_fusion': True,  # 多尺度融合
            'use_bridge': False,
            'use_deep_supervision': False,
        },
        'add_bridge': {
            'use_lidar': True,
            'use_multiscale_fusion': True,
            'use_bridge': True,  # 添加Bridge
            'use_deep_supervision': False,
        },
        'add_deep_sup': {
            'use_lidar': True,
            'use_multiscale_fusion': True,
            'use_bridge': True,
            'use_deep_supervision': True,  # 添加深度监督
        },
        'full': {
            'use_lidar': True,
            'use_multiscale_fusion': True,
            'use_bridge': True,
            'use_deep_supervision': True,
        },
        'optimal': {
            'use_lidar': True,
            'use_multiscale_fusion': False,
            'use_bridge': True,
            'use_deep_supervision': True,
        },
        'transformer_bridge': {
            'use_lidar': True,
            'use_multiscale_fusion': True,
            'use_bridge': True,
            'use_deep_supervision': True,
            'bridge_type': 'transformer',
        },
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    config = configs[config_name]
    return VLLiNet_Ablation(pretrained=pretrained, **config)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试所有消融配置
    print("=" * 80)
    print("VLLiNet 消融实验模型配置")
    print("=" * 80)

    configs = ['baseline', 'add_lidar', 'add_fusion', 'add_bridge', 'add_deep_sup', 'full']

    for config_name in configs:
        model = get_ablation_model(config_name, pretrained=False)
        params = count_parameters(model)

        print(f"\n{config_name:20s} | Params: {params/1e6:6.2f}M")

        # 测试前向传播
        rgb = torch.randn(1, 3, 384, 1248)
        adi = torch.randn(1, 3, 384, 1248)

        with torch.no_grad():
            if model.use_deep_supervision:
                out, aux = model(rgb, adi, return_aux=True)
                print(f"                     | Output: {out.shape}, Aux: {len(aux)} heads")
            else:
                out = model(rgb, adi, return_aux=False)
                print(f"                     | Output: {out.shape}")

    print("\n" + "=" * 80)
