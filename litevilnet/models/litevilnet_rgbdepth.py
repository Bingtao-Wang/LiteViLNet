"""LiteViLNet RGB+Depth model for robot walkable-path experiments.

This module keeps the robot RGB+Depth branch separate from the KITTI/ADI
mainline model while reusing the same lightweight building blocks.
"""

import torch.nn as nn

from .attention_modules import LargeKernelBridge
from .backbone import LiDAREncoder, MobileNetV3Backbone
from .decoder import VLLiNetDecoder
from .fusion_module import MultiScaleFusionModule


class LiteViLNetRGBDepth(nn.Module):
    """LiteViLNet variant whose second input is encoded RGB-aligned depth."""

    def __init__(self, pretrained=True, use_deep_supervision=True, decoder_channels=None):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        self.rgb_backbone = MobileNetV3Backbone(pretrained=pretrained)
        self.depth_encoder = LiDAREncoder(in_channels=3)

        channels_list = [16, 24, 40, 112, 960]
        self.fusion_module = MultiScaleFusionModule(channels_list=channels_list)
        self.bridge = LargeKernelBridge(
            in_channels=960,
            reduced_channels=128,
            num_heads=4,
            mlp_ratio=2.0,
        )

        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16, 16]

        self.decoder = VLLiNetDecoder(
            encoder_channels=channels_list,
            decoder_channels=decoder_channels,
            num_classes=1,
            use_deep_supervision=use_deep_supervision,
        )

    def forward(self, rgb, depth3, return_aux=False):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_encoder(depth3)
        fused_features = self.fusion_module(rgb_features, depth_features)
        fused_features[-1] = self.bridge(fused_features[-1])

        if return_aux and self.use_deep_supervision:
            out, aux_outputs = self.decoder(fused_features, return_aux=True)
            return out, aux_outputs
        return self.decoder(fused_features, return_aux=False)
