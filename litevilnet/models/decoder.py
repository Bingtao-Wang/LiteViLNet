"""
VLLiNet 解码器
与checkpoint完全匹配的版本 - skip_conv为单层Conv2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """
    上采样块 - 与checkpoint匹配
    skip_conv是单层Conv2d（无BN和ReLU）
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Skip connection - 单层Conv2d (与checkpoint匹配)
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        
        # 主卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        # 上采样
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 处理skip connection
        skip = self.skip_conv(skip)
        
        # 拼接并卷积
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class VLLiNetDecoder(nn.Module):
    """
    VLLiNet 解码器 - 与checkpoint完全匹配
    
    结构:
        bottleneck: 960 -> 128
        up4: 128 + 112 -> 64
        up3: 64 + 40 -> 32
        up2: 32 + 24 -> 16
        up1: 16 + 16 -> 16
        head: 16 -> 1
        aux_heads: 3个辅助头 (深度监督)
    """
    
    def __init__(self, 
                 encoder_channels=[16, 24, 40, 112, 960],
                 decoder_channels=[128, 64, 32, 16, 16],
                 num_classes=1,
                 use_deep_supervision=True):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # Bottleneck: 960 -> 128
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[4], decoder_channels[0], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Up blocks
        self.up4 = UpBlock(decoder_channels[0], encoder_channels[3], decoder_channels[1])  # 128+112->64
        self.up3 = UpBlock(decoder_channels[1], encoder_channels[2], decoder_channels[2])  # 64+40->32
        self.up2 = UpBlock(decoder_channels[2], encoder_channels[1], decoder_channels[3])  # 32+24->16
        self.up1 = UpBlock(decoder_channels[3], encoder_channels[0], decoder_channels[4])  # 16+16->16
        
        # 主输出头
        self.head = nn.Conv2d(decoder_channels[4], num_classes, 1, bias=True)
        
        # 深度监督辅助头
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[1], num_classes, 1, bias=True),  # up4输出, 64->1
                nn.Conv2d(decoder_channels[2], num_classes, 1, bias=True),  # up3输出, 32->1
                nn.Conv2d(decoder_channels[3], num_classes, 1, bias=True),  # up2输出, 16->1
            ])
    
    def forward(self, features, return_aux=False):
        """
        Args:
            features: list of 5 tensors [1/2, 1/4, 1/8, 1/16, 1/32]
            return_aux: 是否返回辅助输出
        """
        f1, f2, f3, f4, f5 = features
        
        # Bottleneck
        x = self.bottleneck(f5)
        
        # 上采样
        x = self.up4(x, f4)
        aux1 = x
        
        x = self.up3(x, f3)
        aux2 = x
        
        x = self.up2(x, f2)
        aux3 = x
        
        x = self.up1(x, f1)
        
        # 主输出
        out = self.head(x)
        
        if return_aux and self.use_deep_supervision and hasattr(self, 'aux_heads'):
            aux_outputs = [
                self.aux_heads[0](aux1),
                self.aux_heads[1](aux2),
                self.aux_heads[2](aux3),
            ]
            return out, aux_outputs
        
        return out


# 别名
LiteDecoder = VLLiNetDecoder
