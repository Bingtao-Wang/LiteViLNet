"""
VLLiNet 骨干网络
与checkpoint完全匹配的版本 - LiDAREncoder使用深度可分离卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容新旧版本torchvision
try:
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    NEW_WEIGHTS_API = True
except ImportError:
    from torchvision.models import mobilenet_v3_large
    NEW_WEIGHTS_API = False


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Large 骨干网络
    输出5个尺度的特征，通道数: [16, 24, 40, 112, 960]
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            if NEW_WEIGHTS_API:
                weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
                backbone = mobilenet_v3_large(weights=weights)
            else:
                backbone = mobilenet_v3_large(pretrained=True)
        else:
            if NEW_WEIGHTS_API:
                backbone = mobilenet_v3_large(weights=None)
            else:
                backbone = mobilenet_v3_large(pretrained=False)
        
        features = backbone.features
        
        # 分割成5个stage
        self.stage1 = nn.Sequential(*features[0:2])   # -> 16
        self.stage2 = nn.Sequential(*features[2:4])   # -> 24
        self.stage3 = nn.Sequential(*features[4:7])   # -> 40
        self.stage4 = nn.Sequential(*features[7:13])  # -> 112
        self.stage5 = nn.Sequential(*features[13:17]) # -> 960
        
        self.out_channels = [16, 24, 40, 112, 960]
    
    def forward(self, x):
        features = []
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        x = self.stage4(x)
        features.append(x)
        
        x = self.stage5(x)
        features.append(x)
        
        return features


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # Depthwise: [in_ch, 1, 3, 3] with groups=in_ch
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                    stride=stride, padding=1, 
                                    groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise: [out_ch, in_ch, 1, 1]
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class LiDAREncoder(nn.Module):
    """
    LiDAR/ADI 编码器
    与checkpoint完全匹配 - 使用深度可分离卷积
    
    Checkpoint结构:
        stage1: Conv(3->16) + BN + ReLU, Conv(16->16) + BN + ReLU (stride=2)
        stage2: DepthwiseConv(16) + BN + ReLU + PointwiseConv(16->24) + BN + ReLU (stride=2)
        stage3: DepthwiseConv(24) + BN + ReLU + PointwiseConv(24->40) + BN + ReLU (stride=2)
        stage4: DepthwiseConv(40) + BN + ReLU + PointwiseConv(40->112) + BN + ReLU (stride=2)
        stage5: DepthwiseConv(112) + BN + ReLU + PointwiseConv(112->960) + BN + ReLU (stride=2)
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.out_channels = [16, 24, 40, 112, 960]
        
        # Stage 1: 标准卷积 3 -> 16 (1/2)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2-5: 深度可分离卷积
        # 结构: depthwise(3x3, stride=2) -> bn -> relu -> pointwise(1x1) -> bn -> relu
        
        # Stage 2: 16 -> 24 (1/4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),  # depthwise
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 1, bias=False),  # pointwise
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 24 -> 40 (1/8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 40, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        # Stage 4: 40 -> 112 (1/16)
        self.stage4 = nn.Sequential(
            nn.Conv2d(40, 40, 3, stride=2, padding=1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 112, 1, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True)
        )
        
        # Stage 5: 112 -> 960 (1/32)
        self.stage5 = nn.Sequential(
            nn.Conv2d(112, 112, 3, stride=2, padding=1, groups=112, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
            nn.Conv2d(112, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        
        x = self.stage1(x)
        features.append(x)  # 1/2, 16
        
        x = self.stage2(x)
        features.append(x)  # 1/4, 24
        
        x = self.stage3(x)
        features.append(x)  # 1/8, 40
        
        x = self.stage4(x)
        features.append(x)  # 1/16, 112
        
        x = self.stage5(x)
        features.append(x)  # 1/32, 960
        
        return features
