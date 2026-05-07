"""
VLLiNet 注意力模块
包含 Bottleneck Transformer Bridge 用于恢复 FPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    """ECA-Net: Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class CoordinateAttention(nn.Module):
    """Coordinate Attention"""
    def __init__(self, channels):
        super().__init__()
        if channels <= 112:
            mip = 8
        else:
            mip = 30
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class CrossModalAttention(nn.Module):
    """跨模态注意力"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, rgb_feat, lidar_feat):
        B, C, H, W = rgb_feat.shape
        rgb_global = F.adaptive_avg_pool2d(rgb_feat, 1)
        
        q = self.q_proj(rgb_global).view(B, C, 1)
        k = self.k_proj(lidar_feat).view(B, C, -1)
        v = self.v_proj(lidar_feat).view(B, C, -1)
        
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn / (C ** 0.5), dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, 1, 1)
        out = self.out_proj(out)
        out = out.expand(-1, -1, H, W)
        
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        
        return out + rgb_feat


class FusionModule(nn.Module):
    """单层融合模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        inner_channels = channels // 2

        self.reduce_conv = nn.Conv2d(channels, inner_channels, 1, bias=False)

        self.rgb_enhance = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        self.rgb_attn = ECABlock(inner_channels)
        
        self.lidar_enhance = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        self.lidar_attn = CoordinateAttention(inner_channels)
        
        self.cross_attn = CrossModalAttention(inner_channels)
        
        self.gating = nn.Sequential(
            nn.Conv2d(inner_channels * 2, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.Sigmoid()
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(inner_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb_feat, lidar_feat):
        rgb_feat = self.reduce_conv(rgb_feat)
        lidar_feat = self.reduce_conv(lidar_feat)

        rgb_enhanced = self.rgb_enhance(rgb_feat)
        rgb_enhanced = self.rgb_attn(rgb_enhanced)

        lidar_enhanced = self.lidar_enhance(lidar_feat)
        lidar_enhanced = self.lidar_attn(lidar_enhanced)
        
        cross_feat = self.cross_attn(rgb_enhanced, lidar_enhanced)
        
        concat = torch.cat([rgb_enhanced, lidar_enhanced], dim=1)
        gate = self.gating(concat)
        
        fused = rgb_enhanced * gate + lidar_enhanced * (1 - gate)
        fused = fused + cross_feat
        
        return self.out_conv(fused)


# --- 改名：原 TransformerBridge -> LargeKernelBridge ---
class LargeKernelBridge(nn.Module):
    """
    Large Kernel Bridge (Renamed from TransformerBridge)
    结构: 1x1降维 -> 7x7 Depthwise Conv + Dropout -> 1x1升维
    """
    def __init__(self, in_channels, reduced_channels=128, num_heads=None, mlp_ratio=None, drop=0.2):
        super().__init__()
        # 为了兼容旧代码调用，保留不用的参数 (num_heads, mlp_ratio) 也没关系
        
        self.project_in = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.norm_in = nn.BatchNorm2d(reduced_channels)
        
        # 7x7 大核卷积
        self.spatial_process = nn.Conv2d(
            reduced_channels, reduced_channels, 
            kernel_size=7, padding=3, groups=reduced_channels, bias=False
        )
        self.norm_mid = nn.BatchNorm2d(reduced_channels)
        self.act = nn.GELU()
        
        # Dropout (武器 C)
        self.dropout = nn.Dropout2d(p=drop)
        
        self.project_out = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.norm_out = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        
        x = self.project_in(x)
        x = self.norm_in(x)
        x = self.act(x)
        
        x = self.spatial_process(x)
        x = self.norm_mid(x)
        x = self.act(x)
        
        x = self.dropout(x) # 应用 Dropout
        
        x = self.project_out(x)
        x = self.norm_out(x)
        
        return identity + x


class StandardTransformerBridge(nn.Module):
    """
    标准 Transformer Bridge（用于与 LargeKernelBridge 对比实验）
    结构: LayerNorm -> Multi-Head Self-Attention -> 残差
       -> LayerNorm -> FFN (expansion=4) -> 残差
    """
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4.0, drop=0.2):
        super().__init__()
        self.in_channels = in_channels

        # Pre-norm MHSA
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads,
            dropout=drop, batch_first=True
        )

        # Pre-norm FFN
        self.norm2 = nn.LayerNorm(in_channels)
        hidden_dim = int(in_channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # MHSA block
        normed = self.norm1(x_flat)
        attn_out, _ = self.attn(normed, normed, normed)
        x_flat = x_flat + attn_out

        # FFN block
        normed = self.norm2(x_flat)
        x_flat = x_flat + self.ffn(normed)

        # (B, H*W, C) -> (B, C, H, W)
        return x_flat.transpose(1, 2).reshape(B, C, H, W)