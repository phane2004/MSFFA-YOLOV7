import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """
    3x3 Conv + BN + ReLU, used in encoder/decoder and residual blocks.
    """
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Two ConvBlocks with a skip connection.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class ChannelAttention(nn.Module):
    """
    Global avg pool -> 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
    Produces per-channel weights.
    """
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = torch.mean(x, dim=(2, 3), keepdim=True)  # global average pooling
        weights = self.mlp(gap)
        return x * weights


class PixelAttention(nn.Module):
    """
    Conv -> ReLU -> Conv -> Sigmoid
    Produces per-pixel weights.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pa(x)   # (B, 1, H, W)
        return x * weights


class MSFFA_RestorationSubnet(nn.Module):
    """
    MSFFA restoration subnet.
    Input:  (B, 3, H, W)  foggy RGB image
    Output: (B, 3, H, W)  enhanced RGB image (same size)
    """
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_res_blocks: int = 18):
        super().__init__()

        # Encoder (3 levels)
        self.enc1 = ConvBlock(in_channels, base_channels)          # D1: C
        self.enc2 = ConvBlock(base_channels, base_channels * 2)    # D2: 2C
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)# D3: 4C

        # Downsampling 2x2
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature conversion (18 residual blocks)
        res_blocks = [ResidualBlock(base_channels * 4)
                      for _ in range(num_res_blocks)]
        self.feature_convert = nn.Sequential(*res_blocks)

        # Attention
        self.channel_att = ChannelAttention(base_channels * 4)
        self.pixel_att = PixelAttention(base_channels * 4)

        # Decoder (3 levels) with skip connections (concat)
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 2,
                              base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels,
                              base_channels)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Encoder -----
        d1 = self.enc1(x)                # (B, C, H,   W)
        d2 = self.enc2(self.down(d1))    # (B, 2C, H/2, W/2)
        d3 = self.enc3(self.down(d2))    # (B, 4C, H/4, W/4)

        # ----- Feature Conversion -----
        f = self.feature_convert(d3)     # (B, 4C, H/4, W/4)

        # ----- Attention -----
        f = self.channel_att(f)
        f = self.pixel_att(f)

        # ----- Decoder -----
        u3 = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.dec3(u3)

        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.dec2(u2)

        out = self.dec1(u2)  # (B, 3, H, W)

        return out
