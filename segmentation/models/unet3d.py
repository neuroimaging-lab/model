import torch
from torch import nn
from typing import List, Tuple


"""
3D U-Net implementation for volumetric image segmentation.

This module provides a PyTorch implementation of the 3D U-Net architecture
for medical image segmentation tasks.

source: https://arxiv.org/abs/1606.06650
"""

class ConvBlock(nn.Module):
    """
    Convolutional block with two 3D convolutional layers, batch normalization, and ReLU activation.
    Used as part of the encoder and decoder in the U-Net architecture.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(num_features=out_channels // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block used in the encoder part of the U-Net architecture.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the downsampling block.

        Returns:
            Tuple containing (pooled_features, skip_connection_features)
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class BottleNeck(ConvBlock):
    """
    Bottleneck block at the bottom of the U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels)


class UpBlock(nn.Module):
    """
    Upsampling block used in the decoder part of the U-Net architecture.
    """

    def __init__(self, in_channels: int, res_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels + res_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(num_features=in_channels // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(num_features=in_channels // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat((x, residuals), dim=1)
        return self.conv(x)


class LastBlock(nn.Module):
    """
    Final upsampling block with classification layer.
    """

    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.block = UpBlock(in_channels, out_channels)
        self.conv = nn.Conv3d(
            in_channels=in_channels // 2, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        x = self.block(x, residuals)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net model for volumetric image segmentation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        level_channels: List[int] = [64, 128, 256],
        bottleneck_channels: int = 512,
    ) -> None:
        super().__init__()
        self.down1 = DownBlock(in_channels, level_channels[0])
        self.down2 = DownBlock(level_channels[0], level_channels[1])
        self.down3 = DownBlock(level_channels[1], level_channels[2])
        self.bottleneck = BottleNeck(level_channels[2], bottleneck_channels)
        self.up3 = UpBlock(bottleneck_channels, level_channels[2])
        self.up2 = UpBlock(level_channels[2], level_channels[1])
        self.up1 = LastBlock(level_channels[1], level_channels[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, res1 = self.down1(x)
        x, res2 = self.down2(x)
        x, res3 = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, res3)
        x = self.up2(x, res2)
        x = self.up1(x, res1)
        return x
