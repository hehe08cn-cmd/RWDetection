"""Centerline detection head for runway centerline."""

import torch
import torch.nn as nn


class CenterlineHead(nn.Module):
    """Predicts runway centerline heatmap from backbone features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        img_width: int = 512,
        img_height: int = 288,
    ):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),  # 1 channel: centerline
        )
        self.upsample = nn.Upsample(
            size=(img_height, img_width), mode='bilinear', align_corners=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (B, C, H_f, W_f) backbone features

        Returns:
            heatmap: (B, 1, H, W) sigmoid-activated centerline heatmap
        """
        h = self.conv(features)
        h = self.upsample(h)
        return torch.sigmoid(h)
