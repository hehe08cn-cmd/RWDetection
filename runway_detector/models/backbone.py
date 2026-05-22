"""HRNet backbone wrapper using timm open-source implementation."""

import torch
import torch.nn as nn
import timm
from ..config import BACKBONE, PRETRAINED, HRNET_OUT_INDICES, HRNET_FEATURE_CHANNELS


class HRNetBackbone(nn.Module):
    """HRNet backbone from timm, returns multi-scale feature maps.

    Uses hrnet_w18 by default. Outputs features from the highest-resolution
    stream (stride 2 relative to input), which preserves spatial detail for
    precise heatmap prediction.
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE,
        pretrained: bool = PRETRAINED,
        in_chans: int = 3,
        out_indices: tuple = HRNET_OUT_INDICES,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
        )
        self.out_channels = HRNET_FEATURE_CHANNELS

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            List of feature maps at requested output indices.
            With out_indices=(0,), returns [(B, 64, H/2, W/2)].
        """
        return self.backbone(x)
