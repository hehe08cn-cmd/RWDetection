"""Full RunwayNet model combining HRNet backbone with detection heads.

Architecture:
    Input (B, in_ch, H, W)
    → HRNet-w18 backbone (features at stride 2)
    → Corner head (DSNT) → heatmaps + coords
    → Edge head → left/right edge heatmaps
    → Centerline head → centerline heatmap
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import HRNetBackbone
from .heads.corner_head import DSNTHead
from .heads.edge_head import EdgeHead
from .heads.centerline_head import CenterlineHead
from ..config import (
    NUM_CORNERS, WORKING_SIZE, HRNET_FEATURE_CHANNELS,
    BACKBONE, PRETRAINED,
)


class RunwayNet(nn.Module):
    """Main runway feature detection network.

    Args:
        in_channels: number of input channels (3 RGB, or more with priors)
        stage: training stage (1-4), controls which heads are active
        pretrained: whether to load pretrained weights
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage: int = 1,
        pretrained: bool = PRETRAINED,
    ):
        super().__init__()
        self.stage = stage
        self.in_channels = in_channels
        self.img_w, self.img_h = WORKING_SIZE

        # Backbone
        self.backbone = HRNetBackbone(
            backbone_name=BACKBONE,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        feat_channels = HRNET_FEATURE_CHANNELS

        # Corner head (always active)
        self.corner_head = DSNTHead(
            in_channels=feat_channels,
            num_corners=NUM_CORNERS,
            img_width=self.img_w,
            img_height=self.img_h,
        )

        # Edge head (stage 2+)
        if stage >= 2:
            self.edge_head = EdgeHead(
                in_channels=feat_channels,
                img_width=self.img_w,
                img_height=self.img_h,
            )
            self.centerline_head = CenterlineHead(
                in_channels=feat_channels,
                img_width=self.img_w,
                img_height=self.img_h,
            )
        else:
            self.edge_head = None
            self.centerline_head = None

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            dict with:
                'corners': corner head output dict
                'edges': (B, 2, H, W) edge heatmaps (stage 2+)
                'centerline': (B, 1, H, W) centerline heatmap (stage 2+)
        """
        # Backbone: get highest-res features
        features = self.backbone(x)
        # features is list of feature maps; take the first (highest resolution)
        feat = features[0]  # (B, C, H/2, W/2)

        outputs = {}

        # Corner head
        outputs['corners'] = self.corner_head(feat)

        # Edge and centerline heads (stage 2+)
        if self.stage >= 2:
            outputs['edges'] = self.edge_head(feat)
            outputs['centerline'] = self.centerline_head(feat)

        return outputs

    def train(self, mode: bool = True):
        """Override train() to handle edge/centerline heads."""
        super().train(mode)
        if self.edge_head is not None:
            self.edge_head.train(mode)
        if self.centerline_head is not None:
            self.centerline_head.train(mode)
        return self
