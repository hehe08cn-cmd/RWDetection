"""HRNet backbone wrapper using timm pretrained model.

Matches the detection project's multi-scale fusion pattern: extracts all stages
from timm HRNet, skips stride-2 features, upsamples lower-resolution stages to
1/4 and concatenates them. A 1x1 projection then reduces the fused channels for
downstream heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..config import BACKBONE, PRETRAINED, HRNET_FUSION_CHANNELS


class HRNetBackbone(nn.Module):
    """HRNet backbone from timm with multi-scale feature fusion.

    Extracts features at all 5 scales, skips stride-2 (too expensive),
    upsamples the lower 4 scales to 1/4 resolution, and concatenates.
    A 1x1 conv then projects to a manageable channel count for heads.

    For hrnet_w18_small_v2:
        feat[0]:  64ch @ 1/2  (skipped)
        feat[1]: 128ch @ 1/4
        feat[2]: 256ch @ 1/8  → upsample to 1/4
        feat[3]: 512ch @ 1/16 → upsample to 1/4
        feat[4]:1024ch @ 1/32 → upsample to 1/4
        Fused: 1920ch → projected to HRNET_FUSION_CHANNELS (256)
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE,
        pretrained: bool = PRETRAINED,
        in_chans: int = 3,
        fusion_channels: int = HRNET_FUSION_CHANNELS,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_chans,
        )

        # Channel counts from timm feature_info
        feat_chs = [fi["num_chs"] for fi in self.backbone.feature_info]
        # feat_chs = [64, 128, 256, 512, 1024] for w18_small_v2
        fused_ch = sum(feat_chs[1:])  # skip stride-2: 128+256+512+1024 = 1920
        self.fusion = nn.Conv2d(fused_ch, fusion_channels, 1)
        self.out_channels = [fusion_channels]

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass with multi-scale fusion.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            List containing [(B, fusion_ch, H/4, W/4)] fused feature map
        """
        feats = self.backbone(x)

        # Skip feats[0] (stride 2), use feats[1] (1/4) as base
        high_res = feats[1]
        for i in range(2, len(feats)):
            up = F.interpolate(
                feats[i], size=high_res.shape[2:],
                mode="bilinear", align_corners=True,
            )
            high_res = torch.cat([high_res, up], dim=1)

        high_res = self.fusion(high_res)
        return [high_res]
