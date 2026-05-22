"""Crop-based runway detection model with corner + edge + centerline heads.

Takes 256x256 RGB crop as input, outputs corner coordinates/heatmaps,
left/right edge heatmaps, and centerline heatmap.

Uses timm pretrained HRNet-w18-small with multi-scale feature fusion:
  5-stage extraction → skip stride-2 → upsample+concat to 1/4 → 1x1 proj to 256ch.
Matches the detection project's multi-scale fusion architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HRNetBackbone
from .heads.corner_head import DSNTHead
from .heads.edge_head import EdgeHead
from .heads.centerline_head import CenterlineHead


class CropRunwayNet(nn.Module):
    """Multi-task runway detection on 256x256 crops.

    Shared timm-pretrained HRNet backbone (multi-scale fusion) + three heads:
      - Corner head (DSNT): 4 heatmaps + (4,2) coords in [-1,1]
      - Edge head: 2-channel left/right edge heatmaps
      - Centerline head: 1-channel centerline heatmap
    """

    def __init__(self, in_channels=3, crop_size=256, with_lines=True):
        super().__init__()
        self.crop_size = crop_size
        self.with_lines = with_lines

        # Backbone: timm HRNet-w18-small with multi-scale fusion
        # Outputs 256ch at 1/4 resolution (64x64 for 256x256 input)
        self.backbone = HRNetBackbone(
            backbone_name="hrnet_w18_small_v2.ms_in1k",
            pretrained=True,
            in_chans=in_channels,
        )
        feat_channels = self.backbone.out_channels[0]  # 256 (fusion projection)

        # Corner head
        self.corner_head = DSNTHead(
            in_channels=feat_channels,
            num_corners=4,
            img_width=crop_size,
            img_height=crop_size,
        )

        # Edge and centerline heads
        if with_lines:
            self.edge_head = EdgeHead(
                in_channels=feat_channels,
                img_width=crop_size,
                img_height=crop_size,
            )
            self.centerline_head = CenterlineHead(
                in_channels=feat_channels,
                img_width=crop_size,
                img_height=crop_size,
            )
        else:
            self.edge_head = None
            self.centerline_head = None

    def forward(self, x):
        feats = self.backbone(x)
        feat = feats[0]  # (B, 64, H/2, W/2)

        outputs = self.corner_head(feat)

        if self.with_lines:
            outputs['edge_heatmaps'] = self.edge_head(feat)
            outputs['centerline_heatmap'] = self.centerline_head(feat)

        return outputs

    def compute_loss(self, outputs, gt_heatmaps, gt_coords, visible,
                     gt_edge_heatmaps=None, gt_centerline_heatmap=None):
        pred_hm = outputs["heatmaps"]
        hm_loss = F.mse_loss(pred_hm, gt_heatmaps, reduction="none")
        hm_loss = hm_loss.mean(dim=[2, 3])
        hm_loss = (hm_loss * visible.float()).sum() / visible.float().sum().clamp(min=1)

        pred_coords = outputs["coords"]
        coord_loss = F.l1_loss(pred_coords, gt_coords, reduction="none")
        coord_loss = coord_loss.sum(dim=-1)
        coord_loss = (coord_loss * visible.float()).sum() / visible.float().sum().clamp(min=1)

        total = hm_loss + 0.1 * coord_loss
        result = {"total": total, "heatmap": hm_loss, "coord": coord_loss}

        if self.with_lines and gt_edge_heatmaps is not None and gt_centerline_heatmap is not None:
            edge_loss = F.mse_loss(outputs["edge_heatmaps"], gt_edge_heatmaps)
            cl_loss = F.mse_loss(outputs["centerline_heatmap"], gt_centerline_heatmap)
            line_loss = 0.5 * edge_loss + 0.5 * cl_loss
            result["total"] = total + 0.3 * line_loss
            result["edge"] = edge_loss
            result["centerline"] = cl_loss

        return result
