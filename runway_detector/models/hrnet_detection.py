"""Timm-based HRNet model compatible with detection project checkpoints.

Architecture matches the detection project's hrnet_offset training output:
  - TimmHRNetBackbone: timm HRNet with multi-scale fusion at 1/4 (1920ch)
  - HRNetOutputHead: ConvBlock(1920→960) + Conv2d(960→4) + 4x upsample + DSNT

This exists alongside the native RWDetection models (backbone.py, crop_runway_net.py)
to load checkpoints from the detection project without importing it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Building blocks (copied from detection's hrnet.py — must match checkpoint)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DSNT(nn.Module):
    """Differentiable Spatial to Numerical Transform (with temperature)."""

    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        B, K, H, W = heatmaps.shape
        hm_flat = heatmaps.view(B, K, -1)
        hm_softmax = F.softmax(hm_flat * self.temperature, dim=-1)
        hm = hm_softmax.view(B, K, H, W)

        device = heatmaps.device
        x_grid = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
        y_grid = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)

        px = (hm * x_grid).sum(dim=[2, 3])
        py = (hm * y_grid).sum(dim=[2, 3])
        return torch.stack([px, py], dim=-1)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class TimmHRNetBackbone(nn.Module):
    """Timm HRNet backbone with multi-scale fusion.

    Uses timm's pretrained HRNet, extracts all 5 stages, skips stride-2,
    upsamples+concatenates lower stages to 1/4 resolution.

    For hrnet_w18 / hrnet_w18_small_v2:
        Fused output = 128 + 256 + 512 + 1024 = 1920 channels at 1/4 resolution.
    """

    def __init__(self, model_name: str = "hrnet_w18_small_v2.ms_in1k",
                 pretrained: bool = True, in_chans: int = 3):
        super().__init__()
        # Named 'model' to match detection checkpoint state-dict keys
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_chans,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        # feats[0] = 64ch @ 1/2 (skip), feats[1] = 128ch @ 1/4
        high_res = feats[1]
        for i in range(2, len(feats)):
            up = F.interpolate(
                feats[i], size=high_res.shape[2:],
                mode="bilinear", align_corners=True,
            )
            high_res = torch.cat([high_res, up], dim=1)
        return high_res


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

class HRNetOutputHead(nn.Module):
    """Conv reduction + DSNT for corner coordinate prediction."""

    def __init__(self, in_ch: int, out_ch: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2, 3, 1, 1),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )
        self.dsnt = DSNT()

    def forward(self, x: torch.Tensor):
        heatmaps = self.conv(x)
        heatmaps = F.interpolate(
            heatmaps, scale_factor=4, mode="bilinear", align_corners=True,
        )
        coords = self.dsnt(heatmaps)
        return {"heatmaps": heatmaps, "coords": coords}


# ---------------------------------------------------------------------------
# Full detector (loads detection checkpoint)
# ---------------------------------------------------------------------------

class HRNetOffsetDetector(nn.Module):
    """Full HRNet-Offset detector matching the detection project's checkpoint.

    Usage:
        model = HRNetOffsetDetector()
        ckpt = torch.load("checkpoints/hrnet_offset_best.pt", ...)
        model.load_state_dict(ckpt["model_state_dict"])
    """

    def __init__(self, model_name: str = "hrnet_w18_small_v2.ms_in1k",
                 in_channels: int = 3, out_channels: int = 4,
                 pretrained: bool = False):
        super().__init__()
        self.backbone = TimmHRNetBackbone(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        # 1920ch = 128+256+512+1024 from HRNet stages 1-4
        self.head = HRNetOutputHead(1920, out_channels)

    def forward(self, image: torch.Tensor,
                pnp_heatmaps: torch.Tensor = None) -> dict:
        """Forward pass.

        Args:
            image: (B, 3, H, W) RGB crop
            pnp_heatmaps: ignored (checkpoint is 3ch RGB-only)

        Returns:
            dict with 'heatmaps' (B,4,H,W) and 'coords' (B,4,2) in [-1,1]
        """
        feats = self.backbone(image)
        return self.head(feats)

    def compute_loss(self, outputs: dict, gt_heatmaps: torch.Tensor,
                     gt_coords: torch.Tensor,
                     visible_mask: torch.Tensor) -> dict:
        """Compute training losses (same formulation as detection project)."""
        pred_hm = outputs["heatmaps"]
        hm_loss = F.mse_loss(pred_hm, gt_heatmaps, reduction="none")
        hm_loss = hm_loss.mean(dim=[2, 3])
        hm_loss = (hm_loss * visible_mask.float()).sum() / visible_mask.float().sum().clamp(min=1)

        pred_coords = outputs["coords"]
        coord_loss = F.l1_loss(pred_coords, gt_coords, reduction="none")
        coord_loss = coord_loss.sum(dim=-1)
        coord_loss = (coord_loss * visible_mask.float()).sum() / visible_mask.float().sum().clamp(min=1)

        total = hm_loss + 0.1 * coord_loss
        return {"total": total, "heatmap": hm_loss, "coord": coord_loss}
