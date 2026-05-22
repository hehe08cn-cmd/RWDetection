"""DSNT: Differentiable Spatial to Numerical Transform.

Based on the paper "Numerical Coordinate Regression with Convolutional Neural Networks"
(Nibali et al., 2018). Open-source reference: https://github.com/anibali/dsntnn

DSNT converts heatmaps to (x, y) coordinates differentiably using spatial softmax
followed by soft-argmax (computing expected value over normalized coordinate grids).
"""

import torch
import torch.nn as nn


def spatial_softmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Apply spatial softmax over H*W dimensions.

    Args:
        heatmaps: (B, K, H, W) unnormalized heatmaps

    Returns:
        Normalized heatmaps: (B, K, H, W) where sum over H*W = 1
    """
    B, K, H, W = heatmaps.shape
    h_flat = heatmaps.reshape(B, K, H * W)
    h_sm = torch.softmax(h_flat, dim=-1)
    return h_sm.reshape(B, K, H, W)


def dsnt(heatmaps: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Differentiable Spatial to Numerical Transform.

    Converts heatmaps to normalized coordinates via soft-argmax.

    Args:
        heatmaps: (B, K, H, W) heatmaps for K keypoints
        normalize: if True, apply spatial softmax first

    Returns:
        coords: (B, K, 2) coordinates in [-1, 1] range (normalized to H, W)
    """
    if normalize:
        heatmaps = spatial_softmax(heatmaps)

    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Create normalized coordinate grids
    x_vals = torch.linspace(-1, 1, W, device=device)
    y_vals = torch.linspace(-1, 1, H, device=device)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')
    # (H, W)

    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)

    # Compute expected coordinates
    x_coord = (heatmaps * xx).sum(dim=(2, 3))  # (B, K)
    y_coord = (heatmaps * yy).sum(dim=(2, 3))  # (B, K)

    coords = torch.stack([x_coord, y_coord], dim=-1)  # (B, K, 2)
    return coords


def dsnt_to_pixel(coords: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert DSNT normalized coords [-1, 1] to pixel coordinates.

    Args:
        coords: (B, K, 2) in [-1, 1]
        width: image width in pixels
        height: image height in pixels

    Returns:
        pixel_coords: (B, K, 2) in pixel space
    """
    x = (coords[..., 0] + 1) * 0.5 * (width - 1)  # (B, K)
    y = (coords[..., 1] + 1) * 0.5 * (height - 1)  # (B, K)
    return torch.stack([x, y], dim=-1)


class DSNTHead(nn.Module):
    """Corner detection head using DSNT for coordinate regression.

    Takes HRNet features and produces per-corner heatmaps, then
    converts them to pixel coordinates via DSNT.
    """

    def __init__(
        self,
        in_channels: int,
        num_corners: int = 4,
        hidden_channels: int = 128,
        img_width: int = 512,
        img_height: int = 288,
    ):
        super().__init__()
        self.num_corners = num_corners
        self.img_width = img_width
        self.img_height = img_height

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, num_corners, 1),
        )
        self.upsample = nn.Upsample(
            size=(img_height, img_width), mode='bilinear', align_corners=False
        )

    def forward(self, features: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            features: (B, C, H_f, W_f) HRNet features

        Returns:
            dict with:
                'heatmaps': (B, K, H, W) corner heatmaps
                'coords': (B, K, 2) pixel coordinates from DSNT
                'coords_norm': (B, K, 2) normalized coordinates in [-1, 1]
        """
        h = self.conv(features)
        h = self.upsample(h)  # (B, K, H_img, W_img)

        # DSNT for coordinate extraction
        coords_norm = dsnt(h, normalize=True)  # (B, K, 2) in [-1, 1]
        coords_pixel = dsnt_to_pixel(coords_norm, self.img_width, self.img_height)

        return {
            'heatmaps': h,
            'coords': coords_pixel,
            'coords_norm': coords_norm,
        }
