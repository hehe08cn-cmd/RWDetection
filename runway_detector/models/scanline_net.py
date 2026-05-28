"""ScanlineEdgeNet: per-row edge x-coordinate regression.

Predicts left and right runway edge x-coordinates for each image row,
using a MobileNetV3-small encoder + 1D conv decoder.

Input:  (B, 3, H, W) RGB crop
Output: (B, 2, H) x_left[y], x_right[y] in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScanlineEdgeNet(nn.Module):
    """Predict per-row x-coordinates of left and right runway edges.

    Architecture:
      MobileNetV3-small encoder (trainable)
      -> width collapse (AdaptiveAvgPool2d)
      -> 1D conv decoder along height
      -> per-row (x_left, x_right) in [-1, 1]
    """

    def __init__(self, crop_size: int = 256, freeze_backbone: bool = False):
        super().__init__()
        self.crop_size = crop_size
        self.freeze_backbone = freeze_backbone

        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(weights="DEFAULT")
        self.encoder = backbone.features

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # MobileNetV3-small final features: (B, 576, H/32, W/32) = (B, 576, 8, 8)
        enc_out_ch = 576

        # Width collapse: (B, C, 8, 8) -> (B, C, H, 1) -> (B, C, H)
        self.width_pool = nn.AdaptiveAvgPool2d((crop_size, 1))

        # 1D conv decoder along height (smooth row-wise predictions)
        self.decoder = nn.Sequential(
            nn.Conv1d(enc_out_ch, 128, 7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        feats = self.encoder(x)  # (B, 576, H/32, W/32)

        # Width collapse preserving height
        feats = self.width_pool(feats)  # (B, 576, H, 1)
        feats = feats.squeeze(-1)       # (B, 576, H)

        # 1D conv decoder
        out = self.decoder(feats)       # (B, 2, H)
        return torch.tanh(out)          # clamp to [-1, 1]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self


def fit_line_from_row_points(
    xs: "np.ndarray",
    ys: "np.ndarray",
    weights: "np.ndarray" = None,
    crop_size: int = 256,
) -> tuple | None:
    """Fit line (a,b,c) from per-row (x, y) points using weighted PCA.

    Args:
        xs: (N,) x coordinates in [-1, 1]
        ys: (N,) row indices as y coordinates in [-1, 1] (normalized)
        weights: (N,) optional confidence weights
        crop_size: for coordinate denormalization

    Returns:
        (a, b, c) in crop pixel coords, or None
    """
    if len(xs) < 3:
        return None

    # Denormalize to pixel coords for line fitting
    x_px = (xs + 1.0) * crop_size / 2.0
    y_px = (ys + 1.0) * crop_size / 2.0

    if weights is None:
        weights = np.ones(len(xs))
    total_w = weights.sum()
    if total_w < 1e-6:
        return None

    mean_x = (weights * x_px).sum() / total_w
    mean_y = (weights * y_px).sum() / total_w

    dx = x_px - mean_x
    dy = y_px - mean_y
    cov_xx = (weights * dx * dx).sum() / total_w
    cov_yy = (weights * dy * dy).sum() / total_w
    cov_xy = (weights * dx * dy).sum() / total_w

    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    eigenval_min = (trace - np.sqrt(max(trace * trace - 4 * det, 0))) / 2

    a = cov_xy
    b = eigenval_min - cov_xx
    n = np.sqrt(a * a + b * b)
    if n < 1e-8:
        return None
    a, b = a / n, b / n
    c = -(a * mean_x + b * mean_y)

    if a < 0 or (abs(a) < 1e-8 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def predict_to_lines(
    output: "np.ndarray",
    crop_size: int = 256,
    min_points: int = 10,
    smooth_window: int = 11,
) -> dict:
    """Convert model output to line parameters.

    Args:
        output: (2, H) numpy array, output[0]=x_left per row, output[1]=x_right
        crop_size: crop dimension
        min_points: minimum rows needed to fit a line
        smooth_window: median filter window for outlier removal

    Returns:
        dict with 'left', 'right', 'centerline' keys (a,b,c) or None
    """
    from scipy.ndimage import median_filter

    H = output.shape[1]
    ys_norm = np.linspace(-1, 1, H)

    result = {}
    for i, name in enumerate(['left', 'right']):
        x_pred = output[i]
        # Median filter to remove outliers
        x_smooth = median_filter(x_pred, size=smooth_window)
        # Only use points within valid range (not at boundaries)
        valid = (x_smooth > -0.99) & (x_smooth < 0.99)
        if valid.sum() < min_points:
            result[name] = None
        else:
            result[name] = fit_line_from_row_points(
                x_smooth[valid], ys_norm[valid],
                weights=np.ones(valid.sum()),
                crop_size=crop_size,
            )

    # Centerline from left and right midpoints
    x_left = median_filter(output[0], size=smooth_window)
    x_right = median_filter(output[1], size=smooth_window)
    x_mid = (x_left + x_right) / 2.0
    valid = (x_left > -0.99) & (x_left < 0.99) & (x_right > -0.99) & (x_right < 0.99)
    if valid.sum() >= min_points:
        result['centerline'] = fit_line_from_row_points(
            x_mid[valid], ys_norm[valid],
            weights=np.ones(valid.sum()),
            crop_size=crop_size,
        )
    else:
        result['centerline'] = None

    return result
