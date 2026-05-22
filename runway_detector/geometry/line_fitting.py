"""Weighted PCA line fitting from heatmaps.

Converts 2D heatmaps to line parameters (a, b, c) where ax + by + c = 0, a²+b²=1.
"""

import torch
import numpy as np


def fit_line_weighted_pca(
    heatmap: torch.Tensor, threshold: float = 0.1
) -> dict:
    """Fit a line to a heatmap using weighted PCA.

    Args:
        heatmap: (B, H, W) or (H, W) heatmap tensor
        threshold: pixels with heatmap value < threshold are ignored

    Returns:
        dict with:
            'line_params': (B, 3) or (3,) normalized line params (a, b, c)
            'centroid': (B, 2) or (2,) weighted centroid (x, y)
            'direction': (B, 2) or (2,) line direction vector
            'uncertainty': (B,) or scalar perpendicular spread (sigma²)
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, H, W = heatmap.shape
    device = heatmap.device

    # Create pixel coordinate grids
    x_vals = torch.arange(W, device=device, dtype=torch.float32)
    y_vals = torch.arange(H, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    # Mask: only use pixels above threshold
    mask = (heatmap > threshold).float()
    weights = heatmap * mask  # (B, H, W)

    # Weighted centroid
    total_weight = weights.sum(dim=(1, 2)).clamp(min=1e-8)  # (B,)
    cx = (weights * xx).sum(dim=(1, 2)) / total_weight  # (B,)
    cy = (weights * yy).sum(dim=(1, 2)) / total_weight  # (B,)
    centroid = torch.stack([cx, cy], dim=-1)  # (B, 2)

    # Weighted covariance (2x2 per batch item)
    dx = xx - cx.view(B, 1, 1)  # (B, H, W)
    dy = yy - cy.view(B, 1, 1)

    cov_xx = (weights * dx * dx).sum(dim=(1, 2)) / total_weight
    cov_xy = (weights * dx * dy).sum(dim=(1, 2)) / total_weight
    cov_yy = (weights * dy * dy).sum(dim=(1, 2)) / total_weight

    # Eigendecomposition of 2x2 covariance: larger eigenvalue → line direction
    # For 2x2 matrix [[a,b],[b,c]], eigenvalues:
    # λ = (a+c ± sqrt((a-c)² + 4b²)) / 2
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    discriminant = torch.sqrt((trace**2 - 4 * det).clamp(min=1e-10))

    lambda1 = (trace + discriminant) / 2  # larger eigenvalue (along line)
    lambda2 = (trace - discriminant) / 2  # smaller eigenvalue (perpendicular)

    # Eigenvector for λ₁ (line direction):
    # v = (cov_xy, λ₁ - cov_xx) or (λ₁ - cov_yy, cov_xy)
    # Use the numerically stable version
    vx = cov_xy
    vy = lambda1 - cov_xx
    # Fallback for when cov_xx dominates
    use_alt = (vx.abs() < 1e-6) & (vy.abs() < 1e-6)
    if use_alt.any():
        vx_alt = lambda1 - cov_yy
        vy_alt = cov_xy
        vx = torch.where(use_alt, vx_alt, vx)
        vy = torch.where(use_alt, vy_alt, vy)

    # Normalize direction vector
    v_norm = torch.sqrt(vx**2 + vy**2).clamp(min=1e-8)
    vx = vx / v_norm
    vy = vy / v_norm
    direction = torch.stack([vx, vy], dim=-1)  # (B, 2)

    # Line normal (perpendicular to direction): n = (-vy, vx)
    a = -vy  # (B,)
    b = vx   # (B,)
    # c = -(a*cx + b*cy)
    c_val = -(a * cx + b * cy)

    line_params = torch.stack([a, b, c_val], dim=-1)  # (B, 3)
    uncertainty = lambda2  # perpendicular spread (variance)

    if squeeze:
        line_params = line_params.squeeze(0)
        centroid = centroid.squeeze(0)
        direction = direction.squeeze(0)
        uncertainty = uncertainty.squeeze(0)

    return {
        'line_params': line_params,
        'centroid': centroid,
        'direction': direction,
        'uncertainty': uncertainty,
    }
