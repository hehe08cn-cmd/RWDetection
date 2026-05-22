"""Geometric consistency losses enforcing runway structural constraints."""

import torch
from ..geometry.line_fitting import fit_line_weighted_pca
from ..geometry.vanishing_point import check_vp_consistency


def vanishing_point_loss(
    edge_heatmaps: torch.Tensor, centerline_heatmap: torch.Tensor,
) -> torch.Tensor:
    """Vanishing point consistency loss.

    Left edge, right edge, and centerline should be concurrent
    (intersect at the same vanishing point).

    Uses determinant of the 3x3 line coefficient matrix.
    det = 0 iff three lines are concurrent.

    Args:
        edge_heatmaps: (B, 2, H, W) left and right edge heatmaps
        centerline_heatmap: (B, 1, H, W) centerline heatmap

    Returns:
        scalar loss
    """
    B = edge_heatmaps.shape[0]

    # Fit lines from heatmaps
    line_left = fit_line_weighted_pca(edge_heatmaps[:, 0])['line_params']   # (B, 3)
    line_right = fit_line_weighted_pca(edge_heatmaps[:, 1])['line_params']  # (B, 3)
    line_center = fit_line_weighted_pca(centerline_heatmap[:, 0])['line_params']  # (B, 3)

    # Concurrency check via determinant
    error = check_vp_consistency(line_left, line_right, line_center)
    return error.mean()


def corner_line_consistency_loss(
    corner_coords: torch.Tensor, edge_heatmaps: torch.Tensor,
    corner_visible: torch.Tensor = None,
) -> torch.Tensor:
    """Loss enforcing that corners lie at intersections of adjacent edges.

    Corner layout: 0=bottom_left, 1=top_left, 2=top_right, 3=bottom_right
    - bottom_left (0) at intersection of left_edge and bottom_edge
    - top_left (1) at intersection of left_edge and top_edge
    - etc.

    Simplified: each corner should be close to the corresponding edge line.

    Args:
        corner_coords: (B, 4, 2) corner pixel coordinates
        edge_heatmaps: (B, 2, H, W) [left_edge, right_edge]
        corner_visible: (B, 4) bool mask

    Returns:
        scalar loss
    """
    B = corner_coords.shape[0]
    device = corner_coords.device

    # Fit edge lines
    line_left = fit_line_weighted_pca(edge_heatmaps[:, 0])['line_params']   # (B, 3)
    line_right = fit_line_weighted_pca(edge_heatmaps[:, 1])['line_params']  # (B, 3)

    # Distance from point (x,y) to line ax+by+c=0: |ax+by+c| / sqrt(a²+b²)
    def point_line_dist(coords, line_params):
        x, y = coords[..., 0], coords[..., 1]
        a, b, c = line_params[..., 0], line_params[..., 1], line_params[..., 2]
        return (a * x + b * y + c).abs() / (a**2 + b**2).sqrt().clamp(min=1e-8)

    # Left corners (0, 1) should be on left edge
    dist_left_0 = point_line_dist(corner_coords[:, 0], line_left)
    dist_left_1 = point_line_dist(corner_coords[:, 1], line_left)

    # Right corners (2, 3) should be on right edge
    dist_right_2 = point_line_dist(corner_coords[:, 2], line_right)
    dist_right_3 = point_line_dist(corner_coords[:, 3], line_right)

    dists = torch.stack([dist_left_0, dist_left_1, dist_right_2, dist_right_3], dim=1)  # (B, 4)

    if corner_visible is not None:
        dists = dists * corner_visible.float()
        return dists.sum() / corner_visible.float().sum().clamp(min=1)

    return dists.mean()


def parallel_loss(
    line1: torch.Tensor, line2: torch.Tensor
) -> torch.Tensor:
    """Loss enforcing that two lines are parallel.

    Args:
        line1: (B, 3) line params (a, b, c)
        line2: (B, 3) line params

    Returns:
        scalar loss (1 - |cos(angle)|, 0 when parallel)
    """
    # Line normal direction
    n1 = line1[..., :2]  # (B, 2)
    n2 = line2[..., :2]

    n1_norm = n1 / (n1.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    n2_norm = n2 / (n2.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    cos_angle = (n1_norm * n2_norm).sum(dim=-1).abs()  # |cos(theta)|
    return (1 - cos_angle).mean()
