"""Vanishing point computation from line intersections."""

import torch


def compute_vanishing_point(
    line1: torch.Tensor, line2: torch.Tensor
) -> torch.Tensor:
    """Compute vanishing point from two lines in homogeneous coordinates.

    Args:
        line1: (..., 3) line params (a, b, c) where a*x + b*y + c = 0
        line2: (..., 3) line params

    Returns:
        vp: (..., 3) vanishing point in homogeneous coords (u, v, w)
            Pixel coords = (u/w, v/w) when w != 0.
            When w ≈ 0, the VP is at infinity (parallel in image).
    """
    # Cross product in homogeneous coordinates: l1 × l2 = point
    vp = torch.linalg.cross(line1, line2, dim=-1)
    return vp


def vanishing_point_to_pixel(vp: torch.Tensor) -> torch.Tensor:
    """Convert homogeneous vanishing point to pixel coordinates.

    Args:
        vp: (..., 3) homogeneous coords

    Returns:
        pixel: (..., 2) pixel coords (may be large/far for near-parallel lines)
    """
    w = vp[..., 2:3].clamp(min=1e-10)
    u = vp[..., 0:1] / w
    v = vp[..., 1:2] / w
    return torch.cat([u, v], dim=-1)


def check_vp_consistency(
    line_left: torch.Tensor,
    line_right: torch.Tensor,
    line_center: torch.Tensor,
) -> torch.Tensor:
    """Check if three lines are concurrent (share the same intersection point).

    Three lines (a_i*x + b_i*y + c_i = 0) are concurrent iff the determinant
    of the 3x3 matrix [a_i, b_i, c_i] is zero.

    Args:
        line_left, line_right, line_center: (B, 3) line parameters

    Returns:
        error: (B,) absolute determinant |det([l1; l2; l3])|
    """
    # Stack into (B, 3, 3) matrix
    M = torch.stack([line_left, line_right, line_center], dim=1)  # (B, 3, 3)
    det = torch.linalg.det(M)  # (B,)
    return det.abs()


def lines_angular_error(
    line1: torch.Tensor, line2: torch.Tensor
) -> torch.Tensor:
    """Angular error between two lines (in radians).

    Args:
        line1, line2: (..., 3) line parameters

    Returns:
        angle: (...,) absolute angle between lines in radians
    """
    n1 = line1[..., :2]
    n2 = line2[..., :2]
    n1_n = n1 / (n1.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    n2_n = n2 / (n2.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    dot = (n1_n * n2_n).sum(dim=-1).clamp(-1, 1)
    return torch.acos(dot.abs())
