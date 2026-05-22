"""Coordinate regression loss for DSNT corner outputs."""

import torch


def coord_l1_loss(pred_coords: torch.Tensor, gt_coords: torch.Tensor,
                   visible_mask: torch.Tensor = None) -> torch.Tensor:
    """L1 loss on predicted vs ground truth pixel coordinates.

    Args:
        pred_coords: (B, K, 2) predicted coordinates in pixels
        gt_coords: (B, K, 2) ground truth coordinates in pixels
        visible_mask: (B, K) bool mask

    Returns:
        scalar loss (in pixels)
    """
    diff = (pred_coords - gt_coords).abs().sum(dim=-1)  # (B, K) L1 distance
    if visible_mask is not None:
        diff = diff * visible_mask.float()
        return diff.sum() / visible_mask.float().sum().clamp(min=1)
    return diff.mean()


def coord_nll_loss(pred_coords: torch.Tensor, pred_log_var: torch.Tensor,
                    gt_coords: torch.Tensor,
                    visible_mask: torch.Tensor = None) -> torch.Tensor:
    """Negative log-likelihood loss for coordinate regression with uncertainty.

    L = log(σ²) + (x_pred - x_gt)² / σ²

    Args:
        pred_coords: (B, K, 2) predicted coordinates
        pred_log_var: (B, K) predicted log variance (per corner)
        gt_coords: (B, K, 2) ground truth coordinates
        visible_mask: (B, K) bool mask

    Returns:
        scalar NLL loss
    """
    var = torch.exp(pred_log_var).clamp(min=1e-6)  # (B, K)
    sq_error = ((pred_coords - gt_coords) ** 2).sum(dim=-1)  # (B, K)

    nll = pred_log_var + sq_error / var  # (B, K)

    if visible_mask is not None:
        nll = nll * visible_mask.float()
        return nll.sum() / visible_mask.float().sum().clamp(min=1)
    return nll.mean()
