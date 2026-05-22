"""Heatmap loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_mse_loss(pred: torch.Tensor, gt: torch.Tensor,
                     visible_mask: torch.Tensor = None) -> torch.Tensor:
    """MSE loss between predicted and ground truth heatmaps.

    Args:
        pred: (B, K, H, W) predicted heatmaps (before sigmoid/softmax)
        gt: (B, K, H, W) ground truth heatmaps
        visible_mask: (B, K) bool mask, True for visible landmarks

    Returns:
        scalar loss
    """
    loss = F.mse_loss(pred, gt, reduction='none').mean(dim=(2, 3))  # (B, K)
    if visible_mask is not None:
        loss = loss * visible_mask.float()
        return loss.sum() / visible_mask.float().sum().clamp(min=1)
    return loss.mean()
