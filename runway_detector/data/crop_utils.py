"""Crop and heatmap utilities for PnP-prior-guided runway detection.

Standalone copy from the detection project — no imports from runway_detection.
"""

from typing import Tuple
import numpy as np
import cv2


def gaussian_heatmap(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Generate a single Gaussian heatmap."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hm = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def generate_heatmaps(size: int, points: np.ndarray, sigma: float,
                      visible: np.ndarray) -> np.ndarray:
    """Generate K-channel heatmap from K (x, y) points.

    Args:
        size: heatmap spatial dimension (square)
        points: (K, 2) float coords in [0, size-1]
        sigma: Gaussian sigma
        visible: (K,) bool mask

    Returns:
        (K, size, size) float32 heatmaps
    """
    K = points.shape[0]
    heatmaps = np.zeros((K, size, size), dtype=np.float32)
    for k in range(K):
        if visible[k]:
            heatmaps[k] = gaussian_heatmap(size, points[k, 0], points[k, 1], sigma)
    return heatmaps


def compute_crop_region(points: np.ndarray, visible: np.ndarray,
                        image_w: int, image_h: int,
                        padding: float = 1.0,
                        min_size: int = 128, max_size: int = 512) -> Tuple[int, int, int]:
    """Compute crop bounding box around visible points.

    Returns (cx, cy, half_size) where the crop is [cx-half:w, cy-half:h, cx+half:w, cy+half:h].
    """
    vis_pts = points[visible.astype(bool)]
    if len(vis_pts) == 0:
        vis_pts = points

    x_min = np.clip(vis_pts[:, 0].min(), 0, image_w - 1)
    y_min = np.clip(vis_pts[:, 1].min(), 0, image_h - 1)
    x_max = np.clip(vis_pts[:, 0].max(), 0, image_w - 1)
    y_max = np.clip(vis_pts[:, 1].max(), 0, image_h - 1)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    bbox_w = max(x_max - x_min, 1.0)
    bbox_h = max(y_max - y_min, 1.0)

    half = max(bbox_w, bbox_h) * (1.0 + padding) / 2.0
    half = max(half, min_size / 2.0)
    half = min(half, max_size / 2.0)

    half = min(half, cx, cy, image_w - cx - 1, image_h - cy - 1)
    half = max(half, min_size // 2)

    return int(cx), int(cy), int(half)


def crop_and_resize(image: np.ndarray, cx: int, cy: int, half: int,
                    target_size: int) -> np.ndarray:
    """Crop a square region around (cx, cy) and resize to target_size."""
    x1 = max(0, cx - half)
    x2 = min(image.shape[1], cx + half)
    y1 = max(0, cy - half)
    y2 = min(image.shape[0], cy + half)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((half * 2, half * 2, 3), dtype=np.uint8)

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def transform_points(points: np.ndarray, visible: np.ndarray,
                     cx: int, cy: int, half: int, target_size: int) -> Tuple[np.ndarray, float]:
    """Transform points from original image coords to crop+resize coords.

    Returns (new_points, sigma).
    """
    scale = target_size / (2.0 * half)
    new_points = points.copy()
    new_points[:, 0] = (points[:, 0] - cx) * scale + target_size / 2.0
    new_points[:, 1] = (points[:, 1] - cy) * scale + target_size / 2.0
    sigma = max(1.0, (2.0 * half) / target_size * 2.0)
    return new_points, sigma
