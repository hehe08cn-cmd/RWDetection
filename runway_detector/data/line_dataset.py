"""Dataset for multi-task runway detection training with edge/centerline GT.

Generates corner heatmaps + edge line heatmaps + centerline heatmap
from GPS/IMU-projected corner coordinates.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple

from .crop_utils import (compute_crop_region, crop_and_resize,
                          transform_points, generate_heatmaps, gaussian_heatmap)


def render_line_heatmap(size: int, p1: np.ndarray, p2: np.ndarray,
                        sigma: float) -> np.ndarray:
    """Render a Gaussian band along the line segment from p1 to p2.

    Args:
        size: heatmap spatial dimension (square)
        p1: (2,) start point in [0, size-1]
        p2: (2,) end point in [0, size-1]
        sigma: Gaussian sigma for band width

    Returns:
        (size, size) float32 heatmap
    """
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    # Distance from each pixel to the line segment
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    seg_len_sq = dx * dx + dy * dy + 1e-8

    # Project each pixel onto the line, clamped to [0, 1]
    t = ((xx - p1[0]) * dx + (yy - p1[1]) * dy) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    cx = p1[0] + t * dx
    cy = p1[1] + t * dy

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    hm = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return hm.astype(np.float32)


class LineDataset(Dataset):
    """Dataset generating corner + edge + centerline GT from video frames.

    Detection order: [near_left, near_right, far_left, far_right]
    Our order:        [bottom_left, top_left, top_right, bottom_right]

    Left edge:  near_left → far_left   (det[0] → det[2])
    Right edge: near_right → far_right (det[1] → det[3])
    Centerline: midpoint between left and right edges
    """

    def __init__(self, frames_and_poses: list, gt_gen,
                 crop_size: int = 256, line_sigma: float = 3.0,
                 corner_sigma_ratio: float = 0.08, augment: bool = True):
        self.samples = frames_and_poses
        self.gt_gen = gt_gen
        self.crop_size = crop_size
        self.line_sigma = line_sigma
        self.corner_sigma = max(1.5, corner_sigma_ratio * crop_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        frame, pose = self.samples[idx]
        h, w = frame.shape[:2]

        # Get original-resolution corner coords for crop computation
        # generate_corner_gt returns working-res (512x288) — can't use for crop on 1920x1080 image
        projected = self.gt_gen.project_corners(pose)
        corner_names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
        gt_coords_orig_res = np.zeros((4, 2), dtype=np.float32)
        gt_visible = np.zeros(4, dtype=bool)
        for i, name in enumerate(corner_names):
            pt = projected.get(name)
            if self.gt_gen.check_visibility(pt, w, h):
                gt_coords_orig_res[i] = pt
                gt_visible[i] = True

        # Reorder to detection order for crop computation
        # our [BL, TL, TR, BR] → det [near_L, near_R, far_L, far_R] = [0, 3, 1, 2]
        det_order = np.array([0, 3, 1, 2])
        gt_det = gt_coords_orig_res[det_order]
        vis_det = gt_visible[det_order]

        # Compute crop around GT points (now with original-resolution coords)
        cx, cy, half = compute_crop_region(
            gt_det, vis_det, w, h,
            padding=1.0, min_size=128, max_size=512,
        )

        # Crop and resize image
        crop = crop_and_resize(frame, cx, cy, half, self.crop_size)

        # Transform points to crop coordinates
        gt_crop, _ = transform_points(gt_det, vis_det, cx, cy, half, self.crop_size)

        # Generate corner heatmaps
        gt_heatmaps = generate_heatmaps(self.crop_size, gt_crop,
                                         self.corner_sigma, vis_det)

        # Generate edge heatmaps
        # det order: [near_left, near_right, far_left, far_right] = [0, 1, 2, 3]
        edge_hm = np.zeros((2, self.crop_size, self.crop_size), dtype=np.float32)
        cl_hm = np.zeros((1, self.crop_size, self.crop_size), dtype=np.float32)

        # Left edge: near_left(0) → far_left(2)
        if vis_det[0] and vis_det[2]:
            edge_hm[0] = render_line_heatmap(
                self.crop_size, gt_crop[0], gt_crop[2], self.line_sigma)
        # Right edge: near_right(1) → far_right(3)
        if vis_det[1] and vis_det[3]:
            edge_hm[1] = render_line_heatmap(
                self.crop_size, gt_crop[1], gt_crop[3], self.line_sigma)
        # Centerline: midpoint line
        if vis_det.all():
            mid_near = (gt_crop[0] + gt_crop[1]) / 2.0
            mid_far = (gt_crop[2] + gt_crop[3]) / 2.0
            cl_hm[0] = render_line_heatmap(
                self.crop_size, mid_near, mid_far, self.line_sigma)

        # Normalize corner coords to [-1, 1]
        gt_coords = (gt_crop / self.crop_size) * 2.0 - 1.0

        # Augmentation
        if self.augment:
            crop = self._augment(crop)

        # To tensors
        image_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        gt_hm_tensor = torch.from_numpy(gt_heatmaps)
        gt_coords_tensor = torch.from_numpy(gt_coords).float()
        visible_tensor = torch.from_numpy(vis_det)
        edge_tensor = torch.from_numpy(edge_hm)
        cl_tensor = torch.from_numpy(cl_hm)

        return {
            "image": image_tensor,
            "gt_heatmaps": gt_hm_tensor,
            "gt_coords": gt_coords_tensor,
            "visible": visible_tensor,
            "gt_edge": edge_tensor,
            "gt_centerline": cl_tensor,
        }

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            scale = 0.8 + np.random.rand() * 0.4
            image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            offset = np.random.randint(-15, 16)
            image = np.clip(image.astype(np.int32) + offset, 0, 255).astype(np.uint8)
        return image
