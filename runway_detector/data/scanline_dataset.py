"""Dataset for scanline edge regression training.

Generates per-row x-coordinate GT from PnP-projected edge lines
(sampled along the full 3D edge, not just corner-to-corner).
Model predicts x_left[y], x_right[y] for each row in [-1, 1].

Supports crop noise augmentation to build robustness against PnP errors:
randomly offsets the crop center during training, with GT recomputed
for the noisy crop so the model learns to handle imperfect crops.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .crop_utils import compute_crop_region, crop_and_resize, transform_points


class ScanlineDataset(Dataset):
    """Dataset producing per-row edge x-coordinate GT from PnP edge projection.

    Uses project_edge_lines() which samples 50 points along each 3D edge
    and projects to image space. Accurate at all altitudes, unlike
    corner-to-corner lines which break at low altitude when near corners
    are outside the image.

    Our corner order: [bottom_left, top_left, top_right, bottom_right]
    Left edge:  BL(0) -> TL(1)
    Right edge: BR(3) -> TR(2)
    """

    def __init__(self, frames_and_poses: list, gt_gen,
                 crop_size: int = 256, augment: bool = True,
                 crop_noise_std: float = 0.0):
        """Args:
            frames_and_poses: list of (frame, pose) tuples
            gt_gen: GroundTruthGenerator instance
            crop_size: output crop dimension
            augment: enable photometric augmentation
            crop_noise_std: std of Gaussian noise added to crop center (px).
                           0 = no noise. ~15 for training = simulates ~10m GPS error.
        """
        self.samples = frames_and_poses
        self.gt_gen = gt_gen
        self.crop_size = crop_size
        self.augment = augment
        self.crop_noise_std = crop_noise_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame, pose = self.samples[idx]
        h, w = frame.shape[:2]

        # Get original-resolution corner coords for crop computation
        projected = self.gt_gen.project_corners(pose)
        corner_names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
        gt_coords_orig_res = np.zeros((4, 2), dtype=np.float32)
        gt_visible = np.zeros(4, dtype=bool)
        for i, name in enumerate(corner_names):
            pt = projected.get(name)
            if self.gt_gen.check_visibility(pt, w, h):
                gt_coords_orig_res[i] = pt
                gt_visible[i] = True

        # Detection order: [near_L, near_R, far_L, far_R] = our [0, 3, 1, 2]
        det_order = np.array([0, 3, 1, 2])
        gt_det = gt_coords_orig_res[det_order]
        vis_det = gt_visible[det_order]

        # Compute nominal crop
        cx, cy, half = compute_crop_region(
            gt_det, vis_det, w, h,
            padding=1.0, min_size=128, max_size=512,
        )

        # Crop noise augmentation: random offset to crop center.
        # Simulates PnP error. GT is computed from the noisy crop so the
        # model learns to predict correct edge positions regardless of crop placement.
        if self.crop_noise_std > 0:
            noise_x = np.random.randn() * self.crop_noise_std
            noise_y = np.random.randn() * self.crop_noise_std
            cx = cx + noise_x
            cy = cy + noise_y
            # Clamp so crop stays within image
            cx = max(half, min(w - half - 1, cx))
            cy = max(half, min(h - half - 1, cy))

        crop = crop_and_resize(frame, int(cx), int(cy), half, self.crop_size)

        # PnP-projected edge lines in original image coords
        edge_proj = self.gt_gen.projector.project_edge_lines(pose)
        edge_params_orig = edge_proj.get("edge_line_params", {})

        # Initialize GT arrays
        H = self.crop_size
        x_left_target = np.full(H, np.nan, dtype=np.float32)
        x_right_target = np.full(H, np.nan, dtype=np.float32)
        valid_left = np.zeros(H, dtype=np.float32)
        valid_right = np.zeros(H, dtype=np.float32)

        # Transform edge lines from original to crop coords
        # Uses the (possibly noisy) cx, cy, half for GT computation
        scale = self.crop_size / (2.0 * half)
        offset_x = cx - half
        offset_y = cy - half

        for side, line_key, x_target, valid in [
            ("left", "left", x_left_target, valid_left),
            ("right", "right", x_right_target, valid_right),
        ]:
            orig_line = edge_params_orig.get(line_key)
            if orig_line is None:
                continue
            a, b, c = orig_line

            a_crop = a / scale
            b_crop = b / scale
            c_crop = a * offset_x + b * offset_y + c
            n_crop = np.sqrt(a_crop * a_crop + b_crop * b_crop)
            if n_crop < 1e-8:
                continue
            a_crop, b_crop, c_crop = a_crop / n_crop, b_crop / n_crop, c_crop / n_crop

            for y in range(H):
                if abs(a_crop) < 1e-8:
                    continue
                x = -(b_crop * float(y) + c_crop) / a_crop
                if 0.0 <= x < H:
                    x_target[y] = (x / H) * 2.0 - 1.0
                    valid[y] = 1.0

        # Fill NaN with 0 (masked in loss)
        x_left_target = np.nan_to_num(x_left_target, nan=0.0)
        x_right_target = np.nan_to_num(x_right_target, nan=0.0)

        if self.augment:
            crop = self._augment(crop)

        image_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "x_left": torch.from_numpy(x_left_target),
            "x_right": torch.from_numpy(x_right_target),
            "valid_left": torch.from_numpy(valid_left),
            "valid_right": torch.from_numpy(valid_right),
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
