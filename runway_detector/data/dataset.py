"""PyTorch Dataset for runway detection training and evaluation."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

from .ground_truth import GroundTruthGenerator, load_poses_dict
from .prior_renderer import PriorRenderer
from ..config import (
    WORKING_SIZE, ORIGINAL_SIZE, FRAME_STRIDE, HEATMAP_SIGMA, TRAIN_VIDEOS, TEST_VIDEOS
)


class RunwayDataset(Dataset):
    """Dataset for runway feature detection.

    Each item returns:
        - image: (3, H, W) RGB image tensor, normalized to [0, 1]
        - corner_heatmaps: (4, H, W) GT Gaussian heatmaps
        - corner_coords: (4, 2) GT corner pixel coordinates
        - corner_visible: (4,) boolean mask
        - edge_heatmaps: (2, H, W) GT left/right edge heatmaps
        - centerline_heatmap: (1, H, W) GT centerline heatmap
        - prior: (7, H, W) PnP + geometric prior
        - pose: dict with raw aircraft pose data
        - frame_idx: int
        - is_night: bool
    """

    CORNER_NAMES = ['bottom_left', 'top_left', 'top_right', 'bottom_right']

    def __init__(
        self,
        video_keys: list,
        frame_stride: int = FRAME_STRIDE,
        stage: int = 1,  # 1=corner only, 2=+edges, 3=+prior input, 4=+temporal
        include_prior: bool = False,
        is_train: bool = True,
    ):
        self.video_keys = video_keys
        self.frame_stride = frame_stride
        self.stage = stage
        self.include_prior = include_prior
        self.is_train = is_train

        self.gt_gen = GroundTruthGenerator()
        self.prior_renderer = PriorRenderer()
        self.H, self.W = WORKING_SIZE[1], WORKING_SIZE[0]

        # Load all poses
        self.poses = {}
        from ..config import POSES
        for key in video_keys:
            self.poses[key] = load_poses_dict(POSES[key])

        # Build frame index
        self.samples = self._build_index()

    def _build_index(self) -> list:
        """Build list of (video_key, frame_idx) samples."""
        from ..config import VIDEO3_SPLIT_FRAME, VIDEOS

        # Cache video frame counts to avoid repeated file opens
        self._video_frame_counts = {}
        for key in self.video_keys:
            cap = cv2.VideoCapture(VIDEOS[key])
            self._video_frame_counts[key] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        samples = []
        for key in self.video_keys:
            pose_dict = self.poses[key]
            min_frame = min(pose_dict.keys())
            max_frame = min(max(pose_dict.keys()), self._video_frame_counts[key] - 1)

            for frame_idx in range(min_frame, max_frame + 1, self.frame_stride):
                if frame_idx not in pose_dict:
                    continue

                # video3 cross-validation split: train on early frames, test on late
                if key == "video3":
                    if self.is_train and frame_idx >= VIDEO3_SPLIT_FRAME:
                        continue
                    if not self.is_train and frame_idx < VIDEO3_SPLIT_FRAME:
                        continue

                samples.append((key, frame_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_key, frame_idx = self.samples[idx]
        pose = self.poses[video_key][frame_idx]

        # Load image frame
        image = self._load_frame(video_key, frame_idx)

        # Generate ground truth
        corner_heatmaps = self.gt_gen.generate_corner_heatmaps(pose)
        corner_coords, corner_visible = self.gt_gen.generate_corner_gt(pose)

        # Data augmentation (training only, before tensor conversion)
        if self.is_train:
            image, corner_heatmaps, corner_coords = self._augment(
                image, corner_heatmaps, corner_coords, corner_visible,
            )

        # Determine mode
        agl = self.gt_gen.get_altitude_agl(pose)
        mode = "far_field" if agl > 30 and corner_visible.sum() >= 4 else "near_field"
        is_night = "01.14" in video_key or "night" in video_key.lower()

        sample = {
            'image': torch.from_numpy(image).float(),
            'corner_heatmaps': torch.from_numpy(corner_heatmaps).float(),
            'corner_coords': torch.from_numpy(corner_coords).float(),
            'corner_visible': torch.from_numpy(corner_visible).bool(),
            'frame_idx': frame_idx,
            'video_key': video_key,
            'mode': mode,
            'altitude_agl': agl,
            'is_night': is_night,
            'pose': {k: v for k, v in pose.items()},
        }

        # Stage 2+: edge and centerline GT
        if self.stage >= 2:
            edge_heatmaps = self.gt_gen.generate_edge_gt(pose)
            cl_heatmap = self.gt_gen.generate_centerline_gt(pose)
            sample['edge_heatmaps'] = torch.from_numpy(edge_heatmaps).float()
            sample['centerline_heatmap'] = torch.from_numpy(cl_heatmap).float()

        # Stage 3+: prior input
        if self.include_prior:
            prior = self.prior_renderer.render_full_prior(pose)
            sample['prior'] = torch.from_numpy(prior).float()

        return sample

    def _augment(self, image, heatmaps, coords, visible):
        """Apply data augmentation to improve weather generalization.

        Uses heavy color jitter (simulate day/night/rain variation),
        Gaussian blur (simulate rain/motion blur), and small random
        translations. Coords and heatmaps are adjusted for spatial transforms.
        """
        # --- Color jitter: simulates different weather/lighting ---
        # Brightness: 0.5-1.5x, Contrast: 0.5-1.5x
        brightness = 0.5 + np.random.rand() * 1.0
        contrast = 0.5 + np.random.rand() * 1.0
        image = np.clip((image - 0.5) * contrast + 0.5 * brightness, 0, 1)

        # Saturation jitter
        if np.random.rand() > 0.5:
            gray = image.mean(axis=0, keepdims=True)
            sat = 0.3 + np.random.rand() * 1.4
            image = np.clip(gray + sat * (image - gray), 0, 1)

        # Hue jitter (small)
        if np.random.rand() > 0.7:
            hue_shift = (np.random.rand() - 0.5) * 0.1
            image_rgb = image.copy()
            image = image * (1 + hue_shift)

        # --- Random Gaussian blur: simulates rain/motion blur ---
        if np.random.rand() > 0.3:
            sigma = np.random.rand() * 1.5
            ksize = max(3, int(sigma * 4 + 1) | 1)
            for c in range(image.shape[0]):
                img_c = image[c]
                blurred = cv2.GaussianBlur(img_c, (ksize, ksize), sigma)
                image[c] = blurred

        # --- Random Gaussian noise: simulates sensor noise ---
        if np.random.rand() > 0.5:
            noise_std = np.random.rand() * 0.03
            image = image + np.random.randn(*image.shape).astype(np.float32) * noise_std
            image = np.clip(image, 0, 1)

        # --- Random translation: small shifts to improve robustness ---
        if np.random.rand() > 0.3:
            max_shift = 20  # pixels at working resolution
            dx = int((np.random.rand() - 0.5) * 2 * max_shift)
            dy = int((np.random.rand() - 0.5) * 2 * max_shift)
            if dx != 0 or dy != 0:
                H, W = image.shape[1], image.shape[2]
                # Translate image
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                image = np.stack([
                    cv2.warpAffine(image[c], M, (W, H), borderMode=cv2.BORDER_REFLECT)
                    for c in range(image.shape[0])
                ])
                # Translate heatmaps
                heatmaps = np.stack([
                    cv2.warpAffine(heatmaps[k], M, (W, H))
                    for k in range(heatmaps.shape[0])
                ])
                # Translate corner coordinates
                coords = coords.copy()
                coords[:, 0] += dx
                coords[:, 1] += dy

        return image, heatmaps, coords

    def _load_frame(self, video_key: str, frame_idx: int) -> np.ndarray:
        """Load a single video frame and resize to working resolution.

        Opens a fresh VideoCapture per call. This is safe with DataLoader
        multiprocessing (cv2.VideoCapture is not fork-safe).
        """
        from ..config import VIDEOS

        cap = cv2.VideoCapture(VIDEOS[video_key])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Cannot read frame {frame_idx} from {video_key}")
        cap.release()

        # Resize to working resolution
        frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB, normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        # HWC -> CHW
        frame = np.transpose(frame, (2, 0, 1))

        return frame

def create_dataloaders(
    batch_size: int = 8,
    stage: int = 1,
    include_prior: bool = False,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation dataloaders.

    Uses subset of training frames for validation.
    """
    from ..config import VAL_STRIDE

    # Training set
    train_dataset = RunwayDataset(
        video_keys=TRAIN_VIDEOS,
        frame_stride=FRAME_STRIDE,
        stage=stage,
        include_prior=include_prior,
        is_train=True,
    )

    # Validation: subset of training frames (different stride)
    val_dataset = RunwayDataset(
        video_keys=TRAIN_VIDEOS,
        frame_stride=VAL_STRIDE,
        stage=stage,
        include_prior=include_prior,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
