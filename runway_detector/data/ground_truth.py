"""Ground truth generation from GPS/IMU projection.

Reuses RunwayProjector from correct_projection.py to generate pseudo GT labels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional

from correct_projection import RunwayProjector, load_poses
from ..config import WORKING_SIZE, ORIGINAL_SIZE, HEATMAP_SIGMA


class GroundTruthGenerator:
    """Generates ground truth labels from GPS/IMU projection."""

    def __init__(self):
        self.projector = RunwayProjector()
        self.scale_x = WORKING_SIZE[0] / ORIGINAL_SIZE[0]
        self.scale_y = WORKING_SIZE[1] / ORIGINAL_SIZE[1]

    def project_corners(self, pose: dict) -> Dict[str, Optional[np.ndarray]]:
        """Project runway corners to image coordinates using GPS/IMU pose."""
        return self.projector.project_corners(pose)

    def check_visibility(self, corner_pixel: Optional[np.ndarray],
                         img_width: int, img_height: int) -> bool:
        """Check if a corner is visible (in front of camera and within image)."""
        if corner_pixel is None:
            return False
        if corner_pixel[0] < 0 or corner_pixel[1] < 0:
            return False
        # Allow some margin outside image bounds
        margin = 50
        if (corner_pixel[0] < -margin or corner_pixel[0] > img_width + margin or
            corner_pixel[1] < -margin or corner_pixel[1] > img_height + margin):
            return False
        return True

    def generate_corner_gt(self, pose: dict, img_width: int = ORIGINAL_SIZE[0],
                           img_height: int = ORIGINAL_SIZE[1]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate corner coordinates and visibility mask.

        Returns:
            corner_coords: (4, 2) array of corner pixel coordinates at working resolution
            corner_visible: (4,) boolean array
        """
        projected = self.project_corners(pose)
        corner_names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']

        coords = np.zeros((4, 2), dtype=np.float32)
        visible = np.zeros(4, dtype=bool)

        for i, name in enumerate(corner_names):
            pt = projected.get(name)
            if self.check_visibility(pt, img_width, img_height):
                coords[i] = [pt[0] * self.scale_x, pt[1] * self.scale_y]
                visible[i] = True

        return coords, visible

    def generate_corner_heatmaps(self, pose: dict,
                                  out_shape: Tuple[int, int] = None) -> np.ndarray:
        """Generate Gaussian heatmaps for each corner.

        Returns:
            heatmaps: (4, H, W) array of Gaussian heatmaps at working resolution
        """
        if out_shape is None:
            out_shape = (WORKING_SIZE[1], WORKING_SIZE[0])  # (H, W)

        coords, visible = self.generate_corner_gt(pose)
        H, W = out_shape
        heatmaps = np.zeros((4, H, W), dtype=np.float32)

        for i in range(4):
            if visible[i]:
                heatmaps[i] = self._render_gaussian(coords[i], W, H, HEATMAP_SIGMA)

        return heatmaps

    def generate_edge_gt(self, pose: dict,
                          out_shape: Tuple[int, int] = None) -> np.ndarray:
        """Generate left/right edge ground truth heatmaps.

        Returns:
            edge_maps: (2, H, W) array [left_edge, right_edge] at working resolution
        """
        if out_shape is None:
            out_shape = (WORKING_SIZE[1], WORKING_SIZE[0])

        coords, visible = self.generate_corner_gt(pose)
        H, W = out_shape
        edge_maps = np.zeros((2, H, W), dtype=np.float32)

        # Left edge: top_left -> bottom_left
        if visible[1] and visible[0]:  # top_left and bottom_left
            edge_maps[0] = self._render_line_gaussian(
                coords[1], coords[0], W, H, sigma=6.0)

        # Right edge: top_right -> bottom_right
        if visible[2] and visible[3]:  # top_right and bottom_right
            edge_maps[1] = self._render_line_gaussian(
                coords[2], coords[3], W, H, sigma=6.0)

        return edge_maps

    def generate_centerline_gt(self, pose: dict,
                                out_shape: Tuple[int, int] = None) -> np.ndarray:
        """Generate centerline ground truth heatmap.

        Returns:
            centerline_map: (1, H, W) array at working resolution
        """
        if out_shape is None:
            out_shape = (WORKING_SIZE[1], WORKING_SIZE[0])

        coords, visible = self.generate_corner_gt(pose)
        H, W = out_shape
        cl_map = np.zeros((1, H, W), dtype=np.float32)

        # Centerline: mid(top_left, top_right) -> mid(bottom_left, bottom_right)
        top_visible = visible[1] and visible[2]
        bottom_visible = visible[0] and visible[3]

        if top_visible and bottom_visible:
            top_mid = (coords[1] + coords[2]) / 2
            bottom_mid = (coords[0] + coords[3]) / 2
            cl_map[0] = self._render_line_gaussian(
                top_mid, bottom_mid, W, H, sigma=6.0)
        elif top_visible:
            top_mid = (coords[1] + coords[2]) / 2
            cl_map[0] = self._render_gaussian(top_mid, W, H, HEATMAP_SIGMA)
        elif bottom_visible:
            bottom_mid = (coords[0] + coords[3]) / 2
            cl_map[0] = self._render_gaussian(bottom_mid, W, H, HEATMAP_SIGMA)

        return cl_map

    def get_altitude_agl(self, pose: dict) -> float:
        """Get approximate height above ground level (runway elevation ~25-30m)."""
        runway_elevation = 27.5  # average of runway corner altitudes
        return pose['altitude'] - runway_elevation

    def get_mode(self, pose: dict) -> str:
        """Determine if in far-field or near-field mode."""
        agl = self.get_altitude_agl(pose)
        coords, visible = self.generate_corner_gt(pose)

        # Far field: all 4 corners visible and altitude > threshold
        if agl > 35 and visible.sum() >= 4:
            return "far_field"
        # Near field: low altitude or bottom corners not visible
        elif agl < 30 or not visible[0] or not visible[3]:
            return "near_field"
        else:
            return "far_field"

    @staticmethod
    def _render_gaussian(center: np.ndarray, width: int, height: int,
                          sigma: float) -> np.ndarray:
        """Render a 2D Gaussian blob centered at center."""
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        g = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
        return g.astype(np.float32)

    @staticmethod
    def _render_line_gaussian(pt1: np.ndarray, pt2: np.ndarray,
                               width: int, height: int,
                               sigma: float) -> np.ndarray:
        """Render a Gaussian strip along a line segment from pt1 to pt2."""
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return GroundTruthGenerator._render_gaussian(pt1, width, height, sigma)

        # Project onto line
        t = ((xx - pt1[0]) * dx + (yy - pt1[1]) * dy) / (length**2)
        t = np.clip(t, 0, 1)

        # Closest point on segment
        proj_x = pt1[0] + t * dx
        proj_y = pt1[1] + t * dy

        # Perpendicular distance
        dist = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)

        g = np.exp(-dist**2 / (2 * sigma**2))
        return g.astype(np.float32)


def load_poses_dict(txt_path: str) -> dict:
    """Load poses into a dict keyed by frame index for fast lookup."""
    poses = load_poses(txt_path)
    return {p['frame']: p for p in poses}
