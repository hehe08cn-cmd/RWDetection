"""Prior heatmap rendering from GPS/IMU/PnP projection and temporal filters."""

import numpy as np
from typing import Tuple, Optional

from ..config import WORKING_SIZE, HEATMAP_SIGMA
from .ground_truth import GroundTruthGenerator


class PriorRenderer:
    """Renders prior heatmaps for input to the network.

    Generates:
    - PnP corner prior (4ch): Gaussian heatmaps from GPS/IMU projection
    - Geometric prior (3ch): left edge, right edge, centerline from projection
    - Temporal prior (4ch): corners from EKF prediction (Stage 4)
    - Temporal line prior (3ch): lines from EKF prediction (Stage 4)
    """

    def __init__(self, noise_std: float = 5.0):
        self.gt_gen = GroundTruthGenerator()
        self.noise_std = noise_std  # std for PnP prior Gaussian (simulates GPS error)
        self.H, self.W = WORKING_SIZE[1], WORKING_SIZE[0]

    def render_pnp_prior(self, pose: dict) -> np.ndarray:
        """Render 4-channel PnP corner prior heatmaps.

        Args:
            pose: aircraft pose dict with lat/lon/alt/yaw/pitch/roll

        Returns:
            prior: (4, H, W) array of Gaussian heatmaps
        """
        coords, visible = self.gt_gen.generate_corner_gt(pose)
        prior = np.zeros((4, self.H, self.W), dtype=np.float32)

        # Use larger sigma for prior (simulates GPS uncertainty)
        prior_sigma = HEATMAP_SIGMA * (self.noise_std / 3.0)

        for i in range(4):
            if visible[i]:
                prior[i] = GroundTruthGenerator._render_gaussian(
                    coords[i], self.W, self.H, prior_sigma)

        return prior

    def render_geometric_prior(self, pose: dict) -> np.ndarray:
        """Render 3-channel geometric prior (left edge, right edge, centerline).

        Returns:
            prior: (3, H, W) array [left_edge, right_edge, centerline]
        """
        edge_gt = self.gt_gen.generate_edge_gt(pose, (self.H, self.W))
        cl_gt = self.gt_gen.generate_centerline_gt(pose, (self.H, self.W))
        return np.concatenate([edge_gt, cl_gt], axis=0)  # (3, H, W)

    def render_full_prior(self, pose: dict) -> np.ndarray:
        """Render full 7-channel prior (PnP 4ch + geometric 3ch).

        Returns:
            prior: (7, H, W) array
        """
        pnp = self.render_pnp_prior(pose)
        geom = self.render_geometric_prior(pose)
        return np.concatenate([pnp, geom], axis=0)

    def render_temporal_corner_prior(self, corners: np.ndarray,
                                      uncertainties: Optional[np.ndarray] = None
                                      ) -> np.ndarray:
        """Render 4-channel temporal corner prior from EKF prediction.

        Args:
            corners: (4, 2) predicted corner pixel coordinates
            uncertainties: (4,) per-corner standard deviations (pixels)

        Returns:
            prior: (4, H, W) array
        """
        prior = np.zeros((4, self.H, self.W), dtype=np.float32)
        base_sigma = HEATMAP_SIGMA

        for i in range(4):
            sigma = base_sigma
            if uncertainties is not None and uncertainties[i] > 0:
                sigma = max(sigma, uncertainties[i])
            prior[i] = GroundTruthGenerator._render_gaussian(
                corners[i], self.W, self.H, sigma)

        return prior

    def render_temporal_line_prior(self, line_params: np.ndarray) -> np.ndarray:
        """Render 3-channel temporal line prior.

        Args:
            line_params: (3, 3) array of line parameters (a, b, c) for left, right, center

        Returns:
            prior: (3, H, W) array
        """
        prior = np.zeros((3, self.H, self.W), dtype=np.float32)

        for i in range(3):
            a, b, c = line_params[i]
            if abs(a) + abs(b) < 1e-6:
                continue
            prior[i] = self._render_line_from_params(a, b, c, sigma=8.0)

        return prior

    def _render_line_from_params(self, a: float, b: float, c: float,
                                  sigma: float) -> np.ndarray:
        """Render a Gaussian strip for a line defined by ax + by + c = 0."""
        x = np.arange(self.W, dtype=np.float32)
        y = np.arange(self.H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Perpendicular distance to line
        norm = np.sqrt(a**2 + b**2)
        dist = np.abs(a * xx + b * yy + c) / (norm + 1e-8)

        return np.exp(-dist**2 / (2 * sigma**2)).astype(np.float32)
