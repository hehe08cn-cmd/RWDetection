"""Measurement model bridging SE(3) EKF state to pixel corner observations.

Converts EKF state [x, y, z, vx, vy, vz, phi, theta, psi] in runway frame
to 4 corner pixel coordinates using the RunwayProjector geometry pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pymap3d as pm
from typing import Tuple

from correct_projection import RunwayProjector


class CornerMeasurementModel:
    """Projects EKF state to corner pixel coordinates for measurement update.

    State (9,): [x, y, z, vx, vy, vz, phi, theta, psi]
        x, y, z: position in runway ENU frame (meters)
        phi, theta, psi: roll, pitch, yaw attitude (radians)

    Measurement (8,): [c0_x, c0_y, c1_x, c1_y, c2_x, c2_y, c3_x, c3_y]
        in working resolution (512x288)
    """

    def __init__(self):
        self.projector = RunwayProjector()
        self.transformer = self.projector.transformer
        self.scale_x = 512.0 / 1920.0
        self.scale_y = 288.0 / 1080.0
        self._last_state = None
        self._last_meas = None

    def state_to_pose(self, x: np.ndarray) -> dict:
        """Convert EKF state to aircraft pose dict for RunwayProjector.

        Args:
            x: (9,) or (6,) state vector [x, y, z, (vx, vy, vz,) phi, theta, psi]

        Returns:
            pose dict with latitude, longitude, altitude, yaw, pitch, roll
        """
        pos = x[0:3]
        att = x[6:9] if len(x) >= 9 else x[3:6]

        # Runway position → ENU
        enu = (pos[0] * self.projector.runway_x_axis +
               pos[1] * self.projector.runway_y_axis +
               pos[2] * self.projector.runway_z_axis)

        # ENU → ECEF
        origin_ecef = self.projector.runway_origin_ecef
        lat_rad = np.radians(self.projector.runway_origin_lat)
        lon_rad = np.radians(self.projector.runway_origin_lon)

        R_enu_to_ecef = np.array([
            [-np.sin(lon_rad), -np.sin(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.cos(lon_rad)],
            [np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad) * np.sin(lon_rad)],
            [0, np.cos(lat_rad), np.sin(lat_rad)],
        ])

        ecef = origin_ecef + R_enu_to_ecef @ enu
        lat, lon, alt = pm.ecef2geodetic(ecef[0], ecef[1], ecef[2])

        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'yaw': np.degrees(att[2]),
            'pitch': np.degrees(att[1]),
            'roll': np.degrees(att[0]),
        }

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: state → corner pixel coordinates.

        Args:
            x: (9,) state vector

        Returns:
            z_pred: (8,) predicted corner pixel coordinates at working resolution
        """
        pose = self.state_to_pose(x)
        projected = self.projector.project_corners(pose)

        corner_names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
        z_pred = np.zeros(8, dtype=np.float64)

        for i, name in enumerate(corner_names):
            pt = projected.get(name)
            if pt is not None and pt[0] >= 0 and pt[1] >= 0:
                z_pred[2 * i] = pt[0] * self.scale_x
                z_pred[2 * i + 1] = pt[1] * self.scale_y
            else:
                z_pred[2 * i] = -1.0
                z_pred[2 * i + 1] = -1.0

        self._last_state = x.copy()
        self._last_meas = z_pred.copy()
        return z_pred

    def H(self, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Measurement Jacobian via central finite differences.

        Args:
            x: (9,) state vector
            eps: perturbation for finite differences

        Returns:
            H: (8, 9) Jacobian matrix
        """
        H = np.zeros((8, 9), dtype=np.float64)
        h0 = self.h(x)

        for j in range(9):
            if j in (3, 4, 5):  # velocity doesn't affect measurement
                continue
            dx = np.zeros(9)
            dx[j] = eps
            h_plus = self.h(x + dx)
            h_minus = self.h(x - dx)
            H[:, j] = (h_plus - h_minus) / (2 * eps)

        return H

    def get_measurement_noise(self, uncertainties: np.ndarray = None) -> np.ndarray:
        """Build measurement noise covariance R from per-corner uncertainties.

        Args:
            uncertainties: (4,) per-corner std in pixels, or None for default

        Returns:
            R: (8, 8) diagonal covariance matrix
        """
        if uncertainties is None:
            uncertainties = np.full(4, 5.0)

        R = np.zeros((8, 8), dtype=np.float64)
        for i in range(4):
            var = uncertainties[i] ** 2
            R[2 * i, 2 * i] = var
            R[2 * i + 1, 2 * i + 1] = var
        return R

    def corners_to_measurement(self, corners: np.ndarray) -> np.ndarray:
        """Convert (4, 2) corner array to (8,) measurement vector."""
        z = np.zeros(8, dtype=np.float64)
        for i in range(4):
            z[2 * i] = corners[i, 0]
            z[2 * i + 1] = corners[i, 1]
        return z
