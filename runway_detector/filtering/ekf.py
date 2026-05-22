"""SE(3) Extended Kalman Filter for runway feature temporal smoothing.

State vector (9-dim): [x, y, z, vx, vy, vz, phi, theta, psi]
  - x, y, z: aircraft position in runway ENU frame (meters)
  - vx, vy, vz: velocity in runway frame (m/s)
  - phi, theta, psi: roll, pitch, yaw attitude (radians)

The EKF smooths the 6-DoF pose estimation and provides prior feedback
to the next frame's network input.
"""

import numpy as np
from typing import Optional, Tuple


class SE3EKF:
    """Extended Kalman filter for SE(3) pose tracking.

    Process model: constant velocity for position, constant attitude
    Observation: 4 corner pixel coordinates (8-dim from network detection)
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,  # 30 fps
        pos_std: float = 1.0,     # initial position uncertainty (m)
        vel_std: float = 1.0,     # initial velocity uncertainty (m/s)
        att_std: float = 0.05,    # initial attitude uncertainty (rad ~3 deg)
    ):
        self.dt = dt
        self.dim_x = 9
        self.dim_z = 8  # 4 corners * 2 coords

        # State: [x, y, z, vx, vy, vz, phi, theta, psi]
        self.x = np.zeros(self.dim_x)

        # Covariance matrix
        self.P = np.diag([
            pos_std**2, pos_std**2, pos_std**2,
            vel_std**2, vel_std**2, vel_std**2,
            att_std**2, att_std**2, att_std**2,
        ])

        # Process noise
        self.Q_pos = 0.1**2    # position process noise
        self.Q_vel = 0.5**2    # velocity process noise
        self.Q_att = 0.01**2   # attitude process noise (~0.1 rad²)

        # Observation noise (per corner pixel, will be updated from network)
        self.R_base = 5.0**2  # base pixel variance

        self.I = np.eye(self.dim_x)

    def predict(self, imu_gyro: np.ndarray = None,
                imu_accel: np.ndarray = None) -> np.ndarray:
        """EKF prediction step using IMU measurements.

        Args:
            imu_gyro: (3,) gyroscope angular velocity [wx, wy, wz] in rad/s
            imu_accel: (3,) accelerometer [ax, ay, az] in m/s²

        Returns:
            predicted state (9,)
        """
        # State transition (constant velocity + optional IMU)
        F = np.eye(self.dim_x)
        F[0, 3] = self.dt  # x += vx * dt
        F[1, 4] = self.dt  # y += vy * dt
        F[2, 5] = self.dt  # z += vz * dt

        # IMU attitude update
        if imu_gyro is not None:
            phi, theta, psi = self.x[6], self.x[7], self.x[8]
            # Simplified: Euler integration of gyro
            self.x[6] += imu_gyro[0] * self.dt
            self.x[7] += imu_gyro[1] * self.dt
            self.x[8] += imu_gyro[2] * self.dt

        # Position/velocity update
        self.x = F @ self.x

        # Process noise
        Q = np.diag([
            self.Q_pos, self.Q_pos, self.Q_pos,
            self.Q_vel, self.Q_vel, self.Q_vel,
            self.Q_att, self.Q_att, self.Q_att,
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

        return self.x.copy()

    def update(self, z: np.ndarray, R: np.ndarray,
               h_func, H_func) -> np.ndarray:
        """EKF update step with visual observations.

        Args:
            z: (8,) observation vector [corner0_x, corner0_y, ..., corner3_x, corner3_y]
            R: (8, 8) observation covariance (from network aleatoric uncertainty)
            h_func: function h(x) projecting state to image corners
            H_func: function H(x) returning Jacobian of h (8x9)

        Returns:
            updated state (9,)
        """
        # Predicted observation
        z_pred = h_func(self.x)  # (8,)

        # Jacobian
        H = H_func(self.x)  # (8, 9)

        # Innovation
        y = z - z_pred  # (8,)

        # Innovation covariance
        S = H @ self.P @ H.T + R  # (8, 8)

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)  # (9, 8)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

        return self.x.copy()

    def get_corner_prediction(self, projector) -> Optional[np.ndarray]:
        """Get predicted corner pixel coordinates from current state.

        Uses the RunwayProjector to project corners from state pose.

        Args:
            projector: RunwayProjector instance

        Returns:
            corners: (4, 2) predicted corner pixel coordinates, or None
        """
        try:
            pose = {
                'latitude': 0.0,  # need to convert from runway frame to WGS84
                'longitude': 0.0,
                'altitude': 0.0,
                'yaw': np.degrees(self.x[8]),
                'pitch': np.degrees(self.x[7]),
                'roll': np.degrees(self.x[6]),
            }
            # Simplified: return None for now, implement full conversion later
            return None
        except Exception:
            return None

    def get_pose_state(self) -> dict:
        """Return current pose as a dict."""
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
            'attitude': self.x[6:9].copy(),
            'covariance': self.P.copy(),
        }
