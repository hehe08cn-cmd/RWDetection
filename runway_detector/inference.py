"""Real-time runway detector inference pipeline with HRNet-Offset + EKF.

Uses PnP prior to crop ROI around runway (256x256), feeds RGB crop to
HRNet-Offset for corner refinement, then applies SE(3)-EKF for temporal smoothing.

Usage:
    detector = RunwayInference()
    for frame, pose in video:
        result = detector(frame, pose=pose)
"""

import os
import numpy as np
import torch
import cv2
from typing import Optional

from .models.hrnet_detection import HRNetOffsetDetector
from .data.crop_utils import (compute_crop_region, crop_and_resize,
                               transform_points, generate_heatmaps)
from .config import ORIGINAL_SIZE
from .filtering.ekf import SE3EKF
from .filtering.measurement_model import CornerMeasurementModel

# Default checkpoint: best HRNet-Offset model (copied from detection project)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "hrnet_offset_best.pt",
)


class RunwayInference:
    """End-to-end runway detection inference.

    Uses HRNet-Offset with PnP-prior-guided crop + PnP heatmaps as input
    for high-accuracy corner refinement, plus SE(3)-EKF for temporal smoothing.

    Usage:
        detector = RunwayInference()
        for frame, pose in video:
            result = detector(frame, pose=pose)
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda",
        enable_ekf: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load HRNet-Offset model (local, no detection-project dependency)
        ckpt_path = checkpoint_path or DEFAULT_CHECKPOINT
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        in_channels = ckpt_cfg.get("model", {}).get("in_channels", 3)

        self.model = HRNetOffsetDetector(
            model_name="hrnet_w18_small_v2.ms_in1k",
            in_channels=in_channels,
            out_channels=4,
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.crop_size = ckpt_cfg.get("data", {}).get("crop_size", 256)

        # Measurement model for PnP prior projection
        self.meas_model = CornerMeasurementModel()

        # EKF for temporal smoothing
        self.enable_ekf = enable_ekf
        self.ekf = SE3EKF() if enable_ekf else None
        self._ekf_initialized = False

        # CUDA graph for fast inference replay
        self._cuda_graph = None
        self._static_input = None
        if self.device.type == "cuda":
            self._capture_cuda_graph()

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray, pose: dict = None, use_fp16: bool = False) -> dict:
        """Run detection on a single video frame (BGR, HWC, uint8).

        Args:
            frame: BGR video frame at original resolution
            pose: aircraft pose dict with lat/lon/alt/yaw/pitch/roll.
                  Required for PnP-prior-guided crop + heatmap input.

        Returns:
            dict with:
                corners: (4, 2) corner pixel coords at original resolution
                corners_working: (4, 2) scaled to 512x288 working resolution
                heatmaps: (4, 256, 256) predicted corner heatmaps
                ekf_state: SE(3) pose state (if EKF enabled)
        """
        # Get PnP prior corners at original resolution
        if pose is not None:
            prior_corners = self._get_prior_corners(pose)
        else:
            # Fallback: run full-frame with old model not available — fail gracefully
            prior_corners = None

        if prior_corners is None:
            raise ValueError("PnP prior is required for HRNet-Offset crop inference")

        # Reorder prior corners from our [BL,TL,TR,BR] to detection's [near_left,near_right,far_left,far_right]
        #   det[0]=our[0], det[1]=our[3], det[2]=our[1], det[3]=our[2]
        prior_det = prior_corners[[0, 3, 1, 2]]

        # Crop preprocessing (skip PnP heatmaps for 3ch model)
        img_tensor, _, crop_info = self._preprocess(frame, prior_det)

        # CUDA graph replay (2x faster than regular forward)
        outputs = self._infer(img_tensor)
        corners_orig, heatmaps = self._postprocess(outputs, crop_info)

        result = {
            "corners": corners_orig,                    # (4, 2) at original res
            "heatmaps": heatmaps,                       # (4, 256, 256)
            "corners_original": corners_orig.copy(),    # same at original res
            "corners_raw": prior_corners.copy(),        # PnP prior (before refinement)
        }

        # Scale to working resolution for compatibility
        W_orig, H_orig = ORIGINAL_SIZE
        sx = 512.0 / W_orig
        sy = 288.0 / H_orig
        result["corners"] = corners_orig * np.array([sx, sy])

        # Edge and centerline heatmaps (if model has line heads)
        if "edge_heatmaps" in outputs:
            result["edge_heatmaps"] = outputs["edge_heatmaps"][0].cpu().numpy()
            result["centerline_heatmap"] = outputs["centerline_heatmap"][0].cpu().numpy()
            # Fit lines from heatmaps
            result["edge_lines"] = self._fit_edge_lines(result["edge_heatmaps"], crop_info)
            result["centerline"] = self._fit_centerline(result["centerline_heatmap"], crop_info)

        # EKF update
        if self.ekf is not None:
            self._ekf_update(result["corners"])
            result["ekf_state"] = self.get_ekf_state()

        return result

    def _preprocess(self, frame, prior_corners):
        """Crop frame around prior and prepare model input.

        Returns:
            img_tensor: (1, 3, 256, 256) normalized RGB on device
            pnp_tensor: dummy zeros (1, 4, 256, 256) on device (3ch model ignores this)
            crop_info: dict with cx, cy, half for coordinate mapping
        """
        h, w = frame.shape[:2]

        # Compute square crop around prior points
        visible = np.ones(4, dtype=bool)
        visible[prior_corners[:, 0] < 0] = False
        cx, cy, half = compute_crop_region(
            prior_corners, visible, w, h,
            padding=1.0, min_size=128, max_size=512,
        )

        # Crop and resize to 256x256 (BGR->RGB in one step)
        crop = frame[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
        if crop.size == 0:
            crop = np.zeros((half*2, half*2, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # uint8 to GPU first (1/4 data of float32), then normalize on GPU
        img_tensor = torch.from_numpy(np.ascontiguousarray(crop_rgb.transpose(2, 0, 1))
                                      ).unsqueeze(0).to(self.device).float().div_(255.0)

        return img_tensor, None, {"cx": cx, "cy": cy, "half": half}

    def _capture_cuda_graph(self):
        """Capture CUDA graph for the model with a static input buffer.

        Graph replay eliminates kernel launch overhead for small models like
        HRNet-w18, giving ~2x inference speedup.
        """
        c, h, w = 3, self.crop_size, self.crop_size
        self._static_input = torch.zeros(1, c, h, w, device=self.device)
        # Warm-up
        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self._static_input, None)
        torch.cuda.synchronize()
        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            with torch.inference_mode():
                self._static_output = self.model(self._static_input, None)

    def _infer(self, img_tensor):
        """Run model inference, using CUDA graph replay when available."""
        if self._cuda_graph is not None:
            self._static_input.copy_(img_tensor)
            self._cuda_graph.replay()
            return self._static_output
        else:
            with torch.inference_mode():
                return self.model(img_tensor, None)

    def _postprocess(self, outputs, crop_info):
        """Map model outputs from [-1,1] crop space to original image pixels.

        Args:
            outputs: model dict with 'coords' (1, 4, 2) in [-1, 1]
            crop_info: dict with cx, cy, half

        Returns:
            corners_orig: (4, 2) in original image pixels
            heatmaps: (4, 256, 256) numpy
        """
        coords = outputs["coords"][0].cpu().numpy()  # (4, 2) in [-1, 1]
        heatmaps = outputs["heatmaps"][0].cpu().numpy()  # (4, 256, 256)

        # Reorder from detection's ["near_left","near_right","far_left","far_right"]
        # to ours:                 [bottom_left, top_left, top_right, bottom_right]
        #                            [0,           2,        3,          1        ]
        reorder = [0, 2, 3, 1]
        coords = coords[reorder]
        heatmaps = heatmaps[reorder]

        cx, cy, half = crop_info["cx"], crop_info["cy"], crop_info["half"]

        # [-1, 1] → original image pixel coords
        # x_orig = x_norm * half + cx, y_orig = y_norm * half + cy
        corners_orig = np.zeros_like(coords)
        corners_orig[:, 0] = coords[:, 0] * half + cx
        corners_orig[:, 1] = coords[:, 1] * half + cy

        return corners_orig, heatmaps

    def _get_prior_corners(self, pose: dict) -> Optional[np.ndarray]:
        """Get PnP-prior corner estimates from GPS/IMU pose.

        Returns:
            corners: (4, 2) array at original resolution, or None
        """
        try:
            projected = self.meas_model.projector.project_corners(pose)
            corners = np.zeros((4, 2), dtype=np.float32)
            names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
            valid = False
            for i, name in enumerate(names):
                pt = projected.get(name)
                if pt is not None and pt[0] >= 0 and pt[1] >= 0:
                    corners[i] = pt
                    valid = True
                else:
                    corners[i] = [-1, -1]
            return corners if valid else None
        except Exception:
            return None

    def _ekf_update(self, corners):
        """Update EKF with detected corners (at working resolution 512x288)."""
        if not self._ekf_initialized:
            self._init_ekf_state(corners)
            return

        self.ekf.predict()
        z = self.meas_model.corners_to_measurement(corners)
        R = self.meas_model.get_measurement_noise()
        self.ekf.update(z, R, h_func=self.meas_model.h, H_func=self.meas_model.H)

    def _init_ekf_state(self, corners):
        """Initialize EKF state from first corner detection."""
        sx = 1920.0 / 512.0
        sy = 1080.0 / 288.0
        projected = {}
        names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
        for i, name in enumerate(names):
            projected[name] = np.array([corners[i, 0] * sx, corners[i, 1] * sy])
        pnp_result = self.meas_model.projector.solve_pnp_for_camera_pose(projected)
        if pnp_result is not None:
            pos = pnp_result['aircraft_position_runway']
            att = pnp_result['aircraft_pose_runway']
            self.ekf.x[0] = pos['x_meters']
            self.ekf.x[1] = pos['y_meters']
            self.ekf.x[2] = pos['z_meters']
            self.ekf.x[6] = np.radians(att['roll_deg'])
            self.ekf.x[7] = np.radians(att['pitch_deg'])
            self.ekf.x[8] = np.radians(att['yaw_deg'])
        else:
            self.ekf.x[0:3] = 0.0
            self.ekf.x[6:9] = 0.0
        self._ekf_initialized = True

    def get_ekf_state(self) -> Optional[dict]:
        if self.ekf is not None and self._ekf_initialized:
            state = self.ekf.get_pose_state()
            state['corners_pred'] = self._predict_corners_from_state()
            return state
        return None

    def _fit_edge_lines(self, edge_hm, crop_info):
        """Fit left/right edge line parameters from heatmaps.

        Args:
            edge_hm: (2, 256, 256) edge heatmaps [left, right]
            crop_info: dict with cx, cy, half

        Returns:
            dict with 'left' and 'right' line params (a, b, c) in original pixel coords
        """
        cx, cy, half = crop_info["cx"], crop_info["cy"], crop_info["half"]
        lines = {}
        for idx, label in enumerate(['left', 'right']):
            hm = edge_hm[idx]
            line = self._fit_line_from_heatmap(hm)
            if line is not None:
                # Map from crop [0, 255] to original image coords
                # crop_x_pixel = (hm_x * 2*half / 256) + (cx - half)
                scale = 2.0 * half / 256.0
                offset_x = cx - half
                offset_y = cy - half
                # Map line params: a*x + b*y + c = 0
                # x_orig = scale * x_crop + offset_x
                # y_orig = scale * y_crop + offset_y
                # => a*scale*x_crop + a*offset_x + b*scale*y_crop + b*offset_y + c = 0
                # => a_orig*x_orig + b_orig*y_orig + c_orig = 0
                a, b, c = line
                a_orig = a / scale
                b_orig = b / scale
                c_orig = c - a * offset_x / scale - b * offset_y / scale
                lines[label] = (a_orig, b_orig, c_orig)
            else:
                lines[label] = None
        return lines

    def _fit_centerline(self, cl_hm, crop_info):
        """Fit centerline from heatmap. Returns (a, b, c) in original pixel coords."""
        line = self._fit_line_from_heatmap(cl_hm[0])
        if line is None:
            return None
        cx, cy, half = crop_info["cx"], crop_info["cy"], crop_info["half"]
        a, b, c = line
        scale = 2.0 * half / 256.0
        offset_x = cx - half
        offset_y = cy - half
        a_orig = a / scale
        b_orig = b / scale
        c_orig = c - a * offset_x / scale - b * offset_y / scale
        return (a_orig, b_orig, c_orig)

    @staticmethod
    def _fit_line_from_heatmap(hm, threshold=0.3):
        """Fit line params (a, b, c) from heatmap using weighted PCA.

        Args:
            hm: (H, W) float heatmap
            threshold: minimum value to consider a pixel

        Returns:
            (a, b, c) or None if not enough points
        """
        ys, xs = np.where(hm > threshold)
        if len(ys) < 10:
            return None
        weights = hm[ys, xs]
        total_w = weights.sum()
        if total_w < 1e-6:
            return None
        # Weighted centroid
        cx = (xs * weights).sum() / total_w
        cy = (ys * weights).sum() / total_w
        # Weighted covariance
        dx = xs - cx
        dy = ys - cy
        cov_xx = (dx * dx * weights).sum() / total_w
        cov_yy = (dy * dy * weights).sum() / total_w
        cov_xy = (dx * dy * weights).sum() / total_w
        # PCA: direction is eigenvector of [[cov_xx, cov_xy], [cov_xy, cov_yy]]
        # Normal direction is perpendicular
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        if det < 1e-12:
            return None
        # Eigenvalue for line direction (larger)
        eigval = trace / 2.0 + np.sqrt(max(0, trace**2 / 4.0 - det))
        # Normal vector (a, b) is eigenvector of smaller eigenvalue
        a = cov_xy
        b = eigval - cov_xx
        norm = np.sqrt(a**2 + b**2)
        if norm < 1e-8:
            return None
        a, b = a / norm, b / norm
        c = -(a * cx + b * cy)
        return (a, b, c)

    def _predict_corners_from_state(self) -> Optional[np.ndarray]:
        if not self._ekf_initialized:
            return None
        z_pred = self.meas_model.h(self.ekf.x)
        return z_pred.reshape(4, 2)

    def reset(self):
        if self.ekf is not None:
            self.ekf = SE3EKF()
        self._ekf_initialized = False
        # Re-capture CUDA graph (needed if model weights were updated)
        if self.device.type == "cuda":
            self._capture_cuda_graph()
