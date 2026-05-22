"""Compare three prior modes for crop guidance:
  1. Pure GPS prior (baseline)
  2. Direct pass-through (previous frame detection)
  3. EKF prediction prior (SE(3)-EKF closed loop: predict -> crop -> detect -> update)
"""

import sys, os, numpy as np, cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES, ORIGINAL_SIZE
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference
from runway_detector.filtering.ekf import SE3EKF
from runway_detector.filtering.measurement_model import CornerMeasurementModel


W_orig, H_orig = ORIGINAL_SIZE
SX, SY = 512.0 / W_orig, 288.0 / H_orig


def add_gps_noise(pose, noise_m, rng):
    if noise_m <= 0:
        return pose
    noisy = pose.copy()
    ln = noise_m / 111320.0
    ll = noise_m / (111320.0 * np.cos(np.radians(pose["latitude"])) + 1e-8)
    an = noise_m / 10.0
    noisy["latitude"] += rng.uniform(-ln, ln)
    noisy["longitude"] += rng.uniform(-ll, ll)
    noisy["altitude"] += rng.uniform(-noise_m, noise_m)
    for k in ["yaw", "pitch", "roll"]:
        noisy[k] += rng.uniform(-an, an)
    return noisy


@torch.inference_mode()
def eval_mode_gps(detector, meas_model, frames_cache, gt_cache, test_frames,
                  pose_dict, noise_m, rng, stride=1):
    """Mode 1: Pure GPS/IMU prior (current approach)."""
    errors = [[] for _ in range(4)]
    for fi in test_frames:
        if fi not in gt_cache or fi not in frames_cache:
            continue
        gt_coords, gt_visible = gt_cache[fi]
        frame = frames_cache[fi]
        noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
        prior = detector._get_prior_corners(noisy_pose)
        if prior is None:
            continue
        prior_det = prior[[0, 3, 1, 2]]
        img, pnp, crop_info = detector._preprocess(frame, prior_det)
        outputs = detector.model(img, pnp)
        corners_orig, _ = detector._postprocess(outputs, crop_info)
        pred = corners_orig * np.array([SX, SY])
        for i in range(4):
            if gt_visible[i]:
                errors[i].append(np.linalg.norm(pred[i] - gt_coords[i]))
    return errors


@torch.inference_mode()
def eval_mode_passthrough(detector, meas_model, frames_cache, gt_cache, test_frames,
                          pose_dict, noise_m, rng, stride=1):
    """Mode 2: Direct pass-through of previous frame detection."""
    errors = [[] for _ in range(4)]
    prev_corners_orig = None

    for fi in test_frames:
        if fi not in gt_cache or fi not in frames_cache:
            continue
        gt_coords, gt_visible = gt_cache[fi]
        frame = frames_cache[fi]

        if prev_corners_orig is not None:
            prior = prev_corners_orig.copy()
        else:
            noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
            prior = detector._get_prior_corners(noisy_pose)
            if prior is None:
                continue

        prior_det = prior[[0, 3, 1, 2]]
        img, pnp, crop_info = detector._preprocess(frame, prior_det)
        outputs = detector.model(img, pnp)
        corners_orig, _ = detector._postprocess(outputs, crop_info)
        prev_corners_orig = corners_orig.copy()

        pred = corners_orig * np.array([SX, SY])
        for i in range(4):
            if gt_visible[i]:
                errors[i].append(np.linalg.norm(pred[i] - gt_coords[i]))
    return errors


@torch.inference_mode()
def eval_mode_ekf(detector, meas_model, frames_cache, gt_cache, test_frames,
                  pose_dict, noise_m, rng, stride=1):
    """Mode 3: SE(3)-EKF closed-loop tracking prior.

    Flow: EKF.predict() -> h(x) corner prediction -> crop prior -> detect -> EKF.update()
    Falls back to GPS prior when EKF not yet initialized or prediction fails.
    """
    errors = [[] for _ in range(4)]
    ekf = SE3EKF(dt=stride / 30.0)
    ekf_initialized = False
    prev_z = None  # previous measurement for velocity initialization

    for fi in test_frames:
        if fi not in gt_cache or fi not in frames_cache:
            ekf_initialized = False
            continue
        gt_coords, gt_visible = gt_cache[fi]
        frame = frames_cache[fi]

        # Determine crop prior
        if ekf_initialized:
            # Predict EKF forward and get corner prediction
            ekf.predict()
            z_pred = meas_model.h(ekf.x)  # (8,) at working resolution
            # Convert to original resolution corners for crop
            prior = np.zeros((4, 2), dtype=np.float32)
            valid = True
            for i in range(4):
                px, py = z_pred[2 * i], z_pred[2 * i + 1]
                if px < 0 or py < 0:
                    valid = False
                    break
                prior[i, 0] = px / SX
                prior[i, 1] = py / SY
            if not valid:
                # EKF prediction invalid, fall back to GPS
                noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
                prior = detector._get_prior_corners(noisy_pose)
        else:
            # First frame: bootstrap from GPS
            noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
            prior = detector._get_prior_corners(noisy_pose)

        if prior is None:
            ekf_initialized = False
            continue

        # Detect
        prior_det = prior[[0, 3, 1, 2]]
        img, pnp, crop_info = detector._preprocess(frame, prior_det)
        outputs = detector.model(img, pnp)
        corners_orig, _ = detector._postprocess(outputs, crop_info)

        # EKF update with detection
        corners_work = corners_orig * np.array([SX, SY])
        z = meas_model.corners_to_measurement(corners_work)

        if not ekf_initialized:
            # Initialize EKF from first detection
            _init_ekf_from_detection(ekf, meas_model, corners_work, prev_z)
            ekf_initialized = True
        else:
            R = meas_model.get_measurement_noise()
            ekf.update(z, R, h_func=meas_model.h, H_func=meas_model.H)

        prev_z = z.copy()

        pred = corners_orig * np.array([SX, SY])
        for i in range(4):
            if gt_visible[i]:
                errors[i].append(np.linalg.norm(pred[i] - gt_coords[i]))
    return errors


def _init_ekf_from_detection(ekf, meas_model, corners_work, prev_z):
    """Initialize EKF state from corner detection via PnP back-solve."""
    projected = {}
    names = ['bottom_left', 'top_left', 'top_right', 'bottom_right']
    for i, name in enumerate(names):
        projected[name] = np.array([corners_work[i, 0] / SX, corners_work[i, 1] / SY])
    pnp_result = meas_model.projector.solve_pnp_for_camera_pose(projected)
    if pnp_result is not None:
        pos = pnp_result['aircraft_position_runway']
        att = pnp_result['aircraft_pose_runway']
        ekf.x[0] = pos['x_meters']
        ekf.x[1] = pos['y_meters']
        ekf.x[2] = pos['z_meters']
        ekf.x[6] = np.radians(att['roll_deg'])
        ekf.x[7] = np.radians(att['pitch_deg'])
        ekf.x[8] = np.radians(att['yaw_deg'])
    if prev_z is not None:
        # Estimate velocity from previous measurement
        dt = ekf.dt
        ekf.x[3] = (ekf.x[0] - ekf.x[0]) / max(dt, 1e-6)  # placeholder
    else:
        ekf.x[3:6] = 0.0


def main():
    vk = "video3"
    print("Loading HRNet-Offset...")
    detector = RunwayInference(device="cuda", enable_ekf=False)
    meas_model = CornerMeasurementModel()

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[vk])
    all_test_frames = sorted([f for f in pose_dict if f >= 1250])

    cap = cv2.VideoCapture(VIDEOS[vk])
    frames_cache, gt_cache = {}, {}
    for fi in all_test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames_cache[fi] = frame
            gt_cache[fi] = gt_gen.generate_corner_gt(pose_dict[fi])
    cap.release()

    rng = np.random.RandomState(42)

    modes = [
        ("GPS prior", eval_mode_gps),
        ("Direct pass-through", eval_mode_passthrough),
        ("EKF prediction prior", eval_mode_ekf),
    ]

    for stride, desc in [(1, "30fps"), (5, "6fps"), (10, "3fps")]:
        test_frames = all_test_frames[::stride]
        print(f"\n{'='*65}")
        print(f"Frame stride={stride} ({desc}), {len(test_frames)} frames, noise=5m")
        print(f"{'='*65}")
        print(f"{'Mode':<35} {'Mean':>7} {'Med':>7} {'PCK@3':>7} {'PCK@5':>7}")
        print("-" * 65)

        for label, eval_fn in modes:
            detector.reset()
            err_lists = eval_fn(detector, meas_model, frames_cache, gt_cache,
                                test_frames, pose_dict, 5, rng, stride)
            e = np.concatenate(err_lists)
            full_label = f"{label} (stride={stride})"
            print(f"{full_label:<35} {e.mean():>7.2f} {np.median(e):>7.2f} "
                  f"{(e<3).mean():>6.1%} {(e<5).mean():>6.1%}")

    detector.reset()


if __name__ == "__main__":
    main()
