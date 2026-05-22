"""Compare GPS prior vs tracking prior (previous frame detection) for crop guidance."""

import sys, os, numpy as np, cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


def add_gps_noise(pose, noise_m, rng):
    if noise_m <= 0: return pose
    noisy = pose.copy()
    ln = noise_m / 111320.0
    ll = noise_m / (111320.0 * np.cos(np.radians(pose["latitude"])) + 1e-8)
    an = noise_m / 10.0
    noisy["latitude"] += rng.uniform(-ln, ln)
    noisy["longitude"] += rng.uniform(-ll, ll)
    noisy["altitude"] += rng.uniform(-noise_m, noise_m)
    for k in ["yaw","pitch","roll"]:
        noisy[k] += rng.uniform(-an, an)
    return noisy


@torch.inference_mode()
def eval_tracking(detector, frames_cache, gt_cache, test_frames, pose_dict,
                  noise_m, rng, use_tracking):
    """Eval with GPS prior or tracking prior (previous detection propagated forward)."""
    errors = [[] for _ in range(4)]
    prev_corners = None  # tracking prior from previous frame

    for fi in test_frames:
        gt_coords, gt_visible = gt_cache.get(fi, (None, None))
        frame = frames_cache.get(fi)
        if frame is None or gt_coords is None:
            continue

        if use_tracking and prev_corners is not None:
            # Use previous frame's detected corners as prior (at original resolution)
            # Add small noise to simulate tracking uncertainty
            prior = prev_corners.copy()
        else:
            # Use GPS/IMU prior (possibly noisy)
            noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
            prior = detector._get_prior_corners(noisy_pose)
            if prior is None:
                prev_corners = None
                continue

        # Run detection with this prior (bypassing normal __call__ for control)
        prior_det = prior[[0, 3, 1, 2]]
        img, pnp, crop_info = detector._preprocess(frame, prior_det)
        outputs = detector.model(img, pnp)
        corners_orig, _ = detector._postprocess(outputs, crop_info)

        # Store current detection for next frame's tracking prior
        prev_corners = corners_orig.copy()

        pred = corners_orig * np.array([512.0/1920.0, 288.0/1080.0])
        for i in range(4):
            if gt_visible[i]:
                errors[i].append(np.linalg.norm(pred[i] - gt_coords[i]))
    return errors


def main():
    vk = "video3"
    print("Loading HRNet-Offset...")
    detector = RunwayInference(device="cuda", enable_ekf=False)

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[vk])
    test_frames = sorted([f for f in pose_dict if f >= 1250])

    cap = cv2.VideoCapture(VIDEOS[vk])
    frames_cache, gt_cache = {}, {}
    for fi in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames_cache[fi] = frame
            gt_cache[fi] = gt_gen.generate_corner_gt(pose_dict[fi])
    cap.release()

    rng = np.random.RandomState(42)

    print(f"Tracking prior ablation on {vk} ({len(test_frames)} frames)\n")
    print(f"{'Config':<35} {'Mean':>7} {'Med':>7} {'PCK@3':>7} {'PCK@5':>7}")
    print("-" * 65)

    for noise_m in [0, 5, 10]:
        for use_track, label in [(False, f"GPS prior ({noise_m}m noise)"),
                                  (True,  f"Tracking prior (from prev det)")]:
            detector.reset()
            err_lists = eval_tracking(detector, frames_cache, gt_cache,
                                       test_frames, pose_dict, noise_m, rng, use_track)
            e = np.concatenate(err_lists)
            print(f"{label:<35} {e.mean():>7.2f} {np.median(e):>7.2f} "
                  f"{(e<3).mean():>6.1%} {(e<5).mean():>6.1%}")

    detector.reset()


if __name__ == "__main__":
    main()
