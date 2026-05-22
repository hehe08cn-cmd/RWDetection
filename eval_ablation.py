"""Ablation: HRNet-Offset with vs without PnP prior heatmap input."""

import sys, os, numpy as np, cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


@torch.inference_mode()
def eval_ablation(detector, frames_cache, gt_cache, all_frames, pose_dict, zero_pnp):
    """Eval with or without PnP heatmap input."""
    errors = [[] for _ in range(4)]
    for fi in all_frames:
        gt_coords, gt_visible = gt_cache.get(fi, (None, None))
        frame = frames_cache.get(fi)
        if frame is None or gt_coords is None:
            continue

        prior = detector._get_prior_corners(pose_dict[fi])
        if prior is None:
            continue
        prior_det = prior[[0, 3, 1, 2]]

        img, pnp, crop_info = detector._preprocess(frame, prior_det)
        if zero_pnp:
            pnp.zero_()
        outputs = detector.model(img, pnp)
        corners_orig, _ = detector._postprocess(outputs, crop_info)

        pred = corners_orig * np.array([512.0/1920.0, 288.0/1080.0])
        for i in range(4):
            if gt_visible[i]:
                errors[i].append(np.linalg.norm(pred[i] - gt_coords[i]))
    return errors


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

    # Generate noisy pose dicts for testing
    rng = np.random.RandomState(42)
    noisy_pose_dict = {}
    for nm in [0, 5, 10]:
        noisy_pose_dict[nm] = {fi: add_gps_noise(pose_dict[fi], nm, rng)
                               for fi in test_frames}

    print(f"Ablation on {vk} ({len(test_frames)} frames)\n")
    print(f"{'GPS noise':<12} {'PnP input':<12} {'Mean':>7} {'Med':>7} {'PCK@3':>7} {'PCK@5':>7}")
    print("-" * 55)

    for nm in [0, 5, 10]:
        for zero_pnp, label in [(False, "WITH"), (True, "WITHOUT")]:
            err_lists = eval_ablation(detector, frames_cache, gt_cache,
                                       test_frames, noisy_pose_dict[nm], zero_pnp)
            e = np.concatenate(err_lists)
            print(f"{f'{nm}m':<12} {label:<12} {e.mean():>7.2f} {np.median(e):>7.2f} "
                  f"{(e<3).mean():>6.1%} {(e<5).mean():>6.1%}")

    detector.reset()


if __name__ == "__main__":
    main()
