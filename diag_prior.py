"""Measure PnP prior corner error independently, and test lower model weights."""

import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


def add_gps_noise(pose, noise_m, rng):
    if noise_m <= 0:
        return pose
    noisy = pose.copy()
    lat_noise = noise_m / 111320.0
    lon_noise = noise_m / (111320.0 * np.cos(np.radians(pose["latitude"])) + 1e-8)
    noisy["latitude"] = pose["latitude"] + rng.uniform(-lat_noise, lat_noise)
    noisy["longitude"] = pose["longitude"] + rng.uniform(-lon_noise, lon_noise)
    noisy["altitude"] = pose["altitude"] + rng.uniform(-noise_m, noise_m)
    att_noise = noise_m / 10.0
    for k in ["yaw", "pitch", "roll"]:
        noisy[k] = pose[k] + rng.uniform(-att_noise, att_noise)
    return noisy


def main():
    checkpoint = "checkpoints/best_stage2.pt"
    video_key = "video3"
    device = "cuda"

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[video_key])
    test_frames = sorted([f for f in pose_dict if f >= 1250])

    # Pre-load
    cap = cv2.VideoCapture(VIDEOS[video_key])
    frames_cache, gt_cache = {}, {}
    for fi in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames_cache[fi] = frame
            gt_cache[fi] = gt_gen.generate_corner_gt(pose_dict[fi])
    cap.release()

    detector = RunwayInference(checkpoint, device=device, enable_ekf=False,
                               enable_cuda_graph=True, clone_output=True)

    # 1. Measure prior-only error
    print("=" * 60)
    print("PRIOR-ONLY ERROR (no model)")
    print("=" * 60)
    rng = np.random.RandomState(42)
    for nm in [0, 5, 10, 20]:
        errors = []
        for fi in test_frames:
            gt_coords, gt_visible = gt_cache.get(fi, (None, None))
            if gt_coords is None:
                continue
            noisy = add_gps_noise(pose_dict[fi], nm, rng)
            prior = detector._get_prior_corners(noisy)
            if prior is None:
                continue
            # Scale prior to working res
            prior_w = prior.copy()
            prior_w[:, 0] *= 512.0 / 1920.0
            prior_w[:, 1] *= 288.0 / 1080.0
            for i in range(4):
                if gt_visible[i] and prior[i, 0] >= 0:
                    errors.append(np.linalg.norm(prior_w[i] - gt_coords[i]))
        e = np.array(errors)
        print(f"  GPS noise={nm}m: mean={e.mean():.2f} med={np.median(e):.2f} "
              f"min={e.min():.2f} max={e.max():.2f} N={len(e)}")

    # 2. Test lower model weights
    print(f"\n{'='*60}")
    print("LOWER MODEL WEIGHTS (w=0.30, 0.40, 0.50)")
    print("=" * 60)
    original_fuse = detector._fuse_with_prior

    for w in [0.30, 0.40, 0.50, 0.60]:
        def make_fixed(weight):
            def f(result, prior_corners, output):
                mc = result["corners"].copy()
                sx_w = 512.0/1920.0; sy_w = 288.0/1080.0
                pw = prior_corners.copy(); pw[:, 0] *= sx_w; pw[:, 1] *= sy_w
                fused = weight * mc + (1 - weight) * pw
                result["corners"] = fused
                result["corners_original"] = fused * np.array([1920.0/512, 1080.0/288])
                return result
            return f

        for nm in [0, 5, 10]:
            detector._fuse_with_prior = make_fixed(w)
            rng = np.random.RandomState(42)
            errors = []
            for fi in test_frames:
                frame = frames_cache.get(fi)
                gt_coords, gt_visible = gt_cache.get(fi, (None, None))
                if frame is None or gt_coords is None:
                    continue
                noisy = add_gps_noise(pose_dict[fi], nm, rng)
                result = detector(frame, pose=noisy)
                for i in range(4):
                    if gt_visible[i]:
                        errors.append(np.linalg.norm(result["corners"][i] - gt_coords[i]))
            e = np.array(errors)
            print(f"  w={w:.2f} noise={nm}m: mean={e.mean():.2f} med={np.median(e):.2f} "
                  f"PCK@5={(e<5).mean():.1%}")

    detector._fuse_with_prior = original_fuse
    detector.reset()


if __name__ == "__main__":
    main()
