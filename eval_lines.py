"""Evaluate edge and centerline accuracy derived from corner predictions.

Lines are computed from predicted corners:
  - Left edge:  TL -> BL
  - Right edge: TR -> BR
  - Centerline: mid(TL,TR) -> mid(BL,BR)

Metrics: angle error (deg), endpoint distance (px), line offset (px)
"""

import sys, os, numpy as np, cv2
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference

W_orig, H_orig = 1920, 1080
SX, SY = 512.0 / W_orig, 288.0 / H_orig


def line_from_points(p1, p2):
    """Line params (a, b, c) for ax + by + c = 0, a^2 + b^2 = 1."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-8:
        return None
    a, b = -dy / length, dx / length
    c = -(a * p1[0] + b * p1[1])
    return np.array([a, b, c]), length


def line_angle_error(l1, l2):
    """Angle between two lines in degrees [0, 90]."""
    if l1 is None or l2 is None:
        return None
    dot = abs(l1[0] * l2[0] + l1[1] * l2[1])
    dot = min(dot, 1.0)
    return np.degrees(np.arccos(dot))


def point_line_distance(pt, line):
    """Perpendicular distance from point to line."""
    return abs(line[0] * pt[0] + line[1] * pt[1] + line[2])


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
def eval_lines(detector, frames_cache, gt_cache, all_frames, pose_dict, noise_m, rng,
               use_tracking=False):
    """Evaluate edge and centerline from predicted vs GT corners."""
    left_edge_angle = []
    right_edge_angle = []
    centerline_angle = []
    left_endpoint_dist = []
    right_endpoint_dist = []
    centerline_endpoint_dist = []
    left_offset = []
    right_offset = []
    centerline_offset = []

    prev_corners_orig = None

    for fi in all_frames:
        if fi not in gt_cache or fi not in frames_cache:
            continue
        gt_coords, gt_visible = gt_cache[fi]
        frame = frames_cache[fi]

        if use_tracking and prev_corners_orig is not None:
            prior = prev_corners_orig.copy()
        else:
            noisy_pose = add_gps_noise(pose_dict[fi], noise_m, rng)
            prior = detector._get_prior_corners(noisy_pose)
            if prior is None:
                prev_corners_orig = None
                continue

        prior_det = prior[[0, 3, 1, 2]]
        img_t, _, crop_info = detector._preprocess(frame, prior_det)
        outputs = detector._infer(img_t)
        corners_orig, _ = detector._postprocess(outputs, crop_info)
        prev_corners_orig = corners_orig.copy()

        pred = corners_orig * np.array([SX, SY])  # working resolution

        # GT corners already at working resolution
        # Compute lines from corners
        # Left edge: TL(idx=1) -> BL(idx=0)
        # Right edge: TR(idx=2) -> BR(idx=3)
        # Centerline: mid(TL,TR) -> mid(BL,BR)

        # GT lines
        gt_left = line_from_points(gt_coords[1], gt_coords[0]) if (gt_visible[1] and gt_visible[0]) else (None, None)
        gt_right = line_from_points(gt_coords[2], gt_coords[3]) if (gt_visible[2] and gt_visible[3]) else (None, None)
        gt_cl_top = (gt_coords[1] + gt_coords[2]) / 2 if (gt_visible[1] and gt_visible[2]) else None
        gt_cl_bot = (gt_coords[0] + gt_coords[3]) / 2 if (gt_visible[0] and gt_visible[3]) else None
        gt_cl = line_from_points(gt_cl_top, gt_cl_bot) if (gt_cl_top is not None and gt_cl_bot is not None) else (None, None)

        # Pred lines (derived from predictions — all 4 corners always predicted)
        pred_left = line_from_points(pred[1], pred[0])
        pred_right = line_from_points(pred[2], pred[3])
        pred_cl_top = (pred[1] + pred[2]) / 2
        pred_cl_bot = (pred[0] + pred[3]) / 2
        pred_cl = line_from_points(pred_cl_top, pred_cl_bot)

        # Angle errors
        if gt_left[0] is not None and pred_left[0] is not None:
            ae = line_angle_error(gt_left[0], pred_left[0])
            left_edge_angle.append(ae)

        if gt_right[0] is not None and pred_right[0] is not None:
            ae = line_angle_error(gt_right[0], pred_right[0])
            right_edge_angle.append(ae)

        if gt_cl[0] is not None and pred_cl[0] is not None:
            ae = line_angle_error(gt_cl[0], pred_cl[0])
            centerline_angle.append(ae)

        # Endpoint distance: pred corner to GT line
        if gt_left[0] is not None:
            left_endpoint_dist.append(point_line_distance(pred[0], gt_left[0]))
            left_endpoint_dist.append(point_line_distance(pred[1], gt_left[0]))
            left_offset.append(point_line_distance(pred[1], gt_left[0]))

        if gt_right[0] is not None:
            right_endpoint_dist.append(point_line_distance(pred[2], gt_right[0]))
            right_endpoint_dist.append(point_line_distance(pred[3], gt_right[0]))
            right_offset.append(point_line_distance(pred[2], gt_right[0]))

        if gt_cl[0] is not None:
            centerline_endpoint_dist.append(point_line_distance(pred_cl_top, gt_cl[0]))
            centerline_endpoint_dist.append(point_line_distance(pred_cl_bot, gt_cl[0]))
            centerline_offset.append(point_line_distance(pred_cl_top, gt_cl[0]))

    return {
        'left_edge_angle': left_edge_angle,
        'right_edge_angle': right_edge_angle,
        'centerline_angle': centerline_angle,
        'left_endpoint_dist': left_endpoint_dist,
        'right_endpoint_dist': right_endpoint_dist,
        'centerline_endpoint_dist': centerline_endpoint_dist,
        'left_offset': left_offset,
        'right_offset': right_offset,
        'centerline_offset': centerline_offset,
    }


def print_metric(name, values, unit="px"):
    if not values:
        print(f"  {name:<25} (no data)")
        return
    v = np.array(values)
    print(f"  {name:<25} mean={v.mean():.2f}{unit}  med={np.median(v):.2f}{unit}  "
          f"max={v.max():.2f}{unit}")


def eval_single_video(vk, detector, rng):
    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[vk])
    test_frames = sorted(pose_dict.keys())

    cap = cv2.VideoCapture(VIDEOS[vk])
    frames_cache, gt_cache = {}, {}
    for fi in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames_cache[fi] = frame
            gt_cache[fi] = gt_gen.generate_corner_gt(pose_dict[fi])
    cap.release()

    print(f"\n{'='*70}")
    print(f"LINE FEATURE EVALUATION — {vk} ({len(test_frames)} frames)")
    print(f"{'='*70}")

    for noise_m in [0, 5]:
        for use_track, label in [(False, f"GPS prior ({noise_m}m)"), (True, "Tracking prior")]:
            detector.reset()
            results = eval_lines(detector, frames_cache, gt_cache, test_frames,
                                 pose_dict, noise_m, rng, use_tracking=use_track)
            print(f"\n  [{label}]")
            print_metric("Edge angle error",
                         results['left_edge_angle'] + results['right_edge_angle'], "°")
            print_metric("Centerline angle error", results['centerline_angle'], "°")
            print_metric("Edge endpoint dist",
                         results['left_endpoint_dist'] + results['right_endpoint_dist'], "px")
            print_metric("Centerline endpoint dist", results['centerline_endpoint_dist'], "px")


def main():
    print("Loading HRNet-Offset...")
    detector = RunwayInference(device="cuda", enable_ekf=False)
    rng = np.random.RandomState(42)

    for vk in ["video1", "video2", "video3"]:
        eval_single_video(vk, detector, rng)

    detector.reset()
    print()


if __name__ == "__main__":
    main()
