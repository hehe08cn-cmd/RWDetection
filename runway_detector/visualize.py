"""Generate visualization video with runway detection overlay.

Usage:
    python -m runway_detector.visualize --output output.mp4
"""

import sys
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runway_detector.config import VIDEOS, POSES, WORKING_SIZE, ORIGINAL_SIZE
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override HRNet-Offset checkpoint path")
    parser.add_argument("--video_key", type=str, default="video3")
    parser.add_argument("--output", type=str, default="detection_output.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_ekf", action="store_true")
    parser.add_argument("--test_only", action="store_true",
                        help="Only process test split (frames >= 1250 for video3)")
    return parser.parse_args()


def draw_overlay(frame, result, gt_coords, gt_visible):
    """Draw detection results on frame."""
    vis = frame.copy()
    H_orig, W_orig = frame.shape[:2]
    W_work, H_work = WORKING_SIZE
    sx = W_orig / W_work
    sy = H_orig / H_work

    corners = result["corners"]  # at working resolution
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]  # BGR
    names = ['BL', 'TL', 'TR', 'BR']

    # Draw PnP prior corners (cyan outline)
    corners_raw = result.get("corners_raw")
    if corners_raw is not None:
        raw_w = corners_raw.copy()
        raw_w[:, 0] *= 512.0 / W_orig
        raw_w[:, 1] *= 288.0 / H_orig
        for i in range(4):
            rx, ry = int(raw_w[i][0] * sx), int(raw_w[i][1] * sy)
            if rx >= 0 and ry >= 0:
                cv2.circle(vis, (rx, ry), 5, (255, 255, 0), 1)

    # Draw refined runway outline (yellow)
    for i in range(4):
        p1 = (int(corners[i][0] * sx), int(corners[i][1] * sy))
        p2 = (int(corners[(i + 1) % 4][0] * sx), int(corners[(i + 1) % 4][1] * sy))
        cv2.line(vis, p1, p2, (0, 255, 255), 2)

    # Draw refined corners (yellow filled)
    for i in range(4):
        px, py = int(corners[i][0] * sx), int(corners[i][1] * sy)
        cv2.circle(vis, (px, py), 8, (0, 255, 255), -1)
        cv2.putText(vis, names[i], (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Draw GT corners and error lines
    for i in range(4):
        if gt_visible[i]:
            gx, gy = int(gt_coords[i][0] * sx), int(gt_coords[i][1] * sy)
            cv2.circle(vis, (gx, gy), 6, colors[i], 2)
            px, py = int(corners[i][0] * sx), int(corners[i][1] * sy)
            cv2.line(vis, (px, py), (gx, gy), (255, 255, 255), 1)
            err = np.linalg.norm(corners[i] - gt_coords[i])
            cv2.putText(vis, f"{err:.1f}px", (gx - 30, gy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Draw EKF predicted corners
    ekf_state = result.get("ekf_state")
    if ekf_state is not None and ekf_state.get("corners_pred") is not None:
        ekf_corners = ekf_state["corners_pred"]
        for i in range(4):
            ex = int(ekf_corners[i][0] * sx)
            ey = int(ekf_corners[i][1] * sy)
            cv2.circle(vis, (ex, ey), 5, (255, 0, 255), 1)

    # Info panel
    has_ekf = result.get("ekf_state") is not None
    parts = ["HRNet (yellow)", "PnP prior (cyan)"]
    if has_ekf:
        parts.append("EKF (magenta)")
    parts.append("GT (colored)")
    cv2.putText(vis, " | ".join(parts), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis


def main():
    args = parse_args()

    # Load detector (HRNet-Offset with crop+PnP input)
    print("Loading HRNet-Offset detector...")
    kwargs = {"device": args.device, "enable_ekf": not args.no_ekf}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    detector = RunwayInference(**kwargs)
    print(f"  EKF: {detector.enable_ekf}")

    # Load GT
    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[args.video_key])

    # Build frame list
    all_frames = sorted(pose_dict.keys())
    if args.test_only and args.video_key == "video3":
        all_frames = [f for f in all_frames if f >= 1250]
    frames = all_frames[::args.frame_stride]
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Processing {len(frames)} frames")

    # Open video
    cap = cv2.VideoCapture(VIDEOS[args.video_key])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, 30.0, (1920, 1080))

    # Metrics
    corner_errors = []

    pbar = tqdm(frames, desc="Rendering")
    for frame_idx in pbar:
        if frame_idx >= total:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        pose = pose_dict.get(frame_idx)
        if pose is None:
            continue
        gt_coords, gt_visible = gt_gen.generate_corner_gt(pose)

        # Run inference (HRNet-Offset with PnP crop + heatmap input)
        result = detector(frame, pose=pose)

        # Metrics
        for i in range(4):
            if gt_visible[i]:
                corner_errors.append(
                    np.linalg.norm(result["corners"][i] - gt_coords[i]))

        # Draw overlay
        vis = draw_overlay(frame, result, gt_coords, gt_visible)

        # Frame counter
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 1060),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(vis)

        # Update progress
        if corner_errors:
            pbar.set_postfix({
                'err': f"{np.mean(corner_errors[-20:]):.1f}px",
            })

    cap.release()
    writer.release()
    detector.reset()

    # Summary
    print(f"\nVisualization saved to: {args.output}")
    if corner_errors:
        ce = np.array(corner_errors)
        print(f"Corner error: mean={ce.mean():.2f} median={np.median(ce):.2f} px "
              f"@ 512x288 ({ce.mean()*1920/512:.1f} px @ 1920x1080)")


if __name__ == "__main__":
    main()
