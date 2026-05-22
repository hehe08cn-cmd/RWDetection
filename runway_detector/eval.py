"""Evaluation script for HRNet-Offset runway detector.

Usage:
    python -m runway_detector.eval --video_key video3
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runway_detector.config import VIDEOS, POSES, WORKING_SIZE, NUM_CORNERS
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HRNet-Offset detector")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override default checkpoint")
    parser.add_argument("--video_key", type=str, default="video3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_ekf", action="store_true")
    return parser.parse_args()


def evaluate(args):
    gt_gen = GroundTruthGenerator()

    print("Loading HRNet-Offset detector...")
    kwargs = {"device": args.device, "enable_ekf": not args.no_ekf}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    detector = RunwayInference(**kwargs)
    print(f"  EKF: {detector.enable_ekf}")

    # Load poses
    pose_dict = load_poses_dict(POSES[args.video_key])
    frames = sorted(pose_dict.keys())
    frames = frames[::args.frame_stride]
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"Evaluating {len(frames)} frames")

    # Open video
    cap = cv2.VideoCapture(VIDEOS[args.video_key])

    # Metrics
    corner_errors = []
    per_corner_errors = [[] for _ in range(4)]
    prior_errors = []

    pbar = tqdm(frames, desc="Evaluating")
    for frame_idx in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        pose = pose_dict[frame_idx]
        gt_coords, gt_visible = gt_gen.generate_corner_gt(pose)

        try:
            result = detector(frame, pose=pose)
        except Exception as e:
            print(f"  Error at frame {frame_idx}: {e}")
            continue

        pred = result["corners"]
        prior = result.get("corners_raw")

        for i in range(NUM_CORNERS):
            if gt_visible[i]:
                err = np.linalg.norm(pred[i] - gt_coords[i])
                corner_errors.append(err)
                per_corner_errors[i].append(err)
                if prior is not None:
                    prior_err = np.linalg.norm(
                        prior[i] * np.array([512/1920.0, 288/1080.0]) - gt_coords[i])
                    prior_errors.append(prior_err)

        if corner_errors:
            pbar.set_postfix({'err': f"{np.mean(corner_errors[-20:]):.2f}px"})

    cap.release()
    detector.reset()

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - HRNet-Offset")
    print("=" * 60)

    if corner_errors:
        ce = np.array(corner_errors)
        print(f"\nCorner Detection:")
        print(f"  Mean error:     {ce.mean():.2f} px @ 512x288 "
              f"({ce.mean()*1920/512:.1f} px @ 1920x1080)")
        print(f"  Median error:   {np.median(ce):.2f} px")
        print(f"  Std error:      {ce.std():.2f} px")
        print(f"  Max error:      {ce.max():.2f} px")
        for thresh in [1, 2, 3, 5, 10]:
            print(f"  PCK@{thresh}px:     {(ce < thresh).mean():.1%}")

        names = ['BL', 'TL', 'TR', 'BR']
        print(f"\nPer-corner:")
        for i, name in enumerate(names):
            if per_corner_errors[i]:
                pe = np.array(per_corner_errors[i])
                print(f"  {name}: mean={pe.mean():.2f} median={np.median(pe):.2f} "
                      f"PCK@3={(pe<3).mean():.1%} PCK@5={(pe<5).mean():.1%}")

    if prior_errors:
        pe = np.array(prior_errors)
        print(f"\nPnP Prior (before refinement):")
        print(f"  Mean: {pe.mean():.2f} px  Median: {np.median(pe):.2f} px")

    return {
        'corner_errors': corner_errors,
        'per_corner_errors': per_corner_errors,
        'prior_errors': prior_errors,
    }


if __name__ == "__main__":
    evaluate(parse_args())
