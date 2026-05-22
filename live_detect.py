"""Real-time runway detection visualization.

Draws predicted and GT runway regions as filled quadrilaterals.
Press 'q' to quit, ' ' (space) to pause/resume, 't' to toggle tracking prior.
"""

import sys, os, argparse, time, numpy as np, cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES, ORIGINAL_SIZE
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference

W_orig, H_orig = ORIGINAL_SIZE
SX = 512.0 / W_orig
SY = 288.0 / H_orig

# Display at half resolution for speed (drawing on 960×540 is 4x faster than 1920×1080)
DW, DH = W_orig // 2, H_orig // 2
DSX, DSY = DW / 512.0, DH / 288.0


def main():
    parser = argparse.ArgumentParser(description="Live runway detection")
    parser.add_argument("--video_key", type=str, default="video3")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--no_tracking", action="store_true",
                        help="Use GPS prior instead of tracking prior")
    args = parser.parse_args()

    print(f"Loading HRNet-Offset + CUDA Graph...")
    kwargs = {"device": args.device, "enable_ekf": False}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    detector = RunwayInference(**kwargs)
    print(f"  in_channels: {detector.model.config.model.in_channels}")

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[args.video_key])
    all_frames = sorted(pose_dict.keys())
    frames = all_frames[::args.frame_stride]

    cap = cv2.VideoCapture(VIDEOS[args.video_key])

    use_tracking = not args.no_tracking
    prev_corners_orig = None
    paused = False
    frame_idx = 0
    errors = []
    detect_times = []

    print(f"\nVideo: {args.video_key} ({len(frames)} frames)")
    print(f"Tracking prior: {'ON' if use_tracking else 'OFF (GPS only)'}")
    print("Keys: q=quit, space=pause, t=toggle tracking, <- ->=step\n")

    cv2.namedWindow("Runway Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Runway Detection", 960, 540)

    # Pre-seek to first frame and warm up CUDA graph
    first_fi = frames[0] if frames else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_fi)
    ret, frame = cap.read()
    if ret and first_fi in pose_dict:
        pose = pose_dict[first_fi]
        prior = detector._get_prior_corners(pose)
        prior_det = prior[[0, 3, 1, 2]]
        img_t, _, _ = detector._preprocess(frame, prior_det)
        _ = detector._infer(img_t)
        torch.cuda.synchronize()

    while frame_idx < len(frames):
        fi = frames[frame_idx]

        if not paused:
            # Sequential read (avoid expensive cap.set for each frame)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            needed_skip = fi - current_pos
            if needed_skip < 0 or needed_skip > 60:
                # Large jump (user navigated) — must seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                needed_skip = 0
            for _ in range(needed_skip):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                frame_idx += 1
                continue

            pose = pose_dict.get(fi)
            if pose is None:
                frame_idx += 1
                continue

            # Determine crop prior
            if use_tracking and prev_corners_orig is not None:
                crop_prior = prev_corners_orig.copy()
            else:
                prior = detector._get_prior_corners(pose)
                if prior is None:
                    frame_idx += 1
                    continue
                crop_prior = prior.copy()

            # Run detection
            t0 = time.perf_counter()
            try:
                prior_det = crop_prior[[0, 3, 1, 2]]
                img_t, _, crop_info = detector._preprocess(frame, prior_det)
                outputs = detector._infer(img_t)
                corners_orig, _ = detector._postprocess(outputs, crop_info)
            except Exception as e:
                print(f"  Error frame {fi}: {e}")
                frame_idx += 1
                continue
            detect_ms = (time.perf_counter() - t0) * 1000

            prev_corners_orig = corners_orig.copy()
            pred_work = corners_orig * np.array([SX, SY])

            # GT
            gt_coords, gt_visible = gt_gen.generate_corner_gt(pose)

            # Compute errors
            frame_errs = []
            for i in range(4):
                if gt_visible[i]:
                    err = np.linalg.norm(pred_work[i] - gt_coords[i])
                    frame_errs.append(err)
                    errors.append(err)

            # ---- Draw at half resolution (4x faster) ----
            vis = cv2.resize(frame, (DW, DH), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(vis)

            if gt_visible.all():
                pts = np.int32(gt_coords * [DSX, DSY])
                cv2.fillPoly(overlay, [pts], (0, 180, 0))
            pts = np.int32(pred_work * [DSX, DSY])
            cv2.fillPoly(overlay, [pts], (0, 220, 220))
            vis = cv2.addWeighted(vis, 0.75, overlay, 0.30, 0)

            if gt_visible.all():
                pts = np.int32(gt_coords * [DSX, DSY])
                cv2.polylines(vis, [pts], True, (0, 255, 0), thickness=1)
            pts = np.int32(pred_work * [DSX, DSY])
            cv2.polylines(vis, [pts], True, (0, 255, 255), thickness=1)

            # FPS counter
            detect_times.append(detect_ms)
            if len(detect_times) > 20:
                detect_times.pop(0)
            avg_detect_ms = np.mean(detect_times)
            fps = 1000.0 / max(avg_detect_ms, 1.0)

            # Info bar
            BAR_H = 40
            bar = np.zeros((BAR_H, DW, 3), dtype=np.uint8)
            vis = np.vstack([vis, bar])

            info_lines = [
                f"Frame: {fi}/{all_frames[-1]}  Stride: {args.frame_stride}  "
                f"Prior: {'Track' if (use_tracking and frame_idx > 0) else 'GPS'}  "
                f"{'||' if paused else '>'}",
            ]
            if frame_errs:
                info_lines.append(
                    f"Error: mean={np.mean(frame_errs):.1f}  med={np.median(frame_errs):.1f}  "
                    f"max={np.max(frame_errs):.1f}px  |  Running: {np.mean(errors[-100:]):.1f}px"
                )

            for i, line in enumerate(info_lines):
                cv2.putText(vis, line, (8, DH + 16 + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 2)

            # Right side: GPU + FPS
            rx = DW - 380
            cv2.putText(vis, f"GPU: RTX 3090  |  FPS: {fps:.0f}  ({avg_detect_ms:.0f}ms)",
                       (rx, DH + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 2)
            cv2.putText(vis, "HRNet-w18 + CUDA Graph | 256x256 crop",
                       (rx, DH + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow("Runway Detection", vis)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('t'):
            use_tracking = not use_tracking
            prev_corners_orig = None
            print(f"  Tracking prior: {'ON' if use_tracking else 'OFF'}")
        elif key == 81:  # left arrow
            frame_idx = max(0, frame_idx - 1)
            prev_corners_orig = None
        elif key == 83:  # right arrow
            frame_idx += 1
            prev_corners_orig = None

        if not paused:
            frame_idx += 1

    cap.release()
    detector.reset()
    cv2.destroyAllWindows()

    if errors:
        e = np.array(errors)
        print(f"\nFinal: mean={e.mean():.2f} med={np.median(e):.2f} "
              f"PCK@3={(e<3).mean():.1%} PCK@5={(e<5).mean():.1%} px @ 512x288")


if __name__ == "__main__":
    main()
