"""Real-time runway detection visualization.

Draws predicted and GT runway regions as filled quadrilaterals.
Press 'q' to quit, ' ' (space) to pause/resume, 't' to toggle tracking prior.
"""

import sys, os, argparse, time, numpy as np, cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES, ORIGINAL_SIZE, FAR_FIELD_ALTITUDE_THRESHOLD
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference
from runway_detector.models.scanline_net import predict_to_lines

W_orig, H_orig = ORIGINAL_SIZE
SX = 512.0 / W_orig
SY = 288.0 / H_orig

# Display at full resolution for detail visibility
DW, DH = W_orig, H_orig  # 1920×1080
DSX, DSY = DW / 512.0, DH / 288.0


def main():
    parser = argparse.ArgumentParser(description="Live runway detection")
    parser.add_argument("--video_key", type=str, default="video3")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--no_tracking", action="store_true",
                        help="Use GPS prior instead of tracking prior")
    parser.add_argument("--view", type=str, default="both",
                        choices=["corners", "edges", "both"],
                        help="What to display: corners, edges, or both (default)")
    parser.add_argument("--video", type=str, default=None,
                        help="Custom video path (overrides --video_key)")
    parser.add_argument("--no_edges", action="store_true",
                        help="Disable scanline edge detection (faster)")
    parser.add_argument("--poses", type=str, default=None,
                        help="Custom pose file path (overrides --video_key)")
    args = parser.parse_args()

    # Resolve video/pose paths
    if args.video and args.poses:
        video_path = args.video
        pose_path = args.poses
    elif args.video or args.poses:
        print("Error: must specify both --video and --poses together")
        return
    else:
        video_path = VIDEOS[args.video_key]
        pose_path = POSES[args.video_key]

    print(f"Loading HRNet-Offset + CUDA Graph...")
    kwargs = {"device": args.device, "enable_ekf": False}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    detector = RunwayInference(**kwargs)
    print(f"  model: HRNet-w18-small (3ch RGB input)")

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(pose_path)
    all_frames = sorted(pose_dict.keys())
    frames = all_frames[::args.frame_stride]

    cap = cv2.VideoCapture(video_path)

    use_tracking = not args.no_tracking
    view_mode = args.view
    prev_corners_orig = None
    paused = False
    frame_idx = 0
    errors = []
    detect_times = []

    print(f"\nVideo: {args.video_key} ({len(frames)} frames)")
    print(f"Tracking prior: {'ON' if use_tracking else 'OFF (GPS only)'}")
    print(f"View mode: {view_mode} (keys: c=corners, e=edges, b=both)")
    print("Keys: q=quit, space=pause, t=toggle tracking, <- ->=step\n")

    cv2.namedWindow("Runway Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Runway Detection", DW, DH)

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
            # Sequential read (avoid expensive cap.set for each frame).
            # AVI seek takes ~265ms, read ~21ms/frame → breakeven at ~13 frames.
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            needed_skip = fi - current_pos
            if needed_skip < 0 or needed_skip > 15:
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

                # Scanline edge detector
                agl = pose.get('altitude', 999) - 27.5
                detector_edge_lines = None
                detector_centerline = None
                if not args.no_edges and detector.scanline_net is not None:
                    with torch.no_grad():
                        output = detector.scanline_net(img_t)
                        output_np = output[0].cpu().numpy()
                    fitted = predict_to_lines(output_np, crop_size=256)
                    # Convert lines from crop to original image coords
                    cx, cy, half = crop_info["cx"], crop_info["cy"], crop_info["half"]
                    scale = 256.0 / (2.0 * half)
                    def _crop_line_to_orig(line):
                        if line is None:
                            return None
                        a, b, c = line
                        offset_x = cx - half
                        offset_y = cy - half
                        c_orig = c / scale - a * offset_x - b * offset_y
                        n = np.sqrt(a*a + b*b)
                        return (a/n, b/n, c_orig/n)
                    detector_edge_lines = {
                        "left": _crop_line_to_orig(fitted.get("left")),
                        "right": _crop_line_to_orig(fitted.get("right")),
                    }
                    detector_centerline = _crop_line_to_orig(fitted.get("centerline"))

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

            # GT runway region: green fill + outline
            if gt_visible.all():
                pts_gt = np.int32(gt_coords * [DSX, DSY])
                cv2.fillPoly(overlay, [pts_gt], (0, 160, 0))
                cv2.polylines(vis, [pts_gt], True, (0, 255, 0), thickness=2)

            show_corners = view_mode in ("corners", "both")
            show_edges = view_mode in ("edges", "both")

            # Predicted runway region: cyan fill + yellow outline
            pts_pred = np.int32(pred_work * [DSX, DSY])
            if show_corners:
                cv2.fillPoly(overlay, [pts_pred], (0, 220, 220))
                cv2.polylines(vis, [pts_pred], True, (0, 255, 255), thickness=2)

            vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)

            # Compute vanishing point from left/right edge intersection
            # Line through BL(0)→TL(1) and BR(3)→TR(2)
            def line_through(p, q):
                """Return (a, b, c) for line a*x + b*y + c = 0 through two points."""
                dx = q[0] - p[0]
                dy = q[1] - p[1]
                a, b = -dy, dx
                n = np.sqrt(a*a + b*b)
                return (a/n, b/n, -(a*p[0] + b*p[1])/n)

            def intersect(l1, l2):
                """Intersection of two lines (a,b,c). Returns (x,y) or None if parallel."""
                a1, b1, c1 = l1
                a2, b2, c2 = l2
                det = a1*b2 - a2*b1
                if abs(det) < 1e-8:
                    return None
                return np.array([(b1*c2 - b2*c1)/det, (c1*a2 - c2*a1)/det])

            def clip_line_to_image(l, y_start, y_end):
                """Given line (a,b,c) and y range, return (p0, p1) segment endpoints."""
                a, b, c = l
                pts = []
                for y in [y_start, y_end]:
                    if abs(a) > 1e-8:
                        x = -(b*y + c) / a
                        pts.append((int(x), int(y)))
                return pts[0], pts[1] if len(pts) == 2 else (None, None)

            # Compute vanishing point for edge line clipping
            left_line = line_through(pts_pred[0].astype(float), pts_pred[1].astype(float))
            right_line = line_through(pts_pred[3].astype(float), pts_pred[2].astype(float))
            vp = intersect(left_line, right_line)

            # ---- Scanline edge detector lines (thick solid) ----
            if show_edges and (detector_edge_lines or detector_centerline):
                top_y = max(0.0, float(vp[1])) if (vp is not None and vp[1] > 0) else 0.0
                bot_y = float(DH - 1)
                for line, color in [
                    (detector_edge_lines.get("left") if detector_edge_lines else None, (0, 200, 255)),
                    (detector_edge_lines.get("right") if detector_edge_lines else None, (0, 200, 255)),
                    (detector_centerline, (0, 255, 200)),
                ]:
                    if line is not None:
                        p0, p1 = clip_line_to_image(line, top_y, bot_y)
                        if p0 is not None and p1 is not None:
                            cv2.line(vis, p0, p1, color, thickness=2)

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

            # AGL and edge detector status
            agl_str = f"AGL={agl:.0f}m" if agl < 999 else ""
            view_str = f"View: {view_mode}"
            mode_str = " | ".join(filter(None, [agl_str, view_str]))

            info_lines = [
                f"Frame: {fi}/{all_frames[-1]}  Stride: {args.frame_stride}  "
                f"Prior: {'Track' if (use_tracking and frame_idx > 0) else 'GPS'}  "
                f"{'||' if paused else '>'}  |  {mode_str}",
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
        elif key == ord('c'):
            view_mode = "corners"
            print(f"  View: corners only")
        elif key == ord('e'):
            view_mode = "edges"
            print(f"  View: edges only")
        elif key == ord('b'):
            view_mode = "both"
            print(f"  View: both")
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
