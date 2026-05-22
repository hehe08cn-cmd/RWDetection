"""Interactive tool to measure PnP projection accuracy vs manual annotation.

Usage:
    python annotate_corners.py --video_key video3
    - Click on each runway corner in order: BL, TL, TR, BR
    - Press 'n' for next frame, 'q' to quit
    - Outputs per-frame offset between PnP projection and manual click
"""

import sys, os, argparse, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict

CLICK_POINTS = []
CLICK_ORDER = ['BL (bottom-left)', 'TL (top-left)', 'TR (top-right)', 'BR (bottom-right)']


def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((x, y))
        print(f"  Clicked {CLICK_ORDER[len(CLICK_POINTS)-1]}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_key", type=str, default="video3")
    args = parser.parse_args()

    gt_gen = GroundTruthGenerator()
    pose_dict = load_poses_dict(POSES[args.video_key])
    frames = sorted(pose_dict.keys())[::30]  # every 30 frames (~1 per second)

    cap = cv2.VideoCapture(VIDEOS[args.video_key])

    cv2.namedWindow("Annotation")
    cv2.setMouseCallback("Annotation", click_callback)

    results = []
    idx = 0

    print("Click corners in order: BL(bottom-left), TL(top-left), TR(top-right), BR(bottom-right)")
    print("Press 'n'=next frame, 's'=skip, 'q'=quit\n")

    while idx < len(frames):
        fi = frames[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            idx += 1
            continue

        gt_coords, gt_visible = gt_gen.generate_corner_gt(pose_dict[fi])
        CLICK_POINTS.clear()

        vis = frame.copy()
        # Draw PnP projections
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        names = ['BL', 'TL', 'TR', 'BR']
        for i in range(4):
            if gt_visible[i]:
                px = int(gt_coords[i, 0] / 512.0 * 1920)
                py = int(gt_coords[i, 1] / 288.0 * 1080)
                cv2.circle(vis, (px, py), 6, colors[i], 2)
                cv2.line(vis, (px-12, py), (px+12, py), colors[i], 1)
                cv2.line(vis, (px, py-12), (px, py+12), colors[i], 1)
                cv2.putText(vis, f"{names[i]}(PnP)", (px+8, py-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

        cv2.putText(vis, f"Frame {fi} ({idx+1}/{len(frames)})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis, "Click 4 corners, then n/s/q", (10, 1060),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Annotation", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            idx += 1
            continue
        elif key == ord('n'):
            if len(CLICK_POINTS) == 4:
                offsets = []
                for i in range(4):
                    if gt_visible[i]:
                        gt_px = (gt_coords[i, 0] / 512.0 * 1920,
                                 gt_coords[i, 1] / 288.0 * 1080)
                        off = np.linalg.norm(np.array(CLICK_POINTS[i]) - np.array(gt_px))
                        offsets.append(off)
                results.append((fi, offsets))
                print(f"  Frame {fi}: offsets = {[f'{o:.1f}' for o in offsets]}px, "
                      f"mean={np.mean(offsets):.1f}px")
            idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if results:
        all_offsets = np.concatenate([r[1] for r in results])
        print(f"\n{'='*50}")
        print(f"PnP PROJECTION ACCURACY (vs manual annotation)")
        print(f"{'='*50}")
        print(f"Frames annotated: {len(results)}")
        print(f"Mean offset: {all_offsets.mean():.1f}px @ 1920x1080")
        print(f"Median offset: {np.median(all_offsets):.1f}px")
        print(f"Min/Max: {all_offsets.min():.1f} / {all_offsets.max():.1f}px")
        print(f"≈ {all_offsets.mean()*512/1920:.1f}px @ 512x288 working resolution")
    else:
        print("No annotations recorded.")


if __name__ == "__main__":
    main()
