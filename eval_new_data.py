"""Batch evaluation on new datasets from hdd2_disk.

Usage:
    python eval_new_data.py --data_dir /home/hehe/hdd2_disk --frame_stride 10 --max_frames 200
"""

import sys, os, argparse, glob, numpy as np, cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.inference import RunwayInference


def evaluate_video(video_path, pose_path, detector, gt_gen, args):
    pose_dict = load_poses_dict(pose_path)
    frames = sorted(pose_dict.keys())
    frames = frames[::args.frame_stride]
    if args.max_frames:
        frames = frames[:args.max_frames]

    cap = cv2.VideoCapture(video_path)
    corner_errors = []
    per_corner_errors = [[] for _ in range(4)]
    skipped = 0

    for fi in tqdm(frames, desc=f"  Frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            skipped += 1
            continue

        pose = pose_dict[fi]
        gt_coords, gt_visible = gt_gen.generate_corner_gt(pose)

        # Only evaluate when all 4 corners are visible (far-field)
        if not gt_visible.all():
            skipped += 1
            continue

        try:
            result = detector(frame, pose=pose)
        except Exception:
            skipped += 1
            continue

        pred = result["corners"]
        for i in range(4):
            err = np.linalg.norm(pred[i] - gt_coords[i])
            corner_errors.append(err)
            per_corner_errors[i].append(err)

    cap.release()
    return corner_errors, per_corner_errors, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/hehe/hdd2_disk")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_ekf", action="store_true")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Only evaluate first N videos")
    args = parser.parse_args()

    # Find video/pose pairs
    videos = sorted(glob.glob(os.path.join(args.data_dir, "*.avi")))
    pairs = []
    for v in videos:
        txt = v.replace(".avi", ".txt")
        if os.path.exists(txt):
            pairs.append((v, txt))

    if args.top_n:
        pairs = pairs[:args.top_n]

    print(f"Found {len(pairs)} video-pose pairs")
    print(f"Frame stride: {args.frame_stride}, Max frames/video: {args.max_frames or 'all'}\n")

    # Load detector once
    print("Loading HRNet-Offset detector...")
    kwargs = {"device": args.device, "enable_ekf": not args.no_ekf}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    detector = RunwayInference(**kwargs)
    gt_gen = GroundTruthGenerator()

    all_errors = []
    results = []

    for vi, (video_path, pose_path) in enumerate(pairs):
        name = os.path.basename(video_path).replace(".avi", "")
        print(f"\n[{vi+1}/{len(pairs)}] {name}")

        corner_errors, per_corner, skipped = evaluate_video(
            video_path, pose_path, detector, gt_gen, args)

        if corner_errors:
            ce = np.array(corner_errors)
            all_errors.extend(corner_errors)
            results.append({
                "name": name,
                "frames": len(corner_errors) // 4,
                "mean": ce.mean(),
                "median": np.median(ce),
                "std": ce.std(),
                "pck1": (ce < 1).mean(),
                "pck3": (ce < 3).mean(),
                "pck5": (ce < 5).mean(),
                "per_corner_mean": [np.mean(pc) if pc else 0 for pc in per_corner],
            })
            print(f"  Frames: {len(corner_errors)//4}, Mean: {ce.mean():.2f}px, "
                  f"Median: {np.median(ce):.2f}px, PCK@3: {(ce<3).mean():.1%}")
        else:
            print(f"  No valid results (all frames skipped)")

    detector.reset()

    # Summary
    print("\n" + "=" * 65)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 65)

    if results:
        means = [r["mean"] for r in results]
        print(f"\nPer-video mean errors:")
        for r in results:
            bar = "█" * int(r["mean"] * 10) if r["mean"] < 10 else "!!!"
            print(f"  {r['name'][:50]:50s}  mean={r['mean']:.2f}  "
                  f"PCK@3={r['pck3']:.1%}  {bar}")

        total = np.array(all_errors)
        print(f"\n{'─'*65}")
        print(f"OVERALL ({len(results)} videos, {len(all_errors)//4} frames, "
              f"{len(all_errors)} corners):")
        print(f"  Mean error:   {total.mean():.2f} px @ 512x288 "
              f"({total.mean()*1920/512:.1f} px @ 1920x1080)")
        print(f"  Median error: {np.median(total):.2f} px")
        print(f"  Std error:    {total.std():.2f} px")
        print(f"  Max error:    {total.max():.2f} px")
        for thresh in [1, 2, 3, 5, 10]:
            print(f"  PCK@{thresh}px:     {(total < thresh).mean():.1%}")

        names = ['BL', 'TL', 'TR', 'BR']
        print(f"\n  Per-corner:")
        for i, name in enumerate(names):
            pc = [e for r in results for e in ([r["per_corner_mean"][i]] * r["frames"])]
            if pc:
                print(f"    {name}: mean={np.mean(pc):.2f}")

        # Worst videos
        print(f"\n  Best 3:  ", ", ".join(r['name'][:40] for r in sorted(results, key=lambda x: x['mean'])[:3]))
        print(f"  Worst 3: ", ", ".join(r['name'][:40] for r in sorted(results, key=lambda x: x['mean'])[-3:]))
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
