#!/home/hehe/miniconda3/envs/toolbox/bin/python
"""Train ScanlineEdgeNet for per-row edge x-coordinate regression.

Predicts x_left[y], x_right[y] for each image row from a 256x256 crop.
SmoothL1 loss on valid rows + 2nd-order smoothness regularization.

Usage:
    python train_scanline_net.py --epochs 50 --batch_size 16
"""

import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from runway_detector.config import VIDEOS, POSES
from runway_detector.data.ground_truth import GroundTruthGenerator, load_poses_dict
from runway_detector.data.scanline_dataset import ScanlineDataset
from runway_detector.models.scanline_net import ScanlineEdgeNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoothness_loss(x_pred: torch.Tensor) -> torch.Tensor:
    """2nd-order smoothness: penalize large 2nd derivatives along height."""
    # x_pred: (B, 2, H)
    d2 = x_pred[:, :, 2:] - 2 * x_pred[:, :, 1:-1] + x_pred[:, :, :-2]
    return d2.abs().mean()


def fit_line_params(
    x_pred: torch.Tensor,
    valid: torch.Tensor,
    y: torch.Tensor,
) -> tuple:
    """Differentiable weighted least-squares line fitting: x = k*y + b.

    Args:
        x_pred: (B, H) predicted x in [-1, 1]
        valid: (B, H) binary validity mask
        y: (H,) y coordinates in [-1, 1]

    Returns:
        k: (B,) slope,  b: (B,) intercept
    """
    B = x_pred.shape[0]
    w = valid.float()
    y = y.unsqueeze(0)  # (1, H)

    # Normal equations for min Σ w_i*(k*y_i + b - x_i)²
    # [Σwy²  Σwy ] [k] = [Σwyx]
    # [Σwy   Σw  ] [b]   [Σwx ]
    wy2 = (w * y * y).sum(dim=1)  # (B,)
    wy = (w * y).sum(dim=1)
    sw = w.sum(dim=1)
    wyx = (w * y * x_pred).sum(dim=1)
    wx = (w * x_pred).sum(dim=1)

    det = wy2 * sw - wy * wy
    det = torch.where(det.abs() < 1e-8,
                      torch.ones_like(det) * 1e-8, det)

    k = (wyx * sw - wx * wy) / det
    b = (wy2 * wx - wy * wyx) / det
    return k, b


def vanishing_point_loss(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    valid_left: torch.Tensor,
    valid_right: torch.Tensor,
    crop_size: int = 256,
) -> torch.Tensor:
    """Penalize deviation from common vanishing point.

    Fits lines to left/right/centerline predictions, computes VP from
    left-right intersection, and penalizes centerline's distance to VP.
    """
    B, H = x_left.shape
    y = torch.linspace(-1, 1, H, device=x_left.device)

    k_L, b_L = fit_line_params(x_left, valid_left, y)
    k_R, b_R = fit_line_params(x_right, valid_right, y)

    x_mid = (x_left + x_right) / 2.0
    valid_mid = valid_left * valid_right
    k_C, b_C = fit_line_params(x_mid, valid_mid, y)

    # VP from left-right intersection: k_L*vy + b_L = k_R*vy + b_R
    denom = k_L - k_R
    denom = torch.where(denom.abs() < 1e-8,
                        torch.ones_like(denom) * 1e-8, denom)
    vy = (b_R - b_L) / denom
    vx = k_L * vy + b_L

    # Signed distance from centerline to VP: |vx - k_C*vy - b_C| / sqrt(1+k_C²)
    dist = (vx - k_C * vy - b_C).abs() / torch.sqrt(1.0 + k_C * k_C)

    return dist.mean()


def main():
    parser = argparse.ArgumentParser(description="Train ScanlineEdgeNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_decoder", type=float, default=3e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_weight", type=float, default=0.1)
    parser.add_argument("--vp_weight", type=float, default=0.0,
                        help="Vanishing point consistency loss weight. "
                             "0=disabled. Try 0.05 to start.")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--crop_noise", type=float, default=15.0,
                        help="Crop center noise std (px). 0=no noise. "
                             "15px ~ 10m GPS error, builds PnP robustness.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gt_gen = GroundTruthGenerator()
    pose_dicts = {
        "video1": load_poses_dict(POSES["video1"]),
        "video2": load_poses_dict(POSES["video2"]),
        "video3": load_poses_dict(POSES["video3"]),
    }

    # Train: all frames from video1, video2, video3 (pre-split)
    print("\nBuilding datasets...")
    train_v3_frames = [f for f in sorted(pose_dicts["video3"].keys()) if f < 1250]
    val_frames = [f for f in sorted(pose_dicts["video3"].keys()) if f >= 1250]

    train_samples = []
    for vk in ["video1", "video2"]:
        cap = cv2.VideoCapture(VIDEOS[vk])
        pose_dict = pose_dicts[vk]
        for fi in tqdm(sorted(pose_dict.keys())[::2], desc=f"Loading {vk}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                train_samples.append((frame, pose_dict[fi]))
        cap.release()

    cap = cv2.VideoCapture(VIDEOS["video3"])
    for fi in tqdm(train_v3_frames[::2], desc="Loading video3 (train)"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            train_samples.append((frame, pose_dicts["video3"][fi]))
    cap.release()
    print(f"  Train samples: {len(train_samples)}")

    train_ds = ScanlineDataset(train_samples, gt_gen,
                               crop_size=args.crop_size, augment=True,
                               crop_noise_std=args.crop_noise)

    val_samples = []
    cap = cv2.VideoCapture(VIDEOS["video3"])
    for fi in tqdm(val_frames[::5], desc="Loading val"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            val_samples.append((frame, pose_dicts["video3"][fi]))
    cap.release()
    val_ds = ScanlineDataset(val_samples, gt_gen,
                             crop_size=args.crop_size, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    print("\nBuilding ScanlineEdgeNet...")
    model = ScanlineEdgeNet(crop_size=args.crop_size, freeze_backbone=False)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    backbone_p = sum(p.numel() for p in model.encoder.parameters())
    print(f"Params: {total:,} total ({backbone_p:,} encoder + {total-backbone_p:,} decoder)")

    # Differential LR
    backbone_ids = set(id(p) for p in model.encoder.parameters())
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    decoder_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": decoder_params, "lr": args.lr_decoder},
    ], weight_decay=1e-4)

    def warmup_schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1.0 + np.cos(
            np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_schedule)
    loss_fn = nn.SmoothL1Loss(reduction='none')

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_smooth = 0.0
        epoch_vp = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            x_left_gt = batch["x_left"].to(device, non_blocking=True)
            x_right_gt = batch["x_right"].to(device, non_blocking=True)
            valid_left = batch["valid_left"].to(device, non_blocking=True)
            valid_right = batch["valid_right"].to(device, non_blocking=True)

            pred = model(image)  # (B, 2, H)

            # Per-row L1 loss (masked by validity)
            l1_left = loss_fn(pred[:, 0, :], x_left_gt)
            l1_right = loss_fn(pred[:, 1, :], x_right_gt)
            l1 = (l1_left * valid_left).sum() / (valid_left.sum() + 1e-8) + \
                 (l1_right * valid_right).sum() / (valid_right.sum() + 1e-8)

            smooth = smoothness_loss(pred) * args.smooth_weight
            vp = vanishing_point_loss(pred[:, 0, :], pred[:, 1, :],
                                      valid_left, valid_right) * args.vp_weight
            loss = l1 + smooth + vp

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_smooth += smooth.item()
            epoch_vp += vp.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        n = len(train_dl)
        print(f"Epoch {epoch+1}: loss={epoch_loss/n:.4f} "
              f"(L1={epoch_l1/n:.4f} smooth={epoch_smooth/n:.4f} "
              f"vp={epoch_vp/n:.4f}) "
              f"lr={optimizer.param_groups[1]['lr']:.2e}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                image = batch["image"].to(device)
                x_left_gt = batch["x_left"].to(device)
                x_right_gt = batch["x_right"].to(device)
                valid_left = batch["valid_left"].to(device)
                valid_right = batch["valid_right"].to(device)

                pred = model(image)
                l1_left = loss_fn(pred[:, 0, :], x_left_gt)
                l1_right = loss_fn(pred[:, 1, :], x_right_gt)
                vl = (l1_left * valid_left).sum() / (valid_left.sum() + 1e-8) + \
                     (l1_right * valid_right).sum() / (valid_right.sum() + 1e-8)
                val_loss += vl.item()

        val_loss /= len(val_dl)
        print(f"  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(args.save_dir, "scanline_net_best.pt"))
            print(f"  Saved best (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
