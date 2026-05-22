"""Training script for Runway Detection System.

Stages:
    1: Corner detection only (heatmap + coord loss)
    2: Add edge + centerline heads (heatmap losses)
    3: Add geometric consistency losses
    4: Add temporal EKF + prior feedback

Usage:
    python -m runway_detector.train --stage 1 --epochs 100
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runway_detector.config import (
    CHECKPOINT_DIR, LOG_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, WORKING_SIZE, NUM_CORNERS, LOSS_WEIGHTS,
)
from runway_detector.data.dataset import create_dataloaders
from runway_detector.models.runway_net import RunwayNet
from runway_detector.losses.heatmap_loss import heatmap_mse_loss
from runway_detector.losses.coord_loss import coord_l1_loss
from runway_detector.losses.geometric_loss import (
    vanishing_point_loss, corner_line_consistency_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RunwayNet")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage (1-4)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N batches")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, stage, device, epoch, writer, global_step):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)  # (B, 3, H, W)
        corner_gt = batch['corner_heatmaps'].to(device)  # (B, 4, H, W)
        corner_coords_gt = batch['corner_coords'].to(device)  # (B, 4, 2)
        corner_visible = batch['corner_visible'].to(device)  # (B, 4)

        # Stage 3+: concatenate prior channels to input
        if stage >= 3:
            prior = batch['prior'].to(device)  # (B, 7, H, W)
            images = torch.cat([images, prior], dim=1)  # (B, 10, H, W)

        # Forward
        outputs = model(images)
        corner_outputs = outputs['corners']
        pred_heatmaps = corner_outputs['heatmaps']  # (B, 4, H, W)
        pred_coords = corner_outputs['coords']      # (B, 4, 2)

        # --- Stage 1 losses ---
        loss_hm = heatmap_mse_loss(pred_heatmaps, corner_gt, corner_visible)
        loss_coord = coord_l1_loss(pred_coords, corner_coords_gt, corner_visible)
        loss = (LOSS_WEIGHTS['heatmap'] * loss_hm +
                LOSS_WEIGHTS['coord'] * loss_coord)

        # --- Stage 2+ losses ---
        if stage >= 2:
            edge_gt = batch['edge_heatmaps'].to(device)
            cl_gt = batch['centerline_heatmap'].to(device)
            pred_edges = outputs['edges']
            pred_cl = outputs['centerline']

            loss_edge = heatmap_mse_loss(pred_edges, edge_gt)
            loss_cl = heatmap_mse_loss(pred_cl, cl_gt)
            loss = loss + (LOSS_WEIGHTS['edge'] * loss_edge +
                          LOSS_WEIGHTS['centerline'] * loss_cl)

            loss_components['loss_edge'] = loss_edge.item()
            loss_components['loss_cl'] = loss_cl.item()

        # --- Stage 3+ geometric losses ---
        if stage >= 3:
            loss_vp = vanishing_point_loss(pred_edges, pred_cl)
            loss_corner_line = corner_line_consistency_loss(
                pred_coords, pred_edges, corner_visible
            )
            loss_geom = loss_vp + loss_corner_line
            loss = loss + LOSS_WEIGHTS['geom'] * loss_geom

            loss_components['loss_vp'] = loss_vp.item()
            loss_components['loss_corner_line'] = loss_corner_line.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        loss_components['loss_hm'] = loss_hm.item()
        loss_components['loss_coord'] = loss_coord.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'coord': f"{loss_coord.item():.2f}px",
        })

        # TensorBoard logging
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('train/loss_total', loss.item(), global_step)
            writer.add_scalar('train/loss_heatmap', loss_hm.item(), global_step)
            writer.add_scalar('train/loss_coord', loss_coord.item(), global_step)
        global_step += 1

    avg_loss = total_loss / len(loader)
    return avg_loss, loss_components, global_step


@torch.no_grad()
def validate(model, loader, stage, device):
    """Validate on validation set."""
    model.eval()
    total_coord_error = 0.0
    total_corners = 0

    for batch in tqdm(loader, desc="Validating"):
        images = batch['image'].to(device)
        corner_coords_gt = batch['corner_coords'].to(device)
        corner_visible = batch['corner_visible'].to(device)

        if stage >= 3:
            prior = batch['prior'].to(device)
            images = torch.cat([images, prior], dim=1)

        outputs = model(images)
        pred_coords = outputs['corners']['coords']

        # Per-corner pixel error
        diff = (pred_coords - corner_coords_gt).norm(dim=-1)  # (B, 4)
        diff = diff * corner_visible.float()
        total_coord_error += diff.sum().item()
        total_corners += corner_visible.float().sum().item()

    mean_error = total_coord_error / max(total_corners, 1)
    return mean_error


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Stage: {args.stage}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        stage=args.stage,
        include_prior=(args.stage >= 3),
        num_workers=args.num_workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    in_channels = 10 if args.stage >= 3 else 3
    model = RunwayNet(
        in_channels=in_channels,
        stage=args.stage,
        pretrained=True,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Resume
    start_epoch = 0
    ckpt_stage = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_stage = ckpt.get('stage', 1)
        # Allow cross-stage loading: strict=False to skip new heads
        if ckpt_stage != args.stage:
            # Handle channel count mismatch (e.g. 3ch stage2 → 10ch stage3)
            ckpt_state = ckpt['model_state_dict']
            model_state = model.state_dict()
            # Copy first 3 channels of conv1 weight if going from 3ch to 10ch
            for key in ckpt_state:
                if key in model_state and ckpt_state[key].shape != model_state[key].shape:
                    if 'conv1.weight' in key or 'conv1' in key:
                        c_in = min(ckpt_state[key].shape[1], model_state[key].shape[1])
                        model_state[key][:, :c_in] = ckpt_state[key][:, :c_in]
                        ckpt_state[key] = model_state[key]
            missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
            head_keys = [k for k in missing if 'edge' in k.lower()
                         or 'centerline' in k.lower()]
            if head_keys:
                print(f"Stage {args.stage}: new heads initialized randomly "
                      f"({len(head_keys)} params from stage {ckpt_stage} ckpt)")
            # Reset optimizer for new stage
            print(f"Cross-stage: resetting optimizer for stage {args.stage}")
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {ckpt['epoch']}")

    # Logger
    run_name = f"stage{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    else:
        writer = None
    global_step = 0

    best_val_error = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        avg_loss, loss_components, global_step = train_epoch(
            model, train_loader, optimizer, args.stage, device,
            epoch, writer, global_step,
        )

        # Log
        print(f"\nEpoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Validate
        if epoch % args.val_interval == 0:
            val_error = validate(model, val_loader, args.stage, device)
            print(f"  Val  | Mean corner error: {val_error:.2f} px")
            if writer is not None:
                writer.add_scalar('val/corner_error_px', val_error, epoch)

            # Save best
            if val_error < best_val_error:
                best_val_error = val_error
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_stage{args.stage}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_error': val_error,
                    'stage': args.stage,
                }, ckpt_path)
                print(f"  Saved best model: {ckpt_path}")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"stage{args.stage}_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': best_val_error if best_val_error != float('inf') else None,
                'stage': args.stage,
            }, ckpt_path)

        scheduler.step()

    if writer is not None:
        writer.close()
    print(f"\nTraining complete. Best val error: {best_val_error:.2f} px")


if __name__ == "__main__":
    main()
