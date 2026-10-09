"""
train.py
========
Training pipeline for the Inverse FX Chain Predictor (InverseFXNet).

Trains the network to predict:
  1. Effect Presence: multi-label binary cross-entropy across all 7 effects.
  2. Effect Parameters: masked mean squared error across the 29 continuous parameters.

Supports:
  - Automatic GPU (CUDA / Apple MPS / CPU) acceleration.
  - Checkpoint saving to `checkpoints/best_model.pth`.
  - Comprehensive train and validation loss tracking.

Usage:
------
python3 train.py --epochs 20 --batch_size 16
python3 train.py --train_metadata dataset/synthetic/train/metadata.jsonl --epochs 30
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.dapn_dataset import get_dataloader, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet


def compute_masked_param_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes MSE loss only on parameters corresponding to active effects."""
    eps = 1e-8
    diff_sq = (preds - targets).pow(2) * mask
    loss = diff_sq.sum() / (mask.sum() + eps)
    return loss


def compute_param_mae(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Computes Mean Absolute Error only on active parameters."""
    eps = 1e-8
    diff_abs = (preds - targets).abs() * mask
    mae = float((diff_abs.sum() / (mask.sum() + eps)).item())
    return mae


def main():
    parser = argparse.ArgumentParser(description="Train InverseFXNet on synthetic audio pairs")
    parser.add_argument("--train_metadata", type=str, default="dataset/synthetic/train/metadata.jsonl", help="Train metadata path")
    parser.add_argument("--val_metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Val metadata path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 68)
    print(" Inverse FX Chain Predictor: Neural Network Training")
    print("=" * 68)
    print(f"Device        : {device}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Learning Rate : {args.lr}")
    print(f"Train Dataset : {args.train_metadata}")
    print(f"Val Dataset   : {args.val_metadata}\n")

    # DataLoaders
    train_loader = get_dataloader(args.train_metadata, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.val_metadata, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_loader.dataset)} training samples ({len(train_loader)} batches)")
    print(f"Loaded {len(val_loader.dataset)} validation samples ({len(val_loader)} batches)\n")

    # Model, Optimizer, Scheduler
    model = InverseFXNet().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    criterion_bce = nn.BCEWithLogitsLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = save_dir / "best_model.pth"

    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Training Phase
        # -------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_presence_loss = 0.0
        train_param_loss = 0.0

        for batch in train_loader:
            dry = batch["dry_audio"].to(device)
            wet = batch["wet_audio"].to(device)
            presence_targets = batch["presence_multihot"].to(device)
            param_targets = batch["param_targets"].to(device)
            param_mask = batch["param_mask"].to(device)

            optimizer.zero_grad()

            logits, param_preds = model(dry, wet)

            loss_presence = criterion_bce(logits, presence_targets)
            loss_param = compute_masked_param_loss(param_preds, param_targets, param_mask)

            # Combined Multi-Task Loss
            loss = loss_presence + 2.0 * loss_param

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_presence_loss += loss_presence.item()
            train_param_loss += loss_param.item()

        n_train_batches = len(train_loader)
        avg_train_loss = train_loss / n_train_batches
        avg_train_pres = train_presence_loss / n_train_batches
        avg_train_param = train_param_loss / n_train_batches

        scheduler.step()

        # -------------------------------------------------------------
        # 2. Validation Phase
        # -------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_presence_loss = 0.0
        val_param_loss = 0.0
        total_mae = 0.0
        correct_presence_preds = 0
        total_presence_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                dry = batch["dry_audio"].to(device)
                wet = batch["wet_audio"].to(device)
                presence_targets = batch["presence_multihot"].to(device)
                param_targets = batch["param_targets"].to(device)
                param_mask = batch["param_mask"].to(device)

                logits, param_preds = model(dry, wet)

                loss_presence = criterion_bce(logits, presence_targets)
                loss_param = compute_masked_param_loss(param_preds, param_targets, param_mask)
                loss = loss_presence + 2.0 * loss_param

                val_loss += loss.item()
                val_presence_loss += loss_presence.item()
                val_param_loss += loss_param.item()

                # Accuracy and MAE metrics
                probs = torch.sigmoid(logits)
                preds_binary = (probs >= 0.5).float()
                correct_presence_preds += (preds_binary == presence_targets).sum().item()
                total_presence_preds += presence_targets.numel()

                total_mae += compute_param_mae(param_preds, param_targets, param_mask)

        n_val_batches = len(val_loader)
        avg_val_loss = val_loss / n_val_batches
        avg_val_mae = total_mae / n_val_batches
        val_accuracy = (correct_presence_preds / total_presence_preds) * 100.0
        epoch_time = time.perf_counter() - epoch_start

        # Checkpoint saving
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_mae": avg_val_mae,
                "supported_effects": SUPPORTED_EFFECTS,
            }, best_checkpoint_path)

        best_mark = " ★ Best" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train: {avg_train_loss:.4f} (Pres: {avg_train_pres:.3f}, Param: {avg_train_param:.3f}) | "
            f"Val: {avg_val_loss:.4f} | "
            f"Effect Acc: {val_accuracy:.1f}% | "
            f"Param MAE: {avg_val_mae:.4f}{best_mark}"
        )

    total_time = time.perf_counter() - total_start
    print("=" * 68)
    print(f"Training Complete in {total_time/60.0:.2f} minutes!")
    print(f"Best Model Saved to: {best_checkpoint_path.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
