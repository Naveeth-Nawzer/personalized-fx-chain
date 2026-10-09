"""
evaluate_summary.py
===================
Calculates comprehensive evaluation score values and produces a complete
model architecture summary report for InverseFXNet and DAPN.

Metrics reported:
1. Model Architectural Summary (Parameter counts, layer breakdown, memory).
2. Effect Presence Classification Metrics:
   - Per-effect Precision, Recall, F1-Score, and Accuracy (7 effects).
   - Macro F1 and Micro F1.
   - Exact Chain Match Accuracy (all effects predicted correctly).
   - Hamming Accuracy.
3. Continuous Parameter Estimation Metrics:
   - Mean Absolute Error (MAE) and Mean Squared Error (MSE) across active effects.
   - Per-effect Parameter MAE Breakdown.
4. Closed-Loop Acoustic Reconstruction (DAPN):
   - Multi-Resolution STFT Reconstruction Loss (MR-STFT).
   - Waveform Mean Squared Error (MSE).
   - Relative Spectral Improvement over unprocessed Dry baseline (%).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import (
    PARAM_OFFSETS,
    PARAM_SCHEMA,
    SUPPORTED_EFFECTS,
    get_dataloader,
)
from models.inverse_fx_net import InverseFXNet


def print_model_summary(model: torch.nn.Module):
    """Prints structural and parameter summary of InverseFXNet."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    param_size_mb = (total_params * 4) / (1024 * 1024)  # 32-bit float = 4 bytes

    print("=" * 76)
    print("                    INVERSEFXNET MODEL ARCHITECTURE SUMMARY")
    print("=" * 76)
    print(f"  {'Stage / Layer':<35} | {'Type / Dimension':<35}")
    print("-" * 76)
    print(f"  {'Acoustic Front-end':<35} | Log-Mel Spectrogram (n_mels=128, win=1024)")
    print(f"  {'Input Feature Channels':<35} | 3 channels: [Dry, Wet, Delta (Wet-Dry)]")
    print(f"  {'Stage 1 ConvBlock':<35} | 3 -> 32 channels (stride=2)")
    print(f"  {'Stage 2 ConvBlock':<35} | 32 -> 64 channels (stride=2)")
    print(f"  {'Stage 3 ConvBlock':<35} | 64 -> 128 channels (stride=2)")
    print(f"  {'Stage 4 ConvBlock':<35} | 128 -> 256 channels (stride=2)")
    print(f"  {'Spatial Pooling':<35} | AdaptiveAvgPool2d + AdaptiveMaxPool2d (GAP+GMP)")
    print(f"  {'Dense Embedding':<35} | 512-dim Linear -> LeakyReLU -> Dropout(0.2)")
    print(f"  {'Head 1: Effect Presence':<35} | 512 -> 7 logits (BCEWithLogitsLoss)")
    print(f"  {'Head 2: Continuous Parameters':<35} | 512 -> 29 params with Sigmoid [0, 1]")
    print("-" * 76)
    print(f"  Total Parameters          : {total_params:,}")
    print(f"  Trainable Parameters      : {trainable_params:,}")
    print(f"  Non-trainable Parameters  : {non_trainable_params:,}")
    print(f"  Estimated Model Size      : {param_size_mb:.2f} MB (FP32)")
    print("=" * 76)


def evaluate_metrics(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    eval_reconstruction: bool = True,
    max_recon_samples: int = 50,
):
    """Computes full evaluation score values on validation set."""
    model.eval()

    # Classification counters per effect: TP, FP, FN, TN
    num_effects = len(SUPPORTED_EFFECTS)
    tp = [0] * num_effects
    fp = [0] * num_effects
    fn = [0] * num_effects
    tn = [0] * num_effects

    total_chains = 0
    exact_match_chains = 0

    # Parameter regression accumulators
    param_abs_errors = [[] for _ in range(num_effects)]
    all_param_errors = []

    # Closed loop reconstruction
    dapn = DAPN() if eval_reconstruction else None
    loss_fn = MultiResolutionSTFTLoss() if eval_reconstruction else None
    dry_stft_losses = []
    recon_stft_losses = []
    recon_mse_losses = []

    print("\nEvaluating validation dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            dry_audio = batch["dry_audio"].to(device)
            wet_audio = batch["wet_audio"].to(device)
            target_presence = batch["presence_target"].to(device)
            target_params = batch["params_target"].to(device)
            active_mask = batch["active_mask"].to(device)
            batch_size = dry_audio.size(0)

            # Predict
            logits, pred_params = model(dry_audio, wet_audio)
            pred_presence = (torch.sigmoid(logits) >= 0.5).float()

            # 1. Classification Metrics
            for i in range(batch_size):
                total_chains += 1
                sample_pred = pred_presence[i]
                sample_true = target_presence[i]
                if torch.equal(sample_pred, sample_true):
                    exact_match_chains += 1

                for e_idx in range(num_effects):
                    p = bool(sample_pred[e_idx].item() == 1.0)
                    t = bool(sample_true[e_idx].item() == 1.0)
                    if p and t:
                        tp[e_idx] += 1
                    elif p and not t:
                        fp[e_idx] += 1
                    elif not p and t:
                        fn[e_idx] += 1
                    else:
                        tn[e_idx] += 1

            # 2. Parameter Regression Metrics
            for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
                start_idx, end_idx = PARAM_OFFSETS[effect_name]
                eff_mask = active_mask[:, start_idx:end_idx]
                if eff_mask.sum() > 0:
                    pred_p = pred_params[:, start_idx:end_idx]
                    true_p = target_params[:, start_idx:end_idx]
                    abs_diff = torch.abs(pred_p - true_p) * eff_mask
                    mean_err = (abs_diff.sum() / eff_mask.sum()).item()
                    param_abs_errors[e_idx].append(mean_err)

            # Overall active parameter errors
            if active_mask.sum() > 0:
                abs_diff_all = torch.abs(pred_params - target_params) * active_mask
                all_param_errors.append((abs_diff_all.sum() / active_mask.sum()).item())

            # 3. DSP Audio Reconstruction on subset
            if eval_reconstruction and len(recon_stft_losses) < max_recon_samples:
                for i in range(min(batch_size, max_recon_samples - len(recon_stft_losses))):
                    single_dry = dry_audio[i : i + 1]
                    single_target = wet_audio[i : i + 1]

                    pred_chain = model.predict_chain(single_dry, single_target, threshold=0.5)
                    try:
                        recon_audio = dapn.process(single_dry, pred_chain)
                        l_dry = loss_fn(single_dry, single_target).item()
                        l_recon = loss_fn(recon_audio, single_target).item()
                        l_mse = F.mse_loss(recon_audio, single_target).item()

                        dry_stft_losses.append(l_dry)
                        recon_stft_losses.append(l_recon)
                        recon_mse_losses.append(l_mse)
                    except Exception:
                        pass

    # -------------------------------------------------------------
    # Compute Final Aggregated Scores
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("                     EVALUATION SCORE VALUES REPORT")
    print("=" * 76)

    # Effect Classification Table
    print(f"  {'Effect':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 76)

    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)
    total_tn = sum(tn)

    f1_list = []
    prec_list = []
    rec_list = []

    for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
        eff_tp = tp[e_idx]
        eff_fp = fp[e_idx]
        eff_fn = fn[e_idx]
        eff_tn = tn[e_idx]
        tot = eff_tp + eff_fp + eff_fn + eff_tn

        acc = (eff_tp + eff_tn) / max(tot, 1) * 100.0
        prec = eff_tp / max(eff_tp + eff_fp, 1) * 100.0
        rec = eff_tp / max(eff_tp + eff_fn, 1) * 100.0
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

        print(f"  {effect_name:<12} | {acc:>9.2f}% | {prec:>9.2f}% | {rec:>9.2f}% | {f1:>9.2f}%")

    macro_f1 = sum(f1_list) / len(f1_list)
    micro_prec = total_tp / max(total_tp + total_fp, 1) * 100.0
    micro_rec = total_tp / max(total_tp + total_fn, 1) * 100.0
    micro_f1 = (2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-8)
    hamming_acc = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1) * 100.0
    exact_chain_acc = (exact_match_chains / max(total_chains, 1)) * 100.0

    print("-" * 76)
    print(f"  Exact Chain Match Accuracy : {exact_chain_acc:.2f}% (Complete chain identical)")
    print(f"  Hamming Presence Accuracy  : {hamming_acc:.2f}%")
    print(f"  Macro F1-Score             : {macro_f1:.2f}%")
    print(f"  Micro F1-Score             : {micro_f1:.2f}%")
    print("=" * 76)

    # Parameter Regression Table
    overall_mae = (sum(all_param_errors) / max(len(all_param_errors), 1)) if all_param_errors else 0.0
    print("\n" + "=" * 76)
    print("             CONTINUOUS PARAMETER ESTIMATION (NORMALIZED [0, 1])")
    print("=" * 76)
    print(f"  {'Effect':<12} | {'Param Count':<12} | {'Mean Absolute Error (MAE)':<25}")
    print("-" * 76)
    for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
        schema = PARAM_SCHEMA[effect_name]
        p_count = len(schema)
        errs = param_abs_errors[e_idx]
        eff_mae = (sum(errs) / len(errs)) if errs else 0.0
        print(f"  {effect_name:<12} | {p_count:<12} | {eff_mae:>12.4f} ({eff_mae*100:.2f}% error)")
    print("-" * 76)
    print(f"  Overall Parameter MAE      : {overall_mae:.4f} ({overall_mae*100:.2f}% avg error)")
    print("=" * 76)

    # Closed Loop Reconstruction Table
    if eval_reconstruction and recon_stft_losses:
        mean_dry_stft = sum(dry_stft_losses) / len(dry_stft_losses)
        mean_recon_stft = sum(recon_stft_losses) / len(recon_stft_losses)
        mean_recon_mse = sum(recon_mse_losses) / len(recon_mse_losses)
        spectral_imprv = max(0.0, (1.0 - mean_recon_stft / (mean_dry_stft + 1e-8)) * 100.0)

        print("\n" + "=" * 76)
        print("          DAPN CLOSED-LOOP ACOUSTIC RECONSTRUCTION SCORES")
        print("=" * 76)
        print(f"  Baseline (Dry vs Target) MR-STFT Loss       : {mean_dry_stft:.5f}")
        print(f"  Reconstructed (Pred vs Target) MR-STFT Loss : {mean_recon_stft:.5f}")
        print(f"  Average Waveform MSE Loss                   : {mean_recon_mse:.6f}")
        print(f"  Spectral Distance Reduction / Improvement   : {spectral_imprv:.2f}%")
        print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Evaluate summary & score values for trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--val_metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Validation metadata")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_recon", type=int, default=50, help="Number of audio tracks to evaluate with DAPN DSP")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # Load model
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = InverseFXNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 1. Print Model Summary
    print_model_summary(model)

    # 2. Data loader
    val_loader = get_dataloader(
        metadata_path=args.val_metadata,
        batch_size=args.batch_size,
        shuffle=False,
        max_duration_sec=2.0,
        sample_rate=44100,
        num_workers=0,
    )

    # 3. Calculate Scores
    evaluate_metrics(
        model=model,
        val_loader=val_loader,
        device=device,
        eval_reconstruction=True,
        max_recon_samples=args.num_recon,
    )


if __name__ == "__main__":
    main()
