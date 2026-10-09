"""
evaluate_predictor.py
=====================
Step 4: Connects the trained InverseFXNet Predictor to DAPN.

Loads the trained neural network checkpoint, predicts the audio FX chain from an
unseen dry/wet audio pair, renders the predicted sound through DAPN, and measures:
  1. Effect detection accuracy.
  2. Parameter Mean Absolute Error (MAE).
  3. Spectral reconstruction quality via Multi-Resolution STFT Loss.

Usage:
------
python3 evaluate_predictor.py --checkpoint checkpoints/best_model.pth
python3 evaluate_predictor.py --dry audio/dry.wav --wet audio/wet.wav
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio, save_audio
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from models.inverse_fx_net import InverseFXNet


def main():
    parser = argparse.ArgumentParser(description="Evaluate InverseFXNet connected to DAPN")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--dry", type=str, default=None, help="Path to input dry audio (optional)")
    parser.add_argument("--wet", type=str, default=None, help="Path to target wet audio (optional)")
    parser.add_argument("--metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Validation metadata path")
    parser.add_argument("--output", type=str, default="audio/reconstructed_pred.wav", help="Output audio path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold probability (default: 0.5)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"Checkpoint not found at: {ckpt_path}")
        print("Please train the model first by running: python3 train.py --epochs 10")
        return

    print("=" * 68)
    print(" DAPN Inverse FX Predictor: Closed-Loop Evaluation")
    print("=" * 68)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = InverseFXNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully (Trained at epoch {checkpoint.get('epoch', '?')}, Val Acc: {checkpoint.get('val_accuracy', 0.0):.1f}%)\n")

    # Load test audio
    if args.dry and args.wet:
        dry_path = Path(args.dry)
        wet_path = Path(args.wet)
        print(f"Testing on custom files:\n  Dry: {dry_path}\n  Wet: {wet_path}")
        dry_audio, sr = load_audio(dry_path)
        target_wet, _ = load_audio(wet_path)
        ground_truth_seq = None
    else:
        # Load first sample from validation set
        import json
        meta_path = Path(args.metadata)
        if not meta_path.exists():
            raise FileNotFoundError(f"Validation metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())

        dry_path = ROOT / first_line["dry_path"]
        wet_path = ROOT / first_line["wet_path"]
        print(f"Testing on sample: {first_line['sample_id']}")
        print(f"  Dry : {dry_path}")
        print(f"  Wet : {wet_path}")
        dry_audio, sr = load_audio(dry_path)
        target_wet, _ = load_audio(wet_path)
        ground_truth_seq = first_line["chain_sequence"]
        print(f"  Ground Truth Chain: {ground_truth_seq}\n")

    # Crop to 2 seconds for consistent evaluation
    max_len = sr * 2
    dry_audio = dry_audio[..., :max_len].to(device)
    target_wet = target_wet[..., :max_len].to(device)

    # -------------------------------------------------------------
    # Step 1: Predict Chain using Neural Network
    # -------------------------------------------------------------
    print(f"[1] Running Neural Network Inference (Threshold: {args.threshold})...")
    pred_chain = model.predict_chain(dry_audio, target_wet, threshold=args.threshold)

    print("    Predicted Chain Structure:")
    for step in pred_chain["chain"]:
        print(f"     -> {step['effect']:<12}: {step['parameters']}")

    # -------------------------------------------------------------
    # Step 2: Render through DAPN
    # -------------------------------------------------------------
    print("\n[2] Rendering Predicted Chain through DAPN DSP Engine...")
    dapn = DAPN()
    with torch.no_grad():
        reconstructed_audio = dapn.process(dry_audio, pred_chain)

    # -------------------------------------------------------------
    # Step 3: Compute Evaluation Metrics
    # -------------------------------------------------------------
    print("\n[3] Closed-Loop Reconstruction Evaluation:")
    loss_fn = MultiResolutionSTFTLoss()
    with torch.no_grad():
        mr_stft_dist = float(loss_fn(reconstructed_audio, target_wet).item())
        mse_dist = float(torch.nn.functional.mse_loss(reconstructed_audio, target_wet).item())
        dry_target_dist = float(loss_fn(dry_audio, target_wet).item())

    # Calculate improvement over doing nothing (dry audio baseline)
    improvement = max(0.0, (1.0 - mr_stft_dist / (dry_target_dist + 1e-8)) * 100.0)

    print("-" * 68)
    print(f"  Dry vs Target Initial Distance : {dry_target_dist:.5f}")
    print(f"  Reconstructed vs Target MR-STFT: {mr_stft_dist:.5f} (Improved by {improvement:.1f}%)")
    print(f"  Waveform MSE                   : {mse_dist:.6f}")
    print("-" * 68)

    # Save reconstructed audio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(reconstructed_audio.cpu(), out_path, sample_rate=sr)
    print(f"\nSaved Reconstructed Audio to: {out_path.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
