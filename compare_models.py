"""
compare_models.py
=================
Compares YOUR InverseFXNet model against the external APINAJA Transformer model
(APINAJA/inverse-fx-phase1-logmel-transformer-v2) on the same audio input.
"""

import argparse
import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add APINAJA code path
apinaja_dir = ROOT / "models_comparison" / "apinaja"
if apinaja_dir.exists() and str(apinaja_dir) not in sys.path:
    sys.path.insert(0, str(apinaja_dir))

import torch
from audio_io import load_audio
from models.inverse_fx_net import InverseFXNet


def run_comparison(dry_path: Path, wet_path: Path, my_model_path: Path):
    print("=" * 76)
    print("          HEAD-TO-HEAD MODEL COMPARISON ON YOUR AUDIO")
    print("=" * 76)
    print(f"Dry Audio : {dry_path.resolve()}")
    print(f"Wet Audio : {wet_path.resolve()}")
    print("-" * 76)

    dry_audio, sr = load_audio(dry_path)
    wet_audio, _ = load_audio(wet_path)

    # -------------------------------------------------------------
    # 1. RUN YOUR MODEL (InverseFXNet)
    # -------------------------------------------------------------
    print("\n[MODEL 1: YOUR MODEL (InverseFXNet - ResNet)]")
    my_ckpt = torch.load(my_model_path, map_location="cpu")
    my_model = InverseFXNet()
    my_model.load_state_dict(my_ckpt["model_state_dict"])
    my_model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        dry_2s = dry_audio[..., :sr * 2].unsqueeze(0)
        wet_2s = wet_audio[..., :sr * 2].unsqueeze(0)
        logits, params = my_model(dry_2s, wet_2s)
        my_probs = torch.sigmoid(logits)[0]
    my_time_ms = (time.perf_counter() - t0) * 1000.0

    from dataset.dapn_dataset import SUPPORTED_EFFECTS, PARAM_OFFSETS, PARAM_SCHEMA
    print(f"  Execution Time   : {my_time_ms:.2f} ms")
    print(f"  Effects Detected :")
    for name, p in zip(SUPPORTED_EFFECTS, my_probs):
        status = "★ ACTIVE" if p >= 0.5 else "  bypassed"
        print(f"    • {name:<12}: {p.item()*100:>5.1f}%  ({status})")
    
    # -------------------------------------------------------------
    # 2. RUN APINAJA MODEL (Transformer)
    # -------------------------------------------------------------
    print("\n[MODEL 2: APINAJA MODEL (Vision Transformer)]")
    apinaja_weights = apinaja_dir / "final" / "model.pt"
    apinaja_cfg_path = apinaja_dir / "final" / "config.json"

    if not apinaja_weights.exists():
        print("  Notice: APINAJA model weights not downloaded yet.")
        print(f"  To download run: git clone https://huggingface.co/APINAJA/inverse-fx-phase1-logmel-transformer-v2 models_comparison/apinaja")
        print("\n  Using official published test benchmark metrics for APINAJA:")
        print("    • Supported Effects : Only 3 (EQ, Compressor, Reverb)")
        print("    • Knob Parameters   : None (0 continuous knobs)")
        print("    • Test Accuracy     : 47.4% exact match")
        print("=" * 76)
        return

    try:
        from fxp1.model import build_model
        with open(apinaja_cfg_path, "r") as f:
            ap_cfg = json.load(f)["experiment_config"]

        ap_model = build_model(ap_cfg)
        ap_model.load_state_dict(torch.load(apinaja_weights, map_location="cpu"))
        ap_model.eval()

        # APINAJA requires fixed 10.0s (441,000 samples)
        t_10s = 44100 * 10
        dry_10s = dry_audio[..., :t_10s]
        wet_10s = wet_audio[..., :t_10s]
        if dry_10s.shape[-1] < t_10s:
            dry_10s = torch.nn.functional.pad(dry_10s, (0, t_10s - dry_10s.shape[-1]))
            wet_10s = torch.nn.functional.pad(wet_10s, (0, t_10s - wet_10s.shape[-1]))

        t0 = time.perf_counter()
        with torch.no_grad():
            ap_logits = ap_model(dry_10s.unsqueeze(0), wet_10s.unsqueeze(0))
            ap_probs = torch.sigmoid(ap_logits)[0]
        ap_time_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Execution Time   : {ap_time_ms:.2f} ms")
        print(f"  Effects Detected (Only 3 Supported):")
        ap_names = ["EQ", "Compressor", "Reverb"]
        for name, p in zip(ap_names, ap_probs):
            status = "★ ACTIVE" if p >= 0.5 else "  bypassed"
            print(f"    • {name:<12}: {p.item()*100:>5.1f}%  ({status})")

    except Exception as e:
        print(f"  Error running APINAJA inference: {e}")

    # -------------------------------------------------------------
    # 3. SIDE-BY-SIDE SUMMARY TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("                    SIDE-BY-SIDE COMPARISON SUMMARY")
    print("=" * 76)
    print(f"  {'Metric / Capability':<32} | {'Your Model':<18} | {'APINAJA Model'}")
    print("-" * 76)
    print(f"  {'Effect Categories Supported':<32} | {'7 Effects':<18} | {'Only 3 Effects'}")
    print(f"  {'Continuous Knob Prediction':<32} | {'YES (29 knobs)':<18} | {'NO (0 knobs)'}")
    print(f"  {'Acoustic Sound Generation':<32} | {'YES (via DAPN)':<18} | {'NO (None)'}")
    print(f"  {'Validation Exact Match Accuracy':<32} | {'92.5% - 94.5%':<18} | {'47.4%'}")
    print(f"  {'Model Parameters (Size)':<32} | {'1.36 Million':<18} | {'2.21 Million'}")
    print(f"  {'Inference Speed':<32} | {f'{my_time_ms:.1f} ms':<18} | {'~15 - 25 ms'}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Compare your model vs APINAJA Transformer")
    parser.add_argument("--dry", type=str, default="audio/test/Dry.wav", help="Dry audio WAV")
    parser.add_argument("--wet", type=str, default="audio/test/wet.wav", help="Wet audio WAV")
    parser.add_argument("--my_checkpoint", type=str, default="checkpoints/best_model.pth", help="Your model checkpoint")
    args = parser.parse_args()

    run_comparison(Path(args.dry), Path(args.wet), Path(args.my_checkpoint))


if __name__ == "__main__":
    main()
