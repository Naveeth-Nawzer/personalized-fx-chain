"""
examples/process_file.py
========================
Process any custom WAV audio file through DAPN effects chain, save the output,
and generate a comprehensive audio analysis and test results report.

Usage:
------
python examples/process_file.py --input path/to/your_input.wav --output audio/output.wav
python examples/process_file.py --input path/to/your_input.wav --output audio/output.wav --preset reverb_only
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio, save_audio
from dapn.dapn import DAPN


def compute_audio_metrics(audio: torch.Tensor, name: str = "Audio") -> dict:
    """Compute key signal processing metrics for an audio tensor."""
    eps = 1e-9
    peak_linear = float(audio.abs().max().item())
    peak_db = float(20.0 * torch.log10(torch.tensor(peak_linear + eps)).item())

    rms_linear = float(audio.pow(2).mean().sqrt().item())
    rms_db = float(20.0 * torch.log10(torch.tensor(rms_linear + eps)).item())

    crest_factor_db = peak_db - rms_db if rms_linear > 0 else 0.0
    is_finite = bool(torch.isfinite(audio).all().item())
    is_clipped = bool((audio.abs() > 1.0).any().item())

    return {
        "name": name,
        "is_finite": is_finite,
        "is_clipped": is_clipped,
        "peak_linear": round(peak_linear, 5),
        "peak_dbFS": round(peak_db, 2),
        "rms_linear": round(rms_linear, 5),
        "rms_dbFS": round(rms_db, 2),
        "crest_factor_db": round(crest_factor_db, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Process an audio file through DAPN and generate test results")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--output", "-o", type=str, default="audio/processed_output.wav", help="Path to save processed WAV file")
    parser.add_argument("--preset", "-p", type=str, default="full_chain",
                        choices=["full_chain", "reverb_only", "delay_only", "comp_only", "eq_only"],
                        help="Effect chain preset to apply")
    parser.add_argument("--report", "-r", type=str, default=None,
                        help="Optional custom path for JSON test report (defaults to <output_name>_report.json)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}_report.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    print("=" * 65)
    print(" DAPN Audio Processing & Test Result Evaluation")
    print("=" * 65)

    print(f"\n[1] Loading input audio from: {input_path}")
    audio_in, sr = load_audio(input_path)
    duration_s = audio_in.shape[-1] / sr
    print(f"    Shape: {list(audio_in.shape)} (Channels: {audio_in.shape[0]}, Samples: {audio_in.shape[-1]})")
    print(f"    Sample Rate: {sr} Hz | Duration: {duration_s:.3f} seconds")

    # Define effect chain presets
    presets = {
        "full_chain": {
            "chain": [
                {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.7, "q": 0.5}},
                {"effect": "Compressor", "parameters": {"threshold": 0.5, "ratio": 0.5, "attack": 0.2, "release": 0.4, "makeup_gain": 0.2}},
                {"effect": "Delay", "parameters": {"delay_time": 0.15, "feedback": 0.35, "wet_mix": 0.4, "dry_mix": 0.7}},
                {"effect": "Reverb", "parameters": {"room_size": 0.6, "damping": 0.4, "wet_mix": 0.35, "dry_mix": 0.8, "decay": 0.4}},
            ]
        },
        "reverb_only": {
            "chain": [
                {"effect": "Reverb", "parameters": {"room_size": 0.7, "damping": 0.3, "wet_mix": 0.5, "dry_mix": 0.7, "decay": 0.5}}
            ]
        },
        "delay_only": {
            "chain": [
                {"effect": "Delay", "parameters": {"delay_time": 0.2, "feedback": 0.45, "wet_mix": 0.5, "dry_mix": 0.7}}
            ]
        },
        "comp_only": {
            "chain": [
                {"effect": "Compressor", "parameters": {"threshold": 0.4, "ratio": 0.6, "attack": 0.1, "release": 0.3, "makeup_gain": 0.3}}
            ]
        },
        "eq_only": {
            "chain": [
                {"effect": "EQ", "parameters": {"frequency": 0.6, "gain": 0.8, "q": 0.7}}
            ]
        },
    }

    chain_dict = presets[args.preset]
    print(f"\n[2] Configured FX Chain (Preset: '{args.preset}'):")
    for idx, step in enumerate(chain_dict["chain"], 1):
        print(f"    Step {idx}: {step['effect']:<12} -> {step['parameters']}")

    print("\n[3] Executing DAPN Forward Pass...")
    dapn = DAPN()
    start_time = time.perf_counter()
    audio_out = dapn.process(audio_in, chain_dict)
    elapsed_s = time.perf_counter() - start_time
    rtf = duration_s / elapsed_s if elapsed_s > 0 else 0.0

    print(f"    Execution Time : {elapsed_s * 1000.0:.2f} ms")
    print(f"    Real-Time Speed: {rtf:.1f}x faster than real-time")

    # Metrics
    in_metrics = compute_audio_metrics(audio_in, "Input (Dry)")
    out_metrics = compute_audio_metrics(audio_out, "Output (Wet)")

    # Comparison metrics
    mse = float(torch.nn.functional.mse_loss(audio_out, audio_in).item())
    l1 = float((audio_out - audio_in).abs().mean().item())
    rms_diff_db = round(out_metrics["rms_dbFS"] - in_metrics["rms_dbFS"], 2)

    # Save output audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(audio_out, output_path, sample_rate=sr)
    print(f"\n[4] Output Audio Saved: {output_path.resolve()}")

    # Print Result Summary
    print("\n" + "=" * 65)
    print(" TEST RESULTS & AUDIO ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"  Status                  : {'PASSED (Stable)' if out_metrics['is_finite'] else 'FAILED (NaN/Inf detected)'}")
    print(f"  Numerical Stability     : Finite={out_metrics['is_finite']}, Clipped={out_metrics['is_clipped']}")
    print("-" * 65)
    print(f"  Metric                  |  Input (Dry)     |  Output (Wet)    ")
    print("-" * 65)
    print(f"  Peak Amplitude (linear) |  {in_metrics['peak_linear']:<15} |  {out_metrics['peak_linear']:<15}")
    print(f"  Peak Amplitude (dBFS)   |  {in_metrics['peak_dbFS']:<15} |  {out_metrics['peak_dbFS']:<15}")
    print(f"  RMS Level (linear)      |  {in_metrics['rms_linear']:<15} |  {out_metrics['rms_linear']:<15}")
    print(f"  RMS Level (dBFS)        |  {in_metrics['rms_dbFS']:<15} |  {out_metrics['rms_dbFS']:<15}")
    print(f"  Dynamic Range (Crest dB)|  {in_metrics['crest_factor_db']:<15} |  {out_metrics['crest_factor_db']:<15}")
    print("-" * 65)
    print(f"  RMS Difference (Δ dB)   : {rms_diff_db:+} dB")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (L1): {l1:.6f}")
    print("=" * 65)

    # Export Report JSON
    report_data = {
        "test_status": "PASSED" if out_metrics["is_finite"] else "FAILED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "preset": args.preset,
        "chain": chain_dict["chain"],
        "audio_info": {
            "channels": audio_in.shape[0],
            "samples": audio_in.shape[-1],
            "sample_rate": sr,
            "duration_seconds": round(duration_s, 4),
        },
        "performance": {
            "processing_time_ms": round(elapsed_s * 1000.0, 2),
            "real_time_factor": round(rtf, 1),
        },
        "input_metrics": in_metrics,
        "output_metrics": out_metrics,
        "comparison_metrics": {
            "rms_change_db": rms_diff_db,
            "mse": round(mse, 6),
            "l1_error": round(l1, 6),
        }
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[5] Detailed Test Result Report saved to: {report_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
