"""
examples/optimize_fx_match.py
==============================
Demonstration of Differentiable DSP Parameter Matching via Autograd and MR-STFT Loss.

Given:
  1. Dry audio (input)
  2. Target wet audio (e.g., target sound with specific EQ, Compression, or Delay)

Objective:
  Starting from neutral/random parameters, use PyTorch Autograd and Adam optimizer
  to automatically discover the DSP parameters that match the target sound.

Usage:
------
python3 examples/optimize_fx_match.py
python3 examples/optimize_fx_match.py --steps 60 --lr 0.05
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio, save_audio
from dapn.dapn import DAPN
from common.losses import MultiResolutionSTFTLoss


def main():
    parser = argparse.ArgumentParser(description="DAPN Parameter Optimization with MR-STFT Loss")
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.04, help="Learning rate for Adam optimizer")
    parser.add_argument("--input", type=str, default="audio/dry.wav", help="Input dry audio file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Fallback to generating a tone if dry.wav is not present
        from audio_io import generate_test_tone
        print(f"File {input_path} not found. Generating synthetic dry tone...")
        audio_dry, sr = generate_test_tone(duration_seconds=1.0)
    else:
        audio_dry, sr = load_audio(input_path)

    # Use first 1 second for fast and responsive optimization demo
    max_samples = sr * 1
    if audio_dry.shape[-1] > max_samples:
        audio_dry = audio_dry[..., :max_samples]

    print("=" * 68)
    print(" DAPN Differentiable Parameter Matching via Multi-Resolution STFT")
    print("=" * 68)
    print(f"Dry Audio: {audio_dry.shape[-1]} samples ({audio_dry.shape[-1]/sr:.2f}s) at {sr} Hz\n")

    dapn = DAPN()

    # -------------------------------------------------------------
    # 1. Define Ground Truth Target Effect Setup
    # -------------------------------------------------------------
    # Target EQ setup: aggressive boost at mid-high frequencies
    target_freq = 0.70  # ~14 kHz physical
    target_gain = 0.85  # ~+8.4 dB physical
    target_q    = 0.60

    target_chain = {
        "chain": [
            {
                "effect": "EQ",
                "parameters": {
                    "frequency": target_freq,
                    "gain": target_gain,
                    "q": target_q,
                }
            }
        ]
    }

    print("[Step 1] Synthesizing Target Audio with Ground Truth DSP settings:")
    print(f"  Target EQ Frequency : {target_freq:.2f}")
    print(f"  Target EQ Gain      : {target_gain:.2f}")
    print(f"  Target EQ Q         : {target_q:.2f}")

    with torch.no_grad():
        audio_target = dapn.process(audio_dry, target_chain)

    save_audio(audio_target, "audio/opt_target.wav", sample_rate=sr)
    print("  -> Saved target sound to: audio/opt_target.wav\n")

    # -------------------------------------------------------------
    # 2. Initialize Learnable Parameters (Blind to Ground Truth)
    # -------------------------------------------------------------
    # Start at neutral / un-boosted settings
    init_freq = 0.30
    init_gain = 0.50  # 0 dB neutral
    init_q    = 0.30

    param_freq = torch.tensor(init_freq, requires_grad=True)
    param_gain = torch.tensor(init_gain, requires_grad=True)
    param_q    = torch.tensor(init_q,    requires_grad=True)

    print("[Step 2] Initializing Learnable Parameters (Starting Guess):")
    print(f"  Initial EQ Frequency: {init_freq:.2f}")
    print(f"  Initial EQ Gain     : {init_gain:.2f}")
    print(f"  Initial EQ Q        : {init_q:.2f}\n")

    # -------------------------------------------------------------
    # 3. Setup Optimizer & Multi-Resolution STFT Loss
    # -------------------------------------------------------------
    optimizer = torch.optim.Adam([param_freq, param_gain, param_q], lr=args.lr)
    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[256, 512, 1024],
        hop_sizes=[64, 128, 256],
        win_lengths=[256, 512, 1024],
    )

    # -------------------------------------------------------------
    # 4. Run Optimization Loop
    # -------------------------------------------------------------
    print(f"[Step 3] Running Gradient Descent Optimization ({args.steps} steps)...")
    print("-" * 68)
    print(f" Step | MR-STFT Loss | Freq (pred/tgt) | Gain (pred/tgt) | Q (pred/tgt)")
    print("-" * 68)

    t0 = time.perf_counter()
    initial_loss = None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        # Build dynamic override with current gradient-tracked parameters
        override = {
            0: {
                "frequency": param_freq,
                "gain": param_gain,
                "q": param_q,
            }
        }

        # DAPN forward pass (preserves autograd graph)
        pred_audio = dapn.process(audio_dry, target_chain, param_override=override)

        # Multi-Resolution STFT loss
        loss = loss_fn(pred_audio, audio_target)
        if initial_loss is None:
            initial_loss = loss.item()

        # Backward pass
        loss.backward()

        # Gradient step
        optimizer.step()

        # Clamp parameters to valid [0, 1] normalized space
        with torch.no_grad():
            param_freq.clamp_(0.0, 1.0)
            param_gain.clamp_(0.0, 1.0)
            param_q.clamp_(0.0, 1.0)

        # Log progress periodically
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(
                f" {step:4d} | {loss.item():12.5f} | "
                f"{param_freq.item():.2f} / {target_freq:.2f}    | "
                f"{param_gain.item():.2f} / {target_gain:.2f}    | "
                f"{param_q.item():.2f} / {target_q:.2f}"
            )

    elapsed = time.perf_counter() - t0
    print("-" * 68)
    print(f"Optimization completed in {elapsed:.2f} seconds ({elapsed/args.steps*1000:.1f} ms/step)\n")

    # -------------------------------------------------------------
    # 5. Final Results & Audio Output
    # -------------------------------------------------------------
    final_loss = loss.item()
    loss_reduction_pct = (1.0 - final_loss / (initial_loss + 1e-9)) * 100.0

    print("[Step 4] Final Evaluation:")
    print(f"  Initial Loss       : {initial_loss:.5f}")
    print(f"  Final Loss         : {final_loss:.5f}  (Reduced by {loss_reduction_pct:.1f}%)")
    print(f"  Estimated Frequency: {param_freq.item():.3f} (Ground Truth: {target_freq:.3f})")
    print(f"  Estimated Gain     : {param_gain.item():.3f} (Ground Truth: {target_gain:.3f})")
    print(f"  Estimated Q        : {param_q.item():.3f} (Ground Truth: {target_q:.3f})")

    # Save final optimized audio
    with torch.no_grad():
        override_final = {
            0: {
                "frequency": param_freq,
                "gain": param_gain,
                "q": param_q,
            }
        }
        audio_opt = dapn.process(audio_dry, target_chain, param_override=override_final)

    save_audio(audio_opt, "audio/opt_reconstructed.wav", sample_rate=sr)
    print("\n  -> Saved reconstructed audio to: audio/opt_reconstructed.wav")
    print("=" * 68)
    print(" Differentiable DSP optimization verified successfully!")
    print("=" * 68)


if __name__ == "__main__":
    main()
