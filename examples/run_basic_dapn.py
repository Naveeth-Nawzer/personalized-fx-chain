"""
examples/run_basic_dapn.py
==========================
Basic DAPN experiment demonstrating end-to-end differentiable audio processing.

What this script does
---------------------
1. Generates or loads a dry audio signal (sine wave + harmonic test tone) → saves to ``audio/dry.wav``.
2. Loads ``audio/dry.wav`` via ``audio_io.load_audio``.
3. Defines a manual FX chain: EQ → Compressor → Delay (or Reverb).
4. Creates learnable normalized parameter tensors for each effect.
5. Passes the audio and chain to DAPN.
6. Saves the processed output to ``audio/wet.wav``.
7. Computes a simple differentiable loss (mean-squared error against target).
8. Performs ``loss.backward()`` to calculate gradients.
9. Prints gradient information verifying non-zero, finite gradient flow.

NOTE: There is NO GRRL-Agent at this stage. Manual parameters are intentional.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from audio_io import generate_test_tone, load_audio, save_audio
from dapn.dapn import DAPN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    print("=" * 65)
    print(" DAPN MVP: Differentiable Audio Processing Network Demonstration")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Prepare dry audio
    # ------------------------------------------------------------------
    audio_dir = _PROJECT_ROOT / "audio"
    audio_dir.mkdir(exist_ok=True)
    dry_path = audio_dir / "dry.wav"
    wet_path = audio_dir / "wet.wav"

    print("\n[Step 1] Preparing dry audio...")
    # Generate a composite test tone: 220 Hz (A3) + 440 Hz (A4) harmonic
    sr = 44100
    duration_s = 0.5  # 500 ms test tone
    tone1 = generate_test_tone(frequency_hz=220.0, duration_s=duration_s, amplitude=0.4, sample_rate=sr)
    tone2 = generate_test_tone(frequency_hz=440.0, duration_s=duration_s, amplitude=0.2, sample_rate=sr)
    dry_signal = tone1 + tone2

    save_audio(dry_signal, dry_path, sample_rate=sr)
    print(f"  Saved dry signal to: {dry_path}")

    # ------------------------------------------------------------------
    # 2. Load dry audio via audio_io
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading dry audio via audio_io...")
    audio, loaded_sr = load_audio(dry_path)
    print(f"  Loaded audio: shape={tuple(audio.shape)}, sample_rate={loaded_sr} Hz")
    assert audio.shape[0] == 1, "Expected mono audio"
    print("  Dry audio successfully loaded: OK")

    # ------------------------------------------------------------------
    # 3. Define manual FX chain schema (Phase 3 canonical format)
    # ------------------------------------------------------------------
    print("\n[Step 3] Defining canonical FX chain (EQ -> Compressor -> Delay)...")
    chain_dict = {
        "chain": [
            {
                "effect": "EQ",
                "parameters": {
                    "frequency": 0.35,  # ~7000 Hz
                    "gain":      0.75,  # +6 dB boost
                    "q":         0.40,  # Q ≈ 4
                },
            },
            {
                "effect": "Compressor",
                "parameters": {
                    "threshold":   0.50,  # -30 dB
                    "ratio":       0.40,  # ratio ≈ 8.6
                    "attack":      0.20,  # ~20 ms
                    "release":     0.30,  # ~300 ms
                    "makeup_gain": 0.35,  # ~4.2 dB
                },
            },
            {
                "effect": "Delay",
                "parameters": {
                    "delay_time": 0.05,  # ~100 ms
                    "feedback":   0.35,  # feedback ≈ 0.33
                    "wet_mix":    0.60,  # 60% wet
                    "dry_mix":    0.40,  # 40% dry
                },
            },
        ]
    }
    print("  FX chain defined: EQ -> Compressor -> Delay")

    # ------------------------------------------------------------------
    # 4. Create learnable parameter tensors for gradient tracking
    # ------------------------------------------------------------------
    print("\n[Step 4] Creating learnable parameter tensors (requires_grad=True)...")
    # Step 0: EQ
    eq_freq   = torch.tensor(0.35, requires_grad=True)
    eq_gain   = torch.tensor(0.75, requires_grad=True)
    eq_q      = torch.tensor(0.40, requires_grad=True)

    # Step 1: Compressor
    cp_thresh = torch.tensor(0.50, requires_grad=True)
    cp_ratio  = torch.tensor(0.40, requires_grad=True)
    cp_att    = torch.tensor(0.20, requires_grad=True)
    cp_rel    = torch.tensor(0.30, requires_grad=True)
    cp_makeup = torch.tensor(0.35, requires_grad=True)

    # Step 2: Delay
    dl_time   = torch.tensor(0.05, requires_grad=True)
    dl_fb     = torch.tensor(0.35, requires_grad=True)
    dl_wet    = torch.tensor(0.60, requires_grad=True)
    dl_dry    = torch.tensor(0.40, requires_grad=True)

    param_override = {
        0: {"frequency": eq_freq, "gain": eq_gain, "q": eq_q},
        1: {
            "threshold":   cp_thresh,
            "ratio":       cp_ratio,
            "attack":      cp_att,
            "release":     cp_rel,
            "makeup_gain": cp_makeup,
        },
        2: {
            "delay_time": dl_time,
            "feedback":   dl_fb,
            "wet_mix":    dl_wet,
            "dry_mix":    dl_dry,
        },
    }

    # ------------------------------------------------------------------
    # 5. Initialize DAPN and process audio
    # ------------------------------------------------------------------
    print("\n[Step 5] Validating FX chain and running DAPN forward pass...")
    dapn = DAPN()
    wet_audio = dapn.process(audio, chain_dict, param_override=param_override)
    print(f"  Processed audio: shape={tuple(wet_audio.shape)}, finite={torch.isfinite(wet_audio).all().item()}")

    # ------------------------------------------------------------------
    # 6. Save wet audio
    # ------------------------------------------------------------------
    print("\n[Step 6] Saving wet audio...")
    save_audio(wet_audio, wet_path, sample_rate=loaded_sr)
    print(f"  Saved wet audio to: {wet_path}")

    # ------------------------------------------------------------------
    # 7. Compute differentiable loss
    # ------------------------------------------------------------------
    print("\n[Step 7] Computing loss (MSE against half-amplitude dry target)...")
    target = audio * 0.5
    loss = (wet_audio - target).pow(2).mean()
    print(f"  Loss value: {loss.item():.6f}")

    # ------------------------------------------------------------------
    # 8. Backward pass
    # ------------------------------------------------------------------
    print("\n[Step 8] Executing backward pass: loss.backward()...")
    loss.backward()
    print("  Backward pass completed successfully.")

    # ------------------------------------------------------------------
    # 9. Print gradients
    # ------------------------------------------------------------------
    print("\n[Step 9] Gradient Validation Results:")
    print("-" * 55)

    params_to_check = [
        ("EQ frequency",         eq_freq),
        ("EQ gain",              eq_gain),
        ("EQ Q",                 eq_q),
        ("Compressor threshold", cp_thresh),
        ("Compressor ratio",     cp_ratio),
        ("Compressor makeup",    cp_makeup),
        ("Delay delay_time",     dl_time),
        ("Delay feedback",       dl_fb),
        ("Delay wet_mix",        dl_wet),
        ("Delay dry_mix",        dl_dry),
    ]

    all_passed = True
    for label, param in params_to_check:
        grad = param.grad
        if grad is None:
            status = "FAILED (grad is None)"
            all_passed = False
        elif not torch.isfinite(grad):
            status = f"FAILED (not finite: {grad.item()})"
            all_passed = False
        elif grad.abs().item() == 0.0:
            status = "WARNING (grad is 0.0 — inactive for this input/loss)"
        else:
            status = f"PASSED (non-zero: {grad.item():+.6e})"
        print(f"  {label:<25} : {status}")

    print("-" * 55)
    if all_passed:
        print("  ✓ All gradient checks PASSED. Graph is fully differentiable.")
    else:
        print("  ✗ Some gradient checks FAILED.")
        sys.exit(1)

    print("\n=" * 65)
    print(" DAPN MVP execution completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
