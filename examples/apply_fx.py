"""
examples/apply_fx.py
====================
Apply any chosen DAPN audio effects to any plain (dry) audio file to create
a custom target wet audio for testing and listening.

Usage Examples:
---------------
1. Apply Reverb & Delay:
   python3 examples/apply_fx.py --input audio/dry.wav --effects Reverb Delay --output audio/test_wet.wav

2. Apply Distortion & EQ:
   python3 examples/apply_fx.py --input audio/dry.wav --effects EQ Distortion --output audio/test_wet.wav

3. Apply a Preset (e.g. ambient_space, rock_drive, clean_chorus):
   python3 examples/apply_fx.py --input audio/dry.wav --preset ambient_space --output audio/test_wet.wav

4. Test InverseFXNet Predictor on your newly created audio:
   python3 evaluate_predictor.py --dry audio/dry.wav --wet audio/test_wet.wav
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from audio_io import load_audio, save_audio
from dapn.dapn import DAPN

# Preset configurations
PRESETS = {
    "ambient_space": {
        "chain": [
            {"effect": "Delay", "parameters": {"delay_time": 0.35, "feedback": 0.50, "wet_mix": 0.45, "dry_mix": 0.70}},
            {"effect": "Reverb", "parameters": {"room_size": 0.75, "damping": 0.25, "wet_mix": 0.50, "dry_mix": 0.60, "decay": 0.60}},
        ]
    },
    "rock_drive": {
        "chain": [
            {"effect": "EQ", "parameters": {"frequency": 0.55, "gain": 0.75, "q": 0.60}},
            {"effect": "Distortion", "parameters": {"drive": 0.65, "tone": 0.60, "mix": 0.70, "output_gain": 0.45}},
        ]
    },
    "clean_chorus": {
        "chain": [
            {"effect": "Chorus", "parameters": {"rate": 0.40, "depth": 0.55, "delay": 0.25, "mix": 0.70}},
            {"effect": "Reverb", "parameters": {"room_size": 0.50, "damping": 0.40, "wet_mix": 0.35, "dry_mix": 0.80, "decay": 0.30}},
        ]
    },
    "vocal_polish": {
        "chain": [
            {"effect": "Compressor", "parameters": {"threshold": 0.45, "ratio": 0.50, "attack": 0.20, "release": 0.40, "makeup_gain": 0.30}},
            {"effect": "EQ", "parameters": {"frequency": 0.70, "gain": 0.65, "q": 0.50}},
            {"effect": "Reverb", "parameters": {"room_size": 0.40, "damping": 0.30, "wet_mix": 0.25, "dry_mix": 0.85, "decay": 0.25}},
        ]
    },
}

# Default effect parameters when selected by name
DEFAULT_PARAMS = {
    "EQ": {"frequency": 0.55, "gain": 0.70, "q": 0.50},
    "Compressor": {"threshold": 0.40, "ratio": 0.60, "attack": 0.15, "release": 0.35, "makeup_gain": 0.25},
    "Delay": {"delay_time": 0.25, "feedback": 0.45, "wet_mix": 0.40, "dry_mix": 0.70},
    "Reverb": {"room_size": 0.65, "damping": 0.35, "wet_mix": 0.40, "dry_mix": 0.70, "decay": 0.45},
    "Distortion": {"drive": 0.55, "tone": 0.55, "mix": 0.60, "output_gain": 0.50},
    "Chorus": {"rate": 0.45, "depth": 0.50, "delay": 0.25, "mix": 0.65},
    "Flanger": {"rate": 0.30, "depth": 0.50, "feedback": 0.40, "mix": 0.60},
}


def main():
    parser = argparse.ArgumentParser(description="Apply DAPN audio effects to plain/dry audio")
    parser.add_argument("--input", "-i", type=str, default="audio/dry.wav", help="Input plain/dry WAV file")
    parser.add_argument("--output", "-o", type=str, default="audio/test_wet.wav", help="Output wet WAV file")
    parser.add_argument("--effects", "-e", nargs="+", help="List of effects to apply (e.g. --effects Reverb Delay)")
    parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()), help="Apply a pre-configured effect chain preset")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input audio not found: {input_path}")
        return

    # Build chain config
    if args.preset:
        chain_dict = PRESETS[args.preset]
        print(f"Using Preset: '{args.preset}'")
    elif args.effects:
        chain = []
        for name in args.effects:
            # Capitalize to match DAPN effect naming
            norm_name = name.capitalize()
            if norm_name == "Eq":
                norm_name = "EQ"
            if norm_name not in DEFAULT_PARAMS:
                print(f"Unknown effect: '{name}'. Choose from: {list(DEFAULT_PARAMS.keys())}")
                return
            chain.append({"effect": norm_name, "parameters": DEFAULT_PARAMS[norm_name]})
        chain_dict = {"chain": chain}
    else:
        print("Please specify either --effects (e.g. --effects Reverb Delay) or --preset (e.g. --preset ambient_space)")
        print(f"Available presets: {list(PRESETS.keys())}")
        print(f"Available effects: {list(DEFAULT_PARAMS.keys())}")
        return

    print("=" * 65)
    print(" DAPN Audio Effect Processing")
    print("=" * 65)
    print(f"Loading Dry Audio: {input_path}")
    audio_in, sr = load_audio(input_path)

    print("\nApplying Effects Chain:")
    for idx, step in enumerate(chain_dict["chain"], 1):
        print(f"  [{idx}] {step['effect']:<12} -> {step['parameters']}")

    dapn = DAPN()
    with torch.no_grad():
        audio_out = dapn.process(audio_in, chain_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(audio_out, output_path, sample_rate=sr)

    print(f"\n[OK] Effected (Wet) Audio Saved to: {output_path.resolve()}")
    print("=" * 65)
    print("Next step: To test your trained model predicting this chain, run:")
    print(f"  python3 evaluate_predictor.py --dry {input_path} --wet {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
