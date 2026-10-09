"""
predict_fx.py
=============
Direct CLI tool to:
1. Predict the Audio Effects Chain from any (Dry, Wet) audio pair.
2. Predict and display specific effect knob values (e.g. Chorus parameters)
   in both normalized [0, 1] and physical DSP units.

Usage Examples:
---------------
1. Predict Complete Audio Chain:
   python3 predict_fx.py --dry audio/test/Dry.wav --wet audio/test/wet.wav

2. Predict Chorus Parameter Values specifically:
   python3 predict_fx.py --dry audio/test/Dry.wav --wet audio/test/wet.wav --effect Chorus

3. Predict Any Specific Effect (e.g. EQ, Reverb, Distortion, Delay):
   python3 predict_fx.py --dry audio/test/Dry.wav --wet audio/test/wet.wav --effect Reverb
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from audio_io import load_audio
from common.config_loader import load_effects_config
from common.parameter_mapper import ParameterMapper
from dataset.dapn_dataset import PARAM_OFFSETS, PARAM_SCHEMA, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet


from common.audio_types import EffectParameterSpec


def get_physical_mappers():
    """Load parameter mappers from effects_config.yaml."""
    try:
        cfg = load_effects_config()
        effects_data = cfg.get("effects", {})
        mappers = {}
        for eff_name, eff_dict in effects_data.items():
            mappers[eff_name] = {}
            for p_name, p_info in eff_dict.get("parameters", {}).items():
                spec = EffectParameterSpec(
                    name=p_name,
                    normalized_min=p_info["normalized_range"][0],
                    normalized_max=p_info["normalized_range"][1],
                    physical_min=p_info["physical_range"][0],
                    physical_max=p_info["physical_range"][1],
                    unit=p_info.get("unit", ""),
                    mapping_type=p_info.get("mapping_type", "linear"),
                    required=p_info.get("required", True),
                )
                mappers[eff_name][p_name] = ParameterMapper(spec)
        return mappers
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Predict Audio Effects Chain & Parameter Values using trained InverseFXNet")
    parser.add_argument("--dry", type=str, required=True, help="Path to clean Dry audio WAV")
    parser.add_argument("--wet", type=str, required=True, help="Path to target Wet audio WAV")
    parser.add_argument("--effect", type=str, default=None, choices=SUPPORTED_EFFECTS + [e.lower() for e in SUPPORTED_EFFECTS],
                        help="Inspect specific effect parameter values (e.g. Chorus, Reverb, EQ)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold probability (default: 0.5)")
    args = parser.parse_args()

    dry_path = Path(args.dry)
    wet_path = Path(args.wet)
    ckpt_path = Path(args.checkpoint)

    if not dry_path.exists():
        print(f"Error: Dry audio not found: {dry_path}")
        return
    if not wet_path.exists():
        print(f"Error: Wet audio not found: {wet_path}")
        return
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return

    # Load audio
    dry_audio, sr = load_audio(dry_path)
    wet_audio, _ = load_audio(wet_path)

    # Use first 2 seconds for model inference
    max_len = sr * 2
    dry_in = dry_audio[..., :max_len].unsqueeze(0)
    wet_in = wet_audio[..., :max_len].unsqueeze(0)

    # Load model
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = InverseFXNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, pred_params = model(dry_in, wet_in)
        probs = torch.sigmoid(logits)[0]
        params = pred_params[0]

    mappers = get_physical_mappers()

    # Normalize effect argument if provided
    selected_effect = None
    if args.effect:
        for eff in SUPPORTED_EFFECTS:
            if eff.lower() == args.effect.lower():
                selected_effect = eff
                break

    print("=" * 70)
    print("        INVERSE FX CHAIN PREDICTOR: INFERENCE REPORT")
    print("=" * 70)
    print(f"Dry Audio : {dry_path.resolve()}")
    print(f"Wet Audio : {wet_path.resolve()}")
    print("-" * 70)

    # -------------------------------------------------------------
    # 1. SPECIFIC EFFECT PARAMETER INSPECTION (e.g. CHORUS)
    # -------------------------------------------------------------
    if selected_effect:
        e_idx = SUPPORTED_EFFECTS.index(selected_effect)
        eff_prob = probs[e_idx].item()
        is_active = eff_prob >= args.threshold
        status_str = "ACTIVE (Predicted ON)" if is_active else "BYPASSED (Predicted OFF)"

        print(f"\n>>> [{selected_effect.upper()} PARAMETER VALUE PREDICTION] <<<")
        print(f"  Detection Probability : {eff_prob * 100:.2f}%  --> {status_str}")
        print(f"  Decision Threshold    : {args.threshold * 100:.1f}%\n")

        start_idx, end_idx = PARAM_OFFSETS[selected_effect]
        param_names = PARAM_SCHEMA[selected_effect]
        eff_params_norm = params[start_idx:end_idx]

        print(f"  {'Parameter':<16} | {'Normalized [0, 1]':<18} | {'Physical DSP Value'}")
        print("  " + "-" * 58)

        for p_name, p_norm_tensor in zip(param_names, eff_params_norm):
            p_norm = p_norm_tensor.item()
            # Convert to physical if mapper available
            phys_str = "N/A"
            if selected_effect in mappers and p_name in mappers[selected_effect]:
                mapper = mappers[selected_effect][p_name]
                phys_val = mapper.to_physical(torch.tensor(p_norm)).item()
                unit = getattr(mapper._spec, "unit", "")
                phys_str = f"{phys_val:.3f} {unit}".strip()

            print(f"  {p_name:<16} | {p_norm:>10.4f}           | {phys_str}")

        print("=" * 70)
        return

    # -------------------------------------------------------------
    # 2. COMPLETE AUDIO CHAIN PREDICTION
    # -------------------------------------------------------------
    print("\n[1] ALL EFFECT DETECTION PROBABILITIES:")
    print(f"  {'Effect':<14} | {'Confidence':<12} | {'Decision':<14}")
    print("  " + "-" * 46)

    active_effects = []
    for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
        p = probs[e_idx].item()
        if p >= args.threshold:
            status = "★ ACTIVE"
            active_effects.append((effect_name, p))
        else:
            status = "  bypassed"
        print(f"  {effect_name:<14} | {p * 100:>8.2f}%    | {status}")

    print("\n[2] PREDICTED AUDIO CHAIN SEQUENCE:")
    if active_effects:
        chain_str = " -> ".join([eff[0] for eff in active_effects])
        print(f"  Audio Chain: [ {chain_str} ]")
    else:
        print("  Audio Chain: [ None (No effects detected above threshold) ]")

    print("\n[3] PREDICTED KNOB PARAMETERS FOR ACTIVE EFFECTS:")
    for effect_name, conf in active_effects:
        print(f"\n  [{effect_name}] (Confidence: {conf * 100:.1f}%):")
        start_idx, end_idx = PARAM_OFFSETS[effect_name]
        param_names = PARAM_SCHEMA[effect_name]
        for p_name, p_norm_tensor in zip(param_names, params[start_idx:end_idx]):
            p_norm = p_norm_tensor.item()
            phys_str = ""
            if effect_name in mappers and p_name in mappers[effect_name]:
                mapper = mappers[effect_name][p_name]
                phys_val = mapper.to_physical(torch.tensor(p_norm)).item()
                unit = getattr(mapper._spec, "unit", "")
                phys_str = f"({phys_val:.2f} {unit})"
            print(f"    • {p_name:<14}: {p_norm:.4f}  {phys_str}")

    print("=" * 70)


if __name__ == "__main__":
    main()
