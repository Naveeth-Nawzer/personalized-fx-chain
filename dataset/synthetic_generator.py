"""
dataset/synthetic_generator.py
==============================
Generates synthetic audio training pairs using MUSDB18-HQ clean stems and DAPN.

For each sample:
  1. Picks a random track and stem (vocals, drums, bass, other) from MUSDB18-HQ.
  2. Extracts a fixed-duration slice (e.g., 2.0 seconds).
  3. Rejects silent segments (RMS < -45 dBFS).
  4. Generates a random valid DAPN FX chain (1 to 3 effects) with random parameters.
  5. Renders the wet audio using DAPN.
  6. Saves dry/wet audio files and appends the ground truth entry to metadata.jsonl.

Usage:
------
python3 dataset/synthetic_generator.py --num_samples 50 --split train
"""

import argparse
import json
import math
import random
import sys
import wave
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import save_audio
from common.config_loader import load_effects_config, get_sample_rate
from common.parameter_mapper import ParameterMapper
from common.audio_types import EffectParameterSpec
from dapn.dapn import DAPN

# Stems suitable for dry processing
STEM_NAMES = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

# List of 7 active audio effects in DAPN
SUPPORTED_EFFECTS = [
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "Distortion",
    "Chorus",
    "Flanger",
]


def load_audio_slice(wav_path: Path, offset_seconds: float, duration_seconds: float, target_sr: int) -> torch.Tensor | None:
    """Reads a slice of audio from a WAV file directly without loading the entire file."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()

            start_frame = int(offset_seconds * sr)
            num_frames = int(duration_seconds * sr)

            if start_frame + num_frames > total_frames:
                return None

            wf.setpos(start_frame)
            raw_bytes = wf.readframes(num_frames)

            # Convert to float32
            byte_arr = bytearray(raw_bytes)
            if sampwidth == 2:
                # 16-bit PCM
                samples = torch.frombuffer(byte_arr, dtype=torch.int16).float() / 32768.0
            elif sampwidth == 3:
                # 24-bit PCM
                int_samples = []
                for i in range(0, len(raw_bytes), 3):
                    chunk = raw_bytes[i : i + 3] + (b"\x80" if raw_bytes[i + 2] & 0x80 else b"\x00")
                    val = int.from_bytes(chunk, byteorder="little", signed=True) >> 8
                    int_samples.append(val)
                samples = torch.tensor(int_samples, dtype=torch.float32) / 8388608.0
            elif sampwidth == 4:
                # 32-bit float or int
                samples = torch.frombuffer(byte_arr, dtype=torch.int32).float() / 2147483648.0
            else:
                return None

            # Reshape to (channels, samples)
            samples = samples.view(-1, n_channels).t()

            # Downmix to mono: shape (1, T)
            mono = samples.mean(dim=0, keepdim=True)
            if sr != target_sr:
                import torchaudio
                mono = torchaudio.functional.resample(mono, orig_freq=sr, new_freq=target_sr)
            return mono
    except Exception:
        return None


def generate_random_chain(effects_config: dict, mappers: dict, min_len: int = 1, max_len: int = 3) -> tuple[dict, list[dict]]:
    """Generates a random valid FX chain specification and detailed label structure."""
    chain_length = random.randint(min_len, max_len)
    chosen_effects = random.sample(SUPPORTED_EFFECTS, chain_length)

    chain_dict = {"chain": []}
    effects_meta = []

    for order_idx, fx_name in enumerate(chosen_effects):
        fx_cfg = effects_config["effects"][fx_name]["parameters"]
        step_params_norm = {}
        step_params_phys = {}

        for param_name in fx_cfg.keys():
            # Random normalized value in [0.05, 0.95] to avoid extreme boundary artifacts
            norm_val = round(random.uniform(0.05, 0.95), 4)
            step_params_norm[param_name] = norm_val

            # Compute physical value
            mapper = mappers[fx_name][param_name]
            phys_val = float(mapper.to_physical(torch.tensor(norm_val)).item())
            step_params_phys[param_name] = round(phys_val, 3)

        chain_dict["chain"].append({
            "effect": fx_name,
            "parameters": step_params_norm,
        })

        effects_meta.append({
            "order": order_idx,
            "effect": fx_name,
            "parameters_normalized": step_params_norm,
            "parameters_physical": step_params_phys,
        })

    return chain_dict, effects_meta


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DAPN dataset from MUSDB18-HQ")
    parser.add_argument("--musdb_dir", type=str, default="musdb18hq", help="Path to musdb18hq root")
    parser.add_argument("--output_dir", type=str, default="dataset/synthetic", help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each slice in seconds")
    parser.add_argument("--exclude_vocals", action="store_true", help="Exclude vocal stems and use only instruments (drums, bass, other/guitars/keys)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    active_stems = ["drums.wav", "bass.wav", "other.wav"] if args.exclude_vocals else STEM_NAMES

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    musdb_path = Path(args.musdb_dir).resolve()
    # In musdb18hq, the folders are 'train' and 'test'
    source_split = "train" if args.split in ["train", "val"] else "test"
    split_dir = musdb_path / source_split

    if not split_dir.exists():
        raise FileNotFoundError(f"MUSDB18-HQ split directory not found: {split_dir}")

    # Discover track folders
    tracks = [d for d in split_dir.iterdir() if d.is_dir()]
    if not tracks:
        raise ValueError(f"No tracks found in {split_dir}")

    # If generating val split, use the last 15 tracks of the train set
    if args.split == "val":
        tracks = tracks[-15:]
    elif args.split == "train":
        tracks = tracks[:-15]

    print(f"[{args.split.upper()} Split] Found {len(tracks)} source tracks in {split_dir}")

    # Prepare output directories
    out_dir = Path(args.output_dir) / args.split
    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = out_dir / "metadata.jsonl"

    effects_config = load_effects_config()
    target_sr = get_sample_rate()
    dapn = DAPN()

    # Pre-build parameter mappers for fast label conversion
    mappers = {}
    for fx_name in SUPPORTED_EFFECTS:
        mappers[fx_name] = {}
        for p_name, p_spec in effects_config["effects"][fx_name]["parameters"].items():
            spec = EffectParameterSpec(
                name=p_name,
                normalized_min=float(p_spec["normalized_range"][0]),
                normalized_max=float(p_spec["normalized_range"][1]),
                physical_min=float(p_spec["physical_range"][0]),
                physical_max=float(p_spec["physical_range"][1]),
                unit=p_spec.get("unit", ""),
                mapping_type=p_spec.get("mapping_type", "linear"),
                required=p_spec.get("required", True),
            )
            mappers[fx_name][p_name] = ParameterMapper(spec)

    print(f"Generating {args.num_samples} synthetic samples...")
    generated_count = 0
    attempts = 0
    max_attempts = args.num_samples * 10

    with open(metadata_file, "w", encoding="utf-8") as meta_fp:
        while generated_count < args.num_samples and attempts < max_attempts:
            attempts += 1
            track = random.choice(tracks)
            stem_name = random.choice(active_stems)
            stem_path = track / stem_name

            if not stem_path.exists():
                continue

            # Pick a random offset between 10s and 120s into the track
            offset = random.uniform(10.0, 120.0)
            dry_slice = load_audio_slice(stem_path, offset, args.duration, target_sr)

            if dry_slice is None:
                continue

            # Silence rejection: check RMS level (must be > -45 dBFS)
            rms = float(dry_slice.pow(2).mean().sqrt().item())
            if rms < 0.0056:  # ~ -45 dBFS
                continue

            # Generate random FX chain
            chain_dict, effects_meta = generate_random_chain(effects_config, mappers)

            # Process through DAPN
            try:
                with torch.no_grad():
                    wet_slice = dapn.process(dry_slice, chain_dict)
            except Exception as e:
                continue

            if not torch.isfinite(wet_slice).all():
                continue

            # File naming: sample_000001_dry.wav / sample_000001_wet.wav
            sample_id = f"sample_{generated_count + 1:06d}"
            dry_filename = f"{sample_id}_dry.wav"
            wet_filename = f"{sample_id}_wet.wav"

            dry_path = audio_out_dir / dry_filename
            wet_path = audio_out_dir / wet_filename

            save_audio(dry_slice, dry_path, sample_rate=target_sr)
            save_audio(wet_slice, wet_path, sample_rate=target_sr)

            # Multi-hot vector for presence detection: [EQ, Comp, Delay, Reverb, Dist, Chorus, Flanger]
            presence_multihot = [
                1 if fx in [m["effect"] for m in effects_meta] else 0
                for fx in SUPPORTED_EFFECTS
            ]

            record = {
                "sample_id": sample_id,
                "split": args.split,
                "source_track": track.name,
                "source_stem": stem_name,
                "dry_path": str(dry_path.resolve().relative_to(ROOT.resolve())),
                "wet_path": str(wet_path.resolve().relative_to(ROOT.resolve())),
                "sample_rate": target_sr,
                "duration_seconds": args.duration,
                "chain_sequence": [m["effect"] for m in effects_meta],
                "chain_length": len(effects_meta),
                "effects": effects_meta,
                "ml_targets": {
                    "presence_multihot": presence_multihot,
                    "supported_effects": SUPPORTED_EFFECTS,
                },
            }

            meta_fp.write(json.dumps(record) + "\n")
            generated_count += 1

            if generated_count % 10 == 0 or generated_count == args.num_samples:
                print(f"  [{generated_count}/{args.num_samples}] samples generated...")

    print(f"\nDone! Successfully generated {generated_count} samples in: {out_dir}")
    print(f"Metadata index saved to: {metadata_file}\n")


if __name__ == "__main__":
    main()
