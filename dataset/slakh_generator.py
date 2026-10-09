"""
dataset/slakh_generator.py
==========================
Generates synthetic audio training pairs using Slakh2100-redux clean stems
and the DAPN FX chain — a drop-in replacement for synthetic_generator.py
that sources audio from Slakh instead of MUSDB18-HQ.

Key differences vs. synthetic_generator.py
-------------------------------------------
  - Data source  : Slakh2100-redux FLAC stems (not MUSDB18-HQ WAV stems)
  - Stem discovery: Reads each track's metadata.yaml to list only stems
                    where audio_rendered == true.
  - Split mapping : Slakh has its own train / validation / test folders.
                    The 'omitted/' folder is intentionally excluded (duplicate MIDIs).
  - Audio loading : Pure-Python FLAC header parser extracts a frame-accurate
                    slice without loading the entire file; falls back to a full
                    read + slice when the seek path is unavailable.
  - Instrument filter: Skip 'Sound Effects' / 'Sound effects' stems by default
                    (configurable via --inst_classes).
  - All other behaviour (FX chain generation, silence rejection, output schema,
    metadata.jsonl format) is identical to synthetic_generator.py so that
    dapn_dataset.py / train.py require zero changes.

Usage
-----
# Training split — 5 000 samples
python3 dataset/slakh_generator.py \\
    --slakh_dir   newData/slakh2100_flac_redux \\
    --output_dir  dataset/synthetic_slakh \\
    --split       train \\
    --num_samples 5000 \\
    --duration    2.0  \\
    --seed        42

# Validation split — 1 000 samples
python3 dataset/slakh_generator.py \\
    --slakh_dir   newData/slakh2100_flac_redux \\
    --output_dir  dataset/synthetic_slakh \\
    --split       val \\
    --num_samples 1000 \\
    --duration    2.0  \\
    --seed        123

# Test split — 500 samples
python3 dataset/slakh_generator.py \\
    --slakh_dir   newData/slakh2100_flac_redux \\
    --output_dir  dataset/synthetic_slakh \\
    --split       test \\
    --num_samples 500  \\
    --duration    2.0  \\
    --seed        999
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Add project root to Python path so all internal imports resolve correctly.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from audio_io import save_audio
from common.config_loader import load_effects_config, get_sample_rate
from common.parameter_mapper import ParameterMapper
from common.audio_types import EffectParameterSpec
from dapn.dapn import DAPN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Slakh split folder names. 'omitted' is intentionally excluded — it contains
# tracks with duplicate MIDI files that can cause train/val/test data leakage.
SLAKH_SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "val":   "validation",   # Slakh uses 'validation', not 'val'
    "test":  "test",
}

# Instrument classes to include by default.
# 'Sound Effects' / 'Sound effects' are excluded — they are not musical.
DEFAULT_INST_CLASSES: set[str] = {
    "Guitar",
    "Piano",
    "Bass",
    "Drums",
    "Strings",
    "Strings (continued)",
    "Brass",
    "Reed",
    "Organ",
    "Pipe",
    "Synth Lead",
    "Synth Pad",
    "Chromatic Percussion",
    "Ethnic",
    "Percussive",
}

# 7 effects supported by DAPN and InverseFXNet
SUPPORTED_EFFECTS: list[str] = [
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "Distortion",
    "Chorus",
    "Flanger",
]

# Silence threshold: RMS below this value (~-45 dBFS) → reject slice
RMS_SILENCE_THRESHOLD: float = 0.0056

# Minimum track offset (seconds) before slicing — avoids silence at track start
MIN_OFFSET_SECONDS: float = 10.0

# Maximum track offset (seconds) — avoids over-sampling track endings
MAX_OFFSET_SECONDS: float = 240.0


# ---------------------------------------------------------------------------
# Stem catalogue builder
# ---------------------------------------------------------------------------

def build_stem_catalogue(
    slakh_split_dir: Path,
    allowed_inst_classes: set[str],
) -> list[dict]:
    """Walk all track folders in a Slakh split and return a flat list of
    usable stem descriptors.

    Each descriptor is a dict with:
        track_dir  : Path  — absolute path to the TrackXXXX folder
        stem_id    : str   — e.g. 'S00'
        flac_path  : Path  — absolute path to stems/SXX.flac
        inst_class : str   — e.g. 'Guitar'
        is_drum    : bool
        program    : str   — MIDI program name

    Only stems where ``audio_rendered == True`` and whose ``inst_class``
    is in *allowed_inst_classes* are included.
    """
    catalogue: list[dict] = []
    skipped_unrendered = 0
    skipped_class = 0
    skipped_missing_file = 0

    track_dirs = sorted(p for p in slakh_split_dir.iterdir() if p.is_dir())

    for track_dir in track_dirs:
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue

        stems_info: dict = meta.get("stems", {})

        for stem_id, stem_data in stems_info.items():
            # ── Rule 1: must be rendered (FLAC file exists)
            if not stem_data.get("audio_rendered", False):
                skipped_unrendered += 1
                continue

            # ── Rule 2: instrument class must be in the allowed set
            inst_class: str = stem_data.get("inst_class", "Unknown")
            if inst_class not in allowed_inst_classes:
                skipped_class += 1
                continue

            # ── Rule 3: actual FLAC file must exist on disk
            flac_path = track_dir / "stems" / f"{stem_id}.flac"
            if not flac_path.exists():
                skipped_missing_file += 1
                continue

            catalogue.append({
                "track_dir":  track_dir,
                "stem_id":    stem_id,
                "flac_path":  flac_path,
                "inst_class": inst_class,
                "is_drum":    stem_data.get("is_drum", False),
                "program":    stem_data.get("midi_program_name", "Unknown"),
            })

    print(
        f"  Stem catalogue: {len(catalogue)} usable stems  "
        f"(skipped — unrendered: {skipped_unrendered}, "
        f"wrong class: {skipped_class}, "
        f"missing file: {skipped_missing_file})"
    )
    return catalogue


# ---------------------------------------------------------------------------
# FLAC audio slice loader
# ---------------------------------------------------------------------------

def _parse_flac_streaminfo(path: Path) -> tuple[int, int, int] | None:
    """Parse a FLAC STREAMINFO block and return (sample_rate, n_channels,
    total_samples).  Returns None on any parse error."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"fLaC":
                return None
            while True:
                header = f.read(4)
                if len(header) < 4:
                    break
                is_last   = bool(header[0] & 0x80)
                block_type = header[0] & 0x7F
                length = (header[1] << 16) | (header[2] << 8) | header[3]
                if block_type == 0:  # STREAMINFO
                    data = f.read(length)
                    if len(data) < 18:
                        break
                    # Bits 80–99  : sample rate  (20 bits)
                    sample_rate = (
                        (data[10] << 12) | (data[11] << 4) | (data[12] >> 4)
                    )
                    # Bits 100–102: channels - 1  (3 bits)
                    n_channels = ((data[12] & 0x0E) >> 1) + 1
                    # Bits 108–143: total samples (36 bits)
                    total_samples = (
                        ((data[13] & 0x0F) << 32)
                        | (data[14] << 24)
                        | (data[15] << 16)
                        | (data[16] << 8)
                        | data[17]
                    )
                    if sample_rate > 0:
                        return sample_rate, n_channels, total_samples
                    break
                else:
                    f.seek(length, 1)
                if is_last:
                    break
    except Exception:
        pass
    return None


def load_flac_slice(
    flac_path: Path,
    offset_seconds: float,
    duration_seconds: float,
    target_sr: int,
) -> Optional[torch.Tensor]:
    """Load a mono float32 audio slice from a FLAC file.

    Uses a three-level fallback so generation works regardless of which
    audio codec backends are installed:

    Level 1 — ``torchaudio.load`` with ``frame_offset`` + ``num_frames``
               (fast, zero-copy seek; requires torchcodec or sox backend).
    Level 2 — ``soundfile`` seek-read
               (fast, requires libsndfile; usually available via pip).
    Level 3 — ``torchaudio.load`` full file + Python slice
               (always works if torchaudio is installed; slightly slower
               because the entire file is decoded first).

    Returns ``None`` on any unrecoverable error so the generation loop can
    skip and pick another stem.

    Output shape: ``(1, T)`` float32 in ``[-1, 1]``.
    """
    # ── Pre-check: validate requested window fits inside the file
    info = _parse_flac_streaminfo(flac_path)
    if info is not None:
        sr_file, _, total_samples = info
        start_frame = int(offset_seconds * sr_file)
        num_frames  = int(duration_seconds * sr_file)
        if start_frame + num_frames > total_samples:
            return None  # slice exceeds file length — caller will retry

    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """Downmix (C, T) to (1, T) if needed."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    sr_hint = info[0] if info else target_sr

    # ── Level 1: torchaudio with seeking (fast path)
    try:
        import torchaudio
        waveform, _ = torchaudio.load(
            str(flac_path),
            frame_offset=int(offset_seconds * sr_hint),
            num_frames=int(duration_seconds * sr_hint),
        )
        return _to_mono(waveform)
    except Exception:
        pass  # fall through to next backend

    # ── Level 2: soundfile seek-read (medium path)
    try:
        import soundfile as sf
        start_frame = int(offset_seconds * sr_hint)
        num_frames  = int(duration_seconds * sr_hint)
        data, _ = sf.read(
            str(flac_path),
            frames=num_frames,
            start=start_frame,
            dtype="float32",
            always_2d=True,
        )
        # data: (T, C) → transpose to (C, T)
        waveform = torch.from_numpy(data.T)
        return _to_mono(waveform)
    except Exception:
        pass  # fall through to final backend

    # ── Level 3: torchaudio full-file load + Python slice (slow fallback)
    try:
        import torchaudio
        waveform, sr_loaded = torchaudio.load(str(flac_path))
        start = int(offset_seconds * sr_loaded)
        end   = start + int(duration_seconds * sr_loaded)
        if end > waveform.shape[-1]:
            return None
        waveform = waveform[:, start:end]
        return _to_mono(waveform)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FX chain generator  (identical logic to synthetic_generator.py)
# ---------------------------------------------------------------------------

def generate_random_chain(
    effects_config: dict,
    mappers: dict,
    min_len: int = 1,
    max_len: int = 3,
) -> tuple[dict, list[dict]]:
    """Generate a random valid DAPN FX chain with normalised + physical labels."""
    chain_length = random.randint(min_len, max_len)
    chosen_effects = random.sample(SUPPORTED_EFFECTS, chain_length)

    chain_dict: dict = {"chain": []}
    effects_meta: list[dict] = []

    for order_idx, fx_name in enumerate(chosen_effects):
        fx_cfg = effects_config["effects"][fx_name]["parameters"]
        step_params_norm: dict = {}
        step_params_phys: dict = {}

        for param_name in fx_cfg.keys():
            # Sample in [0.05, 0.95] to avoid boundary DSP artefacts
            norm_val = round(random.uniform(0.05, 0.95), 4)
            step_params_norm[param_name] = norm_val

            mapper = mappers[fx_name][param_name]
            phys_val = float(mapper.to_physical(torch.tensor(norm_val)).item())
            step_params_phys[param_name] = round(phys_val, 3)

        chain_dict["chain"].append({
            "effect":     fx_name,
            "parameters": step_params_norm,
        })

        effects_meta.append({
            "order":                 order_idx,
            "effect":                fx_name,
            "parameters_normalized": step_params_norm,
            "parameters_physical":   step_params_phys,
        })

    return chain_dict, effects_meta


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DAPN training pairs from Slakh2100-redux"
    )
    parser.add_argument(
        "--slakh_dir",
        type=str,
        default="newData/slakh2100_flac_redux",
        help="Path to the slakh2100_flac_redux root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/synthetic_slakh",
        help="Root output directory (split sub-folder is added automatically)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to generate: train | val | test",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of dry/wet pairs to generate",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of each audio slice in seconds",
    )
    parser.add_argument(
        "--min_chain",
        type=int,
        default=1,
        help="Minimum number of effects in a random FX chain",
    )
    parser.add_argument(
        "--max_chain",
        type=int,
        default=3,
        help="Maximum number of effects in a random FX chain",
    )
    parser.add_argument(
        "--inst_classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Whitelist of Slakh inst_class values to sample from. "
            "Defaults to all musical classes (excludes Sound Effects)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── Resolve Slakh split folder
    slakh_split_name = SLAKH_SPLIT_MAP[args.split]
    slakh_root = Path(args.slakh_dir).resolve()
    slakh_split_dir = slakh_root / slakh_split_name

    if not slakh_split_dir.exists():
        raise FileNotFoundError(
            f"Slakh split directory not found: {slakh_split_dir}\n"
            f"  Expected structure: {slakh_root}/<train|validation|test>/TrackXXXXX/"
        )

    # ── Instrument class filter
    allowed_classes: set[str] = (
        set(args.inst_classes) if args.inst_classes else DEFAULT_INST_CLASSES
    )

    # ── Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Output paths
    out_dir       = Path(args.output_dir) / args.split
    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = out_dir / "metadata.jsonl"

    # ── Print run configuration
    print("=" * 68)
    print(" Slakh2100-redux → DAPN Synthetic Dataset Generator")
    print("=" * 68)
    print(f"  Slakh source  : {slakh_split_dir}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Split         : {args.split}  (Slakh folder: '{slakh_split_name}')")
    print(f"  Samples       : {args.num_samples}")
    print(f"  Duration      : {args.duration}s per sample")
    print(f"  FX chain len  : {args.min_chain}–{args.max_chain} effects")
    print(f"  Seed          : {args.seed}")
    print()

    # ── Build stem catalogue
    print("[1/4] Scanning Slakh stems ...")
    stem_catalogue = build_stem_catalogue(slakh_split_dir, allowed_classes)
    if not stem_catalogue:
        raise ValueError(
            "No usable stems found. Check --slakh_dir and --inst_classes."
        )
    print(f"  Ready: {len(stem_catalogue)} stems across "
          f"{len({s['track_dir'] for s in stem_catalogue})} tracks\n")

    # ── Load DAPN and parameter mappers
    print("[2/4] Loading DAPN and effects config ...")
    effects_config = load_effects_config()
    target_sr      = get_sample_rate()
    dapn           = DAPN()

    mappers: dict = {}
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
    print("  DAPN ready.\n")

    # ── Generation loop
    print(f"[3/4] Generating {args.num_samples} samples ...")
    generated_count = 0
    attempts        = 0
    max_attempts    = args.num_samples * 20   # generous budget (Slakh has lots of stems)

    # Track rejection reasons for final diagnostics
    reject_eof     = 0  # slice extends beyond file length
    reject_silence = 0  # RMS below threshold
    reject_dapn    = 0  # DAPN processing error or NaN output

    with open(metadata_file, "w", encoding="utf-8") as meta_fp:
        while generated_count < args.num_samples and attempts < max_attempts:
            attempts += 1

            # ── Pick a random stem
            stem = random.choice(stem_catalogue)
            flac_path: Path = stem["flac_path"]

            # ── Random time offset
            offset = random.uniform(MIN_OFFSET_SECONDS, MAX_OFFSET_SECONDS)

            # ── Load audio slice from FLAC
            dry_slice = load_flac_slice(flac_path, offset, args.duration, target_sr)
            if dry_slice is None:
                reject_eof += 1
                continue

            # ── Silence rejection (~-45 dBFS threshold)
            rms = float(dry_slice.pow(2).mean().sqrt().item())
            if rms < RMS_SILENCE_THRESHOLD:
                reject_silence += 1
                continue

            # ── Generate a random FX chain
            chain_dict, effects_meta = generate_random_chain(
                effects_config, mappers,
                min_len=args.min_chain,
                max_len=args.max_chain,
            )

            # ── Apply the chain via DAPN
            try:
                with torch.no_grad():
                    wet_slice = dapn.process(dry_slice, chain_dict)
            except Exception:
                reject_dapn += 1
                continue

            # Guard against NaN / Inf from degenerate DSP
            if not torch.isfinite(wet_slice).all():
                reject_dapn += 1
                continue

            # ── Save dry and wet WAV files
            sample_id    = f"sample_{generated_count + 1:06d}"
            dry_filename = f"{sample_id}_dry.wav"
            wet_filename = f"{sample_id}_wet.wav"

            dry_path = audio_out_dir / dry_filename
            wet_path = audio_out_dir / wet_filename

            save_audio(dry_slice, dry_path, sample_rate=target_sr)
            save_audio(wet_slice, wet_path, sample_rate=target_sr)

            # ── Build multi-hot presence vector: [EQ, Comp, Delay, Reverb, Dist, Chorus, Flanger]
            active_fx_names = {m["effect"] for m in effects_meta}
            presence_multihot = [
                1 if fx in active_fx_names else 0
                for fx in SUPPORTED_EFFECTS
            ]

            # ── Write metadata record (identical schema to synthetic_generator.py)
            record = {
                "sample_id":      sample_id,
                "split":          args.split,
                # Slakh-specific provenance fields
                "source_dataset": "slakh2100_redux",
                "source_track":   stem["track_dir"].name,
                "source_stem":    stem["stem_id"],
                "inst_class":     stem["inst_class"],
                "is_drum":        stem["is_drum"],
                "midi_program":   stem["program"],
                # Audio
                "dry_path":       str(dry_path.resolve().relative_to(ROOT.resolve())),
                "wet_path":       str(wet_path.resolve().relative_to(ROOT.resolve())),
                "sample_rate":    target_sr,
                "duration_seconds": args.duration,
                # FX chain
                "chain_sequence": [m["effect"] for m in effects_meta],
                "chain_length":   len(effects_meta),
                "effects":        effects_meta,
                # ML targets — same keys as DAPNSyntheticDataset expects
                "ml_targets": {
                    "presence_multihot": presence_multihot,
                    "supported_effects": SUPPORTED_EFFECTS,
                },
            }

            meta_fp.write(json.dumps(record) + "\n")
            generated_count += 1

            # Progress print every 50 samples (or on completion)
            if generated_count % 50 == 0 or generated_count == args.num_samples:
                print(
                    f"  [{generated_count:>{len(str(args.num_samples))}}/{args.num_samples}]"
                    f"  attempts={attempts}"
                    f"  reject(eof={reject_eof}, silence={reject_silence}, dapn={reject_dapn})"
                )

    # ── Final summary
    print()
    print("[4/4] Done!")
    print("=" * 68)
    print(f"  Generated     : {generated_count} / {args.num_samples} samples")
    print(f"  Total attempts: {attempts}")
    print(f"  Rejections    : eof={reject_eof}  silence={reject_silence}  dapn={reject_dapn}")
    print(f"  Output audio  : {audio_out_dir}")
    print(f"  Metadata index: {metadata_file}")
    print("=" * 68)

    if generated_count < args.num_samples:
        print(
            f"\n  ⚠ WARNING: Only {generated_count} samples generated "
            f"(target was {args.num_samples}).\n"
            f"  Try increasing --slakh_dir coverage, adjusting --inst_classes,\n"
            f"  or reducing --duration."
        )

    print(
        f"\n  To train, run:\n"
        f"    python3 train.py \\\n"
        f"      --train_metadata dataset/synthetic_slakh/train/metadata.jsonl \\\n"
        f"      --val_metadata   dataset/synthetic_slakh/val/metadata.jsonl\n"
    )


if __name__ == "__main__":
    main()
