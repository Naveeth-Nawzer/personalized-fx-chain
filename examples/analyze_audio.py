"""
examples/analyze_audio.py
=========================
Extract comprehensive acoustic, dynamic, spectral, and temporal features
from any WAV audio file using pure PyTorch.

Usage:
------
python3 examples/analyze_audio.py --input audio/testFile.wav
python3 examples/analyze_audio.py --input audio/testFile.wav --output audio/testFile_analysis.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio


def extract_audio_features(audio: torch.Tensor, sr: int, file_path: Path) -> dict:
    """Extract comprehensive features from an audio tensor (C, T)."""
    eps = 1e-9
    C, T = audio.shape
    duration_s = T / sr

    # Work with mono downmix for spectral and global dynamics
    audio_mono = audio.mean(dim=0)  # Shape: (T,)

    # -------------------------------------------------------------
    # 1. Temporal & Dynamics Metrics
    # -------------------------------------------------------------
    peak_linear = float(audio_mono.abs().max().item())
    peak_dbfs = float(20.0 * math.log10(max(peak_linear, eps)))

    rms_linear = float(audio_mono.pow(2).mean().sqrt().item())
    rms_dbfs = float(20.0 * math.log10(max(rms_linear, eps)))

    crest_factor_db = peak_dbfs - rms_dbfs if rms_linear > 0 else 0.0
    dc_offset = float(audio_mono.mean().item())

    # Clipping detection (|x| >= 0.999)
    clipped_samples = int((audio_mono.abs() >= 0.999).sum().item())
    clipped_pct = float(clipped_samples / T * 100.0)

    # Zero Crossing Rate (ZCR)
    zero_crossings = int(((audio_mono[:-1] * audio_mono[1:]) < 0).sum().item())
    zcr_rate = float(zero_crossings / (T - 1)) if T > 1 else 0.0

    # Active / Silent ratio (threshold at -60 dBFS)
    silence_thresh = 10.0 ** (-60.0 / 20.0)  # ~0.001
    active_samples = int((audio_mono.abs() > silence_thresh).sum().item())
    active_pct = float(active_samples / T * 100.0)

    # -------------------------------------------------------------
    # 2. Spectral Analysis (via PyTorch STFT)
    # -------------------------------------------------------------
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft, dtype=audio.dtype)

    stft = torch.stft(
        audio_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    magnitudes = stft.abs()  # (freq_bins, frames), freq_bins = n_fft // 2 + 1 = 1025
    freq_bins = magnitudes.shape[0]

    # Frequency bin centers in Hz
    freqs = torch.linspace(0, sr / 2.0, freq_bins, dtype=audio.dtype)

    # Power spectrum
    power = magnitudes.pow(2)
    total_power = power.sum(dim=0, keepdim=True) + eps  # (1, frames)

    # Spectral Centroid (brightness)
    freqs_expanded = freqs.unsqueeze(1)  # (freq_bins, 1)
    centroid_per_frame = (power * freqs_expanded).sum(dim=0) / total_power.squeeze(0)
    spectral_centroid_hz = float(centroid_per_frame.mean().item())

    # Spectral Bandwidth / Spread
    bandwidth_per_frame = torch.sqrt(
        (power * (freqs_expanded - centroid_per_frame.unsqueeze(0)).pow(2)).sum(dim=0)
        / total_power.squeeze(0)
    )
    spectral_bandwidth_hz = float(bandwidth_per_frame.mean().item())

    # Spectral Rolloff (frequency where 85% of cumulative energy is reached)
    cumsum_power = power.cumsum(dim=0)
    rolloff_thresh = 0.85 * total_power
    rolloff_idx = (cumsum_power >= rolloff_thresh).int().argmax(dim=0)
    rolloff_hz_per_frame = freqs[rolloff_idx]
    spectral_rolloff_hz = float(rolloff_hz_per_frame.mean().item())

    # Spectral Flatness (Wiener entropy: Geometric Mean / Arithmetic Mean)
    power_clamped = power.clamp(min=eps)
    geom_mean = torch.exp(torch.log(power_clamped).mean(dim=0))
    arith_mean = power_clamped.mean(dim=0)
    spectral_flatness = float((geom_mean / (arith_mean + eps)).mean().item())

    # Frequency Band Energy Distribution (% of total energy)
    mean_power_spectrum = power.mean(dim=1)  # (freq_bins,)
    total_spectrum_power = float(mean_power_spectrum.sum().item()) + eps

    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 4000.0)
    high_mask = freqs >= 4000.0

    low_energy_pct = float(mean_power_spectrum[low_mask].sum().item() / total_spectrum_power * 100.0)
    mid_energy_pct = float(mean_power_spectrum[mid_mask].sum().item() / total_spectrum_power * 100.0)
    high_energy_pct = float(mean_power_spectrum[high_mask].sum().item() / total_spectrum_power * 100.0)

    # -------------------------------------------------------------
    # Assemble Results
    # -------------------------------------------------------------
    file_size_kb = round(file_path.stat().st_size / 1024.0, 2) if file_path.exists() else 0.0

    return {
        "file_metadata": {
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "file_size_kb": file_size_kb,
            "channels": C,
            "channel_layout": "Mono" if C == 1 else ("Stereo" if C == 2 else f"{C}-channels"),
            "sample_rate_hz": sr,
            "total_samples": T,
            "duration_seconds": round(duration_s, 3),
        },
        "dynamics_and_levels": {
            "peak_linear": round(peak_linear, 5),
            "peak_dbFS": round(peak_dbfs, 2),
            "rms_linear": round(rms_linear, 5),
            "rms_dbFS": round(rms_dbfs, 2),
            "crest_factor_db": round(crest_factor_db, 2),
            "dc_offset": round(dc_offset, 6),
            "clipped_samples_count": clipped_samples,
            "clipped_percentage": round(clipped_pct, 3),
            "active_audio_percentage": round(active_pct, 2),
            "zero_crossing_rate": round(zcr_rate, 4),
        },
        "spectral_characteristics": {
            "spectral_centroid_hz": round(spectral_centroid_hz, 1),
            "spectral_bandwidth_hz": round(spectral_bandwidth_hz, 1),
            "spectral_rolloff_85pct_hz": round(spectral_rolloff_hz, 1),
            "spectral_flatness": round(spectral_flatness, 5),
            "energy_distribution": {
                "low_sub_bass_under_250hz_pct": round(low_energy_pct, 2),
                "midrange_250_to_4000hz_pct": round(mid_energy_pct, 2),
                "high_presence_above_4000hz_pct": round(high_energy_pct, 2),
            },
        },
    }


def print_features(features: dict):
    meta = features["file_metadata"]
    dyn = features["dynamics_and_levels"]
    spec = features["spectral_characteristics"]
    dist = spec["energy_distribution"]

    print("\n" + "=" * 68)
    print("           COMPREHENSIVE AUDIO ANALYSIS REPORT")
    print("=" * 68)

    print(" 1. FILE METADATA")
    print(f"    File Name         : {meta['file_name']}")
    print(f"    Duration          : {meta['duration_seconds']} seconds")
    print(f"    Sample Rate       : {meta['sample_rate_hz']} Hz")
    print(f"    Channels          : {meta['channels']} ({meta['channel_layout']})")
    print(f"    Total Samples     : {meta['total_samples']:,}")
    print(f"    File Size         : {meta['file_size_kb']} KB")

    print("\n 2. DYNAMICS & SIGNAL LEVELS")
    print(f"    Peak Amplitude    : {dyn['peak_linear']} ({dyn['peak_dbFS']} dBFS)")
    print(f"    RMS Energy        : {dyn['rms_linear']} ({dyn['rms_dbFS']} dBFS)")
    print(f"    Dynamic Range     : {dyn['crest_factor_db']} dB (Crest Factor)")
    print(f"    DC Offset         : {dyn['dc_offset']}")
    print(f"    Clipping Status   : {dyn['clipped_samples_count']} clipped samples ({dyn['clipped_percentage']}%)")
    print(f"    Active Audio Ratio: {dyn['active_audio_percentage']}% active (> -60 dBFS)")
    print(f"    Zero Crossing Rate: {dyn['zero_crossing_rate']}")

    print("\n 3. SPECTRAL & FREQUENCY CHARACTERISTICS")
    print(f"    Spectral Centroid : {spec['spectral_centroid_hz']} Hz (Spectral Center/Brightness)")
    print(f"    Spectral Bandwidth: {spec['spectral_bandwidth_hz']} Hz (Spread around centroid)")
    print(f"    Spectral Rolloff  : {spec['spectral_rolloff_85pct_hz']} Hz (85% energy boundary)")
    print(f"    Spectral Flatness : {spec['spectral_flatness']} (0.0 = Pure Tones, 1.0 = White Noise)")
    print("    Frequency Band Energy Distribution:")
    print(f"      - Bass (< 250 Hz)       : {dist['low_sub_bass_under_250hz_pct']:>6.2f} %")
    print(f"      - Mids (250 - 4000 Hz)  : {dist['midrange_250_to_4000hz_pct']:>6.2f} %")
    print(f"      - Highs (> 4000 Hz)     : {dist['high_presence_above_4000hz_pct']:>6.2f} %")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Extract comprehensive details and features from an audio file")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path for extracted details")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    audio, sr = load_audio(input_path)
    features = extract_audio_features(audio, sr, input_path)

    print_features(features)

    # Save to JSON
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_details.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print(f"\n[Saved JSON Details Report]: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
