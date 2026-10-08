"""Audio I/O and dry-clip preprocessing (same steps as the Phase 1 project code).

Convention: audio arrays are ``float32`` shaped ``(channels, samples)``.

Dry clip pipeline:
    load -> resample (only if needed) -> channel conversion
    -> pick a non-silent window of ``duration`` seconds (seeded)
    -> loudness / peak normalisation
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyloudnorm
import soundfile as sf
from pedalboard.io import AudioFile
from scipy.signal import resample_poly

_EPS = 1e-12


class SourceRejected(ValueError):
    """The source file cannot produce a valid clip (too short, silent, unreadable...)."""


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


_BIT_DEPTHS = {"FLOAT": 32, "PCM_24": 24, "PCM_16": 16}


def write_audio(path: Path, audio: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> None:
    """Write a WAV with Pedalboard's writer: unlike libsndfile it adds no timestamped
    PEAK chunk, so the same audio always gives byte-identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(path), "w", sample_rate, audio.shape[0], bit_depth=_BIT_DEPTHS[subtype]) as fh:
        fh.write(np.asarray(audio, dtype=np.float32))


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g, axis=-1).astype(np.float32)


def to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Mono = mean downmix; stereo = duplicate mono / keep the first two channels."""
    if audio.shape[0] == channels:
        return audio
    if channels == 1:
        return audio.mean(axis=0, keepdims=True).astype(np.float32)
    if channels == 2:
        return np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2]
    raise ValueError(f"Unsupported channel count: {channels}")


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def amplitude_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, _EPS)))


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def residual_db(reference: np.ndarray, processed: np.ndarray) -> float:
    """Energy of ``processed - reference`` relative to ``reference`` (dB). Very negative = almost no change."""
    ref = reference.astype(np.float64)
    diff = processed.astype(np.float64) - ref
    return float(10.0 * np.log10((np.sum(diff**2) + _EPS) / (np.sum(ref**2) + _EPS)))


def frame_rms_db(mono: np.ndarray, frame_length: int) -> np.ndarray:
    n_frames = mono.shape[-1] // frame_length
    if n_frames == 0:
        return np.empty(0)
    frames = mono[: n_frames * frame_length].reshape(n_frames, frame_length).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), _EPS))


@dataclass(frozen=True)
class DryClip:
    audio: np.ndarray
    sample_rate: int
    info: dict[str, Any]  # what preprocessing did (stored in metadata)


def prepare_dry_clip(path: Path, audio_cfg: Mapping[str, Any], pre: Mapping[str, Any],
                     rng: np.random.Generator) -> DryClip:
    """Turn a full source file into a fixed-length, normalised dry clip.

    Raises:
        SourceRejected: if the file cannot yield a valid clip.
    """
    sr = int(audio_cfg["sample_rate"])
    channels = int(audio_cfg["channels"])
    length = int(round(float(audio_cfg["duration"]) * sr))

    try:
        audio, orig_sr = load_audio(path)
    except Exception as exc:  # noqa: BLE001
        raise SourceRejected(f"cannot decode audio ({exc})") from exc
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise SourceRejected("empty file or NaN/inf samples")
    orig_channels, orig_frames = audio.shape
    audio = to_channels(resample(audio, orig_sr, sr), channels)
    if audio.shape[1] < length:
        raise SourceRejected(f"shorter than {audio_cfg['duration']} s")

    # Candidate windows and the fraction of non-silent frames in each.
    frame = int(pre["frame_length"])
    hop = max(int(round(float(pre["window_hop_seconds"]) * sr)), 1)
    starts = np.arange(0, audio.shape[1] - length + 1, hop)
    active = (frame_rms_db(audio.mean(axis=0), frame) >= float(pre["silence_threshold_dbfs"])).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(active)])
    first = np.ceil(starts / frame).astype(np.int64)
    last = np.minimum((starts + length) // frame, active.size)
    fractions = (cumulative[last] - cumulative[np.minimum(first, last)]) / np.maximum(last - first, 1)
    qualifying = np.flatnonzero(fractions >= float(pre["min_active_fraction"]))
    if qualifying.size == 0:
        raise SourceRejected(f"no {audio_cfg['duration']} s window with enough non-silent audio "
                             f"(best {fractions.max():.0%})")
    choice = int(qualifying[rng.integers(qualifying.size)])
    start = int(starts[choice])
    clip = np.ascontiguousarray(audio[:, start : start + length], dtype=np.float32)

    clip, loudness, gain_db = normalize(clip, sr, pre)
    info = {
        "original_sample_rate": orig_sr,
        "original_channels": orig_channels,
        "original_duration_s": round(orig_frames / orig_sr, 3),
        "clip_start_s": round(start / sr, 3),
        "clip_end_s": round((start + length) / sr, 3),
        "active_fraction": round(float(fractions[choice]), 4),
        "normalization": pre["normalization"],
        "input_loudness_lufs": None if loudness is None else round(loudness, 3),
        "normalization_gain_db": round(gain_db, 3),
        "dry_peak_dbfs": round(amplitude_to_db(peak(clip)), 3),
    }
    return DryClip(clip, sr, info)


def normalize(clip: np.ndarray, sr: int, pre: Mapping[str, Any]) -> tuple[np.ndarray, float | None, float]:
    """Returns ``(clip, measured_loudness_lufs_or_None, applied_gain_db)``."""
    mode = pre["normalization"]
    if mode == "none":
        return clip, None, 0.0
    clip_peak = peak(clip)
    if clip_peak <= 0.0:
        raise SourceRejected("clip is digital silence")
    if mode == "peak":
        gain_db = float(pre["target_peak_dbfs"]) - amplitude_to_db(clip_peak)
        return (clip * db_to_amplitude(gain_db)).astype(np.float32), None, gain_db
    loudness = float(pyloudnorm.Meter(sr).integrated_loudness(clip.T.astype(np.float64)))
    if not np.isfinite(loudness):
        raise SourceRejected("clip loudness is -inf (silent)")
    gain_db = min(float(pre["target_lufs"]) - loudness,
                  float(pre["peak_ceiling_dbfs"]) - amplitude_to_db(clip_peak))
    return (clip * db_to_amplitude(gain_db)).astype(np.float32), loudness, gain_db
