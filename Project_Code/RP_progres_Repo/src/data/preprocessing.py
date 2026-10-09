"""Standardisation of source audio into fixed-length dry clips.

Steps (each recorded in metadata):
    load -> resample (only if needed) -> channel conversion -> pick a non-silent
    window of ``duration`` seconds -> loudness/peak normalisation.

The window is chosen with an RNG seeded from the global seed and the file's
relative path, so a file always yields the same clip for a given seed,
regardless of selection order.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pyloudnorm

from src.data.discovery import AudioFileInfo
from src.utils.audio import amplitude_to_db, db_to_amplitude, frame_rms_db, load_audio, peak, resample, to_channels


class SourceRejected(ValueError):
    """The source file cannot produce a valid clip (too short, silent, unreadable...)."""


@dataclass(frozen=True)
class PreprocessingSettings:
    sample_rate: int
    channels: int
    duration: float
    normalization: str
    target_lufs: float
    target_peak_dbfs: float
    peak_ceiling_dbfs: float
    silence_threshold_dbfs: float
    min_active_fraction: float
    frame_length: int
    window_hop_seconds: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PreprocessingSettings":
        audio, pre = config["audio"], config["preprocessing"]
        return cls(
            sample_rate=int(audio["sample_rate"]),
            channels=int(audio["channels"]),
            duration=float(audio["duration"]),
            normalization=str(pre["normalization"]),
            target_lufs=float(pre["target_lufs"]),
            target_peak_dbfs=float(pre["target_peak_dbfs"]),
            peak_ceiling_dbfs=float(pre["peak_ceiling_dbfs"]),
            silence_threshold_dbfs=float(pre["silence_threshold_dbfs"]),
            min_active_fraction=float(pre["min_active_fraction"]),
            frame_length=int(pre["frame_length"]),
            window_hop_seconds=float(pre["window_hop_seconds"]),
        )

    @property
    def num_samples(self) -> int:
        return int(round(self.duration * self.sample_rate))


@dataclass(frozen=True)
class PreprocessingRecord:
    """What was done to a source file to obtain its dry clip."""

    original_sample_rate: int
    original_channels: int
    original_duration_s: float
    resampled: bool
    channel_conversion: str
    segment_start_s: float
    segment_end_s: float
    active_fraction: float
    normalization: str
    input_loudness_lufs: float | None
    normalization_gain_db: float
    dry_peak_dbfs: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSource:
    audio: np.ndarray  # (channels, samples) float32
    sample_rate: int
    record: PreprocessingRecord


def segment_seed(seed: int, rel_path: str) -> np.random.SeedSequence:
    """Seed for the clip-window choice of one file (independent of selection order)."""
    digest = int(hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.SeedSequence([seed, digest])


def window_activity(mono: np.ndarray, settings: PreprocessingSettings) -> tuple[np.ndarray, np.ndarray]:
    """Candidate window starts (samples) and the fraction of active frames in each window."""
    n = mono.shape[-1]
    length = settings.num_samples
    frame = settings.frame_length
    hop = max(int(round(settings.window_hop_seconds * settings.sample_rate)), 1)
    starts = np.arange(0, n - length + 1, hop)
    active = (frame_rms_db(mono, frame) >= settings.silence_threshold_dbfs).astype(np.int64)
    if active.size == 0 or starts.size == 0:
        return starts, np.zeros(starts.size)
    cumulative = np.concatenate([[0], np.cumsum(active)])
    first = np.ceil(starts / frame).astype(np.int64)
    last = np.minimum((starts + length) // frame, active.size)  # exclusive
    counts = np.maximum(last - first, 1)
    fractions = (cumulative[last] - cumulative[np.minimum(first, last)]) / counts
    return starts, fractions


def prepare_source(file: AudioFileInfo, settings: PreprocessingSettings, seed: int) -> PreparedSource:
    """Turn a source file into a standardised dry clip.

    Raises:
        SourceRejected: if the file cannot yield a valid clip.
    """
    try:
        audio, orig_sr = load_audio(file.path)
    except Exception as exc:  # noqa: BLE001
        raise SourceRejected(f"cannot decode audio ({exc})") from exc
    if audio.size == 0:
        raise SourceRejected("file contains no samples")
    orig_channels, orig_frames = audio.shape
    if not np.all(np.isfinite(audio)):
        raise SourceRejected("file contains NaN/inf samples")

    audio = resample(audio, orig_sr, settings.sample_rate)
    conversion = "none" if orig_channels == settings.channels else f"{orig_channels}->{settings.channels}"
    audio = to_channels(audio, settings.channels)

    length = settings.num_samples
    if audio.shape[1] < length:
        raise SourceRejected(
            f"shorter than the clip duration ({audio.shape[1] / settings.sample_rate:.2f}s < {settings.duration}s)"
        )

    starts, fractions = window_activity(audio.mean(axis=0), settings)
    qualifying = np.flatnonzero(fractions >= settings.min_active_fraction)
    if qualifying.size == 0:
        best = float(fractions.max()) if fractions.size else 0.0
        raise SourceRejected(
            f"no {settings.duration}s window with >= {settings.min_active_fraction:.0%} non-silent frames "
            f"(best {best:.0%}, threshold {settings.silence_threshold_dbfs} dBFS)"
        )
    rng = np.random.default_rng(segment_seed(seed, file.rel_path))
    choice = int(qualifying[rng.integers(qualifying.size)])
    start = int(starts[choice])
    clip = np.ascontiguousarray(audio[:, start : start + length], dtype=np.float32)

    clip, loudness, gain_db = normalize(clip, settings)
    if not np.all(np.isfinite(clip)):
        raise SourceRejected("normalisation produced non-finite samples")

    record = PreprocessingRecord(
        original_sample_rate=int(orig_sr),
        original_channels=int(orig_channels),
        original_duration_s=round(orig_frames / orig_sr, 4),
        resampled=orig_sr != settings.sample_rate,
        channel_conversion=conversion,
        segment_start_s=round(start / settings.sample_rate, 4),
        segment_end_s=round((start + length) / settings.sample_rate, 4),
        active_fraction=round(float(fractions[choice]), 4),
        normalization=settings.normalization,
        input_loudness_lufs=None if loudness is None else round(loudness, 3),
        normalization_gain_db=round(gain_db, 3),
        dry_peak_dbfs=round(amplitude_to_db(peak(clip)), 3),
    )
    return PreparedSource(clip, settings.sample_rate, record)


def normalize(clip: np.ndarray, settings: PreprocessingSettings) -> tuple[np.ndarray, float | None, float]:
    """Apply the configured normalisation.

    Returns:
        ``(clip, measured_loudness_lufs_or_None, applied_gain_db)``.
    """
    if settings.normalization == "none":
        return clip, None, 0.0
    clip_peak = peak(clip)
    if clip_peak <= 0.0:
        raise SourceRejected("clip is digital silence")
    if settings.normalization == "peak":
        gain_db = settings.target_peak_dbfs - amplitude_to_db(clip_peak)
        return (clip * db_to_amplitude(gain_db)).astype(np.float32), None, gain_db

    meter = pyloudnorm.Meter(settings.sample_rate)
    loudness = float(meter.integrated_loudness(clip.T.astype(np.float64)))
    if not np.isfinite(loudness):
        raise SourceRejected("clip loudness is -inf (silent)")
    gain_db = settings.target_lufs - loudness
    # Respect the peak ceiling: never push the dry peak above it.
    max_gain_db = settings.peak_ceiling_dbfs - amplitude_to_db(clip_peak)
    gain_db = min(gain_db, max_gain_db)
    return (clip * db_to_amplitude(gain_db)).astype(np.float32), loudness, gain_db
