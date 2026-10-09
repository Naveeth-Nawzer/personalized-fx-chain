"""Audio I/O and low-level signal helpers.

Convention used throughout the project: audio arrays are ``float32`` with
shape ``(channels, samples)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_EPS = 1e-12


@dataclass(frozen=True)
class AudioInfo:
    """Basic properties of an audio file, read without decoding the samples."""

    sample_rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def audio_info(path: str | Path) -> AudioInfo:
    """Read sample rate, channel count and length of an audio file."""
    path = Path(path)
    try:
        info = sf.info(str(path))
        return AudioInfo(int(info.samplerate), int(info.channels), int(info.frames))
    except Exception:  # noqa: BLE001 — fall back to pedalboard's decoder (mp3 etc.)
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as fh:
            return AudioInfo(int(fh.samplerate), int(fh.num_channels), int(fh.frames))


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode an audio file.

    Returns:
        ``(audio, sample_rate)`` with ``audio`` shaped ``(channels, samples)``, float32.
    """
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except Exception:  # noqa: BLE001 — fall back to pedalboard's decoder (mp3 etc.)
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as fh:
            data = fh.read(fh.frames)
            return np.ascontiguousarray(data, dtype=np.float32), int(fh.samplerate)


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> None:
    """Write ``(channels, samples)`` audio to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32).T, sample_rate, subtype=subtype, format="WAV")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling along the time axis. Returns the input unchanged if rates match."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    out = resample_poly(audio, target_sr // g, orig_sr // g, axis=-1)
    return out.astype(np.float32)


def to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Convert to mono (mean downmix) or stereo (duplicate mono / keep first two channels)."""
    if audio.shape[0] == channels:
        return audio
    if channels == 1:
        return audio.mean(axis=0, keepdims=True).astype(np.float32)
    if channels == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio, 2, axis=0)
        return audio[:2]
    raise ValueError(f"Unsupported channel count: {channels}")


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def amplitude_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, _EPS)))


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def frame_rms_db(mono: np.ndarray, frame_length: int) -> np.ndarray:
    """RMS level (dBFS) of non-overlapping frames of a 1-D signal. A trailing partial frame is dropped."""
    n_frames = mono.shape[-1] // frame_length
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    frames = mono[: n_frames * frame_length].reshape(n_frames, frame_length).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, _EPS))


def residual_db(reference: np.ndarray, processed: np.ndarray) -> float:
    """Energy of ``processed - reference`` relative to ``reference``, in dB.

    Used as a simple "how much did this effect change the signal" measure.
    Very negative values mean the two signals are nearly identical.
    """
    ref = reference.astype(np.float64)
    diff = processed.astype(np.float64) - ref
    return float(10.0 * np.log10((np.sum(diff**2) + _EPS) / (np.sum(ref**2) + _EPS)))
