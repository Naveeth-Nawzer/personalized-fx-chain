"""
audio_io.py
===========
Audio file I/O utilities — intentionally kept *outside* the
differentiable DSP chain.

Uses the Python standard library `wave` module and `struct` for WAV
reading and writing. This avoids heavy external codec dependencies like
torchcodec / soundfile, while guaranteeing standard 16-bit or 32-bit float
PCM WAV compatibility across all platforms.

Functions
---------
load_audio  — WAV → torch.Tensor (C, T)
save_audio  — torch.Tensor → WAV
generate_test_tone — create a synthetic test tone without a file
"""

from __future__ import annotations

import logging
import math
import struct
import wave
from pathlib import Path

import torch

from common.config_loader import get_sample_rate, get_channels, get_max_duration_seconds

logger = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load a WAV audio file and return a float32 tensor.

    Parameters
    ----------
    path : str | Path
        Path to the WAV audio file.
    target_sample_rate : int, optional
        Target sample rate. Resampled if different.
    target_channels : int, optional
        Target channel count. Mono downmixed if needed.

    Returns
    -------
    audio : torch.Tensor
        Shape ``(C, T)``, dtype float32, values in [-1, 1].
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        sr         = wf.getframerate()
        n_frames   = wf.getnframes()
        raw_bytes  = wf.readframes(n_frames)

    # Decode bytes to float32
    if sampwidth == 2:  # 16-bit signed PCM
        total_samples = n_frames * n_channels
        fmt = f"<{total_samples}h"
        integers = struct.unpack(fmt, raw_bytes)
        waveform = torch.tensor(integers, dtype=torch.float32) / 32768.0
    elif sampwidth == 4:  # 32-bit float or 32-bit int PCM
        total_samples = n_frames * n_channels
        try:
            fmt = f"<{total_samples}f"
            floats = struct.unpack(fmt, raw_bytes)
            waveform = torch.tensor(floats, dtype=torch.float32)
        except Exception:
            fmt = f"<{total_samples}i"
            integers = struct.unpack(fmt, raw_bytes)
            waveform = torch.tensor(integers, dtype=torch.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    # Reshape to (C, T)
    waveform = waveform.view(n_frames, n_channels).t()

    target_sr = target_sample_rate or get_sample_rate()
    target_ch = target_channels    or get_channels()

    # Mix down to target channels
    if target_ch == 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif target_ch > 1 and waveform.shape[0] == 1:
        waveform = waveform.expand(target_ch, -1)

    # Check duration limit
    dur_s = waveform.shape[-1] / sr
    max_dur = get_max_duration_seconds()
    if dur_s > max_dur:
        waveform = waveform[..., :int(max_dur * sr)]

    logger.info("Loaded '%s' — shape=%s, sr=%d", path.name, tuple(waveform.shape), sr)
    return waveform, sr


def save_audio(
    audio: torch.Tensor,
    path: str | Path,
    sample_rate: int | None = None,
) -> None:
    """Save an audio tensor to a 16-bit PCM WAV file.

    Parameters
    ----------
    audio : torch.Tensor
        Shape ``(C, T)`` or ``(B, C, T)``.
    path : str | Path
        Output file path.
    sample_rate : int, optional
        Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or get_sample_rate()
    waveform = audio.detach().cpu()

    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected (C, T) or (B, C, T) tensor, got {tuple(audio.shape)}"
        )

    # Clamp to [-1, 1]
    waveform = waveform.clamp(-1.0, 1.0)
    n_channels, n_frames = waveform.shape

    # Convert to 16-bit PCM
    int_samples = (waveform * 32767.0).round().to(torch.int16)
    # Interleave channels: (C, T) -> (T, C) -> flatten
    interleaved = int_samples.t().contiguous().view(-1).tolist()

    raw_bytes = struct.pack(f"<{len(interleaved)}h", *interleaved)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 2 bytes = 16-bit
        wf.setframerate(sr)
        wf.writeframes(raw_bytes)

    logger.info("Saved audio to '%s' — shape=%s, sr=%d", path, tuple(waveform.shape), sr)


def generate_test_tone(
    frequency_hz: float = 440.0,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> torch.Tensor:
    """Generate a pure sine-wave test tone.

    Returns
    -------
    torch.Tensor
        Shape ``(C, T)``, dtype float32.
    """
    sr  = sample_rate or get_sample_rate()
    ch  = channels    or get_channels()
    T   = int(duration_s * sr)
    t   = torch.linspace(0.0, duration_s, T)
    tone = amplitude * torch.sin(2.0 * math.pi * frequency_hz * t)
    return tone.unsqueeze(0).expand(ch, -1)
