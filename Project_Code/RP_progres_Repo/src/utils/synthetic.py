"""Tiny MUSDB-like synthetic dataset for tests and dry runs.

Creates ``<out>/<Song_XX>/{bass,drums,other,vocals,mixture}.wav`` from tones,
noise bursts and silence gaps. Not meant to sound like music; it only lets
the full pipeline run without the real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

DEFAULT_CATEGORIES: tuple[str, ...] = ("bass", "drums", "other", "vocals")


def _stem(category: str, t: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    beat = 0.5
    phase = (t % beat) / beat
    if category == "bass":
        f0 = rng.uniform(45, 90)
        return 0.4 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.cos(2 * np.pi * phase))
    if category == "drums":
        kick = np.exp(-phase * 12.0) * np.sin(2 * np.pi * 60 * t)
        hat_phase = (t % (beat / 4)) / (beat / 4)
        hats = np.exp(-hat_phase * 4.0) * rng.standard_normal(t.size)
        return 0.6 * kick + 0.08 * hats
    if category == "vocals":
        f0 = rng.uniform(180, 320)
        vibrato = 4.0 * np.sin(2 * np.pi * 5.5 * t)
        voice = sum(np.sin(2 * np.pi * k * (f0 * t + vibrato / (2 * np.pi * 5.5))) / k for k in range(1, 6))
        gate = (np.sin(2 * np.pi * 0.4 * t) > -0.3).astype(float)  # phrases separated by silence
        return 0.25 * voice * gate
    # "other" and any unknown category: a chord
    freqs = rng.uniform(300, 1200, size=3)
    return 0.15 * sum(np.sin(2 * np.pi * f * t) for f in freqs)


def make_toy_dataset(
    out_dir: str | Path,
    num_songs: int = 6,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    include_mixture: bool = True,
    duration: float = 4.0,
    sample_rate: int = 22050,
    channels: int = 2,
    seed: int = 0,
) -> list[Path]:
    """Write a synthetic dataset and return the created file paths."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration * sample_rate))) / sample_rate
    created: list[Path] = []
    for i in range(num_songs):
        song_dir = out_dir / f"Song_{i + 1:02d}"
        song_dir.mkdir(parents=True, exist_ok=True)
        stems = {}
        for category in categories:
            mono = _stem(category, t, sample_rate, rng)
            stems[category] = mono
        if include_mixture:
            stems["mixture"] = 0.5 * sum(stems.values())
        for category, mono in stems.items():
            data = np.stack([mono] * channels, axis=1) if channels > 1 else mono[:, None]
            path = song_dir / f"{category}.wav"
            sf.write(str(path), data.astype(np.float32), sample_rate, subtype="PCM_16")
            created.append(path)
    return created
