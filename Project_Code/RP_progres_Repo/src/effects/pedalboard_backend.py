"""Optional pedalboard (Spotify) backend for EQ, Compressor and Reverb.

pedalboard is imported lazily, so the rest of the project works on machines
where it is not installed or its native module cannot load. Select it with
``effects.backend: pedalboard`` in the config.

Parameter names are the same as the scipy backend; they are mapped onto
pedalboard's arguments here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.effects.base import BackendUnavailableError, BaseEffect


def _import_pedalboard() -> Any:
    try:
        import pedalboard
    except ImportError as exc:  # also covers native DLLs blocked by OS policy
        raise BackendUnavailableError(
            "The 'pedalboard' DSP backend is unavailable: "
            f"{exc}. Install it with `pip install pedalboard`, or set effects.backend: scipy."
        ) from exc
    return pedalboard


def pedalboard_available() -> bool:
    """True if pedalboard can be imported (including its native module)."""
    try:
        _import_pedalboard()
    except BackendUnavailableError:
        return False
    return True


class _PedalboardEffect(BaseEffect):
    backend = "pedalboard"

    def __init__(self) -> None:
        self._pb = _import_pedalboard()

    def _plugins(self, parameters: dict[str, float]) -> list[Any]:
        raise NotImplementedError

    def _process(self, audio: np.ndarray, sample_rate: int, parameters: dict[str, float]) -> np.ndarray:
        board = self._pb.Pedalboard(self._plugins(parameters))
        return board(audio, float(sample_rate), reset=True)


class PedalboardEQEffect(_PedalboardEffect):
    """Single peaking band via ``pedalboard.PeakFilter``."""

    name = "EQ"
    parameter_names = ("frequency", "gain", "q")

    def _plugins(self, parameters: dict[str, float]) -> list[Any]:
        return [self._pb.PeakFilter(cutoff_frequency_hz=parameters["frequency"],
                                    gain_db=parameters["gain"], q=parameters["q"])]


class PedalboardCompressorEffect(_PedalboardEffect):
    """``pedalboard.Compressor`` followed by ``pedalboard.Gain`` for makeup gain."""

    name = "Compressor"
    parameter_names = ("threshold", "ratio", "attack", "release", "makeup_gain")

    def _plugins(self, parameters: dict[str, float]) -> list[Any]:
        return [
            self._pb.Compressor(threshold_db=parameters["threshold"], ratio=parameters["ratio"],
                                attack_ms=parameters["attack"], release_ms=parameters["release"]),
            self._pb.Gain(gain_db=parameters["makeup_gain"]),
        ]


class PedalboardReverbEffect(_PedalboardEffect):
    """Freeverb-style ``pedalboard.Reverb``."""

    name = "Reverb"
    parameter_names = ("room_size", "damping", "wet_mix", "dry_mix", "width")

    def _plugins(self, parameters: dict[str, float]) -> list[Any]:
        return [self._pb.Reverb(room_size=parameters["room_size"], damping=parameters["damping"],
                                wet_level=parameters["wet_mix"], dry_level=parameters["dry_mix"],
                                width=parameters["width"], freeze_mode=0.0)]
