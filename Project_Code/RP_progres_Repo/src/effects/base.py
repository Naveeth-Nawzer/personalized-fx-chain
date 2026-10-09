"""Abstract effect interface.

Every effect exposes the same ``process(audio, sample_rate, parameters)``
method, so the dataset generator never depends on a particular DSP library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping

import numpy as np


class EffectError(RuntimeError):
    """Raised when an effect cannot process the given audio/parameters."""


class BackendUnavailableError(EffectError):
    """Raised when the requested DSP backend cannot be imported on this machine."""


class BaseEffect(ABC):
    """Base class for audio effects.

    Subclasses define ``name`` (canonical effect name, e.g. ``"EQ"``),
    ``backend`` and ``parameter_names``, and implement :meth:`_process`.

    Audio is ``float32`` shaped ``(channels, samples)``. The output always has
    the same shape as the input, so dry and wet stay sample-aligned.
    Processing is stateless: each call starts from a cleared state.
    """

    name: ClassVar[str]
    backend: ClassVar[str]
    parameter_names: ClassVar[tuple[str, ...]]

    def process(self, audio: np.ndarray, sample_rate: int, parameters: Mapping[str, float]) -> np.ndarray:
        """Apply the effect.

        Args:
            audio: ``(channels, samples)`` float array.
            sample_rate: Sample rate in Hz.
            parameters: Physical parameter values keyed by ``parameter_names``.

        Returns:
            Processed audio, float32, same shape as ``audio``.
        """
        if audio.ndim != 2:
            raise EffectError(f"{self.name}: expected audio shaped (channels, samples), got {audio.shape}")
        self._check_parameters(parameters)
        x = np.ascontiguousarray(audio, dtype=np.float32)
        out = self._process(x, int(sample_rate), {k: float(parameters[k]) for k in self.parameter_names})
        out = np.asarray(out, dtype=np.float32)
        if out.shape != audio.shape:
            raise EffectError(f"{self.name}: output shape {out.shape} differs from input shape {audio.shape}")
        if not np.all(np.isfinite(out)):
            raise EffectError(f"{self.name}: produced non-finite samples")
        return out

    def _check_parameters(self, parameters: Mapping[str, float]) -> None:
        missing = set(self.parameter_names) - set(parameters)
        unknown = set(parameters) - set(self.parameter_names)
        if missing or unknown:
            raise EffectError(
                f"{self.name}: invalid parameters (missing={sorted(missing)}, unknown={sorted(unknown)})"
            )

    @abstractmethod
    def _process(self, audio: np.ndarray, sample_rate: int, parameters: dict[str, float]) -> np.ndarray:
        """Backend-specific processing (parameters already validated)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, backend={self.backend!r})"
