"""
common/losses.py
================
Differentiable audio loss functions for DAPN and audio effects optimization.

Implements:
  - SpectralConvergenceLoss
  - LogSTFTMagnitudeLoss
  - STFTLoss (single resolution)
  - MultiResolutionSTFTLoss (MR-STFT)
  - CombinedAudioLoss (MR-STFT + optional waveform L1)

Based on the standard Multi-Resolution STFT loss formulation used in
DDSP (Engel et al.) and Parallel WaveGAN (Yamamoto et al.).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss measuring normalized Frobenius norm difference."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Calculate spectral convergence loss with numerically stable gradients."""
        eps = 1e-8
        diff_sq = (target_mag - pred_mag).pow(2).sum(dim=(-2, -1))
        diff_norm = torch.sqrt(diff_sq + eps)

        target_sq = target_mag.pow(2).sum(dim=(-2, -1))
        target_norm = torch.sqrt(target_sq + eps)

        return (diff_norm / target_norm).mean()


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss measuring L1 distance between log-spectrograms."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Calculate log-magnitude STFT loss.

        Parameters
        ----------
        pred_mag : torch.Tensor
            Predicted magnitude spectrogram, shape (..., F, T).
        target_mag : torch.Tensor
            Target magnitude spectrogram, shape (..., F, T).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        log_pred = torch.log(pred_mag + self.eps)
        log_target = torch.log(target_mag + self.eps)
        return torch.nn.functional.l1_loss(log_pred, log_target)


class STFTLoss(nn.Module):
    """Single-resolution STFT loss combining spectral convergence and log-magnitude."""

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft_magnitude(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude spectrogram.

        Audio shape can be (C, T) or (B, C, T).
        """
        original_shape = audio.shape
        T = original_shape[-1]
        flat_audio = audio.reshape(-1, T)

        # STFT with stable complex magnitude
        stft = torch.stft(
            flat_audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window.to(dtype=audio.dtype, device=audio.device),
            return_complex=True,
        )
        eps = 1e-8
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + eps)

        # Reshape back to (..., F, frames)
        if len(original_shape) > 1:
            return mag.view(*original_shape[:-1], mag.shape[-2], mag.shape[-1])
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate STFT loss between predicted and target audio signals.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (sc_loss, mag_loss)
        """
        pred_mag = self._stft_magnitude(pred)
        target_mag = self._stft_magnitude(target)

        sc = self.sc_loss(pred_mag, target_mag) * self.factor_sc
        mag = self.mag_loss(pred_mag, target_mag) * self.factor_mag
        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss combining multiple FFT resolutions.

    Evaluates spectral distance across fine, medium, and broad time-frequency
    resolutions, capturing both transient precision and harmonic frequency detail.

    Parameters
    ----------
    fft_sizes : list[int], optional
        List of FFT sizes, default [512, 1024, 2048].
    hop_sizes : list[int], optional
        List of hop sizes, default [128, 256, 512].
    win_lengths : list[int], optional
        List of window lengths, default [512, 1024, 2048].
    factor_sc : float, optional
        Weight for spectral convergence loss, default 1.0.
    factor_mag : float, optional
        Weight for log-magnitude loss, default 1.0.
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ) -> None:
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [128, 256, 512]
        if win_lengths is None:
            win_lengths = [512, 1024, 2048]

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "fft_sizes, hop_sizes, and win_lengths must have the same length."
        )

        self.loss_layers = nn.ModuleList([
            STFTLoss(fs, hs, wl, factor_sc, factor_mag)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-resolution STFT loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted audio waveform (C, T) or (B, C, T).
        target : torch.Tensor
            Target audio waveform with identical shape to pred.

        Returns
        -------
        torch.Tensor
            Averaged multi-resolution scalar loss.
        """
        total_sc = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        total_mag = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        for layer in self.loss_layers:
            sc, mag = layer(pred, target)
            total_sc = total_sc + sc
            total_mag = total_mag + mag

        n_resolutions = len(self.loss_layers)
        return (total_sc + total_mag) / n_resolutions


class CombinedAudioLoss(nn.Module):
    """Combines Multi-Resolution STFT Loss with an optional waveform L1 loss."""

    def __init__(
        self,
        mr_stft_weight: float = 1.0,
        l1_weight: float = 0.05,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.mr_stft_weight = mr_stft_weight
        self.l1_weight = l1_weight
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate combined loss and return loss breakdown.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            (total_loss, metrics_dict)
        """
        stft_loss = self.mr_stft_loss(pred, target)
        total_loss = self.mr_stft_weight * stft_loss

        metrics = {
            "mr_stft_loss": float(stft_loss.item()),
            "total_loss": float(total_loss.item()),
        }

        if self.l1_weight > 0.0:
            l1_loss = torch.nn.functional.l1_loss(pred, target)
            total_loss = total_loss + self.l1_weight * l1_loss
            metrics["l1_loss"] = float(l1_loss.item())
            metrics["total_loss"] = float(total_loss.item())

        return total_loss, metrics
