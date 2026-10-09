"""
models/inverse_fx_net.py
========================
Deep Convolutional Neural Network for Inverse Audio Effects Chain Prediction.

Architecture:
  Input: (dry_audio, wet_audio), each shape (B, 1, T) at 44.1 kHz.
  1. Front-end: Computes Log-Mel Spectrograms for dry, wet, and delta (wet - dry).
     Feature tensor: (B, 3, n_mels, time_frames).
  2. Backbone: 4-stage 2D Residual Convolutional Network extracting acoustic representations.
  3. Pooling: Concatenated Global Average & Max Pooling (GAP + GMP) -> Dense Embedding.
  4. Multi-Task Heads:
     - presence_head: (B, 7) Logits for active effect classification (BCE loss).
     - param_head: (B, 29) Continuous normalized parameter predictions in [0, 1] (Sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

from dataset.dapn_dataset import SUPPORTED_EFFECTS, PARAM_SCHEMA, PARAM_OFFSETS, TOTAL_NUM_PARAMS


class ConvBlock(nn.Module):
    """Residual 2D Convolutional Block with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Residual shortcut if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class InverseFXNet(nn.Module):
    """End-to-end Inverse FX Predictor predicting active effects and continuous DSP parameters."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        # 1. Front-end Mel-spectrogram extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # 2. 2D Convolutional Backbone (3 input channels: dry, wet, delta)
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # (B, 32, 32, 87)
            ConvBlock(32, 64, stride=2),   # (B, 64, 16, 44)
            ConvBlock(64, 128, stride=2),  # (B, 128, 8, 22)
            ConvBlock(128, 256, stride=2), # (B, 256, 4, 11)
        )

        # 3. Dense Bottleneck (GAP + GMP = 256 * 2 = 512)
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

        # 4. Multi-Task Output Heads
        # Head A: Effect presence logits (7 effects)
        self.presence_head = nn.Linear(embedding_dim, len(SUPPORTED_EFFECTS))

        # Head B: Continuous parameter values (29 parameters in [0, 1])
        self.param_head = nn.Sequential(
            nn.Linear(embedding_dim, TOTAL_NUM_PARAMS),
            nn.Sigmoid(),
        )

    def extract_features(self, dry_audio: torch.Tensor, wet_audio: torch.Tensor) -> torch.Tensor:
        """Extracts 3-channel Log-Mel spectrograms: [Dry, Wet, Delta]."""
        eps = 1e-6
        # dry_audio: (B, 1, T), wet_audio: (B, 1, T)
        mel_dry = torch.log(self.mel_transform(dry_audio) + eps)
        mel_wet = torch.log(self.mel_transform(wet_audio) + eps)
        mel_delta = mel_wet - mel_dry

        # Concatenate along channel dimension: (B, 3, n_mels, frames)
        features = torch.cat([mel_dry, mel_wet, mel_delta], dim=1)
        return features

    def forward(
        self,
        dry_audio: torch.Tensor,
        wet_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        dry_audio : torch.Tensor, shape (B, 1, T)
        wet_audio : torch.Tensor, shape (B, 1, T)

        Returns
        -------
        presence_logits : torch.Tensor, shape (B, 7)
        param_preds : torch.Tensor, shape (B, 29) in [0, 1]
        """
        x = self.extract_features(dry_audio, wet_audio)
        feats = self.backbone(x)  # (B, 256, H, W)

        # Global Average Pooling + Global Max Pooling
        gap = feats.mean(dim=[-2, -1])
        gmp = feats.amax(dim=[-2, -1])
        pooled = torch.cat([gap, gmp], dim=-1)  # (B, 512)

        emb = self.fc(pooled)  # (B, embedding_dim)

        presence_logits = self.presence_head(emb)
        param_preds = self.param_head(emb)

        return presence_logits, param_preds

    def predict_chain(
        self,
        dry_audio: torch.Tensor,
        wet_audio: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, list[dict]]:
        """Inference helper returning a validated DAPN chain dictionary."""
        self.eval()
        with torch.no_grad():
            if dry_audio.ndim == 2:
                dry_audio = dry_audio.unsqueeze(0)
            if wet_audio.ndim == 2:
                wet_audio = wet_audio.unsqueeze(0)

            logits, params = self.forward(dry_audio, wet_audio)
            probs = torch.sigmoid(logits).squeeze(0)  # (7,)
            param_vals = params.squeeze(0)           # (29,)

            chain_steps = []
            for i, fx_name in enumerate(SUPPORTED_EFFECTS):
                if probs[i].item() >= threshold:
                    start_idx, end_idx = PARAM_OFFSETS[fx_name]
                    param_names = PARAM_SCHEMA[fx_name]
                    step_params = {}
                    for p_i, p_name in enumerate(param_names):
                        val = float(param_vals[start_idx + p_i].item())
                        step_params[p_name] = round(val, 4)

                    chain_steps.append({
                        "effect": fx_name,
                        "parameters": step_params,
                    })

            # If no effect passed threshold, default to identity EQ
            if not chain_steps:
                chain_steps.append({
                    "effect": "EQ",
                    "parameters": {"frequency": 0.5, "gain": 0.5, "q": 0.5},
                })

            return {"chain": chain_steps}
