"""tests/test_losses.py — tests for Multi-Resolution STFT and Combined Audio Losses."""
import pytest
import torch

from common.losses import (
    SpectralConvergenceLoss,
    LogSTFTMagnitudeLoss,
    STFTLoss,
    MultiResolutionSTFTLoss,
    CombinedAudioLoss,
)


@pytest.fixture
def sample_audio_2d():
    torch.manual_seed(42)
    return torch.randn(1, 22050)


@pytest.fixture
def sample_audio_3d():
    torch.manual_seed(42)
    return torch.randn(2, 1, 22050)


def test_identical_audio_loss_is_zero(sample_audio_2d):
    criterion = MultiResolutionSTFTLoss()
    loss = criterion(sample_audio_2d, sample_audio_2d)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_different_audio_loss_is_positive(sample_audio_2d):
    torch.manual_seed(99)
    other_audio = torch.randn_like(sample_audio_2d)
    criterion = MultiResolutionSTFTLoss()
    loss = criterion(sample_audio_2d, other_audio)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_multires_stft_loss_shape_2d(sample_audio_2d):
    criterion = MultiResolutionSTFTLoss()
    target = sample_audio_2d * 0.5
    loss = criterion(sample_audio_2d, target)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_multires_stft_loss_shape_3d(sample_audio_3d):
    criterion = MultiResolutionSTFTLoss()
    target = sample_audio_3d * 0.5
    loss = criterion(sample_audio_3d, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradients_propagate_through_loss(sample_audio_2d):
    criterion = MultiResolutionSTFTLoss()
    pred = sample_audio_2d.clone().detach().requires_grad_(True)
    target = sample_audio_2d * 0.5

    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_combined_audio_loss_metrics(sample_audio_2d):
    criterion = CombinedAudioLoss(mr_stft_weight=1.0, l1_weight=0.1)
    target = sample_audio_2d * 0.8
    total_loss, metrics = criterion(sample_audio_2d, target)

    assert torch.isfinite(total_loss)
    assert "mr_stft_loss" in metrics
    assert "l1_loss" in metrics
    assert "total_loss" in metrics
    assert metrics["total_loss"] > metrics["mr_stft_loss"]


def test_short_audio_handling():
    """Signals shorter than window length (e.g., T=256 or T=1024) must not crash."""
    criterion = MultiResolutionSTFTLoss()
    for length in [128, 256, 512, 1024]:
        short_audio = torch.randn(1, length)
        # Identical
        loss_ident = criterion(short_audio, short_audio)
        assert torch.isfinite(loss_ident)
        assert loss_ident.item() < 1e-4

        # Different
        diff_audio = torch.randn(1, length)
        loss_diff = criterion(short_audio, diff_audio)
        assert torch.isfinite(loss_diff)
        assert loss_diff.item() > loss_ident.item()


def test_short_audio_gradient_flow():
    """Gradients propagate through short audio inputs."""
    criterion = MultiResolutionSTFTLoss()
    pred = torch.randn(1, 256, requires_grad=True)
    target = torch.randn(1, 256)
    loss = criterion(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_loss_ordering_near_vs_distant(sample_audio_2d):
    """Near identical audio has much lower loss than distant audio."""
    criterion = MultiResolutionSTFTLoss()
    slightly_perturbed = sample_audio_2d + 0.01 * torch.randn_like(sample_audio_2d)
    very_different = torch.randn_like(sample_audio_2d) * 2.0

    loss_close = criterion(sample_audio_2d, slightly_perturbed).item()
    loss_far = criterion(sample_audio_2d, very_different).item()

    assert loss_close < loss_far
