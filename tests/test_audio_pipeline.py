"""tests/test_audio_pipeline.py — end-to-end audio pipeline tests."""
import tempfile
from pathlib import Path
import pytest
import torch
from audio_io import load_audio, save_audio, generate_test_tone
from dapn.dapn import DAPN


@pytest.fixture
def dapn():
    return DAPN()


def test_generate_and_save_audio():
    tone = generate_test_tone(frequency_hz=440.0, duration_s=0.5, amplitude=0.5)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test_tone.wav"
        save_audio(tone, wav_path)
        assert wav_path.exists()
        loaded, sr = load_audio(wav_path)
        assert sr == 44100
        assert loaded.shape[0] == 1
        assert abs(loaded.shape[1] - int(0.5 * 44100)) <= 1


def test_end_to_end_wav_processing(dapn):
    """Generate tone -> save -> load -> DAPN process -> save wet.wav -> verify."""
    tone = generate_test_tone(frequency_hz=220.0, duration_s=0.25, amplitude=0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        dry_path = Path(tmpdir) / "dry.wav"
        wet_path = Path(tmpdir) / "wet.wav"

        save_audio(tone, dry_path)
        audio, sr = load_audio(dry_path)

        chain_dict = {
            "chain": [
                {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.7, "q": 0.4}},
                {"effect": "Compressor", "parameters": {
                    "threshold": 0.6, "ratio": 0.3, "attack": 0.2, "release": 0.4, "makeup_gain": 0.3
                }},
                {"effect": "Delay", "parameters": {
                    "delay_time": 0.05, "feedback": 0.3, "wet_mix": 0.5, "dry_mix": 0.5
                }},
            ]
        }

        wet_audio = dapn.process(audio, chain_dict)
        save_audio(wet_audio, wet_path)

        assert wet_path.exists()
        loaded_wet, sr_wet = load_audio(wet_path)
        assert sr_wet == 44100
        assert loaded_wet.shape == audio.shape
        assert torch.isfinite(loaded_wet).all()


def test_load_nonexistent_audio_raises():
    with pytest.raises(FileNotFoundError):
        load_audio("/tmp/definitely_not_a_real_audio_file_12345.wav")


def test_resample_48k_to_44k():
    """Verify 48 kHz audio is correctly resampled to target 44.1 kHz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p48 = Path(tmpdir) / "tone_48k.wav"
        tone = generate_test_tone(frequency_hz=440.0, duration_s=1.0, sample_rate=48000)
        save_audio(tone, p48, sample_rate=48000)

        loaded, sr = load_audio(p48, target_sample_rate=44100)
        assert sr == 44100
        assert abs(loaded.shape[1] - 44100) <= 2
        assert torch.isfinite(loaded).all()


def test_resample_44k_to_48k():
    """Verify 44.1 kHz audio is correctly resampled to target 48 kHz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p44 = Path(tmpdir) / "tone_44k.wav"
        tone = generate_test_tone(frequency_hz=440.0, duration_s=0.5, sample_rate=44100)
        save_audio(tone, p44, sample_rate=44100)

        loaded, sr = load_audio(p44, target_sample_rate=48000)
        assert sr == 48000
        assert abs(loaded.shape[1] - 24000) <= 2
        assert torch.isfinite(loaded).all()


def test_resample_stereo_48k_to_mono_44k():
    """Verify stereo 48 kHz audio is downmixed to mono and resampled to 44.1 kHz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_stereo = Path(tmpdir) / "stereo_48k.wav"
        tone_stereo = generate_test_tone(frequency_hz=330.0, duration_s=0.5, channels=2, sample_rate=48000)
        save_audio(tone_stereo, p_stereo, sample_rate=48000)

        loaded, sr = load_audio(p_stereo, target_sample_rate=44100, target_channels=1)
        assert sr == 44100
        assert loaded.shape[0] == 1
        assert abs(loaded.shape[1] - int(0.5 * 44100)) <= 2
        assert torch.isfinite(loaded).all()
