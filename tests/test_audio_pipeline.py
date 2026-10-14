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
