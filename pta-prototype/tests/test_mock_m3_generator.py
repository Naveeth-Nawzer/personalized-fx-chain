import pytest
import numpy as np
from pta.core.guidance_builder import build_guidance_message
from pta.integration.mock_m3_generator import MockM3Generator
from pta.models.guidance import GuidanceMessage


PREF = np.array([0.0, -0.3, 0.51, 0.15, 0.0, 0.15, 0.0])


def test_generates_new_sample_id():
    gen = MockM3Generator()
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    result = gen.generate(msg)
    assert result["status"] == "REGENERATED"
    assert result["sample_id"] != "sample_A"
    assert result["sample_id"].startswith("sample_")


def test_no_regeneration_on_satisfied():
    gen = MockM3Generator()
    msg = build_guidance_message("U001", "S001", "sample_A", "SATISFIED", PREF)
    result = gen.generate(msg)
    assert result["status"] == "NO_REGENERATION"
    assert result["sample_id"] == "sample_A"


def test_chain_respects_constraints():
    gen = MockM3Generator()
    # Strong reverb preference → constraint on Reverb
    prefs = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", prefs)
    result = gen.generate(msg)

    chain = result["metadata"]["chain"]
    assert any(e["effect"] == "Reverb" for e in chain)


def test_chain_is_deterministic():
    """Same guidance → same chain (needed for reproducibility)."""
    gen = MockM3Generator()
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    r1 = gen.generate(msg)
    r2 = gen.generate(msg)
    assert r1["metadata"]["chain"] == r2["metadata"]["chain"]


def test_fallback_chain_when_no_constraints():
    gen = MockM3Generator()
    # Weak preference → no constraints
    prefs = np.array([0.1, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", prefs)
    result = gen.generate(msg)

    chain = result["metadata"]["chain"]
    assert len(chain) == 1
    assert chain[0]["effect"] == "EQ"


def test_samples_dict_tracks_generations():
    gen = MockM3Generator()
    for cmd in ["MORE_REVERB", "LESS_COMPRESSION", "WIDER"]:
        msg = build_guidance_message("U001", "S001", f"sample_{cmd}", cmd, PREF)
        gen.generate(msg)
    assert len(gen.samples) == 3


def test_metadata_contains_required_fields():
    gen = MockM3Generator()
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    result = gen.generate(msg)

    meta = result["metadata"]
    for field in ["sample_id", "user_id", "session_id", "chain",
                  "audio_checksum", "audio_duration_s", "sample_rate"]:
        assert field in meta


def test_no_audio_tensor_in_metadata():
    """Explicitly verify: mock returns metadata only, no audio."""
    gen = MockM3Generator()
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    result = gen.generate(msg)
    assert "audio" not in result["metadata"]
    assert "audio_tensor" not in result["metadata"]


def test_generate_initial_creates_valid_sample():
    gen = MockM3Generator()
    res = gen.generate_initial("U001", "S001")
    assert res["status"] == "INITIAL_GENERATED"
    assert res["sample_id"].startswith("sample_")
    assert "chain" in res["metadata"]
    assert res["metadata"]["user_id"] == "U001"
    assert res["metadata"]["session_id"] == "S001"


def test_generate_initial_tracks_in_samples_dict():
    gen = MockM3Generator()
    res = gen.generate_initial("U001", "S001")
    assert res["sample_id"] in gen.samples
    assert len(gen.samples) == 1

