"""Mock M1 feature extractor — returns deterministic fake features.

⚠️ This is a MOCK. It returns hardcoded values matching the M1 schema
from the integration PDF. It does not analyze real audio.

Keep thin. PTA currently does not use audio features — this mock exists
only to show the future context-aware path is prepared.
"""


class MockM1Features:
    """Returns deterministic features matching M1's documented schema."""

    # Real M1 features from the PDF (11 fields)
    DEFAULT_FEATURES = {
        "peak_dbFS": -3.5,
        "rms_dbFS": -18.2,
        "crest_factor_db": 14.7,
        "zero_crossing_rate": 0.042,
        "spectral_centroid_hz": 3200.0,
        "spectral_bandwidth_hz": 4500.0,
        "spectral_rolloff_85pct_hz": 7800.0,
        "spectral_flatness": 0.31,
        "low_sub_bass_under_250hz_pct": 0.28,
        "midrange_250_to_4000hz_pct": 0.52,
        "high_presence_above_4000hz_pct": 0.20,
    }

    def extract(self, audio_id: str, source: str = "dry") -> dict:
        """
        Return fake features for an audio_id.

        Args:
            audio_id: identifier for the audio sample
            source: "dry" | "reference" | "generated"

        Returns:
            Dict matching M1 schema (11 features + metadata)
        """
        return {
            **self.DEFAULT_FEATURES,
            "audio_id": audio_id,
            "source": source,
            # subtle variation per source (deterministic, not random)
            "spectral_centroid_hz": (
                3200.0 if source == "dry" else
                3600.0 if source == "reference" else
                3400.0
            ),
        }
