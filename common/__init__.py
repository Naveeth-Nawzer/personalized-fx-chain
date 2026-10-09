"""common/__init__.py — public API for the common package."""
from common.audio_types import AudioMeta, AudioTensor, EffectParameterSpec
from common.config_loader import (
    load_effects_config,
    load_system_config,
    get_sample_rate,
    get_channels,
    get_dtype,
    get_max_duration_seconds,
)
from common.fx_schema import FXChainSpec, FXChainValidator, EffectStep
from common.parameter_mapper import ParameterMapper, build_mappers_for_effect
from common.validation import AudioValidationError, validate_audio_tensor

__all__ = [
    "AudioMeta",
    "AudioTensor",
    "EffectParameterSpec",
    "load_effects_config",
    "load_system_config",
    "get_sample_rate",
    "get_channels",
    "get_dtype",
    "get_max_duration_seconds",
    "FXChainSpec",
    "FXChainValidator",
    "EffectStep",
    "ParameterMapper",
    "build_mappers_for_effect",
    "AudioValidationError",
    "validate_audio_tensor",
]
