"""Production integration adapter bridging PTA to the GRRL (M3) RL agent.

Follows the dependency-injection architecture specified in contracts/pta_grrl_contract.md.
Translates 7-dimensional user preferences and constraints into 16-dimensional
taste representations and action masks accepted by the GRRL environment.
"""
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from pta.core.preference_mapping import (
    PREFERENCE_DIMENSIONS,
    SUPPORTED_EFFECTS,
    validate_preference_dict,
)
from pta.core.guidance_builder import project_taste_vector
from pta.models.schemas import VECTOR_DIM


def validate_preferences(preferences: Union[Dict[str, float], np.ndarray, list]) -> np.ndarray:
    """
    Validate that preferences contain exactly 7 finite values in [-1.0, 1.0].

    Accepts either a dictionary mapping dimension name -> float, or an ordered array/list.
    """
    if isinstance(preferences, dict):
        return validate_preference_dict(preferences)

    values = np.asarray(preferences, dtype=np.float32)
    if values.shape != (VECTOR_DIM,):
        raise ValueError(f"Expected exactly {VECTOR_DIM} preference values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Preferences must be finite numbers (no NaN or Inf)")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError(f"All preference values must be in [-1.0, 1.0]; got {values}")
    return values


def project_taste(preferences: Union[Dict[str, float], np.ndarray, list]) -> np.ndarray:
    """
    Project 7-dimensional preference vector into the 16-dimensional taste input for GRRL.

    Formulation:
      z[0..6]   = raw 7 preference values
      z[7..12]  = 6 pairwise interaction terms
      z[13..15] = 3 statistical aggregate norms (L1 mean, normalized L2, peak absolute)
    """
    p = validate_preferences(preferences)
    # Uses the canonical 7->16 projection shared with guidance_builder
    z_list = project_taste_vector(p)
    taste = np.asarray(z_list, dtype=np.float32)
    assert taste.shape == (16,), f"Expected shape (16,), got {taste.shape}"
    assert np.isfinite(taste).all(), "Projected taste vector must be finite"
    return taste


def validate_constraints(constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate prohibited effects and parameter bounds schema."""
    if constraints is None:
        return {"prohibited_effects": [], "parameter_bounds": {}}

    if not isinstance(constraints, dict):
        raise TypeError("constraints must be a dictionary")

    prohibited = constraints.get("prohibited_effects", [])
    if not isinstance(prohibited, (list, tuple, set)):
        raise TypeError("prohibited_effects must be a list or set")

    for eff in prohibited:
        if eff not in SUPPORTED_EFFECTS:
            raise ValueError(f"Unknown effect in prohibited_effects: {eff}")

    bounds = constraints.get("parameter_bounds", {})
    if not isinstance(bounds, dict):
        raise TypeError("parameter_bounds must be a dictionary")

    for eff, params in bounds.items():
        if eff not in SUPPORTED_EFFECTS:
            raise ValueError(f"Unknown effect in parameter_bounds: {eff}")
        if not isinstance(params, dict):
            raise TypeError(f"Parameters for {eff} must be a dictionary")
        for p_name, interval in params.items():
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"Interval for {eff}.{p_name} must be a 2-element list [low, high]")
            low, high = float(interval[0]), float(interval[1])
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(f"Invalid bound range [{low}, {high}] for {eff}.{p_name}; must be within [0, 1]")

    return {
        "prohibited_effects": list(prohibited),
        "parameter_bounds": bounds,
    }


class GRRLAdapter:
    """
    Production integration adapter connecting PTA to the actual GRRL environment.

    Uses dependency injection for testability and separation from heavy ML frameworks.
    """

    def __init__(
        self,
        env_factory: Optional[Callable[[], Any]] = None,
        taste_projector: Callable[[Any], np.ndarray] = project_taste,
    ):
        """
        Initialize the GRRL adapter.

        Args:
            env_factory: Callable returning a GRRL-compatible environment instance.
            taste_projector: Callable projecting 7D preferences into 16D taste array.
        """
        self.env_factory = env_factory
        self.taste_projector = taste_projector

    def generate(
        self,
        dry_audio: Any,
        target_audio: Any,
        preferences: Union[Dict[str, float], np.ndarray, list],
        constraints: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute generation loop in GRRL conditioned on user preferences and constraints.

        Args:
            dry_audio: Audio tensor or path to dry source.
            target_audio: Audio tensor or path to reference/target.
            preferences: 7-dimensional preference dictionary or vector.
            constraints: Optional hard effect prohibitions and parameter bounds.
            session_id: Optional tracking identifier.

        Returns:
            Dictionary matching contracts/pta_grrl_contract.md.
        """
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        gid = f"gen_{uuid.uuid4().hex[:8]}"
        sample_id = f"sample_{uuid.uuid4().hex[:8]}"

        # 1. Validate inputs
        try:
            valid_p = validate_preferences(preferences)
            valid_c = validate_constraints(constraints)
            taste_vector = self.taste_projector(valid_p)
        except Exception as e:
            return {
                "status": "failed",
                "session_id": sid,
                "generation_id": gid,
                "sample_id": sample_id,
                "error_type": "ValidationError",
                "error_message": str(e),
                "chain": [],
                "metrics": {},
                "audio_ref": None,
            }

        # 2. Check environment factory
        if self.env_factory is None:
            return {
                "status": "failed",
                "session_id": sid,
                "generation_id": gid,
                "sample_id": sample_id,
                "error_type": "ConfigurationError",
                "error_message": "No env_factory configured on GRRLAdapter",
                "chain": [],
                "metrics": {},
                "audio_ref": None,
            }

        # 3. Instantiate and run environment
        try:
            env = self.env_factory()

            # Pass taste vector and constraints to environment reset
            reset_kwargs = {
                "dry_audio": dry_audio,
                "target_audio": target_audio,
                "pta_taste": taste_vector,
                "constraints": valid_c,
            }
            # Allow environments with subset of reset parameters
            try:
                obs = env.reset(**reset_kwargs)
            except TypeError:
                obs = env.reset()
                if hasattr(env, "set_taste"):
                    env.set_taste(taste_vector)
                if hasattr(env, "set_constraints"):
                    env.set_constraints(valid_c)

            # 4. Execute generation episode
            chain = []
            metrics = {}
            audio_ref = None

            if hasattr(env, "run_episode"):
                result = env.run_episode()
                chain = result.get("chain", [])
                metrics = result.get("metrics", {})
                audio_ref = result.get("audio_ref")
            elif hasattr(env, "step"):
                # Standard gym-like loop
                done = False
                steps = 0
                max_steps = getattr(env, "max_steps", 8)
                while not done and steps < max_steps:
                    action = env.sample_action() if hasattr(env, "sample_action") else 0
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    steps += 1
                
                chain = getattr(env, "current_chain", info.get("chain", []))
                metrics = {
                    "steps": steps,
                    "reward": float(reward) if "reward" in locals() else 0.0,
                    **info.get("metrics", {}),
                }
                audio_ref = info.get("audio_ref", f"audio/{gid}.wav")
            else:
                chain = getattr(env, "chain", [])
                metrics = getattr(env, "metrics", {})
                audio_ref = getattr(env, "audio_ref", f"audio/{gid}.wav")

            return {
                "status": "success",
                "session_id": sid,
                "generation_id": gid,
                "sample_id": sample_id,
                "chain": chain,
                "metrics": metrics,
                "audio_ref": audio_ref,
                "applied_taste": taste_vector.tolist(),
                "applied_constraints": valid_c,
            }

        except Exception as e:
            return {
                "status": "failed",
                "session_id": sid,
                "generation_id": gid,
                "sample_id": sample_id,
                "error_type": "ExecutionError",
                "error_message": str(e),
                "chain": [],
                "metrics": {},
                "audio_ref": None,
            }

    def generate_initial(
        self,
        dry_audio: Any,
        target_audio: Any,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience method to generate an initial chain using neutral preferences."""
        neutral_prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
        return self.generate(
            dry_audio=dry_audio,
            target_audio=target_audio,
            preferences=neutral_prefs,
            constraints={"prohibited_effects": [], "parameter_bounds": {}},
            session_id=session_id,
        )
