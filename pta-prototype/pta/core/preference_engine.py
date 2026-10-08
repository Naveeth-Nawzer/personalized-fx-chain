"""The core personalization mechanism: exponential smoothing over preference vectors."""
import numpy as np
from pta.core.feedback_encoder import encode_feedback
from pta.models.schemas import VECTOR_DIM


class PreferenceVectorEngine:
    """
    Maintains and updates a user's preference vector.

    Update rule :
        P_new = (1 - alpha) * P_old + alpha * F

    where F is the one-hot feedback vector and alpha is the learning rate.
    Only the dimensions corresponding to the feedback are updated,
    leaving other dimensions unaffected.
    """

    def __init__(self, user_id: str, alpha: float = 0.25,
                 initial_vector: np.ndarray = None):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.user_id = user_id
        self.alpha = alpha
        self.P_u = (np.array(initial_vector)
                    if initial_vector is not None
                    else np.zeros(VECTOR_DIM))

    def update(self, feedback_command: str) -> dict:
        """
        Apply a feedback command and return before/after states.

        Returns:
            {
                "command": str,
                "before": list,
                "after": list,
                "delta": list,
            }
        """
        before = self.P_u.copy()
        F = encode_feedback(feedback_command)
        
        # Only apply decay and update to the dimensions affected by this feedback
        active_mask = F != 0
        if np.any(active_mask):
            self.P_u[active_mask] = (1 - self.alpha) * self.P_u[active_mask] + self.alpha * F[active_mask]
            
        self.P_u = np.clip(self.P_u, -1.0, 1.0)

        return {
            "command": feedback_command,
            "before": before.tolist(),
            "after": self.P_u.tolist(),
            "delta": (self.P_u - before).tolist(),
        }

    def get_vector(self) -> np.ndarray:
        return self.P_u.copy()

    def reset(self):
        self.P_u = np.zeros(VECTOR_DIM)

    def __repr__(self):
        return f"PreferenceVectorEngine(user={self.user_id}, P_u={self.P_u.round(3).tolist()})"
