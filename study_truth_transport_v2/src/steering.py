"""Pure steering-vector scaling used by model-specific hook runners."""
from __future__ import annotations

import numpy as np


def steering_delta(direction: np.ndarray, target_projection_sd: float, alpha: float) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0 or target_projection_sd <= 0:
        raise ValueError("Need nonzero direction and positive projection scale")
    return float(alpha) * float(target_projection_sd) * direction / norm
