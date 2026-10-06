"""Mass-mean probing primitives."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MassMeanProbe:
    direction: np.ndarray
    threshold: float

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.direction - self.threshold


def fit_mass_mean(x: np.ndarray, y: np.ndarray) -> MassMeanProbe:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=int)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Need finite 2D features and both binary classes")
    negative, positive = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
    direction = positive - negative
    if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Undefined mass-mean direction")
    threshold = float((positive @ direction + negative @ direction) / 2)
    return MassMeanProbe(direction=direction, threshold=threshold)
