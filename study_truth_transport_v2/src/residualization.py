"""Cross-fit-safe linear direction removal primitives."""
from __future__ import annotations

import numpy as np


def project_out(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    x, direction = np.asarray(x, dtype=np.float64), np.asarray(direction, dtype=np.float64)
    denominator = float(direction @ direction)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Direction must be finite and nonzero")
    return x - np.outer(x @ direction / denominator, direction)


def project_out_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x, basis = np.asarray(x, dtype=np.float64), np.asarray(basis, dtype=np.float64)
    q, _ = np.linalg.qr(basis.T)
    return x - (x @ q) @ q.T
