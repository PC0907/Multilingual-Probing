"""Difficulty-stratification and cross-fit direction helpers."""
from __future__ import annotations

import hashlib
import numpy as np

from .probes import MassMeanProbe


def stable_fold(group_id: str, folds: int = 5) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16) % folds


def extreme_indices(labels: np.ndarray, ease: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    easy, hard = [], []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        order = candidates[np.argsort(ease[candidates], kind="mergesort")]
        count = len(order) // 3
        if count < 1:
            raise ValueError("Too few examples for label-balanced difficulty terciles")
        hard.extend(order[:count].tolist())
        easy.extend(order[-count:].tolist())
    return np.asarray(easy, dtype=int), np.asarray(hard, dtype=int)


def difficulty_probe(x: np.ndarray, labels: np.ndarray, ease: np.ndarray) -> MassMeanProbe:
    easy, hard = extreme_indices(labels, ease)
    easy_mean, hard_mean = x[easy].mean(axis=0), x[hard].mean(axis=0)
    direction = easy_mean - hard_mean
    threshold = float((easy_mean @ direction + hard_mean @ direction) / 2)
    return MassMeanProbe(direction=direction, threshold=threshold)


def pooled_difficulty_direction(language_tables: dict[str, dict], allowed_groups: set[str] | None = None,
                                score_key: str = "ease_composite") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Average label-balanced easy-minus-hard directions with languages equally weighted."""
    directions = {}
    for language, table in language_tables.items():
        selected = np.asarray([index for index, gid in enumerate(table["ids"])
                               if allowed_groups is None or gid in allowed_groups], dtype=int)
        x = table["x"][selected]
        labels = table["y"][selected]
        score = np.asarray([table["behavior"][table["ids"][index]][score_key]
                            for index in selected], dtype=float)
        directions[language] = difficulty_probe(x, labels, score).direction
    return np.mean(np.asarray([directions[language] for language in sorted(directions)]), axis=0), directions
