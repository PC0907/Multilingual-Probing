"""Dependency-free metrics and score-transport decomposition."""
from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if not n0 or not n1:
        raise ValueError("AUROC requires both classes")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def balanced_accuracy(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> float:
    y, pred = np.asarray(y, dtype=int), np.asarray(score) >= threshold
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("Cosine requires nonzero vectors")
    return float(np.clip(a @ b / denominator, -1, 1))


def score_transport(y: np.ndarray, score: np.ndarray, source_threshold: float = 0.0) -> dict:
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    positive, negative = score[y == 1], score[y == 0]
    mu_pos, mu_neg = float(positive.mean()), float(negative.mean())
    var_pos, var_neg = float(positive.var(ddof=1)), float(negative.var(ddof=1))
    pooled = float(np.sqrt(((len(positive) - 1) * var_pos + (len(negative) - 1) * var_neg) /
                           (len(positive) + len(negative) - 2)))
    return {
        "mu_positive": mu_pos,
        "mu_negative": mu_neg,
        "sd_positive": float(np.sqrt(var_pos)),
        "sd_negative": float(np.sqrt(var_neg)),
        "pooled_sd": pooled,
        "standardized_separation": (mu_pos - mu_neg) / pooled if pooled else None,
        "global_offset": (mu_pos + mu_neg) / 2 - source_threshold,
        "auroc": auroc(y, score),
        "balanced_accuracy": balanced_accuracy(y, score, source_threshold),
    }


def optimal_balanced_accuracy_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Validation-only threshold; deterministic tie break closest to class midpoint."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    unique = np.unique(score)
    candidates = np.concatenate(([-np.inf], (unique[:-1] + unique[1:]) / 2, [np.inf]))
    values = np.asarray([balanced_accuracy(y, score, threshold) for threshold in candidates])
    best = np.flatnonzero(values == values.max())
    midpoint = float((score[y == 1].mean() + score[y == 0].mean()) / 2)
    chosen = int(best[np.argmin(np.abs(candidates[best] - midpoint))])
    return float(candidates[chosen]), float(values[chosen])


def fit_logistic_calibrator(score: np.ndarray, y: np.ndarray, iterations: int = 100) -> tuple[float, float]:
    """Fit scalar slope/intercept by damped Newton steps without external dependencies."""
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=float)
    design = np.column_stack([score, np.ones(len(score))])
    parameters = np.zeros(2, dtype=float)
    ridge = np.diag([1e-8, 1e-8])
    for _ in range(iterations):
        linear = np.clip(design @ parameters, -40, 40)
        probability = 1 / (1 + np.exp(-linear))
        gradient = design.T @ (probability - y) + ridge @ parameters
        weight = np.maximum(probability * (1 - probability), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + ridge
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(parameters[0]), float(parameters[1])


def calibrated_probabilities(score: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    linear = np.clip(np.asarray(score, dtype=float) * slope + intercept, -40, 40)
    return 1 / (1 + np.exp(-linear))


def calibration_metrics(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict:
    """Adaptive equal-count ECE plus proper scoring rules."""
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    order = np.argsort(probability, kind="mergesort")
    chunks = [chunk for chunk in np.array_split(order, min(bins, len(order))) if len(chunk)]
    ece = sum(len(chunk) / len(y) * abs(float(probability[chunk].mean() - y[chunk].mean()))
              for chunk in chunks)
    return {
        "adaptive_ece": float(ece),
        "adaptive_bins": len(chunks),
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
    }
