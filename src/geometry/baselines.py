"""Floor and ceiling baselines for cross-lingual direction alignment.

A raw cosine similarity between two truth directions is uninterpretable. In a
few-thousand-dimensional space, random directions have expected cosine ~0 with
standard deviation 1/sqrt(d), so for d=4096 anything above ~0.05 is "significant"
against chance while still being nowhere near meaningful. Conversely, two probes
trained on disjoint splits of the *same* language do not reach 1.0 either.

Every alignment number in this project is therefore reported as::

    normalised = (observed - floor) / (ceiling - floor)

where floor comes from `random_direction_floor` and ceiling from
`split_half_ceiling`.

Three subtleties this module exists to enforce:

1.  The ceiling is PER LAYER. Estimation noise differs across depth, so a single
    scalar ceiling would over-correct some layers and under-correct others.

2.  The ceiling must be computed at the SAME SAMPLE SIZE as the cross-lingual
    comparison. A mass-mean direction is an estimate of a population mean, and
    the agreement between two such estimates rises with n. A ceiling built from
    2000 statements per half, compared against cross-lingual directions built
    from 500, will make real alignment look worse than it is. Pass
    `n_per_split` explicitly.

3.  Splits must respect GROUPS. Statement sets contain negation pairs and
    template-derived siblings that share almost all their surface form. If the
    affirmative lands in split A and its negation in split B, the two halves are
    not independent and the ceiling is inflated. Pass `groups` (one id per fact,
    shared across its variants).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "BaselineStats",
    "cosine",
    "random_direction_floor",
    "split_half_ceiling",
    "normalised_alignment",
]

DirectionFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""(activations [n, d], labels [n]) -> direction [d]. See probes/mass_mean.py."""


@dataclass(frozen=True)
class BaselineStats:
    """Summary of a resampled baseline distribution."""

    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    n_samples: int

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "BaselineStats":
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("cannot summarise an empty set of values")
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            p05=float(np.percentile(arr, 5)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            n_samples=int(arr.size),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 if either vector is degenerate (zero norm), which happens when a
    class is empty or all activations are identical — usually a data bug, so the
    caller should check for it rather than silently averaging zeros.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def random_direction_floor(
    dim: int,
    n_pairs: int = 1000,
    seed: int | None = 0,
) -> BaselineStats:
    """Cosine between pairs of random unit vectors in `dim` dimensions.

    This is the chance level. Sampled rather than assumed, because the analytic
    result (mean 0, std ~1/sqrt(dim)) is easy to misremember and cheap to verify.

    Note this floor is isotropic: it assumes random directions are uniform on the
    sphere. Real activation spaces are anisotropic — they have a dominant mean
    direction and a heavily skewed spectrum — so this is a LOWER bound on the
    true chance level. If a stricter floor is needed, resample directions from
    shuffled labels on real activations instead (see `split_half_ceiling` with
    permuted labels), which preserves the space's geometry.
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_pairs, dim))
    b = rng.standard_normal((n_pairs, dim))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return BaselineStats.from_values(np.einsum("ij,ij->i", a, b))


def _grouped_halves(
    n: int,
    groups: np.ndarray | None,
    n_per_split: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two disjoint halves, keeping groups intact."""
    if groups is None:
        idx = rng.permutation(n)
        half = n // 2 if n_per_split is None else n_per_split
        if 2 * half > n:
            raise ValueError(
                f"need {2 * half} examples for two splits of {half}, have {n}"
            )
        return idx[:half], idx[half : 2 * half]

    unique = np.unique(groups)
    rng.shuffle(unique)
    cut = len(unique) // 2
    left = np.flatnonzero(np.isin(groups, unique[:cut]))
    right = np.flatnonzero(np.isin(groups, unique[cut:]))
    if n_per_split is not None:
        if min(len(left), len(right)) < n_per_split:
            raise ValueError(
                f"grouped split gives {len(left)}/{len(right)} examples, "
                f"need {n_per_split} each"
            )
        left = rng.choice(left, n_per_split, replace=False)
        right = rng.choice(right, n_per_split, replace=False)
    return left, right


def split_half_ceiling(
    activations: np.ndarray,
    labels: np.ndarray,
    direction_fn: DirectionFn,
    *,
    groups: np.ndarray | None = None,
    n_repeats: int = 50,
    n_per_split: int | None = None,
    permute_labels: bool = False,
    seed: int | None = 0,
) -> BaselineStats:
    """Agreement between two directions estimated from disjoint splits.

    This is the realistic maximum for a given layer, sample size and model. Two
    probes trained on different halves of the same English data will not agree
    perfectly; whatever they achieve is what cross-lingual alignment should be
    measured against.

    Args:
        activations: [n, d] hidden states at one layer.
        labels: [n] binary, 1 for true statements.
        direction_fn: computes a direction from (activations, labels).
        groups: [n] group ids. Statements derived from the same fact — an
            affirmative and its negation, or several templates over one entity —
            must share a group id, or the halves leak into each other and the
            ceiling comes out too high.
        n_repeats: resamples. The spread across resamples matters as much as the
            mean; a wide spread means the direction is unstable at this sample
            size and the layer's alignment number is not trustworthy.
        n_per_split: examples per half. Set this to the size used in the
            cross-lingual comparison so the ceiling is measured like for like.
        permute_labels: if True, shuffle labels within each split before
            computing directions. This yields an ANISOTROPIC FLOOR — directions
            drawn from real activation geometry but carrying no truth signal.
            Stricter and more honest than `random_direction_floor`; prefer it
            when reporting.
        seed: rng seed.

    Returns:
        Distribution of split-half cosines.
    """
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if activations.ndim != 2:
        raise ValueError(f"activations must be [n, d], got {activations.shape}")
    n, dim = activations.shape
    if labels.shape[0] != n:
        raise ValueError(f"got {n} activations but {labels.shape[0]} labels")
    if groups is not None:
        groups = np.asarray(groups).ravel()
        if groups.shape[0] != n:
            raise ValueError("groups must have one entry per example")

    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(n_repeats):
        left, right = _grouped_halves(n, groups, n_per_split, rng)

        y_left, y_right = labels[left], labels[right]
        if permute_labels:
            y_left = rng.permutation(y_left)
            y_right = rng.permutation(y_right)

        # A split with one class missing gives a meaningless direction.
        if len(np.unique(y_left)) < 2 or len(np.unique(y_right)) < 2:
            continue

        d_left = direction_fn(activations[left], y_left)
        d_right = direction_fn(activations[right], y_right)
        values.append(cosine(d_left, d_right))

    if not values:
        raise RuntimeError(
            "no valid resamples — every split was single-class. Check label "
            "balance, or reduce n_per_split."
        )
    return BaselineStats.from_values(values)


def normalised_alignment(
    observed: float,
    floor: float,
    ceiling: float,
) -> float:
    """Rescale a raw cosine onto the [floor, ceiling] interval.

    0.0 means chance; 1.0 means as aligned as two probes on the same language.
    Values can fall outside [0, 1]: below 0 means worse than chance (rare, and
    usually a sign of a sign-convention bug in `direction_fn`), above 1 means the
    cross-lingual directions agree more than two same-language estimates, which
    happens when the ceiling was computed at a smaller sample size than the
    observation. If you see values above 1, check `n_per_split` before believing
    them.
    """
    denom = ceiling - floor
    if abs(denom) < 1e-12:
        raise ValueError(
            f"ceiling ({ceiling:.4f}) is not above floor ({floor:.4f}); "
            "the probe carries no usable signal at this layer"
        )
    return float((observed - floor) / denom)
