"""Mass-mean truth directions (Marks & Tegmark, COLM 2024).

The direction is arithmetic, not a fit::

    d = mean(activations | true) - mean(activations | false)

Everything the two classes share — syntax, topic, "I am processing a declarative
sentence" — appears in both means and cancels. What survives is the component
that systematically distinguishes true from false statements. That is why the
direction is content-independent: it is not the vector of any word, it is what is
left after averaging the words away.

Two variants are provided:

* `mass_mean_direction` — the raw difference of means. Use this for RQ1 geometry.
  It is the object the paper is about, and it is what should be compared across
  languages.

* `mass_mean_iid_direction` — the difference of means whitened by the pooled
  within-class covariance. Use this for CLASSIFICATION (RQ2 transfer). Marks &
  Tegmark show the whitened version separates better, because activation space is
  anisotropic and the raw difference points partly along high-variance directions
  that carry no truth information.

Do not mix them. Whitening is a linear map that depends on the language's own
covariance, so two whitened directions live in differently-transformed spaces and
their cosine is not a meaningful geometric comparison. Geometry uses raw;
transfer uses whitened. This distinction is load-bearing for the RQ1/RQ2
dissociation: if raw directions diverge while whitened probes still transfer,
that is exactly the result the project is looking for, and it would be invisible
if one variant were used for both.

SIGN CONVENTION
---------------
Fixed once, here, and never overridden: the direction points from FALSE toward
TRUE. Positive projection means "more true".

This matters more than it sounds. Cosine similarity is sign-sensitive, and a
flipped convention in one language turns +0.8 into -0.8, which reads as perfect
anti-alignment rather than perfect alignment. Every direction-producing function
in this repo must obey it, and `check_sign_convention` asserts it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mass_mean_direction",
    "mass_mean_iid_direction",
    "project",
    "classify",
    "check_sign_convention",
    "TRUE_LABEL",
    "FALSE_LABEL",
]

TRUE_LABEL = 1
FALSE_LABEL = 0


def _split_classes(
    activations: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels).ravel()

    if activations.ndim != 2:
        raise ValueError(f"activations must be [n, d], got {activations.shape}")
    if labels.shape[0] != activations.shape[0]:
        raise ValueError(
            f"got {activations.shape[0]} activations but {labels.shape[0]} labels"
        )

    true_mask = labels == TRUE_LABEL
    false_mask = labels == FALSE_LABEL

    unexpected = ~(true_mask | false_mask)
    if unexpected.any():
        bad = np.unique(labels[unexpected])
        raise ValueError(
            f"labels must be {FALSE_LABEL}/{TRUE_LABEL}, found {bad.tolist()}"
        )
    if not true_mask.any() or not false_mask.any():
        raise ValueError(
            f"need both classes; got {int(true_mask.sum())} true, "
            f"{int(false_mask.sum())} false"
        )

    return activations[true_mask], activations[false_mask]


def mass_mean_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """Raw difference of class means. Points from false toward true.

    Args:
        activations: [n, d] hidden states at one layer, one position.
        labels: [n] with 1 = true, 0 = false.
        normalise: return a unit vector. Default True, because magnitude is
            arbitrary for geometry and carries the class-imbalance artefact —
            an unbalanced set produces a shorter difference vector without the
            orientation changing. Set False only when the magnitude is wanted
            for steering-coefficient calibration.

    Returns:
        [d] direction vector.
    """
    x_true, x_false = _split_classes(activations, labels)
    direction = x_true.mean(axis=0) - x_false.mean(axis=0)

    if normalise:
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(
                "class means are identical — the layer carries no linear truth "
                "signal, or the labels are shuffled"
            )
        direction = direction / norm
    return direction


def mass_mean_iid_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    shrinkage: float = 1e-3,
    normalise: bool = True,
) -> np.ndarray:
    """Difference of means whitened by pooled within-class covariance.

    Equivalent to the LDA direction, and to Marks & Tegmark's "mass-mean probe
    with IID assumption". Better for classification; wrong for cross-language
    geometry (see module docstring).

    Args:
        shrinkage: ridge added to the covariance diagonal, as a fraction of the
            mean diagonal value. Required, not optional: with d in the thousands
            and n in the hundreds, the empirical covariance is rank-deficient and
            its inverse is pure noise. If transfer results swing wildly with
            small changes to this value, the sample size is too small and the
            number should not be reported.
    """
    x_true, x_false = _split_classes(activations, labels)
    diff = x_true.mean(axis=0) - x_false.mean(axis=0)

    centred = np.vstack(
        [x_true - x_true.mean(axis=0), x_false - x_false.mean(axis=0)]
    )
    n, dim = centred.shape
    cov = centred.T @ centred / max(n - 2, 1)
    cov.flat[:: dim + 1] += shrinkage * np.trace(cov) / dim

    try:
        direction = np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(cov) @ diff

    if normalise:
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("whitened direction is degenerate")
        direction = direction / norm
    return direction


def project(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Scalar projection of each activation onto the direction.

    Higher means more "true", given the sign convention above.
    """
    activations = np.asarray(activations, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).ravel()
    if activations.shape[-1] != direction.shape[0]:
        raise ValueError(
            f"dimension mismatch: activations {activations.shape[-1]}, "
            f"direction {direction.shape[0]}"
        )
    return activations @ direction


def classify(
    activations: np.ndarray,
    direction: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary predictions from projections.

    The threshold is a SEPARATE parameter from the direction, and this separation
    is the point. Cross-lingual transfer can fail for two unrelated reasons: the
    direction is wrong, or the direction is right but the threshold is
    miscalibrated because the target language's activations sit elsewhere in the
    space with a different mean offset.

    Only the first is a claim about truth representation. Report transfer both
    with the source threshold (realistic deployment: an English-tuned monitor
    applied unchanged) and with a target-refitted threshold (isolates direction
    quality from calibration). The gap between them is itself a result.
    """
    return (project(activations, direction) > threshold).astype(int)


def check_sign_convention(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
) -> bool:
    """Assert the direction points from false toward true on its own data.

    Cheap, and worth calling after every direction is computed. A silent sign
    flip in one language turns strong alignment into strong anti-alignment, and
    the resulting figure looks like a finding rather than a bug.
    """
    x_true, x_false = _split_classes(activations, labels)
    return bool(
        project(x_true, direction).mean() > project(x_false, direction).mean()
    )
