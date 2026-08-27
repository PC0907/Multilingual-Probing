"""Principal angles between truth subspaces (Burger et al., NeurIPS 2024).

Burger et al. find that truth is not one direction but a two-dimensional
subspace: one axis separating true from false, a second separating affirmative
from negated statements. Two languages therefore give two planes, and comparing
planes is not a dot product.

Principal angles are the right tool. Given orthonormal bases A and B of two
k-dimensional subspaces, the singular values of A^T B are the cosines of the
principal angles. The first angle is the smallest angle between any line in A and
any line in B; the second is the smallest among directions orthogonal to that
first pair, and so on.

Reading them:

    both angles near 0 deg    the planes essentially coincide
    one small, one large      the languages share ONE axis but not the other
    both large               unrelated planes

The middle case is the interesting one and is invisible to single-vector cosine.
It would mean, for example, that two languages agree on what makes a statement
true but encode negation differently — a result about compositional structure,
not just about truth.

WHY THIS NEEDS NEGATION PAIRS
-----------------------------
There is no second axis to extract from affirmative statements alone. If the
dataset has no negated counterparts, `truth_subspace` will return a plane whose
second dimension is noise, and the second principal angle will be meaningless
while still printing a number. `truth_subspace` refuses to run without both
statement polarities present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SubspaceComparison",
    "orthonormalise",
    "truth_subspace",
    "principal_angles",
    "compare_subspaces",
    "grassmann_distance",
]

AFFIRMATIVE = 0
NEGATED = 1


@dataclass(frozen=True)
class SubspaceComparison:
    """Result of comparing two subspaces."""

    angles_rad: np.ndarray
    cosines: np.ndarray

    @property
    def angles_deg(self) -> np.ndarray:
        return np.degrees(self.angles_rad)

    @property
    def grassmann(self) -> float:
        """Root-sum-square of the angles: one scalar summary of plane distance."""
        return float(np.sqrt(np.sum(self.angles_rad**2)))

    def __repr__(self) -> str:
        deg = ", ".join(f"{a:.1f}" for a in self.angles_deg)
        return f"SubspaceComparison(angles_deg=[{deg}], grassmann={self.grassmann:.3f})"


def orthonormalise(vectors: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis for the span of the given vectors (columns).

    Uses SVD rather than Gram-Schmidt: the two axes of a truth subspace are
    typically far from orthogonal and can be nearly collinear, which makes
    Gram-Schmidt numerically unstable exactly when it matters.

    Raises if the vectors are effectively rank-deficient — that means the two
    axes you thought you had are really one, and any second principal angle
    computed from the result would be noise.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"expected [d, k], got {vectors.shape}")

    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(s > tol * s[0])) if s[0] > 0 else 0
    if rank < vectors.shape[1]:
        raise ValueError(
            f"vectors span {rank} dimensions, not {vectors.shape[1]} "
            f"(singular values {np.round(s, 6).tolist()}). The axes are "
            "collinear — check that negation pairs are actually present and "
            "that the polarity labels are not constant."
        )
    return u[:, : vectors.shape[1]]


def truth_subspace(
    activations: np.ndarray,
    labels: np.ndarray,
    polarity: np.ndarray,
) -> np.ndarray:
    """Two-dimensional truth subspace from labelled, polarity-marked statements.

    Axis 1 (truth):    mean(true) - mean(false)
    Axis 2 (polarity): mean(affirmative) - mean(negated)

    Both computed as raw differences of means, not whitened, for the same reason
    as `mass_mean_direction`: whitening rotates into a language-specific frame
    and destroys the cross-language comparison.

    Args:
        activations: [n, d] hidden states at one layer.
        labels: [n] 1 = true, 0 = false.
        polarity: [n] 0 = affirmative, 1 = negated.

    Returns:
        [d, 2] orthonormal basis.

    The two raw axes are correlated — negation and falsity are entangled in most
    datasets, since a negated true statement and an affirmative false statement
    can look similar. Orthonormalising does not remove that entanglement, it
    just gives a well-conditioned basis for the plane they span. If the axes are
    nearly collinear, `orthonormalise` raises rather than returning a plane whose
    second dimension is noise.
    """
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    polarity = np.asarray(polarity).ravel()

    if not (activations.shape[0] == labels.shape[0] == polarity.shape[0]):
        raise ValueError(
            f"length mismatch: {activations.shape[0]} activations, "
            f"{labels.shape[0]} labels, {polarity.shape[0]} polarity flags"
        )

    for name, arr, expected in (
        ("labels", labels, (0, 1)),
        ("polarity", polarity, (AFFIRMATIVE, NEGATED)),
    ):
        present = set(np.unique(arr).tolist())
        if not present <= set(expected):
            raise ValueError(f"{name} must be in {expected}, found {sorted(present)}")
        if len(present) < 2:
            raise ValueError(
                f"{name} is constant ({present}). A 2D truth subspace needs both "
                "true and false statements AND both affirmative and negated "
                "forms. Without negation pairs, use mass_mean_direction and "
                "single-vector cosine instead — do not fake a second axis."
            )

    truth_axis = (
        activations[labels == 1].mean(axis=0) - activations[labels == 0].mean(axis=0)
    )
    polarity_axis = (
        activations[polarity == AFFIRMATIVE].mean(axis=0)
        - activations[polarity == NEGATED].mean(axis=0)
    )
    return orthonormalise(np.column_stack([truth_axis, polarity_axis]))


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles in radians between two subspaces, ascending.

    Bases must be orthonormal with columns as basis vectors and the same ambient
    dimension. Subspace dimensions may differ; the number of angles returned is
    the smaller of the two.
    """
    basis_a = np.asarray(basis_a, dtype=np.float64)
    basis_b = np.asarray(basis_b, dtype=np.float64)

    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("bases must be 2-D arrays of shape [d, k]")
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError(
            f"ambient dimension mismatch: {basis_a.shape[0]} vs {basis_b.shape[0]}"
        )

    for name, basis in (("basis_a", basis_a), ("basis_b", basis_b)):
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-6):
            raise ValueError(f"{name} is not orthonormal; call orthonormalise first")

    singular = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    # Guard against 1 + 1e-16 from floating point, which would make arccos nan.
    return np.arccos(np.clip(singular, -1.0, 1.0))


def compare_subspaces(basis_a: np.ndarray, basis_b: np.ndarray) -> SubspaceComparison:
    """Principal angles plus their cosines, as a single result object."""
    angles = principal_angles(basis_a, basis_b)
    return SubspaceComparison(angles_rad=angles, cosines=np.cos(angles))


def grassmann_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Root-sum-square of principal angles — a metric on the Grassmannian.

    Useful as the single number to regress on in RQ2, or to plot as a layer
    profile. Report the individual angles alongside it: a Grassmann distance
    collapses "both axes half-aligned" and "one axis perfect, one orthogonal"
    into the same value, and those are different findings.
    """
    return compare_subspaces(basis_a, basis_b).grassmann
