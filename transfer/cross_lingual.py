"""RQ2: cross-lingual probe transfer.

Train a probe in language A, evaluate it in language B, for every ordered pair.
This is the measurement CrossHallu and Shared Doubt already report; it earns its
place here only because it sits next to RQ1's geometry and the two can disagree.

FOUR DECISIONS THAT DETERMINE WHAT THE GRID MEANS
--------------------------------------------------

1.  WHITENED DIRECTIONS, NOT RAW.
    Transfer is a classification question, so `mass_mean_iid_direction` is the
    right probe — it discounts high-variance nuisance directions that carry no
    truth signal. RQ1 uses the raw direction because whitening rotates into a
    language-specific frame and destroys the geometric comparison. Using one
    variant for both would hide the dissociation the project is looking for: raw
    directions diverging while whitened probes still transfer is exactly the
    CLAS-style result.

2.  TWO THRESHOLDS, ALWAYS REPORTED TOGETHER.
    A probe can fail to transfer for two unrelated reasons: the direction is
    wrong, or the direction is right but the target language's activations sit
    elsewhere in the space so the source threshold lands in the wrong place.
    Only the first says anything about truth representation.

        auroc               threshold-free. The cleanest measure of whether the
                            DIRECTION separates the target classes.
        acc_source_thresh   source-language threshold applied unchanged. This is
                            the realistic deployment number — an English-tuned
                            monitor pointed at Tamil.
        acc_target_thresh   threshold refitted on the target. Isolates direction
                            quality from calibration.

    The gap between the last two is itself a result, and it is the number that
    tells a safety-monitoring audience whether the failure is fixable by
    recalibration or not.

3.  THE DIAGONAL IS A CEILING, NOT A DATA POINT.
    In-language transfer (A -> A) is computed on held-out data and used to
    normalise the off-diagonal: a probe cannot transfer into a language better
    than it works within it. `relative_transfer` divides by the target's own
    diagonal, which separates "the probe does not transfer" from "the probe does
    not work in that language at all". These are different findings and the raw
    grid conflates them.

4.  EQUAL SAMPLE SIZES.
    As in RQ1. A direction estimated from 3000 statements will outperform one
    from 600 regardless of language.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

import numpy as np

from ..probes.mass_mean import (
    mass_mean_direction,
    mass_mean_iid_direction,
    project,
)

__all__ = [
    "TransferResult",
    "auroc",
    "best_threshold",
    "evaluate_transfer",
    "transfer_grid",
    "add_relative_transfer",
]


@dataclass
class TransferResult:
    """One cell of the transfer grid."""

    source: str
    target: str
    layer: int
    auroc: float
    acc_source_thresh: float
    acc_target_thresh: float
    n_source: int
    n_target: int

    @property
    def calibration_gap(self) -> float:
        """How much of the failure is threshold placement rather than direction.

        Large gap: the direction is fine, the monitor just needs recalibrating
        per language. Small gap with low AUROC: the direction itself does not
        transfer, and no amount of calibration will fix it.
        """
        return self.acc_target_thresh - self.acc_source_thresh

    def to_dict(self) -> dict:
        return {**asdict(self), "calibration_gap": self.calibration_gap}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Threshold-free by construction, which is why it is the primary transfer
    metric here: it answers "does this direction order the target statements
    correctly" without entangling the answer with where a cutoff was placed.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both classes, got {n_pos} positive, {n_neg} negative")

    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks within ties, or tied scores bias the statistic.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        summed = np.zeros(len(unique))
        np.add.at(summed, inverse, ranks)
        ranks = (summed / counts)[inverse]

    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy-maximising threshold on the given data.

    Used two ways: fitted on the SOURCE to get the deployment threshold, and
    fitted on the TARGET to get the recalibrated one. Fitting on the target and
    reporting only that number would overstate real-world transfer, since a
    deployed monitor has no target labels to calibrate with.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    order = scores.argsort()
    sorted_scores, sorted_labels = scores[order], labels[order]

    n_pos = int((labels == 1).sum())
    # Predict positive above each candidate cut; sweep all of them.
    true_pos = n_pos - np.cumsum(sorted_labels == 1)
    true_neg = np.cumsum(sorted_labels == 0)
    accuracy = (true_pos + true_neg) / len(labels)

    best = int(accuracy.argmax())
    if best + 1 < len(sorted_scores):
        return float((sorted_scores[best] + sorted_scores[best + 1]) / 2)
    return float(sorted_scores[best])


def evaluate_transfer(
    source_acts: np.ndarray,
    source_labels: np.ndarray,
    target_acts: np.ndarray,
    target_labels: np.ndarray,
    source: str,
    target: str,
    layer: int,
    whiten: bool = True,
) -> TransferResult:
    """Train in source, evaluate in target.

    Args:
        whiten: use the covariance-corrected direction. True for transfer (see
            module docstring); set False only to check whether a transfer result
            survives without whitening, which is a useful robustness check
            precisely because whitening is fitted on the source language.
    """
    direction_fn = mass_mean_iid_direction if whiten else mass_mean_direction
    direction = direction_fn(source_acts, source_labels)

    source_scores = project(source_acts, direction)
    target_scores = project(target_acts, direction)

    threshold_source = best_threshold(source_scores, source_labels)
    threshold_target = best_threshold(target_scores, target_labels)

    return TransferResult(
        source=source,
        target=target,
        layer=layer,
        auroc=auroc(target_scores, target_labels),
        acc_source_thresh=float(
            ((target_scores > threshold_source).astype(int) == target_labels).mean()
        ),
        acc_target_thresh=float(
            ((target_scores > threshold_target).astype(int) == target_labels).mean()
        ),
        n_source=len(source_labels),
        n_target=len(target_labels),
    )


def transfer_grid(
    activations: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    layer: int,
    holdout_fraction: float = 0.5,
    whiten: bool = True,
    seed: int = 0,
) -> list[TransferResult]:
    """Every ordered pair, including the diagonal.

    The diagonal matters and is easy to get wrong: A -> A must be evaluated on
    HELD-OUT data from A, never on the statements the direction was estimated
    from. Training and testing on the same rows produces an inflated diagonal,
    which then makes every off-diagonal cell look worse by comparison and
    manufactures a transfer gap that is really an overfitting gap.

    Args:
        activations: language -> [n, d] at this layer, already subsampled to a
            common size by the caller.
        labels: language -> [n].
        holdout_fraction: share of each language reserved for evaluation.
    """
    languages = sorted(activations)
    rng = np.random.default_rng(seed)

    train, test = {}, {}
    for lang in languages:
        acts, y = activations[lang], labels[lang]
        if len(acts) != len(y):
            raise ValueError(f"{lang}: {len(acts)} activations, {len(y)} labels")
        idx = rng.permutation(len(acts))
        cut = int(len(acts) * (1 - holdout_fraction))
        train[lang] = (acts[idx[:cut]], y[idx[:cut]])
        test[lang] = (acts[idx[cut:]], y[idx[cut:]])

        for name, (_, part) in (("train", train[lang]), ("test", test[lang])):
            if len(np.unique(part)) < 2:
                raise ValueError(
                    f"{lang} {name} split is single-class; increase n or "
                    "check label balance"
                )

    sizes = {len(activations[l]) for l in languages}
    if len(sizes) > 1:
        raise ValueError(
            f"languages have different sample sizes {sizes}. A direction from "
            "more data transfers better regardless of language; subsample to a "
            "common size before calling."
        )

    results = []
    for source, target in itertools.product(languages, repeat=2):
        source_acts, source_y = train[source]
        target_acts, target_y = test[target]
        results.append(
            evaluate_transfer(
                source_acts, source_y, target_acts, target_y,
                source=source, target=target, layer=layer, whiten=whiten,
            )
        )
    return results


def add_relative_transfer(results: list[TransferResult]) -> list[dict]:
    """Normalise each off-diagonal cell by its TARGET's own diagonal.

    relative_transfer = (auroc(A->B) - 0.5) / (auroc(B->B) - 0.5)

    1.0 means the foreign probe does as well in B as B's own probe. Near 0 means
    it does no better than chance. Without this, a low A->B cell is ambiguous
    between "the probe does not transfer" and "nothing works in B, including B's
    own probe" — and those support opposite conclusions about whether truth is
    represented differently in B.
    """
    diagonal = {
        (r.target, r.layer): r.auroc
        for r in results if r.source == r.target
    }

    rows = []
    for r in results:
        own = diagonal.get((r.target, r.layer))
        denominator = (own - 0.5) if own is not None else None
        relative = (
            (r.auroc - 0.5) / denominator
            if denominator and abs(denominator) > 0.02
            else np.nan       # target's own probe is at chance; ratio meaningless
        )
        rows.append({**r.to_dict(), "target_diagonal_auroc": own,
                     "relative_transfer": relative})
    return rows
