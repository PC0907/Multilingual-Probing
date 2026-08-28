"""RQ3: does ablating the English pivot subspace break cross-lingual transfer?

Only run this if src/pivot/verify.py reports a control-adjusted replication. If
the pivot does not exist in the model, removing a subspace defined by a
non-existent phenomenon will still change transfer — because removing any
sizeable subspace changes everything — and the result would be meaningless.

THE EXPERIMENT
--------------
1.  Identify a subspace at middle layers that carries English-ness.
2.  Project it out of the residual stream during the forward pass.
3.  Re-extract activations and re-run cross-lingual transfer.
4.  Ask whether transfer degrades MORE than a matched control ablation does.

Step 4 is the whole experiment. Steps 1-3 alone produce a number that always
looks like a result.

WHY THE CONTROL ABLATION IS NOT OPTIONAL
-----------------------------------------
Removing a k-dimensional subspace from the residual stream damages the model.
Perplexity rises, activations shift, and every probe gets worse — including
probes that have nothing to do with English or with truth. So "transfer dropped
after we ablated the pivot" is not evidence about mediation. It is evidence that
ablation is destructive.

The comparison that carries information is against a control subspace of the
SAME dimensionality, removed at the SAME layers, chosen to be unrelated to
language. Three controls are provided, and they answer different objections:

    random          random k-dimensional subspace. Cheapest, weakest: random
                    directions in an anisotropic space carry little variance, so
                    this under-damages the model and makes the pivot look
                    special by comparison.
    variance_matched a subspace spanned by principal components whose total
                    variance matches the pivot subspace's. This is the honest
                    control — it damages the model as much as the pivot ablation
                    does, without being about language.
    topic           a subspace encoding some other categorical distinction
                    present in the data (statement type, say). Controls for
                    "removing any categorical direction hurts".

Report against `variance_matched` as primary. A pivot effect that disappears
under variance matching was a damage effect.

THE MEDIATION CLAIM NEEDS A THIRD CONDITION
--------------------------------------------
Even a clean differential effect does not establish mediation unless the
ablation leaves in-language probes intact. If ablating the pivot also destroys
the English probe's within-English performance, transfer had nowhere to go and
the finding is about the probe, not about routing. `mediation_evidence` checks
all three quantities together and refuses to call it mediation otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.subspace import orthonormalise

__all__ = [
    "AblationResult",
    "language_subspace",
    "project_out",
    "control_subspace",
    "make_ablation_hook",
    "mediation_evidence",
]


@dataclass
class AblationResult:
    """Transfer under one ablation condition."""

    condition: str
    layers_ablated: tuple[int, ...]
    rank: int
    variance_removed: float          # fraction of total activation variance
    transfer_auroc: float
    source_in_language_auroc: float
    target_in_language_auroc: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def language_subspace(
    activations_by_language: dict[str, np.ndarray],
    pivot_language: str = "en",
    rank: int = 8,
) -> np.ndarray:
    """A subspace separating the pivot language from the others.

    Built as the top principal components of the between-language mean
    differences, which is a linear-discriminant-style construction rather than a
    single mean-difference vector. Language identity is known to be encoded
    strongly and across multiple dimensions, so a rank-1 direction would remove
    only part of it and under-state the ablation's effect.

    Args:
        activations_by_language: language -> [n, d] at one layer. Sample sizes
            must match, or the mean estimates differ in noise and the subspace
            partly encodes "which language had more data".
        rank: dimensionality to remove. Report results across several ranks; a
            conclusion that holds only at one rank is a tuning artefact.

    Returns:
        [d, rank] orthonormal basis.
    """
    sizes = {len(a) for a in activations_by_language.values()}
    if len(sizes) > 1:
        raise ValueError(
            f"languages have different sample sizes {sizes}; subsample to a "
            "common size or the subspace encodes sample size as well as language"
        )
    if pivot_language not in activations_by_language:
        raise ValueError(f"{pivot_language!r} not among the languages provided")

    pivot_mean = activations_by_language[pivot_language].mean(axis=0)
    differences = np.stack([
        pivot_mean - acts.mean(axis=0)
        for language, acts in activations_by_language.items()
        if language != pivot_language
    ])

    if differences.shape[0] < rank:
        # With k languages you get at most k-1 independent difference vectors.
        # Padding with principal components of the pivot language's own spread
        # would silently change what is being removed.
        raise ValueError(
            f"rank {rank} requested but only {differences.shape[0]} language "
            "contrasts available. Reduce the rank or add languages — do not pad "
            "the basis with unrelated directions."
        )

    u, _, _ = np.linalg.svd(differences.T, full_matrices=False)
    return u[:, :rank]


def project_out(activations: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of `basis` from each activation.

        x' = x - B (B^T x)

    `basis` must be orthonormal, or the projection over- or under-removes.
    """
    activations = np.asarray(activations, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-6):
        raise ValueError("basis is not orthonormal; call orthonormalise first")

    return activations - (activations @ basis) @ basis.T


def control_subspace(
    activations: np.ndarray,
    rank: int,
    target_variance: float | None = None,
    kind: str = "variance_matched",
    seed: int = 0,
) -> np.ndarray:
    """A subspace to ablate as a control.

    Args:
        activations: [n, d] pooled activations at this layer, all languages.
        rank: same rank as the pivot subspace.
        target_variance: fraction of total variance the pivot subspace removes.
            Required for `variance_matched`. Matching it is what makes the
            control comparable: an equally destructive ablation that is not about
            language.
        kind: "random" or "variance_matched".

    For `variance_matched`, principal components are selected to approach the
    target variance rather than simply taking the top ones — the leading PCs
    typically carry far more variance than a language subspace, so ablating them
    would over-damage the model and make the pivot look harmless.
    """
    rng = np.random.default_rng(seed)
    activations = np.asarray(activations, dtype=np.float64)
    dim = activations.shape[1]

    if kind == "random":
        return orthonormalise(rng.standard_normal((dim, rank)))

    if kind != "variance_matched":
        raise ValueError(f"unknown control kind {kind!r}")

    if target_variance is None:
        raise ValueError(
            "variance_matched control needs the pivot subspace's variance share; "
            "without it the control is not matched to anything"
        )

    centred = activations - activations.mean(axis=0)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    variance = variance / variance.sum()

    # Walk down the spectrum for a contiguous block of `rank` components whose
    # variance share is closest to the target.
    best_start, best_gap = 0, np.inf
    for start in range(0, len(variance) - rank + 1):
        gap = abs(variance[start : start + rank].sum() - target_variance)
        if gap < best_gap:
            best_start, best_gap = start, gap

    return orthonormalise(right[best_start : best_start + rank].T)


def subspace_variance_share(activations: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of total variance lying in the span of `basis`."""
    centred = np.asarray(activations, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    total = float((centred**2).sum())
    if total == 0:
        return 0.0
    projected = centred @ np.asarray(basis, dtype=np.float64)
    return float((projected**2).sum() / total)


def make_ablation_hook(basis: np.ndarray, layers: tuple[int, ...]):
    """A forward hook that projects out `basis` at the given layers.

    Ablating during the forward pass, rather than post-hoc on cached
    activations, is what makes this causal. Post-hoc removal only changes what
    the probe sees; it does not change what the model computes downstream, so it
    cannot test whether later layers depended on the subspace.

    Returns a hook to register on the relevant decoder blocks with
    `module.register_forward_hook(hook)`.
    """
    import torch

    basis_tensor = torch.as_tensor(np.asarray(basis), dtype=torch.float32)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        b = basis_tensor.to(hidden.device, hidden.dtype)
        cleaned = hidden - (hidden @ b) @ b.T
        if isinstance(output, tuple):
            return (cleaned, *output[1:])
        return cleaned

    hook.layers = layers
    return hook


def mediation_evidence(
    baseline: AblationResult,
    pivot: AblationResult,
    control: AblationResult,
    min_differential: float = 0.05,
) -> dict:
    """Decide whether the ablation supports a mediation claim.

    Three conditions, all required:

      1. Transfer drops under pivot ablation.
      2. It drops MORE than under the variance-matched control — otherwise the
         effect is damage, not routing.
      3. In-language probes survive in both languages. If the ablation destroys
         the within-language probe too, transfer had nothing left to mediate and
         the result is about the probe rather than about the pivot.

    Failing (3) while passing (1) and (2) is the most seductive failure mode,
    because the differential looks clean.
    """
    transfer_drop = baseline.transfer_auroc - pivot.transfer_auroc
    control_drop = baseline.transfer_auroc - control.transfer_auroc
    differential = transfer_drop - control_drop

    in_language_retained = min(
        pivot.source_in_language_auroc - 0.5,
        pivot.target_in_language_auroc - 0.5,
    ) / max(
        min(
            baseline.source_in_language_auroc - 0.5,
            baseline.target_in_language_auroc - 0.5,
        ),
        1e-6,
    )

    conditions = {
        "transfer_dropped": transfer_drop > 0,
        "exceeds_control": differential > min_differential,
        "in_language_survived": in_language_retained > 0.7,
    }
    supported = all(conditions.values())

    if supported:
        verdict = (
            f"Ablating the pivot costs {differential:.3f} AUROC beyond a "
            "variance-matched control while in-language probes survive. "
            "Consistent with the pivot mediating cross-lingual transfer."
        )
    elif not conditions["in_language_survived"]:
        verdict = (
            f"In-language probe performance fell to {in_language_retained:.0%} of "
            "baseline. The ablation broke the probe itself, so the transfer drop "
            "says nothing about mediation. Reduce the rank or ablate at fewer "
            "layers."
        )
    elif not conditions["exceeds_control"]:
        verdict = (
            f"Pivot ablation costs {transfer_drop:.3f}, the variance-matched "
            f"control {control_drop:.3f}; the differential of {differential:.3f} "
            "is within noise. Removing ANY subspace of this size hurts transfer "
            "this much, so there is no evidence the pivot specifically mediates."
        )
    else:
        verdict = "Transfer did not drop under pivot ablation. No mediation."

    return {
        "transfer_drop": transfer_drop,
        "control_drop": control_drop,
        "differential": differential,
        "in_language_retained": in_language_retained,
        "conditions": conditions,
        "mediation_supported": supported,
        "verdict": verdict,
    }
