"""Exact fixed-n, fixed-composition, controlled-overlap allocation for RQ-A."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import hashlib
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    label: int
    stratum: str


def _seed(base: int, *parts: object) -> int:
    value = "|".join(map(str, (base, *parts))).encode()
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def _capped_largest_remainder(weights: dict[str, int], total: int, caps: dict[str, int]) -> dict[str, int]:
    """Allocate exactly total integer units, near-proportionally, without exceeding caps."""
    if total < 0 or set(weights) != set(caps) or any(v < 0 for v in weights.values()) or any(v < 0 for v in caps.values()):
        raise ValueError("Invalid weights, total, or caps")
    if sum(caps.values()) < total:
        raise ValueError(f"Insufficient capped capacity: {sum(caps.values())} < {total}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Positive total weight required")
    ideal = {k: total * weights[k] / weight_sum for k in weights}
    out = {k: min(caps[k], int(math.floor(ideal[k]))) for k in weights}
    while sum(out.values()) < total:
        candidates = [k for k in weights if out[k] < caps[k]]
        if not candidates:
            raise AssertionError("Capacity vanished during allocation")
        chosen = max(candidates, key=lambda k: (ideal[k] - out[k], weights[k], k))
        out[chosen] += 1
    return out


def composition_quota(records: Iterable[GroupRecord], n_per_fit: int = 400) -> dict[str, int]:
    """Choose one fixed stratum composition feasible for two disjoint fits.

    Labels are balanced exactly. Each stratum is capped at floor(pool_size / 2),
    which guarantees the 0%-overlap condition is feasible.
    """
    records = list(records)
    if n_per_fit <= 0 or n_per_fit % 2:
        raise ValueError("n_per_fit must be positive and even")
    if len({r.group_id for r in records}) != len(records):
        raise ValueError("Group IDs must be unique")
    if set(r.label for r in records) != {0, 1}:
        raise ValueError("Both binary labels are required")
    counts = Counter((r.label, r.stratum) for r in records)
    quota: dict[str, int] = {}
    for label in (0, 1):
        weights = {s: n for (y, s), n in counts.items() if y == label}
        caps = {s: n // 2 for s, n in weights.items()}
        selected = _capped_largest_remainder(weights, n_per_fit // 2, caps)
        quota.update({f"{label}|{s}": n for s, n in selected.items() if n})
    if sum(quota.values()) != n_per_fit:
        raise AssertionError("Quota size mismatch")
    return quota


def _overlap_quota(fit_quota: dict[str, int], shared_total: int) -> dict[str, int]:
    if not 0 <= shared_total <= sum(fit_quota.values()):
        raise ValueError("Invalid shared total")
    return _capped_largest_remainder(fit_quota, shared_total, fit_quota)


def build_overlap_plan(
    records: Iterable[GroupRecord],
    *,
    n_per_fit: int = 400,
    overlap_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    repetitions: int = 50,
    seed: int = 20261002,
) -> dict:
    """Return paired allocation plans with exact n, balance, composition and overlap.

    Within a repetition, sample A is fixed across overlap conditions. In each
    stratum, increasing-overlap samples are nested prefixes of sample A; sample
    B's exclusive portion is a prefix of a fixed disjoint ordering. This pairs
    conditions and lowers Monte Carlo noise without changing their marginals.
    """
    records = list(records)
    if repetitions < 1:
        raise ValueError("At least one repetition required")
    fractions = tuple(float(x) for x in overlap_fractions)
    if not fractions or any(x < 0 or x > 1 for x in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError("Overlap fractions must be unique and in [0,1]")
    shared_totals = []
    for fraction in fractions:
        raw = fraction * n_per_fit
        if not float(raw).is_integer():
            raise ValueError("Each requested overlap must produce an integer shared count")
        shared_totals.append(int(raw))
    quota = composition_quota(records, n_per_fit)
    pools: dict[str, list[str]] = defaultdict(list)
    lookup = {}
    for r in records:
        key = f"{r.label}|{r.stratum}"
        pools[key].append(r.group_id)
        lookup[r.group_id] = r
    if set(quota) - set(pools):
        raise AssertionError("Quota references absent strata")

    plans = []
    for repetition in range(repetitions):
        source_by_stratum, exclusive_by_stratum = {}, {}
        for key, q in sorted(quota.items()):
            candidates = np.asarray(sorted(pools[key]), dtype=object)
            rng = np.random.default_rng(_seed(seed, repetition, key))
            candidates = candidates[rng.permutation(len(candidates))].tolist()
            if len(candidates) < 2 * q:
                raise AssertionError("Quota feasibility violated")
            source_by_stratum[key] = candidates[:q]
            exclusive_by_stratum[key] = candidates[q:2 * q]
        source = sorted(x for xs in source_by_stratum.values() for x in xs)
        targets = []
        for fraction, shared_total in zip(fractions, shared_totals):
            overlap_quota = _overlap_quota(quota, shared_total)
            target = []
            for key, q in sorted(quota.items()):
                shared = overlap_quota.get(key, 0)
                target.extend(source_by_stratum[key][:shared])
                target.extend(exclusive_by_stratum[key][:(q - shared)])
            target = sorted(target)
            observed_overlap = len(set(source) & set(target))
            plans.append({
                "repetition": repetition,
                "requested_overlap_fraction": fraction,
                "shared_groups": observed_overlap,
                "source_group_ids": source,
                "target_group_ids": target,
                "overlap_by_stratum": overlap_quota,
            })
    result = {
        "seed": seed,
        "n_per_fit": n_per_fit,
        "n_true_per_fit": n_per_fit // 2,
        "n_false_per_fit": n_per_fit // 2,
        "overlap_fractions": list(fractions),
        "repetitions": repetitions,
        "fit_composition": quota,
        "eligible_groups": sum(len(pools[k]) for k in quota),
        "excluded_groups": sorted(r.group_id for r in records if f"{r.label}|{r.stratum}" not in quota),
        "plans": plans,
    }
    validate_overlap_plan(records, result)
    return result


def validate_overlap_plan(records: Iterable[GroupRecord], plan: dict) -> None:
    records = list(records)
    lookup = {r.group_id: r for r in records}
    n = int(plan["n_per_fit"])
    expected_quota = plan["fit_composition"]
    by_rep = defaultdict(list)
    for row in plan["plans"]:
        by_rep[int(row["repetition"])].append(row)
        a, b = row["source_group_ids"], row["target_group_ids"]
        if len(a) != n or len(b) != n or len(set(a)) != n or len(set(b)) != n:
            raise AssertionError("Fit size or uniqueness failure")
        if set(a) - set(lookup) or set(b) - set(lookup):
            raise AssertionError("Unknown group ID")
        for selected in (a, b):
            labels = Counter(lookup[x].label for x in selected)
            if labels != Counter({0: n // 2, 1: n // 2}):
                raise AssertionError(f"Class imbalance: {labels}")
            composition = Counter(f"{lookup[x].label}|{lookup[x].stratum}" for x in selected)
            if dict(composition) != expected_quota:
                raise AssertionError("Stratum composition changed")
        overlap = len(set(a) & set(b))
        expected = round(float(row["requested_overlap_fraction"]) * n)
        if overlap != expected or overlap != row["shared_groups"]:
            raise AssertionError("Overlap mismatch")
    for rows in by_rep.values():
        sources = {tuple(x["source_group_ids"]) for x in rows}
        if len(sources) != 1:
            raise AssertionError("Source fit must remain fixed within repetition")
