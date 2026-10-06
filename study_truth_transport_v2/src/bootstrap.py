"""Fact-group bootstrap helpers."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def stratified_group_bootstrap_indices(labels: np.ndarray, group_ids: np.ndarray, draws: int, seed: int):
    labels, group_ids = np.asarray(labels, dtype=int), np.asarray(group_ids)
    members = {g: np.flatnonzero(group_ids == g) for g in sorted(set(group_ids.tolist()))}
    strata = defaultdict(list)
    for group, ix in members.items():
        signature = (int((labels[ix] == 0).sum()), int((labels[ix] == 1).sum()))
        strata[signature].append(group)
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        selected = []
        for signature in sorted(strata):
            candidates = strata[signature]
            for j in rng.integers(0, len(candidates), size=len(candidates)):
                selected.extend(members[candidates[int(j)]].tolist())
        yield np.asarray(selected, dtype=int)
