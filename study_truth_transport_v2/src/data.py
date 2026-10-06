"""Dataset_v2 loading and group-level aggregation utilities."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LANGUAGES = ("en", "de", "ar", "hi", "fr", "es")


def load_parallel_dataset(root: Path) -> dict[str, list[dict]]:
    rows = {language: json.loads((root / "data" / f"{language}.json").read_text()) for language in LANGUAGES}
    reference = [(r["id"], r["label"], r["group_id"], r["partition"]) for r in rows["en"]]
    if len(reference) != 2000 or len({x[0] for x in reference}) != 2000:
        raise ValueError("Expected 2,000 unique aligned IDs")
    for language in LANGUAGES:
        observed = [(r["id"], r["label"], r["group_id"], r["partition"]) for r in rows[language]]
        if observed != reference:
            raise ValueError(f"Parallel alignment failure for {language}")
    return rows


def group_index(rows: list[dict]) -> dict[str, np.ndarray]:
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[row["group_id"]].append(index)
    return {key: np.asarray(value, dtype=int) for key, value in sorted(members.items())}


def aggregate_group_activations(activations: np.ndarray, rows: list[dict], partition: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average paraphrase-member activations so each dependency group has weight one."""
    if activations.shape[0] != len(rows):
        raise ValueError("Activation rows do not match dataset rows")
    groups = group_index(rows)
    selected = [(gid, ix) for gid, ix in groups.items() if rows[int(ix[0])]["partition"] == partition]
    vectors, labels, ids = [], [], []
    for gid, ix in selected:
        group_labels = {int(rows[int(i)]["label"]) for i in ix}
        if len(group_labels) != 1:
            raise ValueError(f"Mixed-label dependency group {gid}")
        vectors.append(np.asarray(activations[ix], dtype=np.float64).mean(axis=0))
        labels.append(group_labels.pop())
        ids.append(gid)
    return np.asarray(vectors), np.asarray(labels, dtype=int), ids


def pure_group_activations(activations: np.ndarray, rows: list[dict], partition: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Aggregate pure-label groups and report mixed-label groups instead of failing."""
    if activations.shape[0] != len(rows):
        raise ValueError("Activation rows do not match dataset rows")
    groups = group_index(rows)
    vectors, labels, ids, excluded = [], [], [], []
    for gid, ix in groups.items():
        partitions = {rows[int(i)]["partition"] for i in ix}
        if len(partitions) != 1:
            raise ValueError(f"Dependency group crosses partitions: {gid}")
        if partition not in partitions:
            continue
        group_labels = {int(rows[int(i)]["label"]) for i in ix}
        if len(group_labels) != 1:
            excluded.append(gid)
            continue
        vectors.append(np.asarray(activations[ix], dtype=np.float64).mean(axis=0))
        labels.append(group_labels.pop())
        ids.append(gid)
    return np.asarray(vectors), np.asarray(labels, dtype=int), ids, excluded


def group_lookup(vectors: np.ndarray, labels: np.ndarray, ids: list[str]) -> dict[str, tuple[np.ndarray, int]]:
    if len(vectors) != len(labels) or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Group arrays must be aligned and unique")
    return {gid: (vectors[index], int(labels[index])) for index, gid in enumerate(ids)}
