#!/usr/bin/env python3
"""Repeat headline RQ-A/RQ-C/RQ-D estimates after automatic-risk exclusion."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .analyze_rq_a import evaluate, fit_ids, load_language
from .analyze_rq_b import behavior_maps
from study_truth_transport_v2.src.allocation import GroupRecord, build_overlap_plan
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import auroc, cosine, score_transport
from study_truth_transport_v2.src.probes import fit_mass_mean


def clean_table(table: dict, excluded: set[str]) -> dict:
    keep = np.asarray([gid not in excluded for gid in table["ids"]], dtype=bool)
    ids = [gid for gid, retain in zip(table["ids"], keep) if retain]
    x, y = table["x"][keep], table["y"][keep]
    return {"x": x, "y": y, "ids": ids,
            "lookup": {gid: (row, int(label)) for gid, row, label in zip(ids, x, y)}}


def safe_auc(y: np.ndarray, score: np.ndarray):
    return auroc(y, score) if len(y) and len(np.unique(y)) == 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--group-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    selected = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selected["selected_cache_index"])
    metadata = json.loads(args.group_metadata.read_text(encoding="utf-8"))["records"]
    meta = {row["group_id"]: row for row in metadata}
    excluded = {
        language: {row["group_id"] for row in metadata
                   if row["language"][language]["translation_risk"] or
                   row["language"][language]["researcher_corrected"]}
        for language in LANGUAGES
    }
    raw = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
           for language in LANGUAGES}
    data = {language: {partition: clean_table(raw[language][partition], excluded[language])
                       for partition in ("train", "validation", "test")}
            for language in LANGUAGES}
    behavior = behavior_maps(args.behavior)
    probes = {language: fit_mass_mean(data[language]["train"]["x"], data[language]["train"]["y"])
              for language in LANGUAGES}

    overlap_rows, clean_plans = [], {}
    for left, right in combinations(LANGUAGES, 2):
        original = json.loads((args.allocation_root / f"{left}__{right}.json").read_text())
        records = []
        for gid, annotation in original["group_annotations"].items():
            if gid in excluded[left] or gid in excluded[right] or gid not in meta:
                continue
            row = meta[gid]
            if row["partition"] != "train":
                continue
            clean_stratum = (f"len{annotation['length_bin']}|neg{int(annotation['either_language_negation'])}"
                             f"|num{int(annotation['contains_number'])}")
            records.append(GroupRecord(gid, int(row["label"]), clean_stratum))
        plan = build_overlap_plan(records, repetitions=50, seed=args.seed)
        clean_plans[(left, right)] = plan
        for allocation in plan["plans"]:
            left_probe = fit_ids(data[left]["train"]["lookup"], allocation["source_group_ids"])
            right_probe = fit_ids(data[right]["train"]["lookup"], allocation["target_group_ids"])
            overlap_rows.extend([
                {"source": left, "target": right,
                 "overlap_fraction": allocation["requested_overlap_fraction"],
                 "raw_cosine": cosine(left_probe.direction, right_probe.direction),
                 **evaluate(left_probe, data[right]["test"])},
                {"source": right, "target": left,
                 "overlap_fraction": allocation["requested_overlap_fraction"],
                 "raw_cosine": cosine(left_probe.direction, right_probe.direction),
                 **evaluate(right_probe, data[left]["test"])},
            ])
    grouped = defaultdict(list)
    for row in overlap_rows:
        grouped[row["overlap_fraction"]].append(row)
    rq_a = [{"overlap_fraction": overlap,
             "mean_raw_cosine": float(np.mean([row["raw_cosine"] for row in rows])),
             "mean_transfer_auroc": float(np.mean([row["auroc"] for row in rows])),
             "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
             "n_directional_allocations": len(rows)}
            for overlap, rows in sorted(grouped.items())]

    rq_c = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            scores = probes[source].score(data[target]["test"]["x"])
            rq_c.append({"source": source, "target": target,
                         "direction_cosine": cosine(probes[source].direction, probes[target].direction),
                         "n_target_test_groups": len(scores),
                         **score_transport(data[target]["test"]["y"], scores)})

    rq_d = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            left, right = (source, target) if (source, target) in clean_plans else (target, source)
            plans = [row for row in clean_plans[(left, right)]["plans"] if row["shared_groups"] == 0]
            scores = []
            for allocation in plans:
                ids = allocation["source_group_ids"] if source == left else allocation["target_group_ids"]
                scores.append(fit_ids(data[source]["train"]["lookup"], ids).score(data[target]["test"]["x"]))
            mean_score = np.mean(np.asarray(scores), axis=0)
            ids = data[target]["test"]["ids"]
            source_known = np.asarray([behavior[source][gid]["known"] for gid in ids])
            target_known = np.asarray([behavior[target][gid]["known"] for gid in ids])
            labels = data[target]["test"]["y"]
            quadrants = []
            for name, mask in {"K/K": source_known & target_known,
                               "K/U": source_known & ~target_known,
                               "U/K": ~source_known & target_known,
                               "U/U": ~source_known & ~target_known}.items():
                quadrants.append({"quadrant": name, "n_groups": int(mask.sum()),
                                  "n_true": int(labels[mask].sum()),
                                  "n_false": int((labels[mask] == 0).sum()),
                                  "auroc": safe_auc(labels[mask], mean_score[mask])})
            rq_d.append({"source": source, "target": target, "quadrants": quadrants})

    result = {
        "schema_version": 1,
        "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "exclusion": "language-specific automatic translation-risk OR researcher-corrected group",
        "warning": "Automatic flags are conservative and are not a substitute for bilingual human review.",
        "excluded_groups_by_language_and_partition": {
            language: {partition: len(set(raw[language][partition]["ids"]) & excluded[language])
                       for partition in ("train", "validation", "test")}
            for language in LANGUAGES},
        "rq_a_clean_reallocated": rq_a,
        "rq_c_cells": rq_c,
        "rq_d_pairs": rq_d,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rq_a_rows": len(overlap_rows),
                      "rq_c_cells": len(rq_c), "rq_d_pairs": len(rq_d)}))


if __name__ == "__main__":
    main()
