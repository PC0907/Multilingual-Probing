#!/usr/bin/env python3
"""Evaluate dataset-v2 mass-mean directions on aligned MMMLU candidate truth."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_rq_a import load_language
from .cache_io import sha256
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import auroc, balanced_accuracy
from study_truth_transport_v2.src.probes import fit_mass_mean


def load_external(prompt_path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("complete") or not (cache_dir / "COMPLETE").exists():
        raise ValueError(f"Incomplete external cache: {cache_dir}")
    if metadata["input_sha256"] != sha256(prompt_path):
        raise ValueError("External cache input hash mismatch")
    rows = json.loads(prompt_path.read_text(encoding="utf-8"))
    if metadata["source_ids"] != [row["id"] for row in rows]:
        raise ValueError("External cache row order mismatch")
    x = np.load(cache_dir / "last.npy", mmap_mode="r")
    if list(x.shape) != metadata["shape"] or len(x) != len(rows):
        raise ValueError("External cache shape mismatch")
    y = np.asarray([row["label"] for row in rows], dtype=int)
    return x, y, rows


def paired_accuracy(rows: list[dict], scores: np.ndarray) -> float:
    grouped = defaultdict(dict)
    for row, score in zip(rows, scores):
        grouped[row["group_id"]][int(row["label"])] = float(score)
    if any(set(values) != {0, 1} for values in grouped.values()):
        raise ValueError("Every MMMLU group must contain one correct and one incorrect candidate")
    values = [1.0 if item[1] > item[0] else 0.5 if item[1] == item[0] else 0.0
              for item in grouped.values()]
    return float(np.mean(values))


def bootstrap(rows: list[dict], scores: np.ndarray, draws: int,
              rng: np.random.Generator) -> dict:
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group_id"]].append(index)
    groups = sorted(by_group)
    aucs, pairs = [], []
    for _ in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices, sampled_rows = [], []
        for draw_index, group_index in enumerate(sampled):
            group = groups[int(group_index)]
            for row_index in by_group[group]:
                indices.append(row_index)
                sampled_rows.append({**rows[row_index], "group_id": f"boot-{draw_index}"})
        chosen = np.asarray(indices, dtype=int)
        labels = np.asarray([rows[index]["label"] for index in indices], dtype=int)
        aucs.append(auroc(labels, scores[chosen]))
        pairs.append(paired_accuracy(sampled_rows, scores[chosen]))
    return {
        "auroc_p025": float(np.quantile(aucs, 0.025)),
        "auroc_p975": float(np.quantile(aucs, 0.975)),
        "paired_accuracy_p025": float(np.quantile(pairs, 0.025)),
        "paired_accuracy_p975": float(np.quantile(pairs, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-prompt-root", required=True, type=Path)
    parser.add_argument("--train-cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--external-prompt-root", required=True, type=Path)
    parser.add_argument("--external-cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--external-languages", nargs="+", default=None,
                        help="Target languages present in the external set (X6 INCLUDE has no English)")
    parser.add_argument("--breakdown-field", default=None,
                        help="Optional row field for a descriptive per-category breakdown")
    parser.add_argument("--external-data-label",
                        default="English MMLU plus OpenAI MMMLU professional translations; no fitting")
    parser.add_argument("--analysis-label", default="post-core X3 zero-shot external MMMLU candidate-truth transfer")
    args = parser.parse_args()
    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    train = {lang: load_language(args.train_prompt_root, args.train_cache_root, lang, block_index)
             for lang in LANGUAGES}
    probes = {lang: fit_mass_mean(train[lang]["train"]["x"], train[lang]["train"]["y"])
              for lang in LANGUAGES}
    targets = list(args.external_languages or LANGUAGES)
    external = {lang: load_external(args.external_prompt_root / f"{lang}.json",
                                    args.external_cache_root / lang)
                for lang in targets}
    # Prespecified sensitivity (DEVIATIONS.md, 2 Oct 2026): questions not
    # context-compacted in any of the six languages for this model.
    compacted = {row["group_id"] for _, _, rows in external.values() for row in rows
                 if row.get("compaction", {}).get("applied")}
    cells = []
    rng = np.random.default_rng(args.seed)
    for source in LANGUAGES:
        probe = probes[source]
        for target in targets:
            x, y, rows = external[target]
            scores = probe.score(x)
            subjects = sorted({row["subject"] for row in rows})
            subject_metrics = []
            for subject in subjects:
                index = np.asarray([i for i, row in enumerate(rows) if row["subject"] == subject])
                subject_rows = [rows[i] for i in index]
                subject_metrics.append({
                    "subject": subject,
                    "n_questions": len(index) // 2,
                    "auroc": auroc(y[index], scores[index]),
                    "paired_accuracy": paired_accuracy(subject_rows, scores[index]),
                })
            cell = {
                "source": source, "target": target,
                "n_questions": len(rows) // 2,
                "micro_auroc": auroc(y, scores),
                "paired_candidate_accuracy": paired_accuracy(rows, scores),
                "source_threshold_balanced_accuracy": balanced_accuracy(y, scores),
                "macro_subject_auroc": float(np.mean([item["auroc"] for item in subject_metrics])),
                "macro_subject_paired_accuracy": float(np.mean(
                    [item["paired_accuracy"] for item in subject_metrics])),
                "bootstrap": bootstrap(rows, scores, args.bootstrap_draws, rng),
                "subjects": subject_metrics,
            }
            if args.breakdown_field:
                breakdown = []
                for value in sorted({row[args.breakdown_field] for row in rows}):
                    index = np.asarray([i for i, row in enumerate(rows) if row[args.breakdown_field] == value])
                    breakdown.append({"value": value, "n_questions": len(index) // 2,
                                      "auroc": auroc(y[index], scores[index]),
                                      "paired_accuracy": paired_accuracy([rows[i] for i in index], scores[index])})
                cell["breakdown"] = {"field": args.breakdown_field, "values": breakdown}
            keep = np.asarray([i for i, row in enumerate(rows) if row["group_id"] not in compacted])
            kept_rows = [rows[i] for i in keep]
            cell["uncompacted_subset"] = {
                "n_questions": len(keep) // 2,
                "micro_auroc": auroc(y[keep], scores[keep]),
                "paired_candidate_accuracy": paired_accuracy(kept_rows, scores[keep]),
            }
            cells.append(cell)
    off = [cell for cell in cells if cell["source"] != cell["target"]]
    result = {
        "schema_version": 1,
        "analysis": args.analysis_label,
        "target_languages": targets,
        "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "languages": list(LANGUAGES),
        "direction_fit_data": "generic_claims_2000_v2 train only",
        "external_data": args.external_data_label,
        "off_diagonal_means": {
            key: float(np.mean([cell[key] for cell in off]))
            for key in ("micro_auroc", "paired_candidate_accuracy",
                        "source_threshold_balanced_accuracy", "macro_subject_auroc",
                        "macro_subject_paired_accuracy")
        },
        "context_compaction": {
            "rule": "protocol/DEVIATIONS.md, 2 October 2026 X3 entry",
            "n_questions_compacted_any_language": len(compacted),
            "compacted_group_ids": sorted(compacted),
        },
        "off_diagonal_means_uncompacted_subset": {
            key: float(np.mean([cell["uncompacted_subset"][key] for cell in off]))
            for key in ("micro_auroc", "paired_candidate_accuracy")
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "off_diagonal_means": result["off_diagonal_means"]}))


if __name__ == "__main__":
    main()

