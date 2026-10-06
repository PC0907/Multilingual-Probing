#!/usr/bin/env python3
"""Summarize causal-steering dose responses with dependency-group uncertainty."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ALPHAS = np.asarray([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
NAMED = ("zero_overlap_source", "full_overlap_source", "target_native",
         "difficulty", "language_identity")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "p025": float(np.quantile(array, 0.025)),
            "p50": float(np.quantile(array, 0.5)), "p975": float(np.quantile(array, 0.975)),
            "n": int(len(array))}


def group_slopes(rows: list[dict], outcome: str) -> list[tuple[str, int, float]]:
    by_group = defaultdict(dict)
    for row in rows:
        by_group[(row["group_id"], int(row["label"]))][float(row["alpha"])] = float(row[outcome])
    result = []
    for (group_id, label), values in by_group.items():
        if set(values) != set(ALPHAS.tolist()):
            raise ValueError(f"Incomplete alpha grid for {group_id}: {sorted(values)}")
        y = np.asarray([values[float(alpha)] for alpha in ALPHAS])
        slope = float(np.sum((ALPHAS - ALPHAS.mean()) * (y - y.mean())) /
                      np.sum((ALPHAS - ALPHAS.mean()) ** 2))
        result.append((group_id, label, slope))
    return result


def bootstrap_mean(values: np.ndarray, draws: int, rng) -> dict:
    if len(values) < 2:
        return {"mean": float(values.mean()), "p025": None, "p975": None, "n_groups": len(values)}
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "p025": float(np.quantile(sampled, 0.025)),
            "p975": float(np.quantile(sampled, 0.975)), "n_groups": int(len(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    metadata = json.loads((args.steering_output / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("complete") or not (args.steering_output / "COMPLETE").exists():
        raise ValueError("Steering output is incomplete")
    raw = read_jsonl(args.steering_output / "scores.jsonl")
    baseline = {(row["source"], row["target"], row["group_id"]): row
                for row in raw if row["direction"] == "baseline"}
    expanded = []
    for row in raw:
        if row["direction"] != "baseline":
            expanded.append(row)
    direction_keys = sorted({(row["source"], row["target"], row["direction"]) for row in expanded})
    for source, target, direction in direction_keys:
        ids = {(row["group_id"], int(row["label"])) for row in expanded
               if row["source"] == source and row["target"] == target and row["direction"] == direction}
        for group_id, _ in ids:
            base = baseline[(source, target, group_id)]
            expanded.append({**base, "direction": direction})

    rng = np.random.default_rng(args.seed)
    cells, dose_response = [], []
    grouped = defaultdict(list)
    for row in expanded:
        grouped[(row["source"], row["target"], row["direction"])].append(row)
    for (source, target, direction), rows in sorted(grouped.items()):
        outcome_summaries = {}
        for outcome in ("margin", "probability_true_two_choice"):
            slopes = group_slopes(rows, outcome)
            outcome_summaries[outcome] = {
                "all": bootstrap_mean(np.asarray([value for _, _, value in slopes]),
                                      args.bootstrap_draws, rng),
                "false": bootstrap_mean(np.asarray([value for _, label, value in slopes if label == 0]),
                                        args.bootstrap_draws, rng),
                "true": bootstrap_mean(np.asarray([value for _, label, value in slopes if label == 1]),
                                       args.bootstrap_draws, rng),
            }
        cells.append({"source": source, "target": target, "direction": direction,
                      "direction_type": "random" if direction.startswith("random_") else "named",
                      "slope_per_alpha": outcome_summaries})
        for label_name, label_value in (("all", None), ("false", 0), ("true", 1)):
            subset = rows if label_value is None else [row for row in rows if int(row["label"]) == label_value]
            for alpha in ALPHAS:
                alpha_rows = [row for row in subset if float(row["alpha"]) == float(alpha)]
                dose_response.append({
                    "source": source, "target": target, "direction": direction,
                    "label": label_name, "alpha": float(alpha), "n_groups": len(alpha_rows),
                    "mean_margin": float(np.mean([row["margin"] for row in alpha_rows])),
                    "mean_probability_true_two_choice": float(np.mean(
                        [row["probability_true_two_choice"] for row in alpha_rows])),
                })

    random_controls, named_aggregate = {}, {}
    for outcome in ("margin", "probability_true_two_choice"):
        random_controls[outcome] = {}
        named_aggregate[outcome] = {}
        for label in ("all", "false", "true"):
            random_values = [cell["slope_per_alpha"][outcome][label]["mean"] for cell in cells
                             if cell["direction_type"] == "random"]
            random_controls[outcome][label] = percentile_summary(random_values)
            named_aggregate[outcome][label] = {}
            for direction in NAMED:
                values = [cell["slope_per_alpha"][outcome][label]["mean"] for cell in cells
                          if cell["direction"] == direction]
                named_aggregate[outcome][label][direction] = percentile_summary(values)

    result = {
        "schema_version": 1,
        "steering_metadata": metadata,
        "slope_definition": "within-group OLS slope across alpha in {-4,-2,-1,0,1,2,4}",
        "uncertainty": "percentile bootstrap over dependency groups within each language pair",
        "cells": cells,
        "dose_response": dose_response,
        "named_direction_pair_distribution": named_aggregate,
        "random_direction_pair_draw_distribution": random_controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "dose_response_rows": len(dose_response)}))


if __name__ == "__main__":
    main()
