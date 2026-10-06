#!/usr/bin/env python3
"""Select steering strength from neutral damage, then summarize behavior."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_by(items: list[dict], value: str) -> dict[tuple, float]:
    grouped = defaultdict(list)
    for row in items:
        grouped[(row["source"], row["target"], row["direction"], float(row["alpha"]))].append(float(row[value]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-dir", required=True, type=Path)
    parser.add_argument("--neutral-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--damage-budget", type=float, default=0.05)
    args = parser.parse_args()
    for directory in (args.behavior_dir, args.neutral_dir):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("complete") or not (directory / "COMPLETE").exists():
            raise ValueError(f"Incomplete run: {directory}")

    behavior = rows(args.behavior_dir / "scores.jsonl")
    neutral = rows(args.neutral_dir / "scores.jsonl")
    neutral_mean = mean_by(neutral, "nll")
    behavior_mean = mean_by(behavior, "margin")
    neutral_baseline = {(row["source"], row["target"]): value
                        for (source, target, direction, alpha), value in neutral_mean.items()
                        if direction == "baseline" and alpha == 0
                        for row in [{"source": source, "target": target}]}
    behavior_baseline = {(row["source"], row["target"]): value
                         for (source, target, direction, alpha), value in behavior_mean.items()
                         if direction == "baseline" and alpha == 0
                         for row in [{"source": source, "target": target}]}
    keys = sorted({(row["source"], row["target"], row["direction"])
                   for row in behavior if row["direction"] != "baseline"})
    cells = []
    for source, target, direction in keys:
        magnitudes = sorted({abs(alpha) for s, t, d, alpha in neutral_mean
                             if (s, t, d) == (source, target, direction) and alpha != 0})
        passing = []
        damage_curve = []
        for magnitude in magnitudes:
            pos = neutral_mean.get((source, target, direction, magnitude))
            neg = neutral_mean.get((source, target, direction, -magnitude))
            if pos is None or neg is None:
                continue
            base = neutral_baseline[(source, target)]
            damage = 0.5 * ((pos - base) + (neg - base))
            damage_curve.append({"alpha_magnitude": magnitude, "symmetric_mean_delta_nll": damage})
            if damage <= args.damage_budget:
                passing.append(magnitude)
        selected = max(passing) if passing else None
        effect = None
        if selected is not None:
            pos = behavior_mean[(source, target, direction, selected)]
            neg = behavior_mean[(source, target, direction, -selected)]
            base = behavior_baseline[(source, target)]
            effect = {
                "symmetric_slope": (pos - neg) / (2 * selected),
                "positive_alpha_margin_change": pos - base,
                "negative_alpha_margin_change": neg - base,
            }
        cells.append({
            "source": source, "target": target, "direction": direction,
            "direction_type": "random" if direction.startswith("random_") else "named",
            "damage_budget": args.damage_budget,
            "selected_alpha_magnitude": selected,
            "effect": effect,
            "damage_curve": damage_curve,
        })

    aggregate = {}
    for direction in sorted({cell["direction"] for cell in cells if cell["direction_type"] == "named"}):
        subset = [cell for cell in cells if cell["direction"] == direction]
        slopes = [cell["effect"]["symmetric_slope"] for cell in subset if cell["effect"] is not None]
        aggregate[direction] = {
            "pairs": len(subset),
            "pairs_with_safe_nonzero_alpha": len(slopes),
            "mean_safe_alpha": float(np.mean([cell["selected_alpha_magnitude"] for cell in subset
                                               if cell["selected_alpha_magnitude"] is not None])) if slopes else None,
            "mean_symmetric_slope": float(np.mean(slopes)) if slopes else None,
        }
    random_slopes = [cell["effect"]["symmetric_slope"] for cell in cells
                     if cell["direction_type"] == "random" and cell["effect"] is not None]
    random_summary = ({"n": len(random_slopes), "p025": float(np.quantile(random_slopes, 0.025)),
                       "p50": float(np.quantile(random_slopes, 0.5)),
                       "p975": float(np.quantile(random_slopes, 0.975))}
                      if random_slopes else {"n": 0, "p025": None, "p50": None, "p975": None})
    result = {
        "schema_version": 1,
        "analysis": "post-core X2 damage-budgeted steering",
        "damage_budget_mean_delta_nll": args.damage_budget,
        "selection": "largest symmetric alpha passing neutral continuation NLL budget; labels unseen",
        "aggregate": aggregate,
        "random_safe_slope_distribution": random_summary,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells), "aggregate": aggregate}))


if __name__ == "__main__":
    main()

