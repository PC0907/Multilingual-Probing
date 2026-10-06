#!/usr/bin/env python3
"""Aggregate checkpoint-wise overlap and transfer results for exploratory RQ-F."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--model-prefix", default="pythia-1.4b-step")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoints = []
    for directory in sorted(args.results_root.glob(args.model_prefix + "*"),
                            key=lambda path: int(path.name.split("step")[-1])):
        step = int(directory.name.split("step")[-1])
        selection = json.loads((directory / "layer_selection.json").read_text())
        rqa = json.loads((directory / "rq_a" / "summary.json").read_text())
        rqc = json.loads((directory / "rq_c.json").read_text())
        zero = [cell for cell in rqa["cells"] if cell["overlap_fraction"] == 0]
        full = [cell for cell in rqa["cells"] if cell["overlap_fraction"] == 1]
        off_c = [cell for cell in rqc["cells"] if cell["source"] != cell["target"]]
        checkpoints.append({
            "step": step,
            "model_id": directory.name,
            "resolved_revision": json.loads((directory / "checkpoint.json").read_text())["resolved_revision"],
            "selected_block_number": selection["selected_block_number"],
            "mean_validation_auroc": float(max(selection["mean_validation_auroc_by_block"])),
            "zero_overlap_cosine": float(np.mean([cell["metrics"]["raw_cosine"]["mean"] for cell in zero])),
            "full_overlap_cosine": float(np.mean([cell["metrics"]["raw_cosine"]["mean"] for cell in full])),
            "overlap_cosine_difference": float(np.mean([cell["metrics"]["raw_cosine"]["mean"] for cell in full]) -
                                               np.mean([cell["metrics"]["raw_cosine"]["mean"] for cell in zero])),
            "zero_overlap_auroc": float(np.mean([cell["metrics"]["auroc"]["mean"] for cell in zero])),
            "full_overlap_auroc": float(np.mean([cell["metrics"]["auroc"]["mean"] for cell in full])),
            "full_train_cross_auroc": float(np.mean([cell["auroc"] for cell in off_c])),
            "full_train_direction_cosine": float(np.mean([cell["direction_cosine"] for cell in off_c])),
        })
    if not checkpoints:
        raise ValueError("No checkpoint results found")
    result = {
        "schema_version": 1,
        "analysis": "post-core X5 exploratory RQ-F Pythia training trajectory",
        "scope": "one Pythia-1.4B-deduped trajectory; validation-selected layer independently per checkpoint",
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoints": len(checkpoints)}))


if __name__ == "__main__":
    main()
