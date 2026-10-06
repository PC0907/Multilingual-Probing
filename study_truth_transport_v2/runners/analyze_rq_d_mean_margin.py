#!/usr/bin/env python3
"""RQ-D sensitivity using per-token-normalized answer likelihood margins."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_rq_a import fit_ids, load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import auroc


def mean_margin_known(behavior_root: Path, language: str) -> dict[str, bool]:
    rows = [json.loads(line) for line in (behavior_root / language / "scores.jsonl").read_text().splitlines()
            if line.strip()]
    facts = defaultdict(list)
    for row in rows:
        facts[row["id"]].append(row)
    fact_known = {}
    fact_group = {}
    for claim_id, values in facts.items():
        values.sort(key=lambda row: row["template_index"])
        label = int(values[0]["label"])
        margins = [float(row["mean_margin"]) for row in values]
        predictions = [int(value >= 0) for value in margins]
        fact_known[claim_id] = sum(prediction == label for prediction in predictions) >= 2 and len(set(predictions)) == 1
        fact_group[claim_id] = values[0]["group_id"]
    grouped = defaultdict(list)
    for claim_id, known in fact_known.items():
        grouped[fact_group[claim_id]].append(known)
    return {group_id: bool(all(values)) for group_id, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior-root", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    known = {language: mean_margin_known(args.behavior_root, language) for language in LANGUAGES}
    pairs = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            path = args.allocation_root / f"{source}__{target}.json"
            source_side = "source_group_ids"
            if not path.exists():
                path = args.allocation_root / f"{target}__{source}.json"
                source_side = "target_group_ids"
            allocation = json.loads(path.read_text())
            scores = []
            for plan in allocation["plans"]:
                if plan["shared_groups"] == 0:
                    scores.append(fit_ids(data[source]["train"]["lookup"],
                                          plan[source_side]).score(data[target]["test"]["x"]))
            score = np.mean(np.asarray(scores), axis=0)
            ids, labels = data[target]["test"]["ids"], data[target]["test"]["y"]
            source_known = np.asarray([known[source][gid] for gid in ids])
            target_known = np.asarray([known[target][gid] for gid in ids])
            quadrants = []
            for name, mask in {"K/K": source_known & target_known,
                               "K/U": source_known & ~target_known,
                               "U/K": ~source_known & target_known,
                               "U/U": ~source_known & ~target_known}.items():
                y = labels[mask]
                quadrants.append({"quadrant": name, "n_groups": int(mask.sum()),
                                  "n_true": int(y.sum()), "n_false": int((y == 0).sum()),
                                  "auroc": auroc(y, score[mask]) if len(np.unique(y)) == 2 else None})
            pairs.append({"source": source, "target": target, "quadrants": quadrants})
    result = {"schema_version": 1, "model_id": args.model_id,
              "selected_block_number": block_index + 1,
              "known_definition": ">=2/3 correct and consistent sign using per-token-normalized full-answer log likelihood",
              "note": "Sensitivity to unequal localized answer-token lengths; primary RQ-D uses summed sequence log likelihood.",
              "known_rate_test": {language: float(np.mean([known[language][gid]
                                                             for gid in data[language]["test"]["ids"]]))
                                  for language in LANGUAGES},
              "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs)}))


if __name__ == "__main__":
    main()
