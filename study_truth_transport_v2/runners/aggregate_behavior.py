#!/usr/bin/env python3
"""Aggregate template scores into fact- and dependency-group difficulty records."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from study_truth_transport_v2.src.data import LANGUAGES


COMPONENTS = ("correct_fraction", "aligned_margin", "consistent_prediction",
              "negative_entropy", "negative_surprisal")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_language(scores: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["id"]].append(row)
    facts = []
    for claim_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["template_index"])
        if [r["template_index"] for r in rows] != [0, 1, 2]:
            raise ValueError(f"Expected templates 0,1,2 for {claim_id}")
        invariant = ("group_id", "partition", "label", "surface_language", "statement_surprisal")
        for field in invariant:
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Inconsistent {field} for {claim_id}")
        label = int(rows[0]["label"])
        predictions = [int(r["prediction"]) for r in rows]
        sum_margins = [float(r["sum_margin"]) for r in rows]
        aligned = [(2 * label - 1) * value for value in sum_margins]
        correct_count = sum(int(r["correct"]) for r in rows)
        consistent = int(len(set(predictions)) == 1)
        facts.append({
            "id": claim_id, "group_id": rows[0]["group_id"], "partition": rows[0]["partition"],
            "label": label, "surface_language": rows[0]["surface_language"],
            "correct_count": correct_count, "correct_fraction": correct_count / 3,
            "mean_sum_margin": float(np.mean(sum_margins)),
            "aligned_margin": float(np.mean(aligned)),
            "consistent_prediction": consistent,
            "known": bool(correct_count >= 2 and consistent),
            "mean_entropy": float(np.mean([r["response_entropy_two_choice"] for r in rows])),
            "negative_entropy": -float(np.mean([r["response_entropy_two_choice"] for r in rows])),
            "statement_surprisal": float(rows[0]["statement_surprisal"]),
            "negative_surprisal": -float(rows[0]["statement_surprisal"]),
            "template_predictions": predictions,
            "template_sum_margins": sum_margins,
        })
    return facts


def add_composite(facts: list[dict]) -> dict:
    train = [row for row in facts if row["partition"] == "train"]
    parameters, used = {}, []
    for component in COMPONENTS:
        values = np.asarray([row[component] for row in train], dtype=float)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        if sd <= 0:
            # A component that is constant over the training claims carries no difficulty information for this
            # language (e.g. a model whose three templates never agree): it is left out of the composite and
            # recorded (DEVIATIONS.md, v5 Mistral entry). Raise only if nothing is left.
            parameters[component] = {"train_mean": mean, "train_sd": sd, "excluded": "zero variance in train split"}
            continue
        parameters[component] = {"train_mean": mean, "train_sd": sd,
                                 "orientation": "larger means easier/more confidently known"}
        used.append(component)
    if not used:
        raise ValueError("All difficulty components have zero variance")
    for row in facts:
        z = [(float(row[component]) - parameters[component]["train_mean"]) /
             parameters[component]["train_sd"] for component in used]
        row["difficulty_component_z"] = dict(zip(used, z))
        row["ease_composite"] = float(np.mean(z))
        row["difficulty_composite"] = -row["ease_composite"]
    return parameters


def aggregate_groups(facts: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in facts:
        grouped[row["group_id"]].append(row)
    result = []
    for group_id, rows in sorted(grouped.items()):
        partitions = {row["partition"] for row in rows}
        labels = {int(row["label"]) for row in rows}
        if len(partitions) != 1:
            raise ValueError(f"Group crosses partitions: {group_id}")
        result.append({
            "group_id": group_id,
            "partition": next(iter(partitions)),
            "label": next(iter(labels)) if len(labels) == 1 else None,
            "mixed_label": len(labels) != 1,
            "n_claim_rows": len(rows),
            "known": bool(all(row["known"] for row in rows)),
            "known_rule": "all same-group claim rows satisfy >=2/3 correct and prediction sign consistent",
            "ease_composite": float(np.mean([row["ease_composite"] for row in rows])),
            "difficulty_composite": float(np.mean([row["difficulty_composite"] for row in rows])),
            "components": {component: float(np.mean([row[component] for row in rows]))
                           for component in COMPONENTS},
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--behavior-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"schema_version": 1, "model_id": args.model_id,
              "composite_components": list(COMPONENTS), "languages": {}}
    for language in LANGUAGES:
        directory = args.behavior_root / language
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("complete") or not (directory / "COMPLETE").exists():
            raise ValueError(f"Behavior output incomplete: {language}")
        facts = aggregate_language(read_jsonl(directory / "scores.jsonl"))
        parameters = add_composite(facts)
        groups = aggregate_groups(facts)
        result["languages"][language] = {
            "standardization": parameters,
            "n_facts": len(facts),
            "n_groups": len(groups),
            "known_rate_by_partition": {
                partition: float(np.mean([row["known"] for row in groups if row["partition"] == partition]))
                for partition in ("train", "validation", "test")},
            "facts": facts,
            "groups": groups,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "languages": list(LANGUAGES)}))


if __name__ == "__main__":
    main()
