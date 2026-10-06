#!/usr/bin/env python3
"""Prepare frozen unit steering directions from caches without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_rq_a import fit_ids, load_language
from .analyze_rq_b import behavior_maps
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.difficulty import pooled_difficulty_direction
from study_truth_transport_v2.src.probes import fit_mass_mean


DEFAULT_PAIRS = ("en:de", "en:ar", "en:hi", "en:fr", "en:es", "ar:en")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Cannot normalize zero direction")
    return np.asarray(vector, dtype=np.float64) / norm


def average_probe_direction(lookup: dict, plans: list[dict], side: str) -> np.ndarray:
    directions = []
    key = "source_group_ids" if side == "left" else "target_group_ids"
    for plan in plans:
        directions.append(unit(fit_ids(lookup, plan[key]).direction))
    return unit(np.mean(np.asarray(directions), axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS))
    parser.add_argument("--random-directions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    behavior = behavior_maps(args.behavior)
    train_tables = {language: {**data[language]["train"], "behavior": behavior[language]}
                    for language in LANGUAGES}
    difficulty_direction, _ = pooled_difficulty_direction(train_tables)
    rng = np.random.default_rng(args.seed)
    arrays, manifest_pairs = {}, []
    for pair_text in args.pairs:
        source, target = pair_text.split(":")
        allocation_path = args.allocation_root / f"{source}__{target}.json"
        if allocation_path.exists():
            left, right, side = source, target, "left"
        else:
            allocation_path = args.allocation_root / f"{target}__{source}.json"
            left, right, side = target, source, "right"
        allocation = json.loads(allocation_path.read_text())
        zero = [p for p in allocation["plans"] if p["shared_groups"] == 0]
        full = [p for p in allocation["plans"] if p["shared_groups"] == 400]
        common = sorted(set(data[source]["train"]["ids"]) & set(data[target]["train"]["ids"]))
        source_lookup, target_lookup = data[source]["train"]["lookup"], data[target]["train"]["lookup"]
        language_direction = (np.mean([target_lookup[gid][0] for gid in common], axis=0) -
                              np.mean([source_lookup[gid][0] for gid in common], axis=0))
        directions = {
            "zero_overlap_source": average_probe_direction(source_lookup, zero, side),
            "full_overlap_source": average_probe_direction(source_lookup, full, side),
            "target_native": unit(fit_mass_mean(data[target]["train"]["x"], data[target]["train"]["y"]).direction),
            "difficulty": unit(difficulty_direction),
            "language_identity": unit(language_direction),
        }
        for draw in range(args.random_directions):
            directions[f"random_{draw:02d}"] = unit(rng.normal(size=len(difficulty_direction)))
        pair_key = f"{source}__{target}"
        target_validation = data[target]["validation"]["x"]
        entries = []
        for name, direction in directions.items():
            arrays[f"{pair_key}__{name}"] = np.asarray(direction, dtype=np.float32)
            scale = float(np.std(target_validation @ direction, ddof=1))
            entries.append({"name": name, "array_key": f"{pair_key}__{name}",
                            "target_projection_sd": scale,
                            "truth_cosine": float(direction @ directions["target_native"])})
        manifest_pairs.append({"source": source, "target": target, "directions": entries,
                               "zero_overlap_repetitions": len(zero), "full_overlap_repetitions": len(full)})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "directions.npz", **arrays)
    manifest = {"schema_version": 1, "model_id": args.model_id,
                "selected_block_number": block_index + 1,
                "pairs": manifest_pairs, "random_seed": args.seed,
                "direction_scaling": "all directions unit normalized; delta = alpha * target-validation projection SD * direction"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "pairs": len(manifest_pairs),
                      "directions": len(arrays)}))


if __name__ == "__main__":
    main()
