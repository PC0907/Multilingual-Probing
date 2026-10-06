#!/usr/bin/env python3
"""Select and freeze one decoder block per model from mean validation AUROC."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .cache_io import assert_row_alignment, load_complete_cache

from study_truth_transport_v2.src.data import LANGUAGES, pure_group_activations
from study_truth_transport_v2.src.metrics import auroc
from study_truth_transport_v2.src.probes import fit_mass_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path,
                        help="Contains one complete cache directory per language")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    per_language = {}
    block_count = None
    cache_provenance = {}
    for language in LANGUAGES:
        prompt_path = args.prompt_root / f"{language}.activation.json"
        rows = json.loads(prompt_path.read_text(encoding="utf-8"))
        cache, metadata = load_complete_cache(args.cache_root / language, prompt_path)
        assert_row_alignment(rows, metadata)
        if block_count is None:
            block_count = cache.shape[1]
        elif block_count != cache.shape[1]:
            raise ValueError("Languages have different decoder-block counts")
        scores = []
        for block_index in range(block_count):
            train_x, train_y, _, train_excluded = pure_group_activations(cache[:, block_index, :], rows, "train")
            val_x, val_y, _, val_excluded = pure_group_activations(cache[:, block_index, :], rows, "validation")
            probe = fit_mass_mean(train_x, train_y)
            scores.append(auroc(val_y, probe.score(val_x)))
        per_language[language] = scores
        cache_provenance[language] = {
            "cache": str((args.cache_root / language).resolve()),
            "input_sha256": metadata["input_sha256"],
            "model": metadata["model"],
            "configuration_and_tokenizer_sha256": metadata["configuration_and_tokenizer_sha256"],
            "train_mixed_label_groups_excluded": train_excluded,
            "validation_mixed_label_groups_excluded": val_excluded,
        }
    matrix = np.asarray([per_language[language] for language in LANGUAGES])
    means = matrix.mean(axis=0)
    selected_index = int(np.argmax(means))
    result = {
        "schema_version": 1,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": args.model_id,
        "criterion": "argmax decoder block of unweighted mean within-language validation AUROC across six languages",
        "tie_rule": "first/earliest decoder block",
        "embedding_excluded": True,
        "selected_block_number": selected_index + 1,
        "selected_cache_index": selected_index,
        "mean_validation_auroc_by_block": means.tolist(),
        "validation_auroc_by_language": per_language,
        "languages_equal_weight": True,
        "cache_provenance": cache_provenance,
        "test_data_inspected_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model_id, "selected_block_number": selected_index + 1,
                      "mean_validation_auroc": float(means[selected_index])}))


if __name__ == "__main__":
    main()
