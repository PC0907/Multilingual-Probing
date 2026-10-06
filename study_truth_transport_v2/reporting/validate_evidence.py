#!/usr/bin/env python3
"""Fail closed unless every report input has the expected completed shape."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODELS = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inventory = {}
    for model in MODELS:
        root = args.results_root / model
        expected = {
            "layer_selection.json": None,
            "rq_a/summary.json": 150,
            "rq_a/inference.json": 150,
            "rq_b.json": 36,
            "rq_b_sensitivities.json": 3,
            "rq_c.json": 36,
            "rq_d.json": 30,
            "rq_d_mean_margin.json": 30,
            "rq_a_controls.json": 15,
            "all_layers.json": None,
            "steering_summary.json": 150,
            "translation_sensitivity.json": 36,
            "steering_sensitivity_layer_minus_2_summary.json": 60,
            "steering_sensitivity_layer_plus_2_summary.json": 60,
            "steering_sensitivity_all_statement_tokens_summary.json": 60,
            "steering_neutral_flores/summary.json": None,
        }
        model_files = {}
        for name, count in expected.items():
            path = root / name
            if name == "rq_d_mean_margin.json" and not path.exists():
                continue  # v2-only sensitivity analysis (DEVIATIONS.md, v3 entry); not rerun on v3
            if not path.exists():
                raise FileNotFoundError(path)
            payload = read(path)
            if count is not None:
                if name.startswith("rq_a/"):
                    actual = len(payload["cells"])
                elif name == "rq_b.json":
                    actual = len(payload["truth_pair_results"])
                elif name == "rq_b_sensitivities.json":
                    actual = len(payload["variants"])
                elif name in {"rq_c.json", "translation_sensitivity.json"}:
                    actual = len(payload["cells"] if name == "rq_c.json" else payload["rq_c_cells"])
                elif name in {"rq_d.json", "rq_d_mean_margin.json"}:
                    actual = len(payload["pairs"])
                elif name == "rq_a_controls.json":
                    actual = len(payload["pair_controls"])
                else:
                    actual = len(payload["cells"])
                if actual != count:
                    raise ValueError(f"{path}: expected {count}, got {actual}")
            model_files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        layer = read(root / "layer_selection.json")
        all_layers = read(root / "all_layers.json")
        if layer["selected_block_number"] != all_layers["selected_block_number"]:
            raise ValueError(f"Layer mismatch for {model}")
        inventory[model] = {
            "selected_block_number": layer["selected_block_number"],
            "n_layers": len(all_layers["layers"]),
            "files": model_files,
        }
    # Fourth family (X4): RQ-A/B/C replication only.
    for model in ["mistral-7b-v0.3"]:
        root = args.results_root / model
        files = {}
        for name, count in {"rq_a/inference.json": 150, "rq_b.json": 36, "rq_c.json": 36,
                            "rq_a_controls.json": 15, "layer_selection.json": None}.items():
            path = root / name
            payload = read(path)
            if count is not None:
                actual = len(payload["truth_pair_results"] if name == "rq_b.json" else
                             payload["pair_controls"] if name == "rq_a_controls.json" else payload["cells"])
                if actual != count:
                    raise ValueError(f"{path}: expected {count}, got {actual}")
            files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        inventory[model] = {"selected_block_number": read(root / "layer_selection.json")["selected_block_number"],
                            "files": files}
    # Post-core extensions for every family.
    expected_ext = {"alignment_baselines.json": ("cells", 30), "lsi_latent/results.json": ("cells", 36),
                    "lsi_latent/inference.json": (None, None), "mmmlu_transfer.json": ("cells", 36),
                    "include_transfer.json": ("cells", 30), "damage_budgeted_steering.json": ("cells", 42)}
    for model in MODELS + ["mistral-7b-v0.3"]:
        for name, (key, count) in expected_ext.items():
            path = args.results_root / model / "extensions" / name
            payload = read(path)
            if key is not None and len(payload[key]) != count:
                raise ValueError(f"{path}: expected {count} {key}, got {len(payload[key])}")
            inventory[model].setdefault("extensions", {})[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    trajectory = args.results_root / "rq_f_pythia_trajectory.json"
    if trajectory.exists():
        checkpoints = read(trajectory)["checkpoints"]
        if [row["step"] for row in checkpoints] != [0, 1000, 4000, 16000, 64000, 143000]:
            raise ValueError("RQ-F trajectory does not contain the six frozen checkpoints")
        inventory["pythia-1.4b-deduped"] = {"rq_f_pythia_trajectory.json": digest(trajectory)}
    result = {"complete": True, "models": inventory}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "models": len(inventory), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
