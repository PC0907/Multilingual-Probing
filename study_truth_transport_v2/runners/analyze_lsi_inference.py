#!/usr/bin/env python3
"""Paired bootstrap comparison of the saved LSI-style latent probe and raw probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .analyze_alignment_baselines import paired_macro_bootstrap
from .analyze_rq_a import load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.probes import fit_mass_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--lsi-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    if not (args.lsi_dir / "COMPLETE").exists():
        raise ValueError("LSI-style model is incomplete")

    import torch
    from torch import nn

    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    result = json.loads((args.lsi_dir / "results.json").read_text(encoding="utf-8"))
    hidden_dim = int(result["hyperparameters"]["latent_dim"])
    data = {lang: load_language(args.prompt_root, args.cache_root, lang, block_index)
            for lang in LANGUAGES}
    width = int(data[LANGUAGES[0]]["train"]["x"].shape[1])

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(width, 1024), nn.ReLU(),
                                         nn.Linear(1024, 512), nn.ReLU(),
                                         nn.Linear(512, hidden_dim))
            self.decoder = nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(),
                                         nn.Linear(1024, 512), nn.ReLU(),
                                         nn.Linear(512, width))

    model = AE().to("cuda:0")
    model.load_state_dict(torch.load(args.lsi_dir / "best_model.pt",
                                     map_location="cuda:0", weights_only=True))
    model.eval()

    def encode(array: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(array), args.batch_size):
                tensor = torch.as_tensor(np.asarray(array[start:start + args.batch_size]),
                                         dtype=torch.float32, device="cuda:0")
                chunks.append(model.encoder(tensor).float().cpu().numpy())
        return np.concatenate(chunks)

    latent = {lang: {part: encode(data[lang][part]["x"]) for part in ("train", "test")}
              for lang in LANGUAGES}
    raw_probes = {lang: fit_mass_mean(data[lang]["train"]["x"], data[lang]["train"]["y"])
                  for lang in LANGUAGES}
    latent_probes = {lang: fit_mass_mean(latent[lang]["train"], data[lang]["train"]["y"])
                     for lang in LANGUAGES}
    records = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            records.append({
                "source": source, "target": target,
                "ids": list(data[target]["test"]["ids"]),
                "y": np.asarray(data[target]["test"]["y"], dtype=int),
                "scores": {
                    "raw_source_mass_mean": raw_probes[source].score(data[target]["test"]["x"]),
                    "lsi_latent": latent_probes[source].score(latent[target]["test"]),
                },
            })
    inference = paired_macro_bootstrap(records, args.bootstrap_draws, args.seed)
    payload = {
        "schema_version": 1,
        "analysis": "paired group bootstrap of LSI-style latent versus raw mass mean",
        "model_id": args.model_id,
        "bootstrap_draws": args.bootstrap_draws,
        "comparison": inference["lsi_latent"],
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparison": payload["comparison"]}))


if __name__ == "__main__":
    main()
