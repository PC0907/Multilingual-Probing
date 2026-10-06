#!/usr/bin/env python3
"""Train an LSI-style shared AE and evaluate mass-mean probes in latent space."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from .analyze_rq_a import load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import cosine, score_transport
from study_truth_transport_v2.src.probes import fit_mass_mean


def split_group(group_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"lsi-ae-v1|{seed}|{group_id}".encode()).hexdigest()[:16], 16)
    return "dev" if value % 5 == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.output_dir.exists():
        raise ValueError(f"Refusing existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    data = {lang: load_language(args.prompt_root, args.cache_root, lang, block_index)
            for lang in LANGUAGES}
    common = sorted(set.intersection(*(set(data[lang]["train"]["ids"]) for lang in LANGUAGES)))
    width = int(data[LANGUAGES[0]]["train"]["x"].shape[1])
    states = np.stack([
        np.asarray([data[lang]["train"]["lookup"][gid][0] for gid in common], dtype=np.float32)
        for lang in LANGUAGES
    ])
    pair_index = [(i, j, k) for i in range(len(LANGUAGES)) for j in range(i + 1, len(LANGUAGES))
                  for k, gid in enumerate(common)]
    train_index = [item for item in pair_index if split_group(common[item[2]], args.seed) == "train"]
    dev_index = [item for item in pair_index if split_group(common[item[2]], args.seed) == "dev"]

    class PairDataset(Dataset):
        def __init__(self, index):
            self.index = index

        def __len__(self):
            return len(self.index)

        def __getitem__(self, item):
            left, right, group = self.index[item]
            return states[left, group], states[right, group]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(width, 1024), nn.ReLU(),
                                         nn.Linear(1024, 512), nn.ReLU(),
                                         nn.Linear(512, args.hidden_dim))
            self.decoder = nn.Sequential(nn.Linear(args.hidden_dim, 1024), nn.ReLU(),
                                         nn.Linear(1024, 512), nn.ReLU(),
                                         nn.Linear(512, width))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        def encode(self, x, noisy=False):
            if noisy and self.training:
                x = x + x * torch.randn_like(x) * 0.3
            return self.encoder(x)

    model = AE().to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(PairDataset(train_index), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    dev_loader = DataLoader(PairDataset(dev_index), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    def loss_batch(x1, x2, noisy):
        x1, x2 = x1.to("cuda:0"), x2.to("cuda:0")
        z1, z2 = model.encode(x1, noisy=noisy), model.encode(x2, noisy=noisy)
        r1, r2 = model.decoder(z1), model.decoder(z2)
        reconstruction = (F.huber_loss(r1, x1) + F.huber_loss(r2, x2) +
                          F.huber_loss(r1, x2) + F.huber_loss(r2, x1))
        z1c, z2c = z1 - z1.mean(0, keepdim=True), z2 - z2.mean(0, keepdim=True)
        alignment = F.mse_loss(F.normalize(z1c, dim=-1), F.normalize(z2c, dim=-1))

        def variance(z):
            std = torch.sqrt((z - z.mean(0, keepdim=True)).var(0) + 1e-4)
            return F.relu(1.0 - std).mean()

        def covariance(z):
            z = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
            cov = z.T @ z / max(1, z.shape[0] - 1)
            off = cov - torch.diag(torch.diag(cov))
            return off.pow(2).sum() / z.shape[1]

        return reconstruction + 0.1 * alignment + variance(z1) + variance(z2) + covariance(z1) + covariance(z2)

    history, best, stale = [], float("inf"), 0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for x1, x2 in train_loader:
            loss = loss_batch(x1, x2, noisy=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(float(loss.item()))
        model.eval()
        dev_loss = []
        with torch.inference_mode():
            for x1, x2 in dev_loader:
                dev_loss.append(float(loss_batch(x1, x2, noisy=False).item()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_loss)),
                  "dev_loss": float(np.mean(dev_loss))}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best - record["dev_loss"] > 1e-5:
            best, stale = record["dev_loss"], 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(best_path, map_location="cuda:0", weights_only=True))
    model.eval()

    def encode(array: np.ndarray) -> np.ndarray:
        output = []
        with torch.inference_mode():
            for start in range(0, len(array), args.batch_size):
                tensor = torch.as_tensor(np.asarray(array[start:start + args.batch_size]),
                                         dtype=torch.float32, device="cuda:0")
                output.append(model.encode(tensor).cpu().numpy())
        return np.concatenate(output)

    encoded = {lang: {part: encode(data[lang][part]["x"])
                      for part in ("train", "test")}
               for lang in LANGUAGES}
    probes = {lang: fit_mass_mean(encoded[lang]["train"], data[lang]["train"]["y"])
              for lang in LANGUAGES}
    cells = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            scores = probes[source].score(encoded[target]["test"])
            cells.append({
                "source": source, "target": target,
                "direction_cosine": cosine(probes[source].direction, probes[target].direction),
                **score_transport(data[target]["test"]["y"], scores),
            })
    off = [cell for cell in cells if cell["source"] != cell["target"]]
    result = {
        "schema_version": 1,
        "analysis": "post-core X1 LSI-style shared-AE latent mass-mean transfer",
        "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "languages": list(LANGUAGES),
        "n_parallel_groups": len(common),
        "n_training_pairs": len(train_index),
        "n_dev_pairs": len(dev_index),
        "label_use": "AE is label blind; labels only fit downstream mass-mean directions",
        "scope": "same-dataset frozen-layer adaptation of released LSI AE training; not TED-trained end-to-end QA LSI",
        "hyperparameters": {
            "latent_dim": args.hidden_dim, "max_epochs": args.epochs,
            "actual_epochs": len(history), "patience": args.patience,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "numeric_precision": "float32 adaptation (released implementation uses float64)",
            "speckle_noise_std": 0.3,
            "loss": "four-way Huber reconstruction + 0.1 alignment + variance + covariance",
        },
        "best_dev_loss": best,
        "off_diagonal_means": {
            key: float(np.mean([cell[key] for cell in off]))
            for key in ("direction_cosine", "auroc", "balanced_accuracy", "standardized_separation")
        },
        "history": history,
        "cells": cells,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "COMPLETE").write_text("LSI-style latent probe complete.\n")
    print(json.dumps({"output": str(args.output_dir),
                      "off_diagonal_means": result["off_diagonal_means"]}))


if __name__ == "__main__":
    main()
