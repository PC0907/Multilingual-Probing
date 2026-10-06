#!/usr/bin/env python3
"""X11: label-blind automatic translation-quality scores with LaBSE.

Cosine similarity between each English statement and its translation (CLS
pooler output, L2-normalized). Per language, dependency groups whose lowest
member similarity is in the bottom 10% are flagged. Labels and model outputs
are never read (protocol/X10_X11_PREREGISTRATION.md).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

LANGS = ("de", "ar", "hi", "fr", "es")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flag-quantile", type=float, default=0.10)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if args.device == "cuda" and (not visible or "," in visible):
        raise RuntimeError("Expose exactly one GPU")
    device = "cuda:0" if args.device == "cuda" else "cpu"
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(device)

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 64):
            batch = tokenizer(texts[start:start + 64], padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(device)
            with torch.inference_mode():
                pooled = model(**batch).pooler_output.float()
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def rows(lang: str) -> list[dict]:
        return json.loads((args.prompt_root / f"{lang}.activation.json").read_text(encoding="utf-8"))

    english = rows("en")
    english_vectors = embed([r["raw_statement"] for r in english])
    result = {"schema_version": 1, "model": str(args.model), "flag_quantile": args.flag_quantile,
              "protocol": "protocol/X10_X11_PREREGISTRATION.md", "languages": {}}
    for lang in LANGS:
        target = rows(lang)
        if [r["id"] for r in target] != [r["id"] for r in english]:
            raise ValueError(f"Row alignment failure for {lang}")
        sims = (embed([r["raw_statement"] for r in target]) * english_vectors).sum(axis=1)
        by_group = defaultdict(list)
        for row, value in zip(target, sims):
            by_group[row["group_id"]].append(float(value))
        group_min = {g: min(v) for g, v in by_group.items()}
        cutoff = float(np.quantile(list(group_min.values()), args.flag_quantile))
        flagged = sorted(g for g, v in group_min.items() if v <= cutoff)
        result["languages"][lang] = {"cutoff": cutoff, "n_groups": len(group_min), "n_flagged": len(flagged),
                                     "mean_similarity": float(np.mean(sims)), "flagged_group_ids": flagged,
                                     "group_min_similarity": group_min}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({lang: {k: v for k, v in item.items() if k in ("cutoff", "n_flagged", "mean_similarity")}
                      for lang, item in result["languages"].items()}))


if __name__ == "__main__":
    main()
