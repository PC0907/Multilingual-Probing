#!/usr/bin/env python3
"""Measure off-target FLORES next-token loss under all-token steering."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np


ALPHAS = (-4.0, 0.0, 4.0)
DEFAULT_DIRECTIONS = ("zero_overlap_source", "full_overlap_source", "target_native",
                      "random_00", "random_01", "random_02", "random_03", "random_04")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--neutral-root", required=True, type=Path)
    parser.add_argument("--steering-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--direction-names", nargs="*", default=list(DEFAULT_DIRECTIONS))
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.steering_dir / "manifest.json").read_text())
    directions = np.load(args.steering_dir / "directions.npz")
    metadata = {"complete": False, "status": "initializing", "alphas": list(ALPHAS),
                "directions": args.direction_names,
                "intervention": "all non-special neutral-text input tokens after frozen primary block",
                "outcome": "mean next-token NLL over non-special targets; exp(mean NLL) is perplexity"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.eval().requires_grad_(False).to("cuda:0")
        layers = model.model.layers
        block_index = int(manifest["selected_block_number"]) - 1
        state = {"delta": None, "mask": None}

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden + state["mask"].unsqueeze(-1).to(hidden.dtype) * state["delta"].view(1, 1, -1).to(hidden.dtype)
            return (modified,) + output[1:] if isinstance(output, tuple) else modified

        handle = layers[block_index].register_forward_hook(hook)
        writer = (args.output / "scores.jsonl").open("w", encoding="utf-8")
        for pair in manifest["pairs"]:
            source, target = pair["source"], pair["target"]
            records = json.loads((args.neutral_root / f"{target}.json").read_text())
            entries = {entry["name"]: entry for entry in pair["directions"]
                       if entry["name"] in args.direction_names}
            settings = [("baseline", 0.0, np.zeros(model.config.hidden_size, dtype=np.float32), 0.0)]
            for name in args.direction_names:
                entry = entries[name]
                vector = directions[entry["array_key"]]
                for alpha in (-4.0, 4.0):
                    settings.append((name, alpha, vector, float(entry["target_projection_sd"])))
            for direction, alpha, vector, scale in settings:
                state["delta"] = torch.as_tensor(alpha * scale * vector, device="cuda:0")
                for start in range(0, len(records), args.batch_size):
                    batch_rows = records[start:start + args.batch_size]
                    batch = tokenizer([row["text"] for row in batch_rows], padding=True,
                                      add_special_tokens=True, return_tensors="pt").to("cuda:0")
                    mask = batch["attention_mask"].bool()
                    for special_id in tokenizer.all_special_ids:
                        mask &= batch["input_ids"].ne(special_id)
                    state["mask"] = mask
                    with torch.inference_mode():
                        logits = model(**batch, use_cache=False, return_dict=True).logits.float()
                        logp = torch.log_softmax(logits[:, :-1], dim=-1)
                    targets = batch["input_ids"][:, 1:]
                    valid = batch["attention_mask"][:, 1:].bool()
                    for special_id in tokenizer.all_special_ids:
                        valid &= targets.ne(special_id)
                    token_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    sums = (token_nll * valid).sum(dim=1)
                    counts = valid.sum(dim=1)
                    for row, total, count in zip(batch_rows, sums.tolist(), counts.tolist()):
                        writer.write(json.dumps({"source": source, "target": target,
                                                 "id": row["id"], "direction": direction,
                                                 "alpha": alpha, "sum_nll": total,
                                                 "token_count": count,
                                                 "mean_nll": total / count},
                                                ensure_ascii=False, allow_nan=False) + "\n")
                writer.flush()
        writer.close()
        handle.remove()
        rows = [json.loads(line) for line in (args.output / "scores.jsonl").read_text().splitlines()]
        baseline = {(row["source"], row["target"], row["id"]): row["mean_nll"]
                    for row in rows if row["direction"] == "baseline"}
        grouped = {}
        for row in rows:
            if row["direction"] == "baseline":
                continue
            key = (row["source"], row["target"], row["direction"], row["alpha"])
            grouped.setdefault(key, []).append(row["mean_nll"] - baseline[(row["source"], row["target"], row["id"])])
        summary = [{"source": key[0], "target": key[1], "direction": key[2], "alpha": key[3],
                    "mean_delta_nll": float(np.mean(values)),
                    "mean_perplexity_ratio": float(np.exp(np.mean(values))), "n_sentences": len(values)}
                   for key, values in sorted(grouped.items())]
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        metadata.update({"complete": True, "status": "complete", "summary_cells": len(summary),
                         "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30})
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "COMPLETE").write_text("Neutral steering control complete.\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        metadata.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
