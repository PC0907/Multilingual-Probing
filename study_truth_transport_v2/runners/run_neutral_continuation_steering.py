#!/usr/bin/env python3
"""Measure next-token neutral-text damage under final-prefix-token steering."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_ALPHAS = (-4.0, -2.0, -1.0, -0.5, -0.25, -0.125,
                  0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_DIRECTIONS = ("zero_overlap_source", "target_native",
                      "random_00", "random_01", "random_02", "random_03", "random_04")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prefix_target(tokenizer, text: str) -> tuple[list[int], int]:
    content = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(content) < 2:
        raise ValueError("Neutral sentence needs at least two content tokens")
    prefix_content, target = content[:-1], int(content[-1])
    full = tokenizer.prepare_for_model(prefix_content, add_special_tokens=True,
                                       truncation=False, return_attention_mask=False)["input_ids"]
    if not full:
        raise ValueError("Tokenizer produced empty prefix")
    # Some tokenizers append EOS in prepare_for_model. Remove trailing special
    # tokens so the intervention remains on the final content token.
    while full and full[-1] in tokenizer.all_special_ids and full[-1] != tokenizer.bos_token_id:
        full.pop()
    if not full:
        raise ValueError("No prefix token remains after removing trailing specials")
    return full, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--neutral-root", required=True, type=Path)
    parser.add_argument("--steering-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--direction-names", nargs="*", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--pairs", nargs="*", help="Optional source:target subset")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.steering_dir / "manifest.json").read_text(encoding="utf-8"))
    directions = np.load(args.steering_dir / "directions.npz")
    metadata = {
        "complete": False,
        "status": "initializing",
        "alphas": args.alphas,
        "directions": args.direction_names,
        "intervention": "final content token of neutral prefix after frozen primary block",
        "outcome": "NLL of held-out final content token",
        "cuda_visible_devices": visible,
    }
    atomic_json(args.output / "metadata.json", metadata)
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
        with (args.output / "scores.partial.jsonl").open("w", encoding="utf-8") as writer:
            for pair in manifest["pairs"]:
                source, target = pair["source"], pair["target"]
                if args.pairs is not None and f"{source}:{target}" not in args.pairs:
                    continue
                records = json.loads((args.neutral_root / f"{target}.json").read_text(encoding="utf-8"))
                prepared = [(row, *prefix_target(tokenizer, row["text"])) for row in records]
                entries = {entry["name"]: entry for entry in pair["directions"]
                           if entry["name"] in args.direction_names}
                missing = set(args.direction_names) - set(entries)
                if missing:
                    raise ValueError(f"Missing directions for {source}:{target}: {sorted(missing)}")
                settings = [("baseline", 0.0, np.zeros(model.config.hidden_size, dtype=np.float32), 0.0)]
                for name in args.direction_names:
                    entry = entries[name]
                    vector = directions[entry["array_key"]]
                    settings.extend((name, alpha, vector, float(entry["target_projection_sd"]))
                                    for alpha in args.alphas if alpha != 0)
                for direction, alpha, vector, scale in settings:
                    state["delta"] = torch.as_tensor(alpha * scale * vector, device="cuda:0")
                    for start in range(0, len(prepared), args.batch_size):
                        subset = prepared[start:start + args.batch_size]
                        sequences = [item[1] for item in subset]
                        targets = torch.as_tensor([item[2] for item in subset], device="cuda:0")
                        batch = tokenizer.pad(
                            {"input_ids": sequences, "attention_mask": [[1] * len(x) for x in sequences]},
                            padding=True, return_tensors="pt").to("cuda:0")
                        last = batch["attention_mask"].sum(dim=1) - 1
                        state["mask"] = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
                        state["mask"][torch.arange(len(subset), device="cuda:0"), last] = True
                        with torch.inference_mode():
                            logits = model(**batch, use_cache=False, return_dict=True).logits.float()
                            chosen = logits[torch.arange(len(subset), device="cuda:0"), last]
                            nll = -torch.log_softmax(chosen, dim=-1).gather(1, targets[:, None]).squeeze(1)
                        for (row, _, _), value in zip(subset, nll.tolist()):
                            writer.write(json.dumps({
                                "source": source, "target": target, "id": row["id"],
                                "direction": direction, "alpha": alpha, "nll": value,
                            }, ensure_ascii=False, allow_nan=False) + "\n")
                    writer.flush()
        handle.remove()
        partial = args.output / "scores.partial.jsonl"
        partial.replace(args.output / "scores.jsonl")
        metadata.update({
            "complete": True,
            "status": "complete",
            "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        })
        atomic_json(args.output / "metadata.json", metadata)
        (args.output / "COMPLETE").write_text("Matched-scope neutral steering complete.\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        metadata.update({"complete": False, "status": "failed",
                         "error_type": type(error).__name__, "error": str(error)})
        atomic_json(args.output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()

