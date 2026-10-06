#!/usr/bin/env python3
"""Run frozen-layer causal steering for prespecified language pairs and controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


ALPHAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statement_positions(tokenizer, prompt: str, statement: str) -> tuple[list[int], list[int]]:
    start = prompt.find(statement)
    if start < 0 or prompt.find(statement, start + 1) >= 0:
        raise ValueError("Raw statement must occur exactly once in behavioral prompt")
    end = start + len(statement)
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    candidates = [index for index, (left, right) in enumerate(offsets)
                  if right > left and left < end and right > start]
    if not candidates:
        raise ValueError("Could not align final statement token")
    return encoded["input_ids"], candidates


def select_test_records(records: list[dict], n_per_pair: int) -> list[dict]:
    candidates = {}
    for row in records:
        if row["template_index"] == 0 and row["partition"] == "test":
            candidates.setdefault(row["group_id"], row)
    selected = []
    for label in (0, 1):
        pool = [row for row in candidates.values() if int(row["label"]) == label]
        pool.sort(key=lambda row: hashlib.sha256(row["group_id"].encode()).hexdigest())
        count = n_per_pair // 2
        if len(pool) < count:
            raise ValueError("Insufficient test groups for balanced steering subset")
        selected.extend(pool[:count])
    selected.sort(key=lambda row: row["group_id"])
    return selected


def score_answer(model, tokenizer, records, answer_field: str, state: dict,
                 device: str, token_scope: str):
    import torch
    sequences, targets, starts, lengths = [], [], [], []
    statement_token_positions = []
    for row in records:
        prompt_ids, positions = statement_positions(tokenizer, row["prompt"], row["raw_statement"])
        answer_ids = tokenizer(" " + row[answer_field], add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise ValueError("Empty answer continuation")
        sequences.append(prompt_ids + answer_ids)
        targets.append(max(positions))
        statement_token_positions.append(positions)
        starts.append(len(prompt_ids))
        lengths.append(len(answer_ids))
    batch = tokenizer.pad({"input_ids": sequences, "attention_mask": [[1] * len(x) for x in sequences]},
                          padding=True, return_tensors="pt").to(device)
    target_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
    for row, positions in enumerate(statement_token_positions):
        chosen = [max(positions)] if token_scope == "final" else positions
        target_mask[row, torch.as_tensor(chosen, device=device)] = True
    state["target_mask"] = target_mask
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    result = []
    for row, (start, length) in enumerate(zip(starts, lengths)):
        token_ids = batch["input_ids"][row, start:start + length]
        positions = torch.arange(start - 1, start + length - 1, device=device)
        result.append(float(log_probs[row, positions, token_ids].sum().item()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--steering-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=100)
    parser.add_argument("--block-offset", type=int, default=0)
    parser.add_argument("--token-scope", choices=("final", "all-statement"), default="final")
    parser.add_argument("--direction-names", nargs="*")
    parser.add_argument("--pairs", nargs="*", help="Optional source:target subset")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS),
                        help="Intervention grid; zero baseline is stored once and need not be listed")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.test_groups % 2:
        raise ValueError("test-groups must be even")
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.mkdir(parents=True)
    manifest_path = args.steering_dir / "manifest.json"
    directions_path = args.steering_dir / "directions.npz"
    manifest = json.loads(manifest_path.read_text())
    directions = np.load(directions_path)
    metadata = {"complete": False, "status": "initializing", "model": str(args.model.resolve()),
                "steering_manifest_sha256": sha256(manifest_path),
                "directions_sha256": sha256(directions_path), "alphas": list(args.alphas),
                "test_groups_per_pair": args.test_groups, "batch_size": args.batch_size,
                "intervention_location": f"after decoder block at {args.token_scope} token scope",
                "block_offset_from_frozen_primary": args.block_offset,
                "direction_name_filter": args.direction_names,
                "pair_filter": args.pairs,
                "cuda_visible_devices": visible}
    atomic_json(args.output / "metadata.json", metadata)
    started = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Fast tokenizer with offset mapping required")
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.eval().requires_grad_(False).to("cuda:0")
        layers = getattr(getattr(model, "model", None), "layers", None)
        block_index = int(manifest["selected_block_number"]) - 1 + args.block_offset
        if layers is None or not (0 <= block_index < len(layers)):
            raise ValueError("Unsupported model layer layout")
        state = {"delta": None, "target_mask": None}

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            modified += (state["target_mask"].unsqueeze(-1).to(modified.dtype) *
                         state["delta"].to(modified.dtype).view(1, 1, -1))
            return (modified,) + output[1:] if isinstance(output, tuple) else modified

        handle = layers[block_index].register_forward_hook(hook)
        partial = args.output / "scores.partial.jsonl"
        completed_settings = 0
        with partial.open("w", encoding="utf-8") as writer:
            for pair in manifest["pairs"]:
                source, target = pair["source"], pair["target"]
                if args.pairs is not None and f"{source}:{target}" not in args.pairs:
                    continue
                records = json.loads((args.prompt_root / f"{target}.behavior.json").read_text())
                selected = select_test_records(records, args.test_groups)
                settings = [("baseline", 0.0, np.zeros(model.config.hidden_size, dtype=np.float32), 0.0)]
                for entry in pair["directions"]:
                    if args.direction_names is not None and entry["name"] not in args.direction_names:
                        continue
                    vector = directions[entry["array_key"]]
                    for alpha in args.alphas:
                        if alpha != 0:
                            settings.append((entry["name"], alpha, vector, entry["target_projection_sd"]))
                for direction_name, alpha, vector, scale in settings:
                    state["delta"] = torch.as_tensor(alpha * scale * vector, device="cuda:0")
                    for start in range(0, len(selected), args.batch_size):
                        batch = selected[start:start + args.batch_size]
                        true_ll = score_answer(model, tokenizer, batch, "answer_true", state,
                                               "cuda:0", args.token_scope)
                        false_ll = score_answer(model, tokenizer, batch, "answer_false", state,
                                                "cuda:0", args.token_scope)
                        for row, true_value, false_value in zip(batch, true_ll, false_ll):
                            margin = true_value - false_value
                            probability = float(1 / (1 + np.exp(np.clip(-margin, -700, 700))))
                            writer.write(json.dumps({
                                "source": source, "target": target, "id": row["id"],
                                "group_id": row["group_id"], "label": int(row["label"]),
                                "direction": direction_name, "alpha": alpha,
                                "sum_logp_true": true_value, "sum_logp_false": false_value,
                                "margin": margin, "probability_true_two_choice": probability,
                            }, ensure_ascii=False, allow_nan=False) + "\n")
                    writer.flush()
                    completed_settings += 1
                    metadata.update({"status": "running", "completed_settings": completed_settings,
                                     "current_pair": f"{source}:{target}",
                                     "current_direction": direction_name, "current_alpha": alpha,
                                     "seconds_so_far": time.time() - started,
                                     "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30})
                    atomic_json(args.output / "metadata.json", metadata)
                    print(json.dumps({"pair": f"{source}:{target}", "direction": direction_name,
                                      "alpha": alpha, "settings": completed_settings}), flush=True)
        handle.remove()
        scores = args.output / "scores.jsonl"
        partial.replace(scores)
        metadata.update({"complete": True, "status": "complete", "seconds": time.time() - started,
                         "scores_sha256": sha256(scores),
                         "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30})
        atomic_json(args.output / "metadata.json", metadata)
        (args.output / "COMPLETE").write_text("Steering scores complete.\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        metadata.update({"complete": False, "status": "failed", "error_type": type(error).__name__,
                         "error": str(error), "seconds": time.time() - started})
        atomic_json(args.output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
