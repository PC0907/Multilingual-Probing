#!/usr/bin/env python3
"""Score three multilingual true/false judgment prompts and statement surprisal."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except Exception as error:
        return f"unavailable: {error}"


def continuation_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Empty answer tokenization: {answer!r}")
    return ids


def batch_conditional_loglikelihood(model, tokenizer, prompts: list[str], answers: list[str], device: str):
    import torch
    sequences, starts, answer_lengths = [], [], []
    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = continuation_ids(tokenizer, answer)
        if not prompt_ids:
            raise ValueError("Empty prompt tokenization")
        sequences.append(prompt_ids + answer_ids)
        starts.append(len(prompt_ids))
        answer_lengths.append(len(answer_ids))
    batch = tokenizer.pad({"input_ids": sequences,
                           "attention_mask": [[1] * len(row) for row in sequences]},
                          padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    sums, means = [], []
    for row, (start, length) in enumerate(zip(starts, answer_lengths)):
        token_ids = batch["input_ids"][row, start:start + length]
        positions = torch.arange(start - 1, start + length - 1, device=device)
        selected = log_probs[row, positions, token_ids]
        sums.append(float(selected.sum().item()))
        means.append(float(selected.mean().item()))
    return sums, means, answer_lengths


def batch_statement_surprisal(model, tokenizer, statements: list[str], device: str):
    import torch
    sequences = tokenizer(statements, padding=True, truncation=False, add_special_tokens=True,
                          return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**sequences, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = sequences["input_ids"][:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    valid = sequences["attention_mask"][:, 1:].bool()
    special_ids = set(tokenizer.all_special_ids)
    for special_id in special_ids:
        valid &= targets.ne(special_id)
    if not bool(valid.sum(dim=1).min() > 0):
        raise ValueError("Statement contains no predictable non-special token")
    losses = -(token_log_probs * valid).sum(dim=1) / valid.sum(dim=1)
    return losses.cpu().numpy().astype(float).tolist(), valid.sum(dim=1).cpu().numpy().astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.mkdir(parents=True)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not records or len(records) % 3:
        raise ValueError("Expected three aligned templates per fact")
    identities = [(r["id"], r["template_index"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate fact/template identity")
    metadata = {"complete": False, "status": "initializing", "input": str(args.input.resolve()),
                "input_sha256": sha256(args.input), "model": str(args.model.resolve()),
                "n_prompt_records": len(records), "batch_size": args.batch_size,
                "cuda_visible_devices": visible, "gpu_before": gpu_snapshot(),
                "score_definition": "full continuation sequence conditional log likelihood; continuation tokenized from one leading space plus frozen answer string",
                "primary_margin": "sum_logp_true_minus_sum_logp_false",
                "secondary_margin": "mean_logp_true_minus_mean_logp_false",
                "statement_surprisal": "mean negative log likelihood over predictable non-special statement tokens"}
    atomic_json(args.output / "metadata.json", metadata)
    started = time.time()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no padding or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.eval().requires_grad_(False).to("cuda:0")
        metadata.update({"torch": torch.__version__, "transformers": transformers.__version__,
                         "tokenizer_class": type(tokenizer).__name__, "gpu_name": torch.cuda.get_device_name(0),
                         "answer_tokenization": {}})
        for language_answer in sorted({(r["answer_true"], r["answer_false"]) for r in records}):
            metadata["answer_tokenization"]["|".join(language_answer)] = {
                "true": continuation_ids(tokenizer, language_answer[0]),
                "false": continuation_ids(tokenizer, language_answer[1])}
        atomic_json(args.output / "metadata.json", metadata)
        output_rows = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            prompts = [row["prompt"] for row in batch]
            true_answers = [row["answer_true"] for row in batch]
            false_answers = [row["answer_false"] for row in batch]
            true_sum, true_mean, true_lengths = batch_conditional_loglikelihood(model, tokenizer, prompts, true_answers, "cuda:0")
            false_sum, false_mean, false_lengths = batch_conditional_loglikelihood(model, tokenizer, prompts, false_answers, "cuda:0")
            for row, ts, tm, tl, fs, fm, fl in zip(batch, true_sum, true_mean, true_lengths,
                                                   false_sum, false_mean, false_lengths):
                margin = ts - fs
                probability_true = float(1.0 / (1.0 + np.exp(np.clip(-margin, -700, 700))))
                entropy = float(-(probability_true * np.log(max(probability_true, 1e-300)) +
                                  (1 - probability_true) * np.log(max(1 - probability_true, 1e-300))))
                output_rows.append({
                    "id": row["id"], "group_id": row["group_id"], "partition": row["partition"],
                    "label": int(row["label"]), "surface_language": row["surface_language"],
                    "template_index": int(row["template_index"]),
                    "sum_logp_true": ts, "sum_logp_false": fs, "sum_margin": margin,
                    "mean_logp_true": tm, "mean_logp_false": fm, "mean_margin": tm - fm,
                    "true_answer_tokens": tl, "false_answer_tokens": fl,
                    "probability_true_two_choice": probability_true,
                    "response_entropy_two_choice": entropy,
                    "prediction": int(margin >= 0), "correct": int((margin >= 0) == bool(row["label"])),
                })
            if start == 0 or start + len(batch) == len(records) or (start + len(batch)) % 256 == 0:
                metadata.update({"status": "scoring_answers", "n_completed_prompt_records": start + len(batch),
                                 "seconds_so_far": time.time() - started,
                                 "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30})
                atomic_json(args.output / "metadata.json", metadata)

        unique = {}
        for row in records:
            unique.setdefault(row["id"], row)
        unique_rows = list(unique.values())
        surprisal = {}
        for start in range(0, len(unique_rows), args.batch_size):
            batch = unique_rows[start:start + args.batch_size]
            losses, counts = batch_statement_surprisal(model, tokenizer,
                                                       [row["raw_statement"] for row in batch], "cuda:0")
            for row, loss, count in zip(batch, losses, counts):
                surprisal[row["id"]] = {"statement_surprisal": loss, "statement_scored_tokens": count}
        for row in output_rows:
            row.update(surprisal[row["id"]])
        results_path = args.output / "scores.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        metadata.update({"complete": True, "status": "complete", "seconds": time.time() - started,
                         "scores_sha256": sha256(results_path), "n_unique_facts": len(unique),
                         "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30,
                         "gpu_after": gpu_snapshot()})
        atomic_json(args.output / "metadata.json", metadata)
        (args.output / "COMPLETE").write_text("Behavioral scoring complete.\n", encoding="utf-8")
        print(json.dumps({"complete": str(args.output), "records": len(output_rows),
                          "seconds": metadata["seconds"]}))
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
