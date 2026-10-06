#!/usr/bin/env python3
"""Extract one frozen decoder-block state on exactly one visible GPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .extract_prompted_states import content_end_positions, validate_records


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-number", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("Expose exactly one GPU")
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    records = json.loads(args.input.read_text(encoding="utf-8"))
    validate_records(records)
    args.output.mkdir(parents=True)
    metadata = {
        "complete": False, "status": "initializing", "input_sha256": sha256(args.input),
        "source_ids": [row["id"] for row in records], "n": len(records),
        "block_number": args.block_number, "model": str(args.model.resolve()),
        "cuda_visible_devices": visible,
    }
    atomic_json(args.output / "metadata.json", metadata)
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        if not 1 <= args.block_number <= int(config.num_hidden_layers):
            raise ValueError("Block number outside model depth")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = [row["sentence"] for row in records]
        all_ids = tokenizer(prompts, add_special_tokens=True, truncation=False)["input_ids"]
        content_ids = tokenizer(prompts, add_special_tokens=False, truncation=False)["input_ids"]
        positions = content_end_positions(all_ids, content_ids)
        lengths = [len(ids) for ids in all_ids]
        if max(lengths) > min(args.max_seq_length, int(config.max_position_embeddings)):
            raise ValueError("At least one example exceeds frozen context limit")
        model = AutoModel.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.eval().requires_grad_(False).to("cuda:0")
        layers = getattr(model, "layers", None)
        if layers is None:
            layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != int(config.num_hidden_layers):
            raise RuntimeError("Unsupported decoder-layer layout")
        state = {}

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            state["captured"] = hidden.detach()

        handle = layers[args.block_number - 1].register_forward_hook(hook)
        output = np.lib.format.open_memmap(
            args.output / "last.partial.npy", mode="w+", dtype=np.float32,
            shape=(len(records), int(config.hidden_size)))
        for start in range(0, len(records), args.batch_size):
            end = min(start + args.batch_size, len(records))
            batch = tokenizer(prompts[start:end], padding=True, truncation=False,
                              add_special_tokens=True, return_tensors="pt").to("cuda:0")
            state.clear()
            with torch.inference_mode():
                model(**batch, use_cache=False, return_dict=True)
            hidden = state.pop("captured")
            local_positions = torch.as_tensor(positions[start:end], device="cuda:0")
            chosen = hidden[torch.arange(end - start, device="cuda:0"), local_positions]
            output[start:end] = chosen.float().cpu().numpy()
            output.flush()
            metadata.update({"status": "running", "n_completed": end,
                             "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30})
            atomic_json(args.output / "metadata.json", metadata)
        handle.remove()
        del output
        (args.output / "last.partial.npy").replace(args.output / "last.npy")
        array = np.load(args.output / "last.npy", mmap_mode="r")
        metadata.update({
            "complete": True, "status": "complete", "shape": list(array.shape),
            "dtype": str(array.dtype), "pooling": "final non-special content token",
            "peak_cuda_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        })
        atomic_json(args.output / "metadata.json", metadata)
        (args.output / "COMPLETE").write_text("Selected-layer extraction complete.\n")
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

