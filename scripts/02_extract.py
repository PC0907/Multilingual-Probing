#!/usr/bin/env python
"""Extract and cache activations for one (model, language, dataset) triple.

Closes the gap between src/data/ and the analysis scripts. Reads an index
produced by build_statements.to_frame, runs forward passes, and writes one
memory-mapped .npy per layer plus a meta.json describing how it was built.

STREAMING, NOT ACCUMULATING
---------------------------
`src.activations.extract.extract_activations` collects every layer in memory and
concatenates at the end. That is fine for a small run and wasteful for a real
one: a 33-layer model over 20k statements at 4096 dims is roughly 11 GB in
float32, held all at once, before anything is written.

This script preallocates a memmap per layer and fills it batch by batch, so peak
memory is one batch rather than the whole corpus. It also makes the run
resumable: a partially-written cache can be detected and skipped rather than
restarted.

ENTITY SPANS ARE CONVERTED HERE
--------------------------------
The index carries CHARACTER spans, because templates.py is model-agnostic. Token
spans depend on the tokenizer, so the conversion happens at extraction time using
`return_offsets_mapping`. If a statement's entity was truncated away, the row is
recorded in the manifest rather than silently pooled at an arbitrary position.

Statements are NOT sorted by length before batching. Length-sorted batches change
the padding pattern per batch, and with left padding that changes which absolute
positions hold real tokens. Position embeddings then differ between a statement
batched with short neighbours and the same statement batched with long ones,
which introduces a systematic difference correlated with statement length — and
length correlates with the entity labels being swapped. Fixed order costs some
throughput and removes the confound.

Usage::

    python scripts/02_extract.py \\
        --model meta-llama/Llama-3.1-8B \\
        --index data/processed/ta/index_B.csv \\
        --out   data/processed/ta/llama-3.1-8b \\
        --pooling entity_last --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import ExtractionConfig, pool_tokens   # noqa: E402
from src.data.templates import char_span_to_token_span              # noqa: E402


def open_layer_memmaps(
    out_dir: Path, layers: list[int], n_rows: int, dim: int
) -> dict[int, np.memmap]:
    """Preallocate one on-disk float32 array per layer.

    float32 regardless of the model's inference dtype. A mass-mean direction is a
    difference of two nearly-equal means, which is exactly where bf16's ~8-bit
    mantissa loses the signal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        layer: np.lib.format.open_memmap(
            out_dir / f"layer_{layer:02d}.npy",
            mode="w+", dtype=np.float32, shape=(n_rows, dim),
        )
        for layer in layers
    }


def resolve_entity_spans(
    index: pd.DataFrame,
    offset_mappings: list[list[tuple[int, int]]],
    batch_rows: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Convert character spans to token spans for one batch.

    Returns the spans plus the row ids that failed, so truncation losses are
    counted in the manifest instead of disappearing.
    """
    spans, failed = [], []
    for position, row_id in enumerate(batch_rows):
        char_span = (
            int(index.loc[row_id, "object_char_start"]),
            int(index.loc[row_id, "object_char_end"]),
        )
        try:
            start, end = char_span_to_token_span(offset_mappings[position], char_span)
            spans.append((start, end))
        except ValueError:
            failed.append(int(row_id))
            spans.append((0, 1))       # placeholder; row excluded downstream
    return spans, failed


def write_meta(
    out_dir: Path,
    config: ExtractionConfig,
    index_path: Path,
    n_rows: int,
    n_layers: int,
    dim: int,
    truncated_rows: list[int],
) -> None:
    """Record everything needed to decide whether two caches are comparable."""
    (out_dir / "meta.json").write_text(json.dumps({
        "config": config.to_dict(),
        "index_path": str(index_path),
        "n_statements": n_rows,
        "n_layers": n_layers,
        "hidden_dim": dim,
        "truncated_entity_rows": truncated_rows,
        "n_truncated": len(truncated_rows),
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pooling", default="last",
                        choices=["last", "mean", "entity_last"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16",
                        help="inference dtype; activations are cached as float32 regardless")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and (args.out / "meta.json").exists() and not args.overwrite:
        raise SystemExit(
            f"{args.out} already holds a complete cache. Pass --overwrite to "
            "rebuild, or point --out elsewhere. Silently reusing a cache built "
            "with different settings is the easiest way to compare "
            "non-comparable activations."
        )

    index = (
        pd.read_parquet(args.index) if args.index.suffix == ".parquet"
        else pd.read_csv(args.index)
    ).reset_index(drop=True)

    for column in ("text", "label"):
        if column not in index.columns:
            raise SystemExit(f"{args.index}: missing required column {column!r}")
    if args.pooling == "entity_last":
        missing = [c for c in ("object_char_start", "object_char_end")
                   if c not in index.columns]
        if missing:
            raise SystemExit(
                f"entity_last pooling needs {missing}. These are recorded at "
                "template render time and cannot be recovered — regenerate the "
                "dataset with build_statements.to_frame."
            )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device,
    )
    model.eval()

    config = ExtractionConfig(
        model_id=args.model,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        inference_dtype=args.dtype,
    )

    texts = index["text"].tolist()
    n_rows = len(texts)

    # One probe batch to learn the shape before preallocating.
    probe = tokenizer(texts[:1], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(model.device)
    with torch.inference_mode():
        shape_out = model(**probe, output_hidden_states=True)
    n_layers = len(shape_out.hidden_states)
    dim = shape_out.hidden_states[0].shape[-1]
    del shape_out

    layers = list(range(n_layers))
    memmaps = open_layer_memmaps(args.out, layers, n_rows, dim)
    print(f"{args.model}: {n_layers} layers, dim {dim}, {n_rows} statements")
    print(f"cache size ~{n_rows * n_layers * dim * 4 / 1e9:.1f} GB float32")

    truncated: list[int] = []
    for start in range(0, n_rows, args.batch_size):
        batch_rows = np.arange(start, min(start + args.batch_size, n_rows))
        batch_texts = [texts[i] for i in batch_rows]

        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=(args.pooling == "entity_last"),
        )
        offsets = encoded.pop("offset_mapping", None)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        spans = None
        if args.pooling == "entity_last":
            spans, failed = resolve_entity_spans(
                index, [list(map(tuple, o.tolist())) for o in offsets], batch_rows
            )
            truncated.extend(failed)

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)

        mask = encoded["attention_mask"].cpu().numpy()
        for layer in layers:
            hidden = outputs.hidden_states[layer].to(torch.float32).cpu().numpy()
            memmaps[layer][batch_rows] = pool_tokens(
                hidden, mask, args.pooling, spans
            )
        del outputs

        if start % (args.batch_size * 50) == 0:
            print(f"  {start + len(batch_rows)}/{n_rows}")

    for array in memmaps.values():
        array.flush()

    write_meta(args.out, config, args.index, n_rows, n_layers, dim, truncated)

    print(f"\nwrote {args.out}")
    if truncated:
        print(f"WARNING: {len(truncated)} statements had their object entity "
              f"truncated at max_length={args.max_length}. Those rows were pooled "
              "at a placeholder position and MUST be excluded before analysis — "
              "their ids are in meta.json. Raising --max-length is usually the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
