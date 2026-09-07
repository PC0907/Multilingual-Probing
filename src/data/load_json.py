#!/usr/bin/env python
"""Convert a statement dataset into the index CSV the pipeline reads.

Bridges hand-authored and downloaded datasets into the schema
`scripts/02_extract.py` consumes, which was designed around
`build_statements.to_frame` output from template generation.

Accepts JSON (`[{id, sentence, label, tag}, ...]`) and CSV (`statement,label`),
covering both `data/raw/json_merged.json` and the Truth_is_Universal and
geometry-of-truth CSVs.

Four things happen here that cannot be fixed after extraction.

LABEL CONVENTION
----------------
The repo convention is `1 = true`, set by `Statement.label` in templates.py and
assumed by every mass-mean computation downstream. `json_merged.json` and the
Truth_is_Universal CSVs already use it. The earlier German sports files use
`0 = true` and MUST be loaded with `--true-label 0`, which inverts them on the
way in.

Getting this wrong does not raise and does not change any accuracy number. It
silently negates every direction, so cosines between a correctly-loaded and an
incorrectly-loaded language come out near -1 and look like a dramatic finding.
The flag is required rather than defaulted for that reason.

GROUPING
--------
Hand-authored datasets contain near-minimal pairs: "Die Elbe fliesst durch
Hamburg" and "Die Elbe fliesst durch Muenchen" differ in one entity. A random
train/test split puts one in each half, and the probe can score well by
recognising the shared prefix rather than by reading truth.

`group_id` is assigned from the first N words. `baselines.split_half_ceiling`
keeps a group intact, so variants of one fact land on the same side. The
heuristic over-groups sometimes (two unrelated facts opening identically),
costing a little effective sample size, and under-groups when variants are
phrased differently, which leaks. Over-grouping is the safe direction, so the
prefix is deliberately short.

Crowd-sourced sets like CommonClaim have no minimal pairs, so nearly every
statement becomes a singleton group. That is correct, not a failure: grouped and
random splits then coincide, which is itself worth seeing in the output.

BALANCED SUBSAMPLING
--------------------
`--sample-n` draws equally from each class rather than proportionally. A plain
random subsample would carry the source's class prior across, and a prior
mismatch between two languages is a confound in exactly the cross-language
comparison this is meant to enable.

ORDER
-----
`02_extract.py` reads the index in file order and does not sort, because batch
composition changes the left-padding pattern and therefore which absolute
positions hold real tokens. Whatever order this file writes IS the extraction
order and becomes part of the measurement.

Source files are often grouped by topic, so unshuffled every batch would be
topically homogeneous and padding patterns would correlate with topic. That is a
direct confound for the topic-transfer matrix. Rows are therefore shuffled
exactly once here, with a recorded seed, and never again.

Usage::

    python -m src.data.load_json \\
        --input data/raw/json_merged.json \\
        --output data/processed/de/index.csv \\
        --language de --true-label 1

    python -m src.data.load_json \\
        --input /tmp/tiu/datasets/common_claim_true_false.csv \\
        --output data/processed/en/index.csv \\
        --language en --true-label 1 --sample-n 1996
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

__all__ = ["load_records", "build_index", "NEG_MARKERS"]

# Polarity here is INFERRED, not authored: a statement false for reasons other
# than negation is correctly counted affirmative, but a negation construction
# outside this list is missed. Extend per language. English markers are included
# so the same inference runs on the English sets.
NEG_MARKERS = [
    # German
    "nie", "kein", "keine", "keinen", "keiner", "nicht", "niemals",
    "ausschließlich", "ausschliesslich", "nur", "völlig", "voellig",
    # English
    "not", "never", "no", "none", "cannot", "n't", "neither", "nor", "only",
]

AFFIRMATIVE, NEGATED = 0, 1
TEXT_KEYS = ("sentence", "statement", "text")


def has_negation(text: str, markers: list[str] = NEG_MARKERS) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(m)}\b", low) for m in markers)


def subject_key(text: str, n_words: int = 4) -> str:
    """Leakage-group key: the opening words, normalised."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return " ".join(words[:n_words])


def load_records(path: Path) -> tuple[list[str], list[int], list[str], list]:
    """Read JSON or CSV. Both map onto (texts, labels, tags, ids)."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        text_key = next((k for k in TEXT_KEYS if k in df.columns), None)
        if text_key is None:
            raise SystemExit(
                f"{path}: no text column; expected one of {TEXT_KEYS}, "
                f"found {list(df.columns)}"
            )
        if "label" not in df.columns:
            raise SystemExit(f"{path}: no 'label' column")
        bad = df[~df["label"].isin([0, 1])]
        if len(bad):
            raise SystemExit(
                f"{path}: {len(bad)} rows have a label outside 0/1, first at "
                f"row {bad.index[0]}. Fix the source rather than coercing here."
            )
        texts = df[text_key].astype(str).str.strip().tolist()
        labels = df["label"].astype(int).tolist()
        tags = (df["tag"].astype(str).tolist() if "tag" in df.columns
                else ["untagged"] * len(df))
        ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
        return texts, labels, tags, ids

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON array")
    if not data:
        raise SystemExit(f"{path}: empty")

    text_key = next((k for k in TEXT_KEYS if k in data[0]), None)
    if text_key is None:
        raise SystemExit(
            f"{path}: no text field; expected one of {TEXT_KEYS}, "
            f"found {sorted(data[0])}"
        )

    texts, labels, tags, ids = [], [], [], []
    for i, r in enumerate(data):
        if text_key not in r or "label" not in r:
            raise SystemExit(f"{path} row {i}: missing {text_key!r} or 'label'")
        if r["label"] not in (0, 1):
            raise SystemExit(
                f"{path} row {i}: label is {r['label']!r}; want 0 or 1. "
                "Fix the source rather than coercing here."
            )
        texts.append(str(r[text_key]).strip())
        labels.append(int(r["label"]))
        tags.append(str(r.get("tag", "")) or "untagged")
        ids.append(r.get("id", i))
    return texts, labels, tags, ids


def build_index(
    texts: list[str],
    labels: list[int],
    tags: list[str],
    ids: list,
    language: str,
    true_label: int,
    seed: int,
    group_words: int,
    drop_duplicates: bool = True,
    sample_n: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the index frame and a provenance manifest."""
    df = pd.DataFrame({
        "source_id": ids,
        "text": texts,
        "raw_label": labels,
        "tag": tags,
    })

    # Convention: 1 = true, repo-wide.
    df["label"] = df["raw_label"] if true_label == 1 else 1 - df["raw_label"]

    n_before = len(df)
    dropped_duplicates = []
    if drop_duplicates:
        dupes = df[df.duplicated("text", keep="first")]
        dropped_duplicates = dupes["source_id"].tolist()
        df = df.drop_duplicates("text", keep="first")

    # A statement appearing with BOTH labels is a source error, not a duplicate.
    conflicts = (
        df.groupby("text")["label"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if conflicts:
        raise SystemExit(
            f"{len(conflicts)} statements appear with contradictory labels, "
            f"e.g. {conflicts[:3]}. Resolve in the source file."
        )

    # Balanced subsample. Equal draws per class, not proportional -- see the
    # module docstring on why the class prior must not travel across.
    n_sampled_from = None
    if sample_n is not None and sample_n < len(df):
        n_sampled_from = len(df)
        per_class = sample_n // 2
        parts = []
        for lab in (0, 1):
            pool = df[df["label"] == lab]
            if len(pool) < per_class:
                raise SystemExit(
                    f"cannot draw {per_class} statements with label={lab}: "
                    f"only {len(pool)} available"
                )
            parts.append(pool.sample(n=per_class, random_state=seed))
        df = pd.concat(parts)

    df["group_id"] = df["text"].map(lambda t: subject_key(t, group_words))
    df["polarity"] = df["text"].map(
        lambda t: NEGATED if has_negation(t) else AFFIRMATIVE
    )
    df["language"] = language
    df["template_id"] = f"native.{language}"

    # The single permitted shuffle. Recorded, reproducible, never repeated.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))

    sizes = Counter(df["group_id"])
    cells = Counter(zip(df["label"], df["polarity"]))
    manifest = {
        "language": language,
        "true_label_in_source": true_label,
        "shuffle_seed": seed,
        "group_words": group_words,
        "n_input": n_before,
        "n_output": len(df),
        "sampled_from": n_sampled_from,
        "sample_n": sample_n,
        "dropped_duplicate_ids": dropped_duplicates,
        "n_groups": len(sizes),
        "largest_group": max(sizes.values()),
        "singleton_groups": sum(1 for v in sizes.values() if v == 1),
        "tags": dict(Counter(df["tag"])),
        "cells": {f"label={a},polarity={b}": c for (a, b), c in cells.items()},
        "has_entity_spans": False,
        "pooling_supported": ["last", "mean"],
    }
    return df[[
        "row_id", "source_id", "text", "label", "polarity",
        "language", "group_id", "template_id", "tag",
    ]], manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--language", required=True)
    ap.add_argument("--true-label", required=True, type=int, choices=[0, 1],
                    help="which label value means TRUE in the SOURCE file "
                         "(json_merged.json and Truth_is_Universal = 1, "
                         "the German sports files = 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-words", type=int, default=4)
    ap.add_argument("--sample-n", type=int, default=None,
                    help="balanced subsample to this many rows, half per class")
    ap.add_argument("--keep-duplicates", action="store_true")
    args = ap.parse_args()

    texts, labels, tags, ids = load_records(args.input)
    df, manifest = build_index(
        texts, labels, tags, ids, args.language, args.true_label,
        args.seed, args.group_words,
        drop_duplicates=not args.keep_duplicates,
        sample_n=args.sample_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"wrote {args.output}  ({manifest['n_output']} rows, "
          f"{len(manifest['dropped_duplicate_ids'])} duplicates dropped)")
    if manifest["sampled_from"]:
        print(f"balanced subsample: {manifest['n_output']} drawn from "
              f"{manifest['sampled_from']}")
    print(f"wrote {manifest_path}")
    print(f"\ngroups: {manifest['n_groups']}  largest {manifest['largest_group']}  "
          f"singletons {manifest['singleton_groups']}")
    print("cells:", manifest["cells"])
    print("tags: ", manifest["tags"])
    print("\nNo entity spans in this source, so entity_last pooling is "
          "unavailable. Extract with --pooling last.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())