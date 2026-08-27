#!/usr/bin/env python
"""Week one: the English split-half ceiling.

Produces the per-layer curve that every cross-lingual alignment number in this
project is normalised against. Until it exists, no cosine and no principal angle
is interpretable: a raw English-Tamil cosine of 0.4 is near-perfect if two
English probes only manage 0.45, and near-total divergence if they manage 0.95.

Needs no multilingual data. Runs in an afternoon. Do it before anything else.

What it emits, per layer:

    ceiling      cosine between directions from two disjoint English splits
    floor        cosine between directions from label-permuted splits, which
                 preserves the real anisotropic geometry of the space while
                 destroying the truth signal
    separation   ceiling - floor. Where this is near zero the layer carries no
                 usable linear truth signal and MUST be excluded from the layer
                 sweep. Normalising by a near-zero denominator produces enormous
                 meaningless numbers, and those are the ones that end up in a
                 figure.
    ceiling_sd   spread across resamples. Wide spread means the direction is
                 unstable at this sample size; the layer's alignment number is
                 not trustworthy however good its mean looks.

Usage::

    python scripts/01_ceiling.py \\
        --activations data/processed/en/llama-3.1-8b \\
        --index       data/processed/en/index.parquet \\
        --n-per-split 500 \\
        --out         results/tables/ceiling_en_llama-3.1-8b.csv

`--n-per-split` is not cosmetic. Split-half agreement rises with sample size, so
the ceiling must be measured at the SAME size used for the cross-lingual
directions. Set it to the smallest per-language dataset in the study, and use
that same value in scripts/03_geometry.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import ActivationCache          # noqa: E402
from src.geometry.baselines import cosine, split_half_ceiling  # noqa: E402
from src.probes.mass_mean import mass_mean_direction          # noqa: E402


REQUIRED_COLUMNS = ("label", "group_id")


def load_index(path: Path) -> pd.DataFrame:
    """Load statement metadata and check it can support a grouped split.

    `group_id` must be shared by every statement derived from the same fact —
    an affirmative and its negation, several templates over one entity. Without
    it, the two halves leak into each other through near-duplicate surface forms
    and the ceiling comes out inflated, which then makes every cross-lingual
    number look worse than it is.
    """
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"index is missing required columns: {missing}")

    labels = set(df["label"].unique().tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"label must be 0/1, found {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError("index contains only one class")

    n_groups = df["group_id"].nunique()
    if n_groups == len(df):
        print(
            "  WARNING: every statement has a unique group_id. If this dataset "
            "contains negation pairs, they are NOT grouped and the ceiling will "
            "be inflated. Check the generation code.",
            file=sys.stderr,
        )
    return df


def ceiling_for_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_per_split: int | None,
    n_repeats: int,
    seed: int,
) -> dict:
    """Ceiling, permuted floor and separation for a single layer."""
    common = dict(
        activations=activations,
        labels=labels,
        direction_fn=mass_mean_direction,
        groups=groups,
        n_repeats=n_repeats,
        n_per_split=n_per_split,
        seed=seed,
    )
    ceiling = split_half_ceiling(**common, permute_labels=False)
    floor = split_half_ceiling(**common, permute_labels=True)

    return {
        "ceiling": ceiling.mean,
        "ceiling_sd": ceiling.std,
        "ceiling_p05": ceiling.p05,
        "ceiling_p95": ceiling.p95,
        "floor": floor.mean,
        "floor_sd": floor.std,
        "floor_p95": floor.p95,
        "separation": ceiling.mean - floor.mean,
        # A layer is usable if its ceiling clears the floor's upper tail. This is
        # deliberately conservative: a layer whose real signal sits inside the
        # noise band of the permuted control cannot support a normalised
        # alignment claim.
        "usable": bool(ceiling.p05 > floor.p95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True, type=Path,
                        help="ActivationCache directory for English")
    parser.add_argument("--index", required=True, type=Path,
                        help="parquet/csv with label, group_id per statement")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-split", type=int, default=None,
                        help="examples per half; match the cross-lingual runs")
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = load_index(args.index)
    labels = index["label"].to_numpy()
    groups = index["group_id"].to_numpy()

    cache = ActivationCache(args.activations)
    layers = args.layers or cache.available_layers()
    if not layers:
        raise SystemExit(f"no cached layers found in {args.activations}")

    print(f"model    : {cache.config.model_id}")
    print(f"pooling  : {cache.config.pooling}")
    print(f"n        : {len(index)} statements, {index['group_id'].nunique()} groups")
    print(f"split    : {args.n_per_split or 'half of available'} per side, "
          f"{args.n_repeats} resamples")
    print()
    print(f"{'layer':>5} {'ceiling':>9} {'±sd':>7} {'floor':>8} {'sep':>8}  usable")

    rows = []
    for layer in layers:
        acts = np.asarray(cache.load_layer(layer, mmap=True), dtype=np.float64)
        if acts.shape[0] != len(index):
            raise ValueError(
                f"layer {layer} has {acts.shape[0]} rows but the index has "
                f"{len(index)}. The cache and the index are out of sync — "
                "regenerate rather than trying to align them."
            )

        stats = ceiling_for_layer(
            acts, labels, groups, args.n_per_split, args.n_repeats, args.seed
        )
        stats["layer"] = layer
        rows.append(stats)

        print(f"{layer:>5} {stats['ceiling']:>9.3f} {stats['ceiling_sd']:>7.3f} "
              f"{stats['floor']:>8.3f} {stats['separation']:>8.3f}  "
              f"{'yes' if stats['usable'] else 'NO'}")

    df = pd.DataFrame(rows).set_index("layer").sort_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)

    sidecar = args.out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "config": cache.config.to_dict(),
        "n_statements": int(len(index)),
        "n_groups": int(index["group_id"].nunique()),
        "n_per_split": args.n_per_split,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
    }, indent=2))

    usable = df.index[df["usable"]].tolist()
    print()
    print(f"wrote {args.out}")
    if not usable:
        print("NO LAYER IS USABLE. The probe finds nothing above the permuted "
              "floor anywhere. Before concluding anything about the model, check "
              "pooling (a padding-side error returns the pad embedding "
              "everywhere) and check that labels are not shuffled.")
        return 1

    best = df["ceiling"].idxmax()
    print(f"usable layers : {usable[0]}–{usable[-1]} ({len(usable)} of {len(df)})")
    print(f"peak ceiling  : layer {best} at {df.loc[best, 'ceiling']:.3f}")
    print()
    print("Interpretation: the peak is the best agreement two probes on the SAME "
          "language achieve. No cross-lingual pair should be expected to beat it, "
          "and a cross-lingual value near it is strong alignment, not weak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
