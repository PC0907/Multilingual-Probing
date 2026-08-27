#!/usr/bin/env python
"""RQ2: cross-lingual transfer grid, joined against RQ1 geometry.

Produces two tables.

    transfer_<model>.csv     one row per (source, target, layer): AUROC,
                             accuracy at both thresholds, relative transfer
    dissociation_<model>.csv the join that the paper is actually about — one row
                             per language pair with RQ1's normalised alignment
                             beside RQ2's relative transfer

The second table is the point. RQ1 alone says whether directions look alike; RQ2
alone says whether probes work across languages. The claim the project rests on
is that these can come apart, and it is only visible when they sit in the same
frame. If the correlation between them is weak, that IS the finding, and it is a
stronger one than confirming the obvious story.

Usage::

    python scripts/04_transfer.py \\
        --cache en=data/processed/en/llama hi=data/processed/hi/llama ... \\
        --index en=data/processed/en/index_A.csv ... \\
        --ceiling  results/tables/ceiling_en_llama.csv \\
        --geometry results/tables/geometry_llama.csv \\
        --n-per-language 500 \\
        --out-dir results/tables

Note the asymmetry between the two measurements, which has to be handled rather
than ignored: RQ1 alignment is symmetric (cos(a,b) == cos(b,a)) while transfer is
directional (en->ta need not equal ta->en). The join averages the two directions
for the scatter and keeps both in the full grid, because a large directional
asymmetry is itself informative — it usually means one language's probe is
simply better, not that the pair is asymmetrically aligned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import ActivationCache               # noqa: E402
from src.transfer.crosslingual import (                           # noqa: E402
    add_relative_transfer,
    transfer_grid,
)


def parse_mapping(pairs: list[str], flag: str) -> dict[str, Path]:
    mapping = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"{flag} expects lang=path, got {item!r}")
        lang, path = item.split("=", 1)
        mapping[lang] = Path(path)
    return mapping


def load_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def balanced_subsample(index: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Balanced row indices, same contract as scripts/03_geometry.py.

    Must match 03_geometry's subsampling, or the alignment and transfer numbers
    in the dissociation table are computed from different data and the scatter
    compares two things that were never measured on the same statements.
    """
    rng = np.random.default_rng(seed)
    per_class = n // 2
    picked = []
    for label in (0, 1):
        rows = index.index[index["label"] == label].to_numpy()
        if len(rows) < per_class:
            raise ValueError(
                f"need {per_class} statements with label={label}, have {len(rows)}"
            )
        picked.append(rng.choice(rows, per_class, replace=False))
    return np.concatenate(picked)


def build_dissociation(
    transfer: pd.DataFrame,
    geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Join symmetric alignment onto averaged bidirectional transfer.

    Both are taken at each pair's own best layer rather than a fixed layer.
    Fixing a layer would penalise pairs whose shared structure peaks elsewhere,
    and the layer at which a pair aligns is itself a reported quantity.
    """
    off_diagonal = transfer[transfer["source"] != transfer["target"]].copy()
    off_diagonal["pair"] = off_diagonal.apply(
        lambda r: tuple(sorted([r["source"], r["target"]])), axis=1
    )

    # Best layer per direction, then average the two directions of each pair.
    best_per_direction = (
        off_diagonal.loc[off_diagonal.groupby(["source", "target"])["auroc"].idxmax()]
    )
    pair_transfer = (
        best_per_direction.groupby("pair")
        .agg(
            relative_transfer=("relative_transfer", "mean"),
            auroc=("auroc", "mean"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
            calibration_gap=("calibration_gap", "mean"),
            transfer_layer=("layer", "median"),
        )
        .reset_index()
    )
    pair_transfer["direction_asymmetry"] = (
        pair_transfer["auroc_max"] - pair_transfer["auroc_min"]
    )

    geo = geometry.copy()
    geo["pair"] = geo.apply(
        lambda r: tuple(sorted([r["language_a"], r["language_b"]])), axis=1
    )
    best_geo = geo.loc[geo.groupby("pair")["cosine_normalised"].idxmax()]
    pair_geometry = best_geo[
        ["pair", "cosine_normalised", "cosine_raw", "layer"]
    ].rename(columns={"layer": "alignment_layer"})

    merged = pair_transfer.merge(pair_geometry, on="pair", how="inner")
    merged["language_a"] = merged["pair"].apply(lambda p: p[0])
    merged["language_b"] = merged["pair"].apply(lambda p: p[1])
    return merged.drop(columns=["pair"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True, metavar="LANG=DIR")
    parser.add_argument("--index", nargs="+", required=True, metavar="LANG=PATH")
    parser.add_argument("--ceiling", required=True, type=Path)
    parser.add_argument("--geometry", type=Path, default=None,
                        help="output of 03_geometry.py; enables the dissociation table")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-per-language", type=int, required=True)
    parser.add_argument("--no-whiten", action="store_true",
                        help="robustness check: transfer without covariance correction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    caches = parse_mapping(args.cache, "--cache")
    indices = parse_mapping(args.index, "--index")
    if set(caches) != set(indices):
        raise SystemExit(f"--cache and --index disagree: {set(caches) ^ set(indices)}")

    ceiling = pd.read_csv(args.ceiling).set_index("layer")
    layers = [int(l) for l in ceiling.index[ceiling["usable"]]]
    if not layers:
        raise SystemExit("no usable layers in the ceiling table")

    meta_path = args.ceiling.with_suffix(".meta.json")
    if meta_path.exists():
        built_at = json.loads(meta_path.read_text()).get("n_per_split")
        if built_at is not None and built_at != args.n_per_language:
            print(
                f"NOTE: ceiling built at n={built_at}, this run uses "
                f"n={args.n_per_language}. Transfer itself does not use the "
                "ceiling, but the geometry column in the dissociation table "
                "does, so the two halves of the scatter would not be comparable.",
                file=sys.stderr,
            )

    reference = ActivationCache(next(iter(caches.values()))).config
    subsamples, label_map = {}, {}
    for lang, cache_dir in caches.items():
        cache = ActivationCache(cache_dir)
        reference.assert_compatible(cache.config)
        index = load_index(indices[lang])
        rows = balanced_subsample(index, args.n_per_language, args.seed)
        subsamples[lang] = (cache, rows)
        label_map[lang] = index.loc[rows, "label"].to_numpy()
        print(f"{lang}: {len(index)} statements -> {len(rows)} sampled")

    results = []
    for layer in layers:
        acts = {
            lang: np.asarray(cache.load_layer(layer, mmap=True), dtype=np.float64)[rows]
            for lang, (cache, rows) in subsamples.items()
        }
        results.extend(
            add_relative_transfer(
                transfer_grid(
                    acts, label_map, layer=layer,
                    whiten=not args.no_whiten, seed=args.seed,
                )
            )
        )

    transfer = pd.DataFrame(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    transfer_path = args.out_dir / "transfer.csv"
    transfer.to_csv(transfer_path, index=False)

    peak = transfer.loc[transfer.groupby(["source", "target"])["auroc"].idxmax()]
    print()
    print("AUROC at each pair's best layer (row = trained, col = tested):")
    print(peak.pivot(index="source", columns="target", values="auroc").round(3).to_string())

    diagonal = peak[peak["source"] == peak["target"]]
    weak = diagonal[diagonal["auroc"] < 0.6]["target"].tolist()
    if weak:
        print(f"\nWARNING: in-language probes are near chance for {weak}. Transfer "
              "INTO these languages is uninterpretable — a low cell means the "
              "language has no readable truth signal at all, not that transfer "
              "failed. Exclude them or report separately.")

    print()
    print("Calibration gap (target-refit minus source-threshold accuracy):")
    off = peak[peak["source"] != peak["target"]]
    print(off.pivot(index="source", columns="target",
                    values="calibration_gap").round(3).to_string())
    print("Large values mean the DIRECTION transfers and only the threshold is "
          "wrong — a recalibration problem, not a representational one.")

    if args.geometry and args.geometry.exists():
        geometry = pd.read_csv(args.geometry)
        dissociation = build_dissociation(transfer, geometry)
        dissociation_path = args.out_dir / "dissociation.csv"
        dissociation.to_csv(dissociation_path, index=False)

        print()
        print("DISSOCIATION — RQ1 alignment vs RQ2 transfer, per pair:")
        print(dissociation[[
            "language_a", "language_b", "cosine_normalised",
            "relative_transfer", "alignment_layer", "transfer_layer",
            "direction_asymmetry",
        ]].round(3).to_string(index=False))

        if len(dissociation) >= 3:
            r = dissociation["cosine_normalised"].corr(
                dissociation["relative_transfer"]
            )
            print(f"\nPearson r(alignment, transfer) = {r:.3f} "
                  f"over {len(dissociation)} pairs")
            print("With this few pairs the correlation is descriptive only — no "
                  "confidence interval from 6 points is worth reporting. The "
                  "individual pairs are the evidence; the scatter is the figure.")
        print(f"\nwrote {dissociation_path}")
    else:
        print("\nno --geometry given; skipping the dissociation table (which is "
              "the table the paper is about)")

    print(f"wrote {transfer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
