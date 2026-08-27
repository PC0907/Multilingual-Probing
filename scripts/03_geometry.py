#!/usr/bin/env python
"""RQ1: per-layer cross-lingual alignment of truth directions.

Trains a mass-mean direction INDEPENDENTLY in each language — no transfer
anywhere in this script — then compares the directions pairwise and normalises
against the same-language ceiling from scripts/01_ceiling.py.

Emits one row per (language_a, language_b, layer):

    cosine_raw              uninterpretable on its own; recorded for the appendix
    cosine_normalised       (raw - floor) / (ceiling - floor). THE number.
    angle_1, angle_2        principal angles between the 2D truth subspaces
    grassmann               root-sum-square of those angles
    ceiling, floor          what the normalisation used, carried alongside so a
                            reader can recompute without the sidecar

Usage::

    python scripts/03_geometry.py \\
        --cache en=data/processed/en/llama  hi=data/processed/hi/llama \\
                ur=data/processed/ur/llama  ta=data/processed/ta/llama \\
        --index en=data/processed/en/index_A.csv  hi=... \\
        --ceiling results/tables/ceiling_en_llama.csv \\
        --n-per-language 500 \\
        --out results/tables/geometry_llama.csv

WHY --n-per-language IS NOT OPTIONAL
------------------------------------
A mass-mean direction is an estimate of a difference of population means, and two
such estimates agree better the more data each had. If Hindi has 3000 statements
and Tamil 600, the English-Hindi cosine will beat English-Tamil for reasons that
have nothing to do with representation. Every direction here is computed from
exactly `n_per_language` statements, and that value must match the `--n-per-split`
used when the ceiling was built. The script checks the second condition and
refuses if they differ.

SUBSPACES ARE OPTIONAL, AND SILENTLY SO WHEN POLARITY IS MISSING
----------------------------------------------------------------
Principal angles need both affirmative and negated statements. Where the index
lacks a `polarity` column, or a language has only one polarity, the subspace
columns are left empty rather than filled with a number derived from a
rank-deficient basis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import ActivationCache            # noqa: E402
from src.geometry.baselines import cosine, normalised_alignment  # noqa: E402
from src.geometry.subspace import compare_subspaces, truth_subspace  # noqa: E402
from src.probes.mass_mean import (                             # noqa: E402
    check_sign_convention,
    mass_mean_direction,
)


def parse_mapping(pairs: list[str], flag: str) -> dict[str, Path]:
    """Parse `lang=path` arguments into a dict."""
    mapping = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"{flag} expects lang=path, got {item!r}")
        lang, path = item.split("=", 1)
        mapping[lang] = Path(path)
    return mapping


def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"{path}: no label column")
    return df


def subsample(
    index: pd.DataFrame, n: int, seed: int
) -> np.ndarray:
    """Row indices for a balanced, group-respecting subsample of size n.

    Balanced because an unbalanced mass-mean direction is a noisier estimate on
    the smaller side, and the imbalance would differ across languages. Grouped
    because splitting a fact's four cells across the sample boundary is harmless
    here but keeping them together makes the subsample reproducible against the
    ceiling's grouped splits.
    """
    rng = np.random.default_rng(seed)
    per_class = n // 2
    picked = []
    for label in (0, 1):
        rows = index.index[index["label"] == label].to_numpy()
        if len(rows) < per_class:
            raise ValueError(
                f"need {per_class} statements with label={label}, have {len(rows)}. "
                "Lower --n-per-language, or note that this language cannot be "
                "compared at the same sample size as the others."
            )
        picked.append(rng.choice(rows, per_class, replace=False))
    return np.concatenate(picked)


def directions_for_language(
    cache: ActivationCache,
    index: pd.DataFrame,
    layers: list[int],
    n_per_language: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray | None]]:
    """Mass-mean direction and 2D truth subspace per layer, for one language."""
    rows = subsample(index, n_per_language, seed)
    labels = index.loc[rows, "label"].to_numpy()
    has_polarity = (
        "polarity" in index.columns
        and index.loc[rows, "polarity"].nunique() == 2
    )
    polarity = index.loc[rows, "polarity"].to_numpy() if has_polarity else None

    directions: dict[int, np.ndarray] = {}
    subspaces: dict[int, np.ndarray | None] = {}

    for layer in layers:
        acts = np.asarray(cache.load_layer(layer, mmap=True), dtype=np.float64)
        if acts.shape[0] != len(index):
            raise ValueError(
                f"layer {layer} has {acts.shape[0]} rows, index has {len(index)}"
            )
        acts = acts[rows]

        direction = mass_mean_direction(acts, labels)
        if not check_sign_convention(acts, labels, direction):
            # Should be impossible given mass_mean's construction. If it fires,
            # the labels are inverted somewhere upstream, and every cosine in
            # this run would have had its sign flipped.
            raise RuntimeError(
                f"sign convention violated at layer {layer} — labels are "
                "probably inverted in the index"
            )
        directions[layer] = direction

        if has_polarity:
            try:
                subspaces[layer] = truth_subspace(acts, labels, polarity)
            except ValueError:
                # Rank-deficient: truth and polarity axes are collinear here.
                subspaces[layer] = None
        else:
            subspaces[layer] = None

    return directions, subspaces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True, metavar="LANG=DIR")
    parser.add_argument("--index", nargs="+", required=True, metavar="LANG=PATH")
    parser.add_argument("--ceiling", required=True, type=Path,
                        help="output of scripts/01_ceiling.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-language", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usable-only", action="store_true", default=True,
                        help="restrict to layers the ceiling marked usable")
    args = parser.parse_args()

    caches = parse_mapping(args.cache, "--cache")
    indices = parse_mapping(args.index, "--index")

    missing = set(caches) ^ set(indices)
    if missing:
        raise SystemExit(f"--cache and --index disagree on languages: {missing}")

    ceiling_df = pd.read_csv(args.ceiling).set_index("layer")
    meta_path = args.ceiling.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        built_at = meta.get("n_per_split")
        if built_at is not None and built_at != args.n_per_language:
            raise SystemExit(
                f"the ceiling was built at n_per_split={built_at} but this run "
                f"uses n_per_language={args.n_per_language}. Split-half agreement "
                "depends on sample size, so normalising against a ceiling from a "
                "different size produces values that are systematically wrong — "
                "above 1.0 if the ceiling was smaller, deflated if larger. "
                "Rerun 01_ceiling.py at the matching size."
            )

    layers = [int(l) for l in ceiling_df.index]
    if args.usable_only:
        usable = [int(l) for l in ceiling_df.index[ceiling_df["usable"]]]
        if not usable:
            raise SystemExit(
                "no layer was marked usable in the ceiling table. Cross-lingual "
                "alignment cannot be normalised anywhere. Investigate extraction "
                "before proceeding."
            )
        skipped = len(layers) - len(usable)
        layers = usable
        if skipped:
            print(f"restricting to {len(layers)} usable layers ({skipped} skipped)")

    # Verify all caches were built the same way before comparing anything.
    reference = ActivationCache(next(iter(caches.values()))).config
    per_language = {}
    for lang, cache_dir in caches.items():
        cache = ActivationCache(cache_dir)
        reference.assert_compatible(cache.config)
        index = load_index(indices[lang])
        print(f"{lang}: {len(index)} statements, layers {cache.available_layers()[:3]}...")
        per_language[lang] = directions_for_language(
            cache, index, layers, args.n_per_language, args.seed
        )

    rows = []
    for lang_a, lang_b in itertools.combinations(sorted(caches), 2):
        dir_a, sub_a = per_language[lang_a]
        dir_b, sub_b = per_language[lang_b]

        for layer in layers:
            raw = cosine(dir_a[layer], dir_b[layer])
            ceiling = float(ceiling_df.loc[layer, "ceiling"])
            floor = float(ceiling_df.loc[layer, "floor"])
            try:
                normalised = normalised_alignment(raw, floor, ceiling)
            except ValueError:
                normalised = np.nan

            row = {
                "language_a": lang_a,
                "language_b": lang_b,
                "layer": layer,
                "cosine_raw": raw,
                "cosine_normalised": normalised,
                "ceiling": ceiling,
                "floor": floor,
                "angle_1_deg": np.nan,
                "angle_2_deg": np.nan,
                "grassmann": np.nan,
            }

            if sub_a[layer] is not None and sub_b[layer] is not None:
                comparison = compare_subspaces(sub_a[layer], sub_b[layer])
                row["angle_1_deg"] = float(comparison.angles_deg[0])
                row["angle_2_deg"] = float(comparison.angles_deg[1])
                row["grassmann"] = comparison.grassmann

            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print()
    print(f"{'pair':>10} {'peak layer':>11} {'normalised':>11} {'raw':>7}")
    for (a, b), group in df.groupby(["language_a", "language_b"]):
        best = group.loc[group["cosine_normalised"].idxmax()]
        print(f"{a + '-' + b:>10} {int(best['layer']):>11} "
              f"{best['cosine_normalised']:>11.3f} {best['cosine_raw']:>7.3f}")

    above_one = int((df["cosine_normalised"] > 1.0).sum())
    if above_one:
        print(f"\nWARNING: {above_one} values exceed 1.0. Cross-lingual directions "
              "should not agree more than two same-language estimates. Check that "
              "the ceiling and this run used the same sample size and the same "
              "grouped-split settings.")

    print(f"\nwrote {args.out}")
    print("Reminder: cosine_normalised is the reportable number. A raw cosine of "
          "0.3 can be either strong or negligible depending on the ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
