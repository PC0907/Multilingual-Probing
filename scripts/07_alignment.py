#!/usr/bin/env python
"""Cross-lingual alignment of truth directions, with floor and ceiling.

RQ1, single-vector form. Answers: do the truth directions of two languages point
the same way, and how much of the available agreement do they achieve?

A raw cosine is uninterpretable on its own. In 4096 dimensions random directions
sit near 0 with sd ~1/sqrt(d), so any positive value looks impressive; and two
probes fitted on disjoint halves of the SAME language do not reach 1.0 either.
Every number here is therefore reported against both bounds:

    normalised = (observed - floor) / (ceiling - floor)

MATCHED SAMPLE SIZE
-------------------
A mass-mean direction is an estimate of a difference of population means, and two
such estimates agree better as n grows. The ceiling comes from split halves, so
each of its directions sees n/2 statements. Comparing that against a
cross-lingual cosine computed from the FULL n would understate the ceiling and
push normalised alignment above 1.

So the observed cosine is also resampled at the same per-language n, giving a
distribution rather than a point estimate. The full-n cosine is reported too, but
the normalised figure uses the matched one.

ATTENUATION
-----------
Each language has its own ceiling: German directions may be less stable than
English ones, and a cross-lingual cosine is limited by both. The normalisation
therefore divides by the geometric mean of the two ceilings, which is the
standard correction for attenuation, sqrt(r_xx * r_yy). Both ceilings are
reported separately as well, because a large gap between them is itself a
finding: it means one language's direction is estimated far more reliably than
the other's, and any raw comparison is dominated by that.

FLOOR
-----
Not the isotropic random-vector floor. Directions are fitted on the real
activations with labels permuted within each split, so they inherit the space's
anisotropy and mean offset while carrying no truth signal. Strictly higher, and
more honest, than assuming uniformity on the sphere.

Usage::

    python scripts/07_alignment.py \\
        --cache-a data/processed/de_v2/llama-3.1-8b --index-a data/processed/de_v2/index.csv --name-a de \\
        --cache-b data/processed/en/llama-3.1-8b    --index-b data/processed/en/index.csv    --name-b en \\
        --out results/de_en_alignment --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.baselines import (                                  # noqa: E402
    cosine, split_half_ceiling, normalised_alignment, random_direction_floor,
)
from src.probes.mass_mean import (                                    # noqa: E402
    mass_mean_direction, check_sign_convention,
)


def load(cache: Path, index: Path):
    meta = json.loads((cache / "meta.json").read_text())
    df = pd.read_csv(index)
    if len(df) != meta["n_statements"]:
        raise SystemExit(
            f"{index} has {len(df)} rows but {cache} holds "
            f"{meta['n_statements']}. These are not the same dataset -- row i "
            "of the cache would not be statement i of the index."
        )
    layers = sorted(int(p.stem.split("_")[1]) for p in cache.glob("layer_*.npy"))
    return meta, df, layers


def resampled_cross_cosine(
    Xa, ya, Xb, yb, n_per_split, n_repeats, rng,
) -> list[float]:
    """Cosine between directions fitted on independent subsamples of each side.

    Matched to the ceiling's per-direction sample size so the two are comparable.
    Subsampling is NOT grouped here: within one language the two directions are
    not being compared to each other, so group leakage across languages is not
    possible. Grouping matters for the ceiling, where both halves come from the
    same corpus.
    """
    out = []
    for _ in range(n_repeats):
        ia = rng.choice(len(ya), n_per_split, replace=False)
        ib = rng.choice(len(yb), n_per_split, replace=False)
        if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2:
            continue
        try:
            da = mass_mean_direction(Xa[ia], ya[ia])
            db = mass_mean_direction(Xb[ib], yb[ib])
        except ValueError:
            continue          # degenerate layer, e.g. embeddings of a shared token
        out.append(cosine(da, db))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-a", required=True, type=Path)
    ap.add_argument("--index-a", required=True, type=Path)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--cache-b", required=True, type=Path)
    ap.add_argument("--index-b", required=True, type=Path)
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta_a, df_a, layers_a = load(args.cache_a, args.index_a)
    meta_b, df_b, layers_b = load(args.cache_b, args.index_b)

    # Caches built differently are not comparable. assert_compatible only checks
    # a subset, so the two that silently ruin a cosine are checked by hand.
    for field in ("model_id", "pooling"):
        va, vb = meta_a["config"][field], meta_b["config"][field]
        if va != vb:
            raise SystemExit(
                f"incompatible caches: {field} is {va!r} vs {vb!r}. Directions "
                "from differently-extracted activations are not comparable."
            )
    if meta_a["config"]["model_id"] != meta_b["config"]["model_id"]:
        raise SystemExit("different models; a cosine between them is meaningless")

    shared = sorted(set(layers_a) & set(layers_b))
    layers = [l for l in shared if l % args.stride == 0]
    if not layers:
        raise SystemExit("no layers in common between the two caches")

    ya = df_a["label"].to_numpy()
    yb = df_b["label"].to_numpy()
    ga = df_a["group_id"].to_numpy()
    gb = df_b["group_id"].to_numpy()

    n_per_split = min(len(ya), len(yb)) // 2
    rng = np.random.default_rng(args.seed)

    print(f"{args.name_a}: {len(ya)} statements, {len(np.unique(ga))} groups")
    print(f"{args.name_b}: {len(yb)} statements, {len(np.unique(gb))} groups")
    print(f"model  : {meta_a['config']['model_id']}  "
          f"pooling={meta_a['config']['pooling']}")
    print(f"matched n per direction: {n_per_split}   resamples: {args.repeats}")
    iso = random_direction_floor(meta_a.get("hidden_dim", 4096), seed=args.seed)
    print(f"isotropic reference (not used for normalising): "
          f"{iso.mean:+.4f} +/- {iso.std:.4f}\n")

    rows = []
    hdr = (f"{'layer':>6} {'cos_full':>9} {'cos_match':>10} {'floor':>8} "
           f"{'ceil_'+args.name_a:>9} {'ceil_'+args.name_b:>9} {'normalised':>11}")
    print(hdr)
    print("-" * len(hdr))

    for layer in layers:
        Xa = np.array(np.load(args.cache_a / f"layer_{layer:02d}.npy", mmap_mode="r"),
                      dtype=np.float64)
        Xb = np.array(np.load(args.cache_b / f"layer_{layer:02d}.npy", mmap_mode="r"),
                      dtype=np.float64)

        try:
            da_full = mass_mean_direction(Xa, ya)
            db_full = mass_mean_direction(Xb, yb)
        except ValueError as e:
            print(f"{layer:>6}   skipped: {e}")
            del Xa, Xb
            continue
        # A silent sign flip turns +0.8 into -0.8 and reads as a finding.
        if not check_sign_convention(Xa, ya, da_full):
            raise SystemExit(f"layer {layer}: {args.name_a} direction has wrong sign")
        if not check_sign_convention(Xb, yb, db_full):
            raise SystemExit(f"layer {layer}: {args.name_b} direction has wrong sign")
        cos_full = cosine(da_full, db_full)

        matched = resampled_cross_cosine(Xa, ya, Xb, yb, n_per_split,
                                         args.repeats, rng)
        cos_match = float(np.mean(matched))
        cos_match_sd = float(np.std(matched, ddof=1))

        try:
            ceil_a = split_half_ceiling(
                Xa, ya, mass_mean_direction, groups=ga,
                n_repeats=args.repeats, n_per_split=None, seed=args.seed)
            ceil_b = split_half_ceiling(
                Xb, yb, mass_mean_direction, groups=gb,
                n_repeats=args.repeats, n_per_split=None, seed=args.seed)
            floor = split_half_ceiling(
                Xa, ya, mass_mean_direction, groups=ga,
                n_repeats=args.repeats, n_per_split=None,
                permute_labels=True, seed=args.seed)
        except (ValueError, RuntimeError) as e:
            print(f"{layer:>6}   skipped: {e}")
            del Xa, Xb
            continue

        # Attenuation correction: a cross-lingual cosine is bounded by the
        # reliability of BOTH directions, not just one.
        ceil_geom = float(np.sqrt(max(ceil_a.mean, 0.0) * max(ceil_b.mean, 0.0)))
        try:
            norm = normalised_alignment(cos_match, floor.mean, ceil_geom)
        except ValueError:
            norm = float("nan")

        flag = "" if ceil_geom >= 0.30 else "   (low ceiling)"
        print(f"{layer:>6} {cos_full:>9.3f} {cos_match:>10.3f} {floor.mean:>8.3f} "
              f"{ceil_a.mean:>9.3f} {ceil_b.mean:>9.3f} {norm:>11.3f}{flag}")

        rows.append({
            "layer": layer,
            "depth_fraction": layer / (max(layers_a) if layers_a else 1),
            "cosine_full_n": cos_full,
            "cosine_matched_n": cos_match,
            "cosine_matched_sd": cos_match_sd,
            "floor_mean": floor.mean, "floor_sd": floor.std,
            f"ceiling_{args.name_a}_mean": ceil_a.mean,
            f"ceiling_{args.name_a}_sd": ceil_a.std,
            f"ceiling_{args.name_b}_mean": ceil_b.mean,
            f"ceiling_{args.name_b}_sd": ceil_b.std,
            "ceiling_geometric": ceil_geom,
            "normalised_alignment": norm,
        })
        del Xa, Xb

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    # Peak is chosen by RELIABILITY, not by the ratio. Where the ceiling is near
    # the floor the denominator is tiny and the normalised value explodes: that
    # is a layer with no usable direction, not a layer with strong alignment.
    RELIABLE = 0.30
    reliable = df[df["ceiling_geometric"] >= RELIABLE]
    if reliable.empty:
        print(f"\nNo layer reaches a ceiling of {RELIABLE}. The direction is not "
              "stable across resamples at any depth, so no normalised alignment "
              "here is trustworthy. Reporting the highest-ceiling layer only.")
        best = df.loc[df["ceiling_geometric"].idxmax()]
    else:
        best = reliable.loc[reliable["ceiling_geometric"].idxmax()]
    df["reliable"] = df["ceiling_geometric"] >= RELIABLE
    summary = {
        "pair": [args.name_a, args.name_b],
        "model": meta_a["config"]["model_id"],
        "pooling": meta_a["config"]["pooling"],
        "n_per_direction": int(n_per_split),
        "peak_layer": int(best["layer"]),
        "peak_normalised_alignment": float(best["normalised_alignment"]),
        "peak_cosine_matched": float(best["cosine_matched_n"]),
        "peak_floor": float(best["floor_mean"]),
        "peak_ceiling_geometric": float(best["ceiling_geometric"]),
        "reliability_threshold": RELIABLE,
        "n_reliable_layers": int(df["reliable"].sum()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    print(f"\npeak normalised alignment: {best['normalised_alignment']:.3f} "
          f"at layer {int(best['layer'])}")
    print(f"  raw cosine {best['cosine_matched_n']:.3f}, "
          f"floor {best['floor_mean']:.3f}, "
          f"ceiling {best['ceiling_geometric']:.3f}")

    print(f"  (peak chosen among the {int(df['reliable'].sum())} layers with "
          f"ceiling >= {RELIABLE}; elsewhere the ratio is unstable)")

    low = df[[f"ceiling_{args.name_a}_mean", f"ceiling_{args.name_b}_mean"]].max().min()
    if low < 0.5:
        print(f"\nNOTE: the best ceiling either language reaches is {low:.3f}. "
              "Directions are only moderately stable across resamples of their "
              "own data, so treat the normalised figures as indicative.")

    print(f"\nwrote {args.out.with_suffix('.csv')} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())