#!/usr/bin/env python
"""In-language probe accuracy over layer depth, with the controls that make it
interpretable.

This is the go/no-go for everything downstream. If a mass-mean probe cannot
separate true from false statements within a single language, no cross-lingual
geometry computed from those directions means anything.

Runs on CPU from the cached memmaps. No GPU, no queue.

WHAT IS MEASURED
----------------
At each layer, a mass-mean direction (Marks & Tegmark) is fitted on a training
fold and evaluated on a held-out fold:

    theta = mean(activations | true) - mean(activations | false)

Classification projects a held-out activation onto theta and thresholds at the
midpoint of the two training-set projections. There is no learned scale and no
regularisation -- that is the point. A direction that classifies well is a
direction that exists in the representation, not one fitted into existence.

FOUR CONTROLS, ALL LOAD-BEARING
-------------------------------
1. GROUPED SPLITS. 1302 of 1996 German statements share a `group_id` with at
   least one other, because they are near-minimal pairs ("Die Elbe fliesst durch
   Hamburg" / "... durch Muenchen"). A random split puts one in train and one in
   test, and the probe scores well by recognising the shared prefix. GroupKFold
   keeps them together. Random-split accuracy is also reported, and the gap
   between the two IS the leakage a naive analysis would have bought.

2. SHUFFLED LABELS. Labels permuted within the training fold only. Must land at
   chance. Anything else means the split leaks or the evaluation is wrong, and
   the real numbers cannot be trusted either.

3. MEAN CENTRING. The residual stream carries a large shared offset, so a raw
   difference of class means is partly a difference in that offset. Marks &
   Tegmark centre before fitting. Both are reported: if they diverge sharply the
   direction is dominated by the offset rather than by truth.

4. PER-TAG ACCURACY. Whether one topic carries the result. A layer profile that
   looks healthy in aggregate but rests on `geography` alone is a different
   finding from one that holds across all four.

Usage::

    python scripts/06_layer_sweep.py \\
        --cache data/processed/de/llama-3.1-8b \\
        --index data/processed/de/index.csv \\
        --out   results/de_layer_sweep \\
        --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mass_mean_direction(X: np.ndarray, y: np.ndarray, centre: bool = True):
    """Fit theta and a decision threshold on training data.

    Returns (theta, threshold, mu). `mu` is the centring vector, which must be
    applied to test data as well -- fitting centred and evaluating uncentred is
    a silent way to get chance performance.
    """
    mu = X.mean(axis=0) if centre else np.zeros(X.shape[1], dtype=X.dtype)
    Xc = X - mu
    theta = Xc[y == 1].mean(axis=0) - Xc[y == 0].mean(axis=0)
    norm = np.linalg.norm(theta)
    if norm < 1e-12:
        raise ValueError("degenerate direction: the two class means coincide")
    theta = theta / norm
    proj = Xc @ theta
    # Midpoint between class means in projection. No fitted scale.
    threshold = 0.5 * (proj[y == 1].mean() + proj[y == 0].mean())
    return theta, threshold, mu


def score(X, y, theta, threshold, mu) -> float:
    proj = (X - mu) @ theta
    return float(((proj > threshold).astype(int) == y).mean())


def grouped_folds(groups: np.ndarray, n_splits: int, rng: np.random.Generator):
    """GroupKFold-equivalent: whole groups assigned to folds, largest first.

    Written out rather than imported so the assignment is visible: greedy
    largest-group-first balancing keeps fold sizes close when group sizes are
    uneven, which they are here (singletons up to groups of 8).
    """
    unique, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    fold_of_group, load = {}, np.zeros(n_splits)
    for gi in order:
        f = int(np.argmin(load))
        fold_of_group[unique[gi]] = f
        load[f] += counts[gi]
    assignment = np.array([fold_of_group[g] for g in groups])
    return [(np.where(assignment != f)[0], np.where(assignment == f)[0])
            for f in range(n_splits)]


def random_folds(n: int, n_splits: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    chunks = np.array_split(idx, n_splits)
    return [(np.setdiff1d(idx, c), c) for c in chunks]


def cv_accuracy(X, y, folds, centre=True, shuffle_labels=False,
                rng: np.random.Generator | None = None) -> float:
    accs = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        if shuffle_labels:
            # Permute the TRAINING labels only. Permuting both would preserve
            # the association and the control would pass vacuously.
            y_train = rng.permutation(y_train)
        try:
            theta, thr, mu = mass_mean_direction(X[train_idx], y_train, centre)
        except ValueError:
            accs.append(0.5)
            continue
        accs.append(score(X[test_idx], y[test_idx], theta, thr, mu))
    return float(np.mean(accs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=4,
                    help="report every Nth layer; the cache holds all of them")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    index = pd.read_csv(args.index)
    meta = json.loads((args.cache / "meta.json").read_text())
    n_rows = meta["n_statements"]
    if len(index) != n_rows:
        raise SystemExit(
            f"index has {len(index)} rows but the cache holds {n_rows}. These "
            "are not the same dataset; row i of the cache would not be "
            "statement i of the index."
        )

    y = index["label"].to_numpy()
    groups = index["group_id"].to_numpy()
    tags = index["tag"].to_numpy() if "tag" in index.columns else None

    available = sorted(int(p.stem.split("_")[1])
                       for p in args.cache.glob("layer_*.npy"))
    layers = [l for l in available if l % args.stride == 0]

    print(f"cache   : {args.cache}")
    print(f"model   : {meta['config']['model_id']}  "
          f"pooling={meta['config']['pooling']}  "
          f"dtype={meta['config'].get('inference_dtype', 'unrecorded')}")
    print(f"n       : {n_rows}  ({int((y == 1).sum())} true / "
          f"{int((y == 0).sum())} false)")
    print(f"groups  : {len(np.unique(groups))}")
    print(f"layers  : {len(available)} cached, reporting {len(layers)} "
          f"at stride {args.stride}\n")

    g_folds = grouped_folds(groups, args.folds, rng)
    r_folds = random_folds(n_rows, args.folds, rng)

    rows = []
    header = f"{'layer':>6} {'grouped':>9} {'random':>8} {'uncentred':>10} {'shuffled':>9}"
    if tags is not None:
        header += "   " + " ".join(f"{t[:8]:>9}" for t in sorted(set(tags)))
    print(header)
    print("-" * len(header))

    for layer in layers:
        X = np.array(np.load(args.cache / f"layer_{layer:02d}.npy", mmap_mode="r"))

        grouped = cv_accuracy(X, y, g_folds, centre=True)
        random_ = cv_accuracy(X, y, r_folds, centre=True)
        uncentred = cv_accuracy(X, y, g_folds, centre=False)
        shuffled = cv_accuracy(X, y, g_folds, centre=True,
                               shuffle_labels=True, rng=rng)

        row = {"layer": layer, "grouped": grouped, "random": random_,
               "uncentred": uncentred, "shuffled": shuffled,
               "depth_fraction": layer / (len(available) - 1)}

        line = f"{layer:>6} {grouped:>9.3f} {random_:>8.3f} {uncentred:>10.3f} {shuffled:>9.3f}"
        if tags is not None:
            for t in sorted(set(tags)):
                m = tags == t
                sub = [(np.intersect1d(tr, np.where(m)[0]),
                        np.intersect1d(te, np.where(m)[0])) for tr, te in g_folds]
                acc = cv_accuracy(X, y, sub, centre=True)
                row[f"tag_{t}"] = acc
                line += f" {acc:>9.3f}"
        print(line)
        rows.append(row)
        del X

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    best = df.loc[df["grouped"].idxmax()]
    summary = {
        "cache": str(args.cache),
        "model": meta["config"]["model_id"],
        "pooling": meta["config"]["pooling"],
        "n_statements": int(n_rows),
        "n_groups": int(len(np.unique(groups))),
        "peak_layer": int(best["layer"]),
        "peak_depth_fraction": float(best["depth_fraction"]),
        "peak_grouped_accuracy": float(best["grouped"]),
        "peak_random_accuracy": float(best["random"]),
        "leakage_gap": float(best["random"] - best["grouped"]),
        "mean_shuffled_accuracy": float(df["shuffled"].mean()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    print(f"\npeak: layer {int(best['layer'])} "
          f"({best['depth_fraction']:.0%} depth), grouped {best['grouped']:.3f}")
    print(f"leakage gap (random - grouped): {summary['leakage_gap']:+.3f}")
    print(f"shuffled-label mean: {summary['mean_shuffled_accuracy']:.3f}")

    if summary["mean_shuffled_accuracy"] > 0.56:
        print("\nWARNING: the shuffled-label control is above chance. Something "
              "leaks; the real numbers are not trustworthy until this is fixed.")
    if best["grouped"] < 0.65:
        print("\nWARNING: peak accuracy is low. Before concluding anything about "
              "the model, check the dataset audit -- a probe cannot beat the "
              "quality of its labels.")

    print(f"\nwrote {args.out.with_suffix('.csv')} and "
          f"{args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())