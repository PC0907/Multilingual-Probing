"""Shared analysis helpers: loading caches, the mass-mean probe, metrics, group bootstrap."""
import json, sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C

_EN = json.load(open(C.DATA / "en.json"))
LABELS = np.array([r["label"] for r in _EN], dtype=int)
PART = np.array([r["partition"] for r in _EN])
GROUPS = np.array([r["group_id"] for r in _EN])
TRAIN, VAL, TEST = PART == "train", PART == "validation", PART == "test"
_, GROUP_IDX = np.unique(GROUPS, return_inverse=True)


@lru_cache(maxsize=64)
def load_acts(model, cond):
    """[N, L, d] float32 (memory-mapped)."""
    return np.load(C.ACTS / model / f"{cond}.npy", mmap_mode="r")


def layer(model, cond, l):
    """Activations after decoder block l (1-indexed), [N, d] float64."""
    return np.asarray(load_acts(model, cond)[:, l - 1, :], dtype=np.float64)


def meta(model, cond):
    return json.load(open(C.ACTS / model / f"{cond}.meta.json"))


def available(model):
    return [c for c in C.CONDITIONS if (C.ACTS / model / f"{c}.npy").exists()]


# ----------------------------------------------------------------- mass-mean probe
def massmean(X, y):
    """direction = mu_true - mu_false; threshold = midpoint of projected class means."""
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    w = mu1 - mu0
    b = 0.5 * (mu1 @ w + mu0 @ w)
    return w, b


def auroc(scores, y):
    """Rank AUROC (Mann-Whitney), ties averaged."""
    from scipy.stats import rankdata
    r = rankdata(scores)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bal_acc(scores, y, thr):
    pred = scores > thr
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def dprime(scores, y):
    s1, s0 = scores[y == 1], scores[y == 0]
    return (s1.mean() - s0.mean()) / np.sqrt(0.5 * (s1.var(ddof=1) + s0.var(ddof=1)))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def group_bootstrap_idx(mask, n_boot, rng):
    """Resample whole dependency groups among rows in `mask`; yields row-index arrays."""
    rows = np.where(mask)[0]
    g = GROUP_IDX[rows]
    ug = np.unique(g)
    by_g = {k: rows[g == k] for k in ug}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        yield np.concatenate([by_g[k] for k in pick])
