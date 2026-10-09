"""Core transfer matrices at each model's frozen block (all 18 conditions, all ordered pairs).

Per (source s, target t):
  auroc[s,t]     AUROC of the s-direction on t test rows (diagonal = within-condition)
  ba[s,t]        balanced accuracy at the frozen s threshold
  dprime[s,t]    d' of the s-direction scores on t test rows
  cos_full[s,t]  cosine of full-train directions (same 1,200 facts in both: overlap-contaminated)
  cos_zero[s,t]  zero-overlap cosine: s and t directions from disjoint halves of the train groups
  ceil[c]        split-half reliability of c's direction (cosine of its two disjoint-half directions)
  cos_dis[s,t]   cos_zero / sqrt(ceil_s * ceil_t)  (disattenuated zero-overlap cosine)

Uncertainty: 1,000 draws. Each draw resamples test dependency groups (scores, paired across all
cells) and, independently, train dependency groups (directions); the zero-overlap halves are a
fresh random partition of the drawn groups. The same seed gives the same draws in every model.

Usage: python analysis_core.py [model ...]
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K

N_BOOT = 1000
N_SPLIT_POINT = 200          # random half-splits averaged for the zero-overlap point estimate


def auroc_batch(scores, y):
    """scores [..., n], y [n] -> AUROC [...] (ties averaged)."""
    r = rankdata(scores, axis=-1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[..., y == 1].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def ba_batch(scores, y, thr):
    """scores [S,T,n], thr [S] -> balanced accuracy [S,T]."""
    pred = scores > thr[:, None, None]
    return 0.5 * (pred[..., y == 1].mean(-1) + (~pred[..., y == 0]).mean(-1))


def dprime_batch(scores, y):
    s1, s0 = scores[..., y == 1], scores[..., y == 0]
    return (s1.mean(-1) - s0.mean(-1)) / np.sqrt(0.5 * (s1.var(-1, ddof=1) + s0.var(-1, ddof=1)))


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def run(model):
    t0 = time.time()
    sel = json.load(open(C.RESULTS / model / "layer_selection.json"))
    L = sel["selected_block"]
    conds = [c for c in C.CONDITIONS if (C.ACTS / model / f"{c}.npy").exists()]
    S = len(conds)
    y = K.LABELS
    tr, te = np.where(K.TRAIN)[0], np.where(K.TEST)[0]
    ytr, yte = y[tr], y[te]

    # train group bookkeeping: per-group class sums -> any resampled/split direction is a matmul
    gtr = K.GROUP_IDX[tr]
    ug, ginv = np.unique(gtr, return_inverse=True)
    G = len(ug)
    M1 = np.zeros((G, len(tr)), np.float32); M1[ginv, np.arange(len(tr))] = (ytr == 1)
    M0 = np.zeros((G, len(tr)), np.float32); M0[ginv, np.arange(len(tr))] = (ytr == 0)
    n1g, n0g = M1.sum(1), M0.sum(1)

    Xte = np.zeros((S, len(te), 0), np.float32)
    S1, S0, W, B = [], [], [], []
    for i, c in enumerate(conds):
        X = K.layer(model, c, L).astype(np.float32)
        Xtr = X[tr]
        S1.append(M1 @ Xtr); S0.append(M0 @ Xtr)
        w, b = K.massmean(X[tr].astype(np.float64), ytr)
        W.append(w.astype(np.float32)); B.append(b)
        if i == 0:
            Xte = np.zeros((S, len(te), X.shape[1]), np.float32)
        Xte[i] = X[te]
    S1, S0 = np.stack(S1), np.stack(S0)              # [S, G, d]
    W, B = np.stack(W), np.array(B)                  # [S, d], [S]
    d = W.shape[1]
    print(f"[{model}] block {L}, {S} conds, d={d}, G_train={G}  loaded {time.time()-t0:.0f}s", flush=True)

    # ---------------- point estimates
    scores = np.einsum("sd,tnd->stn", W, Xte)         # [S, T, n_test]
    auroc = auroc_batch(scores, yte)
    ba = ba_batch(scores, yte, B)
    dpr = dprime_batch(scores, yte)
    Wu = unit(W)
    cos_full = Wu @ Wu.T

    def half_dirs(cA, cB):
        """counts per group for halves A and B ([..., G]) -> unit directions [..., S, d]."""
        out = []
        for cnt in (cA, cB):
            mu1 = np.einsum("bg,sgd->bsd", cnt, S1) / (cnt @ n1g)[:, None, None]
            mu0 = np.einsum("bg,sgd->bsd", cnt, S0) / (cnt @ n0g)[:, None, None]
            out.append(unit(mu1 - mu0))
        return out

    rng = np.random.default_rng(C.SEED)
    cz, ce = [], []
    for chunk in np.array_split(np.arange(N_SPLIT_POINT), 4):
        m = np.stack([rng.permutation(G) < G // 2 for _ in chunk]).astype(np.float32)
        A, Bh = half_dirs(m, 1 - m)
        x = np.einsum("bsd,btd->bst", A, Bh)
        cz.append(0.5 * (x + x.transpose(0, 2, 1))); ce.append(np.einsum("bsd,bsd->bs", A, Bh))
    cz, ce = np.concatenate(cz), np.concatenate(ce)
    cos_zero, ceil = cz.mean(0), ce.mean(0)
    cos_dis = (cz / np.sqrt(np.clip(ce[:, :, None] * ce[:, None, :], 1e-6, None))).mean(0)
    print(f"[{model}] point estimates {time.time()-t0:.0f}s", flush=True)

    # ---------------- bootstrap
    rng = np.random.default_rng(C.SEED + 1)
    te_draws = list(K.group_bootstrap_idx(K.TEST, N_BOOT, rng))
    pos = {r: k for k, r in enumerate(te)}
    b_auroc = np.zeros((N_BOOT, S, S), np.float32); b_ba = np.zeros_like(b_auroc)
    b_dpr = np.zeros_like(b_auroc)
    for k, rows in enumerate(te_draws):
        ii = np.array([pos[r] for r in rows])
        sc, yy = scores[..., ii], yte[ii]
        b_auroc[k] = auroc_batch(sc, yy); b_ba[k] = ba_batch(sc, yy, B); b_dpr[k] = dprime_batch(sc, yy)
    print(f"[{model}] test bootstrap {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(C.SEED + 2)
    b_full = np.zeros((N_BOOT, S, S), np.float32); b_zero = np.zeros_like(b_full)
    b_ceil = np.zeros((N_BOOT, S), np.float32); b_dis = np.zeros_like(b_full)
    for chunk in np.array_split(np.arange(N_BOOT), 20):
        cnt = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in chunk]).astype(np.float32)
        half = np.stack([rng.permutation(G) < G // 2 for _ in chunk]).astype(np.float32)
        mu1 = np.einsum("bg,sgd->bsd", cnt, S1) / (cnt @ n1g)[:, None, None]
        mu0 = np.einsum("bg,sgd->bsd", cnt, S0) / (cnt @ n0g)[:, None, None]
        F = unit(mu1 - mu0)
        b_full[chunk] = np.einsum("bsd,btd->bst", F, F)
        A, Bh = half_dirs(cnt * half, cnt * (1 - half))
        x = np.einsum("bsd,btd->bst", A, Bh)
        z = 0.5 * (x + x.transpose(0, 2, 1)); e = np.einsum("bsd,bsd->bs", A, Bh)
        b_zero[chunk] = z; b_ceil[chunk] = e
        b_dis[chunk] = z / np.sqrt(np.clip(e[:, :, None] * e[:, None, :], 1e-6, None))
    print(f"[{model}] train bootstrap {time.time()-t0:.0f}s", flush=True)

    out = C.RESULTS / model / "core.npz"
    np.savez_compressed(out, conds=np.array(conds), block=L,
                        auroc=auroc, ba=ba, dprime=dpr, cos_full=cos_full, cos_zero=cos_zero,
                        ceil=ceil, cos_dis=cos_dis, thr=B,
                        b_auroc=b_auroc, b_ba=b_ba, b_dprime=b_dpr, b_cos_full=b_full,
                        b_cos_zero=b_zero, b_ceil=b_ceil, b_cos_dis=b_dis)
    print(f"[{model}] wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(C.MODELS)):
        run(m)
