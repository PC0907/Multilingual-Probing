"""Paper RQ-A on the five low-resource languages: shared facts vs cosine vs transfer.

Design (as in the paper, Section 4.1 / Table 2):
  * For every unordered language pair (A, B) and overlap level f in {0, .25, .5, .75, 1}, draw
    N_ALLOC allocations. Each allocation gives A and B training sets of exactly 400 dependency
    groups (train split); a fraction f of the groups is shared, the rest is unique to each side.
    Training size is fixed, only the share of matched fact identities changes.
  * Mass-mean direction per side at the model's frozen block.
  * cosine(d_A, d_B)                         -> unordered pair
  * AUROC of d_A on B's test rows, and of d_B on A's test rows -> ordered pairs (all combinations)
  * Slope = OLS of the metric on f (per unit overlap fraction), per pair; pooled = mean over pairs.
  * Uncertainty: 1,000 hierarchical draws. Each draw resamples the allocations within every
    (pair, level) cell and resamples test dependency groups (shared by all ordered pairs).
    Per-pair two-sided bootstrap p-values, Holm-adjusted within model and metric.

Usage: python analysis_overlap.py <set> [model ...]     set = new5 | orig6
"""
import json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K
from analysis_core import auroc_batch, unit

SETS = {"new5": ["ur", "ne", "gu", "pa", "mr"], "orig6": C.ORIGINAL6}
LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
N_ALLOC, N_BOOT, SIZE = 50, 1000, 400


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def slope(y):
    """y [..., n_levels] -> OLS slope on LEVELS."""
    x = LEVELS - LEVELS.mean()
    return (y * x).sum(-1) / (x ** 2).sum()


def run(setname, model):
    t0 = time.time()
    langs = SETS[setname]
    L = int(json.load(open(C.RESULTS / model / "layer_selection.json"))["selected_block"])
    y = K.LABELS
    tr, te = np.where(K.TRAIN)[0], np.where(K.TEST)[0]
    ytr, yte = y[tr], y[te]
    gtr = K.GROUP_IDX[tr]
    ug, ginv = np.unique(gtr, return_inverse=True); G = len(ug)
    M1 = np.zeros((G, len(tr)), np.float32); M1[ginv, np.arange(len(tr))] = ytr == 1
    M0 = np.zeros((G, len(tr)), np.float32); M0[ginv, np.arange(len(tr))] = ytr == 0
    n1g, n0g = M1.sum(1), M0.sum(1)
    S1, S0, Xte = {}, {}, {}
    for c in langs:
        X = K.layer(model, c, L).astype(np.float32)
        S1[c], S0[c], Xte[c] = M1 @ X[tr], M0 @ X[tr], X[te]

    # allocations: the same random draws for every model (seeded by pair and level)
    pairs = list(combinations(langs, 2))
    nL = len(LEVELS)
    cosv = np.zeros((len(pairs), nL, N_ALLOC))
    scores = {}                                   # (src, tgt) -> [nL, N_ALLOC, n_test]
    for p, (a, b) in enumerate(pairs):
        for li, f in enumerate(LEVELS):
            rng = np.random.default_rng([C.SEED, 77, langs.index(a), langs.index(b), li])
            k = int(round(SIZE * f)); u = SIZE - k
            mA = np.zeros((N_ALLOC, G), np.float32); mB = np.zeros_like(mA)
            for r in range(N_ALLOC):
                perm = rng.permutation(G)
                shared, ua, ub = perm[:k], perm[k:k + u], perm[k + u:k + 2 * u]
                mA[r, np.r_[shared, ua]] = 1; mB[r, np.r_[shared, ub]] = 1
            dA = mA @ S1[a] / (mA @ n1g)[:, None] - mA @ S0[a] / (mA @ n0g)[:, None]
            dB = mB @ S1[b] / (mB @ n1g)[:, None] - mB @ S0[b] / (mB @ n0g)[:, None]
            cosv[p, li] = np.einsum("rd,rd->r", unit(dA), unit(dB))
            scores.setdefault((a, b), np.zeros((nL, N_ALLOC, len(te)), np.float32))[li] = dA @ Xte[b].T
            scores.setdefault((b, a), np.zeros((nL, N_ALLOC, len(te)), np.float32))[li] = dB @ Xte[a].T
    dirs = list(scores)
    SC = np.stack([scores[d] for d in dirs])                     # [D, nL, A, n_test]
    au = auroc_batch(SC, yte)                                    # [D, nL, A]
    print(f"[{setname}/{model}] block {L}: allocations done {time.time()-t0:.0f}s", flush=True)

    # point estimates
    cos_lvl, au_lvl = cosv.mean(-1), au.mean(-1)                 # [P, nL], [D, nL]
    cos_sl, au_sl = slope(cos_lvl), slope(au_lvl)

    # hierarchical bootstrap
    rng = np.random.default_rng(C.SEED + 77)
    pos = {r: i for i, r in enumerate(te)}
    b_cos = np.zeros((N_BOOT, len(pairs))); b_au = np.zeros((N_BOOT, len(dirs)))
    b_cos_lvl = np.zeros((N_BOOT, nL)); b_au_lvl = np.zeros((N_BOOT, nL))
    b_au_unp = np.zeros((N_BOOT, len(dirs)))          # conservative: independent test draw per level
    extra = [[np.array([pos[r] for r in rows]) for rows in K.group_bootstrap_idx(K.TEST, nL, rng)]
             for _ in range(N_BOOT)]
    for bi, rows in enumerate(K.group_bootstrap_idx(K.TEST, N_BOOT, rng)):
        ii = np.array([pos[r] for r in rows])
        pick_c = rng.integers(0, N_ALLOC, (len(pairs), nL, N_ALLOC))
        pick_a = rng.integers(0, N_ALLOC, (len(dirs), nL, N_ALLOC))
        cl = np.take_along_axis(cosv, pick_c, -1).mean(-1)
        a_all = auroc_batch(SC[..., ii], yte[ii])                # [D, nL, A]
        al = np.take_along_axis(a_all, pick_a, -1).mean(-1)
        b_cos[bi], b_au[bi] = slope(cl), slope(al)
        b_cos_lvl[bi], b_au_lvl[bi] = cl.mean(0), al.mean(0)
        au_u = np.stack([auroc_batch(SC[:, li][..., extra[bi][li]], yte[extra[bi][li]]) for li in range(nL)], 1)
        b_au_unp[bi] = slope(np.take_along_axis(au_u, pick_a, -1).mean(-1))
    print(f"[{setname}/{model}] bootstrap done {time.time()-t0:.0f}s", flush=True)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def pval(v):
        return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))

    pc = [pval(b_cos[:, p]) for p in range(len(pairs))]; pa = [pval(b_au[:, d]) for d in range(len(dirs))]
    hc, ha = holm(pc), holm(pa)
    res = dict(model=model, set=setname, block=L, langs=langs, levels=LEVELS.tolist(), n_alloc=N_ALLOC, size=SIZE,
               cos_slope=float(cos_sl.mean()), cos_slope_ci=ci(b_cos.mean(1)),
               auroc_slope=float(au_sl.mean()), auroc_slope_ci=ci(b_au.mean(1)),
               auroc_slope_ci_unpaired=ci(b_au_unp.mean(1)),
               n_auroc_sig_unpaired=int((holm([pval(b_au_unp[:, d]) for d in range(len(dirs))]) < 0.05).sum()),
               cos_by_level=cos_lvl.mean(0).tolist(), cos_by_level_ci=[ci(b_cos_lvl[:, i]) for i in range(nL)],
               auroc_by_level=au_lvl.mean(0).tolist(), auroc_by_level_ci=[ci(b_au_lvl[:, i]) for i in range(nL)],
               n_cos_sig=int((hc < 0.05).sum()), n_cos_pairs=len(pairs),
               n_cos_sig_pos=int(((hc < 0.05) & (cos_sl > 0)).sum()),
               n_auroc_sig=int((ha < 0.05).sum()), n_auroc_pairs=len(dirs),
               pairs={f"{a}-{b}": dict(cos_slope=float(cos_sl[p]), cos_ci=ci(b_cos[:, p]), p_holm=float(hc[p]),
                                        cos_0=float(cos_lvl[p, 0]), cos_100=float(cos_lvl[p, -1]))
                      for p, (a, b) in enumerate(pairs)},
               directed={f"{a}->{b}": dict(auroc_slope=float(au_sl[d]), auroc_ci=ci(b_au[:, d]), p_holm=float(ha[d]),
                                            auroc_0=float(au_lvl[d, 0]), auroc_100=float(au_lvl[d, -1]))
                         for d, (a, b) in enumerate(dirs)})
    out = C.RESULTS / model / f"overlap_{setname}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{setname}/{model}] cos slope {res['cos_slope']:+.4f} {res['cos_slope_ci']}  "
          f"AUROC slope {res['auroc_slope']:+.4f} {res['auroc_slope_ci']} unpaired {res['auroc_slope_ci_unpaired']}  "
          f"sig cos {res['n_cos_sig']}/{len(pairs)} AUROC {res['n_auroc_sig']}/{len(dirs)}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    setname = sys.argv[1]
    for m in (sys.argv[2:] or list(C.MODELS)):
        run(setname, m)
