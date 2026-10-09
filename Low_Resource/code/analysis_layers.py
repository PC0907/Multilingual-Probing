"""RQ11/RQ12: every decoder block, all 18 conditions, all ordered pairs.

Per block l: auroc[l,s,t] (full-train directions, test rows), cos_zero[l,s,t] and ceil[l,c]
(mean over N_SPLIT random disjoint halves of the train groups), and a 200-draw test-group
bootstrap of AUROC (paired across cells and blocks: the same test draws at every block).
Validation-set AUROC is stored too (val_auroc) so that any layer-picking uses validation only.

Usage: python analysis_layers.py [model ...]
"""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K
from analysis_core import auroc_batch, unit

N_SPLIT, N_BOOT = 40, 200


def run(model):
    t0 = time.time()
    conds = [c for c in C.CONDITIONS if (C.ACTS / model / f"{c}.npy").exists()]
    S = len(conds)
    y = K.LABELS
    tr, va, te = np.where(K.TRAIN)[0], np.where(K.VAL)[0], np.where(K.TEST)[0]
    ytr = y[tr]
    A = {c: np.load(C.ACTS / model / f"{c}.npy") for c in conds}                    # [N, L, d] in RAM
    L = A[conds[0]].shape[1]
    print(f"[{model}] loaded {S} conds x {L} blocks in {time.time()-t0:.0f}s", flush=True)

    gtr = K.GROUP_IDX[tr]
    ug, ginv = np.unique(gtr, return_inverse=True); G = len(ug)
    rng = np.random.default_rng(C.SEED)
    halves = np.stack([rng.permutation(G) < G // 2 for _ in range(N_SPLIT)])      # [N_SPLIT, G]
    rowhalf = halves[:, ginv]                                                        # [N_SPLIT, n_tr]
    rng = np.random.default_rng(C.SEED + 1)
    te_draws = list(K.group_bootstrap_idx(K.TEST, N_BOOT, rng))
    pos = {r: k for k, r in enumerate(te)}
    te_draws = [np.array([pos[r] for r in rows]) for rows in te_draws]

    auroc = np.zeros((L, S, S)); val_auroc = np.zeros((L, S, S))
    cos_zero = np.zeros((L, S, S)); ceil = np.zeros((L, S)); cos_full = np.zeros((L, S, S))
    b_auroc = np.zeros((N_BOOT, L, S, S), np.float32)
    for l in range(L):
        Xtr = np.stack([A[c][tr, l] for c in conds]).astype(np.float64)              # [S, n_tr, d]
        Xva = np.stack([A[c][va, l] for c in conds])
        Xte = np.stack([A[c][te, l] for c in conds])
        W = Xtr[:, ytr == 1].mean(1) - Xtr[:, ytr == 0].mean(1)                     # [S, d]
        Wf = W.astype(np.float32)
        sc = np.einsum("sd,tnd->stn", Wf, Xte)
        auroc[l] = auroc_batch(sc, y[te])
        val_auroc[l] = auroc_batch(np.einsum("sd,tnd->stn", Wf, Xva), y[va])
        Wu = unit(W); cos_full[l] = Wu @ Wu.T
        for k, ii in enumerate(te_draws):
            b_auroc[k, l] = auroc_batch(sc[..., ii], y[te][ii])
        z = np.zeros((S, S)); e = np.zeros(S)
        for h in rowhalf:
            m1a, m0a = h & (ytr == 1), h & (ytr == 0)
            m1b, m0b = ~h & (ytr == 1), ~h & (ytr == 0)
            Da = unit(Xtr[:, m1a].mean(1) - Xtr[:, m0a].mean(1))
            Db = unit(Xtr[:, m1b].mean(1) - Xtr[:, m0b].mean(1))
            x = Da @ Db.T
            z += 0.5 * (x + x.T); e += np.einsum("sd,sd->s", Da, Db)
        cos_zero[l], ceil[l] = z / N_SPLIT, e / N_SPLIT
        if l % 6 == 0:
            print(f"[{model}] block {l+1}/{L}  {time.time()-t0:.0f}s", flush=True)
    out = C.RESULTS / model / "layers.npz"
    np.savez_compressed(out, conds=np.array(conds), auroc=auroc, val_auroc=val_auroc,
                        cos_zero=cos_zero, cos_full=cos_full, ceil=ceil, b_auroc=b_auroc)
    print(f"[{model}] wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(C.MODELS)):
        run(m)
