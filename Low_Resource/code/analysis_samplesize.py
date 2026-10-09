"""RQ6: how many target-language labels does a native probe need to match zero-shot transfer?

At the frozen block, for every target condition t and n in N_PER_CLASS, draw n true and n false
train rows of t (at most one row per dependency group), fit a mass-mean direction, and score
t's test rows (REPS draws per n; the same row draws are reused for every model). Compare with
zero-shot AUROC of full-train source directions (from core.npz). The break-even n for source s
is the smallest n whose mean native AUROC reaches the zero-shot AUROC of s->t (log-linear
interpolation between grid points; reported as >600 if never reached).

Also: "source + few target" mixing (pooled: source direction + n-shot target direction, each
unit-normalised then averaged) to ask whether a little target data on top of transfer helps.

Usage: python analysis_samplesize.py [model ...]
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K
from analysis_core import auroc_batch, unit

N_PER_CLASS = [2, 4, 8, 16, 32, 64, 128, 256, 600]
REPS = 100
MIX_SOURCES = ["en", "hi"]


def draws():
    """Group-unique row draws per n (shared across models and conditions)."""
    rng = np.random.default_rng(C.SEED + 6)
    tr = np.where(K.TRAIN)[0]
    y = K.LABELS
    out = {}
    for n in N_PER_CLASS:
        reps = []
        for _ in range(REPS if n < 600 else 1):
            rows = []
            for cls in (1, 0):
                cand = rng.permutation(tr[y[tr] == cls])
                seen, pick = set(), []
                for r in cand:
                    g = K.GROUP_IDX[r]
                    if n < 600 and g in seen:
                        continue
                    seen.add(g); pick.append(r)
                    if len(pick) == n:
                        break
                rows.append(pick)
            reps.append((np.array(rows[0]), np.array(rows[1])))
        out[n] = reps
    return out


def run(model, D):
    t0 = time.time()
    core = np.load(C.RESULTS / model / "core.npz")
    conds = list(core["conds"]); L = int(core["block"])
    te = np.where(K.TEST)[0]; yte = K.LABELS[te]
    X = {c: K.layer(model, c, L).astype(np.float32) for c in conds}
    Wsrc = {}
    tr = np.where(K.TRAIN)[0]
    for s in MIX_SOURCES:
        Wsrc[s] = unit(K.massmean(X[s][tr].astype(np.float64), K.LABELS[tr])[0])
    res = {"n": N_PER_CLASS, "native": {}, "mix": {s: {} for s in MIX_SOURCES}}
    for t in conds:
        Xte = X[t][te]
        nat, mix = [], {s: [] for s in MIX_SOURCES}
        for n in N_PER_CLASS:
            Ws = np.stack([X[t][p].mean(0) - X[t][q].mean(0) for p, q in D[n]])      # [R, d]
            a = auroc_batch(Ws @ Xte.T, yte)
            nat.append([float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))])
            for s in MIX_SOURCES:
                Wm = unit(unit(Ws) + Wsrc[s][None])
                am = auroc_batch(Wm @ Xte.T, yte)
                mix[s].append(float(am.mean()))
        res["native"][t] = nat
        for s in MIX_SOURCES:
            res["mix"][s][t] = mix[s]
    # break-even against every source's zero-shot AUROC
    logn = np.log(N_PER_CLASS)
    be = {}
    for j, t in enumerate(conds):
        curve = np.array([v[0] for v in res["native"][t]])
        be[t] = {}
        for i, s in enumerate(conds):
            if s == t:
                continue
            target = core["auroc"][i, j]
            k = np.where(curve >= target)[0]
            if len(k) == 0:
                be[t][s] = None
            elif k[0] == 0:
                be[t][s] = float(N_PER_CLASS[0])
            else:
                a, b = k[0] - 1, k[0]
                f = (target - curve[a]) / (curve[b] - curve[a])
                be[t][s] = float(np.exp(logn[a] + f * (logn[b] - logn[a])))
    res["break_even"] = be
    json.dump(res, open(C.RESULTS / model / "samplesize.json", "w"), indent=1)
    print(f"[{model}] samplesize done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    D = draws()
    for m in (sys.argv[1:] or list(C.MODELS)):
        run(m, D)
