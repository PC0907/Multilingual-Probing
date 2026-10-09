"""Answer the research questions from core.npz / layers.npz / samplesize.json / features.json.

Writes results/summary/<rq>.json (machine-readable, with 95% bootstrap intervals) and
results/summary/tables.md (human-readable), and figures/*.png.

All intervals are percentile intervals of the 1,000 paired group-bootstrap draws in core.npz
(test-group draws for AUROC/BA/d', train-group draws for cosines). A contrast's interval comes
from the same draws for both terms, so it is paired.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = C.RESULTS / "summary"; OUT.mkdir(parents=True, exist_ok=True)
C.FIGURES.mkdir(parents=True, exist_ok=True)
MODELS = list(C.MODELS)
LAB = C.MODEL_LABEL
NEW5 = ["ur", "mr", "ne", "gu", "pa"]
TRANSLIT = [c for c, m in C.CONDITIONS.items() if m["kind"] == "translit"]
MD = []


def md(s=""):
    MD.append(s)


def load(model):
    z = np.load(C.RESULTS / model / "core.npz")
    d = {k: z[k] for k in z.files}
    d["conds"] = [str(c) for c in d["conds"]]
    d["ix"] = {c: i for i, c in enumerate(d["conds"])}
    d["feat"] = json.load(open(C.RESULTS / model / "features.json"))
    # transfer ratio: share of the target's own above-chance AUROC that the source recovers
    diag = np.diag(d["auroc"])
    d["ratio"] = (d["auroc"] - 0.5) / (diag[None, :] - 0.5)
    bd = np.einsum("bii->bi", d["b_auroc"])
    d["b_ratio"] = (d["b_auroc"] - 0.5) / (bd[:, None, :] - 0.5)
    # reliability masks: a ratio needs a target whose own direction is clearly above chance, and a
    # disattenuated cosine needs split-half ceilings far enough from zero to divide by
    bad_t = diag < 0.55
    d["ratio"][:, bad_t] = np.nan; d["b_ratio"][:, :, bad_t] = np.nan
    ce = np.clip(d["ceil"], 0, None)
    # unreliable if the point ceiling or the bootstrap 2.5th-percentile ceiling is below 0.3
    ce = np.minimum(ce, np.percentile(d["b_ceil"], 2.5, axis=0))
    bad_c = (ce[:, None] < 0.3) | (ce[None, :] < 0.3)
    d["cos_dis"] = np.where(bad_c, np.nan, d["cos_dis"]); d["b_cos_dis"] = np.where(bad_c[None], np.nan, d["b_cos_dis"])
    d["unreliable_dir"] = [c for c, v in zip(d["conds"], ce) if v < 0.3]
    return d


D = {m: load(m) for m in MODELS}
DIST = json.load(open(C.RESULTS / "lang_distances.json"))


def est(m, metric, f):
    """point, lo, hi of f(matrix) for one model and metric."""
    d = D[m]
    p = f(d[metric], d["ix"])
    if not np.isfinite(p):
        return [float("nan")] * 3
    b = np.array([f(x, d["ix"]) for x in d["b_" + metric]])
    return [float(p), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


def cell(s, t):
    return lambda M, ix: M[ix[s], ix[t]]


def diff(a, b):
    return lambda M, ix: a(M, ix) - b(M, ix)


def fmt(e, sign=False):
    p, lo, hi = e
    if not np.isfinite(p):
        return "n/a"
    f = "{:+.3f}" if sign else "{:.3f}"
    star = "*" if (sign and (lo > 0 or hi < 0)) else ""
    return (f.format(p) + f" [{lo:+.3f},{hi:+.3f}]" + star) if sign else f"{p:.3f} [{lo:.3f},{hi:.3f}]"


def table(header, rows):
    md("| " + " | ".join(header) + " |")
    md("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        md("| " + " | ".join(str(x) for x in r) + " |")
    md()


def save(name, obj):
    json.dump(obj, open(OUT / f"{name}.json", "w"), indent=1)


# =============================================================== overview
def overview():
    md("## Overview: frozen blocks, within-condition AUROC, and features")
    res = {}
    for m in MODELS:
        d = D[m]
        res[m] = {c: dict(within=est(m, "auroc", cell(c, c)), ceil=float(d["ceil"][d["ix"][c]]),
                          **{k: d["feat"]["cond"][c][k] for k in ["fertility", "tok_per_char", "bpc"]})
                  for c in d["conds"]}
    conds = D[MODELS[0]]["conds"]
    md("Within-condition test AUROC (own direction, 400 test rows; 95% CI). Block: " +
       ", ".join(f"{LAB[m]} {int(D[m]['block'])}" for m in MODELS))
    table(["condition"] + [LAB[m] for m in MODELS],
          [[c] + [fmt(res[m][c]["within"]) for m in MODELS] for c in conds])
    md("Tokenizer fertility (mean statement tokens) / bits per character (lower = more familiar):")
    table(["condition"] + [LAB[m] for m in MODELS],
          [[c] + [f"{res[m][c]['fertility']:.1f} / {res[m][c]['bpc']:.2f}" for m in MODELS] for c in conds])
    save("overview", res)

    # heatmaps
    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    for ax, m in zip(axes, MODELS):
        d = D[m]
        im = ax.imshow(d["auroc"], vmin=0.5, vmax=0.9, cmap="viridis")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=90, fontsize=8)
        ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=8)
        ax.set_title(f"{LAB[m]} (block {int(d['block'])})"); ax.set_xlabel("target"); ax.set_ylabel("source")
        for i in range(len(conds)):
            for j in range(len(conds)):
                ax.text(j, i, f"{d['auroc'][i,j]*100:.0f}", ha="center", va="center", fontsize=5.5,
                        color="w" if d["auroc"][i, j] < 0.7 else "k")
    fig.colorbar(im, ax=axes, shrink=0.6, label="AUROC")
    fig.savefig(C.FIGURES / "auroc_matrices.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# =============================================================== RQ1 / RQ3 / RQ5 / RQ14: script
INTERVENTIONS = [
    # (source, native target, transliterated target, expectation for script-match)
    ("hi", "ur", "ur_Deva", "into source script"),
    ("en", "ur", "ur_Latn", "into source script"),
    ("ar", "ur", "ur_Deva", "out of source script"),
    ("ar", "ur", "ur_Latn", "out of source script"),
    ("ur", "pa", "pa_Arab", "into source script"),
    ("hi", "pa", "pa_Deva", "into source script"),
    ("hi", "gu", "gu_Deva", "into source script"),
    ("hi", "mr", "mr_Gujr", "out of source script"),
    ("en", "hi", "hi_Latn", "into source script"),
    ("ur", "hi", "hi_Latn", "control (neither)"),
    ("en", "ur", "ur_Deva", "control (neither)"),
    ("en", "pa", "pa_Arab", "control (neither)"),
    ("en", "gu", "gu_Deva", "control (neither)"),
]


DID = [  # (match source, control source, native target, transliterated target, relation)
    ("hi", "en", "ur", "ur_Deva", "gains match"),
    ("en", "hi", "ur", "ur_Latn", "gains match"),
    ("ar", "en", "ur", "ur_Deva", "loses match"),
    ("ur", "en", "pa", "pa_Arab", "gains match"),
    ("ar", "en", "pa", "pa_Arab", "gains match"),
    ("hi", "en", "pa", "pa_Deva", "gains match"),
    ("hi", "en", "gu", "gu_Deva", "gains match"),
    ("gu", "en", "mr", "mr_Gujr", "gains match"),
    ("hi", "en", "mr", "mr_Gujr", "loses match"),
    ("en", "ur", "hi", "hi_Latn", "gains match"),
    ("en", "ar", "ur", "ur_Latn", "gains match"),
    ("hi", "ar", "ur", "ur_Deva", "gains match"),
]


def rq1():
    md("## RQ1. Does truth-direction transfer depend on script?")
    md("### (a) Hindi source into related languages with different scripts")
    res = {"hi_to": {}, "interventions": {}}
    rows = []
    for t in ["mr", "ne", "ur", "gu", "pa"]:
        r = [t, C.CONDITIONS[t]["script"]]
        res["hi_to"][t] = {}
        for m in MODELS:
            e = est(m, "auroc", cell("hi", t)); q = est(m, "ratio", cell("hi", t))
            res["hi_to"][t][m] = dict(auroc=e, ratio=q, cos_dis=est(m, "cos_dis", cell("hi", t)))
            r.append(f"{e[0]:.3f} ({q[0]:.2f})")
        rows.append(r)
    md("AUROC of the Hindi direction on each target (transfer ratio = (AUROC-0.5)/(target's own AUROC-0.5) in parentheses):")
    table(["target", "script"] + [LAB[m] for m in MODELS], rows)

    md("Paired contrasts, same source, Devanagari target minus other-script target:")
    rows = []
    for a, b in [("mr", "ur"), ("mr", "gu"), ("ne", "ur"), ("ne", "gu"), ("mr", "pa")]:
        rows.append([f"HI->{a} minus HI->{b}"] + [fmt(est(m, "auroc", diff(cell("hi", a), cell("hi", b))), True) for m in MODELS])
    table(["contrast (AUROC)"] + [LAB[m] for m in MODELS], rows)

    md("### (b) Script interventions: same sentences, target re-written in another script")
    md("Delta = (transliterated target) minus (native target), same source. Positive when the "
       "transliteration moves the target *into* the source's script means script match helps. "
       "Reported for AUROC, transfer ratio and disattenuated zero-overlap cosine; * = 95% CI excludes 0.")
    for metric, name in [("auroc", "Delta AUROC"), ("ratio", "Delta transfer ratio"), ("cos_dis", "Delta disattenuated cosine")]:
        rows = []
        for s, t0, t1, kind in INTERVENTIONS:
            key = f"{s}->{t0} => {t1}"
            res["interventions"].setdefault(key, {"kind": kind})
            r = [key, kind]
            for m in MODELS:
                e = est(m, metric, diff(cell(s, t1), cell(s, t0)))
                res["interventions"][key].setdefault(m, {})[metric] = e
                r.append(fmt(e, True))
            rows.append(r)
        md(f"**{name}**")
        table(["intervention", "script move"] + [LAB[m] for m in MODELS], rows)

    md("### (c) Script-match effect isolated by difference-in-differences")
    md("A transliterated target is harder for *every* source (see the control rows above and the within-condition "
       "table below), so raw deltas mix a target-degradation effect with any script-match effect. The DiD removes the "
       "former: [A(match->translit) - A(match->native)] - [A(control->translit) - A(control->native)]. "
       "Positive = the source whose script the target moved into (or, for 'loses match', away from) gains relative to a "
       "control source whose script relation did not change. Expected sign: + for 'gains match', - for 'loses match'.")
    for metric, name in [("auroc", "DiD AUROC"), ("cos_dis", "DiD disattenuated cosine"), ("cos_zero", "DiD zero-overlap cosine (raw)")]:
        rows = []
        for m_src, c_src, t0, t1, kind in DID:
            key = f"{m_src} vs {c_src}: {t0} => {t1}"
            f = lambda M, ix, a=m_src, b=c_src, u=t0, v=t1: (M[ix[a], ix[v]] - M[ix[a], ix[u]]) - (M[ix[b], ix[v]] - M[ix[b], ix[u]])
            r = [key, kind]
            for m in MODELS:
                e = est(m, metric, f)
                res.setdefault("did", {}).setdefault(key, {"kind": kind}).setdefault(m, {})[metric] = e
                r.append(fmt(e, True))
            rows.append(r)
        md(f"**{name}**")
        table(["match source vs control: target change", "script relation"] + [LAB[m] for m in MODELS], rows)
    md("Pooled over the 'gains match' rows (mean DiD AUROC across the rows, same draws):")
    rows = []
    gains = [x for x in DID if x[4] == "gains match"]
    for m in MODELS:
        f = lambda M, ix: np.mean([(M[ix[a], ix[v]] - M[ix[a], ix[u]]) - (M[ix[b], ix[v]] - M[ix[b], ix[u]]) for a, b, u, v, _ in gains])
        e = est(m, "auroc", f); ec = est(m, "cos_zero", f)
        res.setdefault("did_pooled", {})[m] = dict(auroc=e, cos_zero=ec)
        rows.append([LAB[m], fmt(e, True), fmt(ec, True)])
    table(["model", "mean DiD AUROC (gains match)", "mean DiD zero-overlap cosine"], rows)

    md("Within-condition AUROC change under transliteration (is the transliterated text itself still decodable?):")
    rows = []
    for t1 in TRANSLIT:
        t0 = C.CONDITIONS[t1]["base"]
        rows.append([f"{t1} minus {t0}"] + [fmt(est(m, "auroc", diff(cell(t1, t1), cell(t0, t0))), True) for m in MODELS])
        for m in MODELS:
            res.setdefault("within_delta", {}).setdefault(t1, {})[m] = est(m, "auroc", diff(cell(t1, t1), cell(t0, t0)))
    table(["within-condition"] + [LAB[m] for m in MODELS], rows)
    save("rq1_script", res)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    keys = list(res["did"])
    for k, m in enumerate(MODELS):
        v = np.array([res["did"][key][m]["auroc"] for key in keys] + [res["did_pooled"][m]["auroc"]])
        yy = np.arange(len(keys) + 1) + (k - 1.5) * 0.18
        ax.errorbar(v[:, 0], yy, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, label=LAB[m])
    ax.axvline(0, color="k", lw=0.8); ax.axhline(len(keys) - 0.5, color="grey", lw=0.6, ls=":")
    ax.set_yticks(range(len(keys) + 1))
    ax.set_yticklabels([f"{k}  [{res['did'][k]['kind']}]" for k in keys] + ["POOLED (gains match)"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("script-match difference-in-differences, AUROC")
    ax.set_title("Does moving the target into the source's script help that source?")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(C.FIGURES / "script_match_did.png", dpi=130); plt.close(fig)

    # figure: intervention deltas
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    keys = [f"{s}->{t0} => {t1}" for s, t0, t1, _ in INTERVENTIONS]
    for ax, metric, title in zip(axes, ["auroc", "cos_dis"], ["Delta AUROC", "Delta disattenuated zero-overlap cosine"]):
        for k, m in enumerate(MODELS):
            v = np.array([res["interventions"][key][m][metric] for key in keys])
            yy = np.arange(len(keys)) + (k - 1.5) * 0.18
            ax.errorbar(v[:, 0], yy, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, label=LAB[m])
        ax.axvline(0, color="k", lw=0.8); ax.set_title(title)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{k}  [{INTERVENTIONS[i][3]}]" for i, k in enumerate(keys)], fontsize=8)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "script_interventions.png", dpi=130); plt.close(fig)
    return res


def rq3():
    md("## RQ3. Geometry (cosine) versus function (AUROC) under script change")
    res = {}
    rows = []
    for m in MODELS:
        d = D[m]; S = len(d["conds"])
        off = ~np.eye(S, dtype=bool)
        sp = lambda a, b: spearmanr(a, b, nan_policy="omit").statistic
        r_full = sp(d["cos_full"][off], d["auroc"][off])
        r_dis = sp(d["cos_dis"][off], d["auroc"][off])
        r_ratio = sp(d["cos_dis"][off], d["ratio"][off])
        res[m] = dict(rho_full_auroc=float(r_full), rho_dis_auroc=float(r_dis), rho_dis_ratio=float(r_ratio),
                      mean_cos_full=float(d["cos_full"][off].mean()), mean_cos_zero=float(d["cos_zero"][off].mean()),
                      mean_cos_dis=float(np.nanmean(d["cos_dis"][off])))
        rows.append([LAB[m], f"{r_full:.2f}", f"{r_dis:.2f}", f"{r_ratio:.2f}",
                     f"{res[m]['mean_cos_full']:.3f} / {res[m]['mean_cos_zero']:.3f} / {res[m]['mean_cos_dis']:.3f}"])
    md("Across all 306 ordered pairs of the 18 conditions: Spearman correlation of geometry with function.")
    table(["model", "rho(full cos, AUROC)", "rho(disatt. cos, AUROC)", "rho(disatt. cos, ratio)",
           "mean cos full / zero-overlap / disatt."], rows)

    # dissociation index per intervention: sign agreement of delta-cos and delta-AUROC
    r1 = json.load(open(OUT / "rq1_script.json"))
    rows = []
    for key, v in r1["did"].items():
        r = [key]
        for m in MODELS:
            a, c = v[m]["auroc"], v[m]["cos_zero"]
            sg = lambda e: "n/a" if not np.isfinite(e[0]) else ("+" if e[1] > 0 else ("-" if e[2] < 0 else "0"))
            sa, sc = sg(a), sg(c)
            r.append(f"cos {sc} / AUROC {sa}")
        rows.append(r)
    md("Direction of significant script-match DiD (95% CI) in zero-overlap cosine vs AUROC (rows as in RQ1c):")
    table(["intervention"] + [LAB[m] for m in MODELS], rows)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, m in zip(axes, MODELS):
        d = D[m]; S = len(d["conds"]); off = ~np.eye(S, dtype=bool)
        tl = np.array([[C.CONDITIONS[s]["kind"] == "translit" or C.CONDITIONS[t]["kind"] == "translit"
                        for t in d["conds"]] for s in d["conds"]])
        ax.scatter(d["cos_dis"][off & ~tl], d["auroc"][off & ~tl], s=10, label="native-native")
        ax.scatter(d["cos_dis"][off & tl], d["auroc"][off & tl], s=10, label="involves transliteration")
        ax.set_xlabel("disattenuated zero-overlap cosine"); ax.set_ylabel("cross AUROC"); ax.set_title(LAB[m])
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "cos_vs_auroc.png", dpi=130); plt.close(fig)
    save("rq3_geometry", res)


def rq5_rq14():
    md("## RQ5. Transliteration: fertility change vs geometry change vs transfer change")
    res = {}
    rows = []
    for t1 in TRANSLIT:
        t0 = C.CONDITIONS[t1]["base"]
        res[t1] = {}
        for m in MODELS:
            f = D[m]["feat"]["cond"]
            dfert = f[t1]["fertility"] / f[t0]["fertility"]
            dbpc = f[t1]["bpc"] - f[t0]["bpc"]
            # mean change in incoming transfer from all native non-same-language sources
            srcs = [s for s in C.NATIVE if C.CONDITIONS[s]["lang"] != C.CONDITIONS[t1]["lang"]]
            fa = lambda M, ix: np.nanmean([M[ix[s], ix[t1]] - M[ix[s], ix[t0]] for s in srcs])
            res[t1][m] = dict(fert_ratio=dfert, d_bpc=dbpc, d_in_auroc=est(m, "auroc", fa),
                              d_in_ratio=est(m, "ratio", fa), d_in_cos=est(m, "cos_zero", fa),
                              d_within=est(m, "auroc", diff(cell(t1, t1), cell(t0, t0))))
        rows.append([f"{t1} vs {t0}"] + [
            f"x{res[t1][m]['fert_ratio']:.2f} / {res[t1][m]['d_bpc']:+.2f} / {res[t1][m]['d_in_cos'][0]:+.3f} / {res[t1][m]['d_in_auroc'][0]:+.3f}"
            for m in MODELS])
    md("Per transliteration: fertility ratio / Delta bits-per-char / Delta mean incoming zero-overlap cosine / "
       "Delta mean incoming AUROC (incoming = from every native source of a different language):")
    table(["transliteration"] + [LAB[m] for m in MODELS], rows)
    rows = []
    for t1 in TRANSLIT:
        rows.append([t1] + [fmt(res[t1][m]["d_in_auroc"], True) + "<br>cos " + fmt(res[t1][m]["d_in_cos"], True) for m in MODELS])
    md("Same with 95% CIs:")
    table(["transliteration"] + [LAB[m] for m in MODELS], rows)
    # correlation across 7 transliterations x 4 models
    x = np.array([[res[t][m]["fert_ratio"], res[t][m]["d_bpc"], res[t][m]["d_in_cos"][0], res[t][m]["d_in_auroc"][0],
                   res[t][m]["d_within"][0]] for t in TRANSLIT for m in MODELS])
    cc = {"fert_vs_dAUROC": spearmanr(np.log(x[:, 0]), x[:, 3]).statistic,
          "bpc_vs_dAUROC": spearmanr(x[:, 1], x[:, 3]).statistic,
          "dcos_vs_dAUROC": spearmanr(x[:, 2], x[:, 3]).statistic,
          "fert_vs_dcos": spearmanr(np.log(x[:, 0]), x[:, 2]).statistic,
          "fert_vs_dwithin": spearmanr(np.log(x[:, 0]), x[:, 4]).statistic}
    md("Spearman correlations over the 28 (transliteration x model) cells: " +
       ", ".join(f"{k} = {v:+.2f}" for k, v in cc.items()))
    md()
    res["_corr"] = {k: float(v) for k, v in cc.items()}
    save("rq5_translit", res)

    md("## RQ14. Romanization (Latin-script Hindi and Urdu)")
    out = {}
    rows = []
    latin_src = ["en", "de", "fr", "es"]
    for m in MODELS:
        e1 = est(m, "auroc", lambda M, ix: np.mean([M[ix[s], ix["hi_Latn"]] - M[ix[s], ix["hi"]] for s in latin_src]))
        e2 = est(m, "auroc", lambda M, ix: np.mean([M[ix[s], ix["ur_Latn"]] - M[ix[s], ix["ur"]] for s in latin_src]))
        e3 = est(m, "auroc", diff(cell("hi_Latn", "ur_Latn"), cell("hi", "ur")))
        e4 = est(m, "auroc", diff(cell("hi", "hi_Latn"), cell("hi", "hi")))
        e5 = est(m, "auroc", lambda M, ix: np.mean([M[ix["hi_Latn"], ix[t]] - M[ix["hi"], ix[t]] for t in latin_src]))
        out[m] = dict(latin_src_to_hiLatn_minus_hi=e1, latin_src_to_urLatn_minus_ur=e2,
                      hiLatn_to_urLatn_minus_hi_to_ur=e3, hi_dir_on_hiLatn_minus_hi=e4,
                      hiLatn_dir_to_latin_minus_hi_dir=e5)
        rows.append([LAB[m], fmt(e1, True), fmt(e2, True), fmt(e3, True), fmt(e4, True), fmt(e5, True)])
    table(["model", "EN/DE/FR/ES -> hi_Latn minus -> hi", "EN/DE/FR/ES -> ur_Latn minus -> ur",
           "hi_Latn->ur_Latn minus hi->ur", "HI direction on hi_Latn minus on hi", "hi_Latn direction -> Latin langs minus HI direction"], rows)
    save("rq14_roman", out)


# =============================================================== RQ2: regression
def pair_table(m, conds):
    d = D[m]; f = d["feat"]
    rows = []
    for s in conds:
        for t in conds:
            ls, lt = C.CONDITIONS[s]["lang"], C.CONDITIONS[t]["lang"]
            if ls == lt:
                continue
            rows.append(dict(s=s, t=t, ls=ls, lt=lt,
                             auroc=d["auroc"][d["ix"][s], d["ix"][t]], ratio=d["ratio"][d["ix"][s], d["ix"][t]],
                             cos=d["cos_dis"][d["ix"][s], d["ix"][t]],
                             same_script=float(C.CONDITIONS[s]["script"] == C.CONDITIONS[t]["script"]),
                             gen=DIST["fam"][ls][lt], syn=DIST["syntax_knn"][ls][lt],
                             bpc_s=f["cond"][s]["bpc"], bpc_t=f["cond"][t]["bpc"],
                             jac=f["token_jaccard"][s][t]))
    return rows


PRED = ["same_script", "gen", "syn", "bpc_s", "bpc_t", "jac"]


def ols(rows, y):
    rows = [r for r in rows if np.isfinite(r[y])]
    X = np.array([[r[p] for p in PRED] for r in rows]); Y = np.array([r[y] for r in rows])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Z, (Y - Y.mean()) / Y.std(), rcond=None)
    pred = Z @ beta; r2 = 1 - ((Y - Y.mean()) / Y.std() - pred).var() / 1.0
    return beta[1:], r2


def rq2():
    md("## RQ2. What predicts transfer: script, relatedness, resource, token overlap?")
    md("OLS on ordered pairs of conditions with different underlying languages; predictors standardised "
       "(same-script indicator, lang2vec genetic and syntactic distance, source and target bits-per-character "
       "[resource/familiarity proxy, lower = better known], token-type Jaccard overlap). Outcome standardised, so "
       "coefficients are in SD units. 95% intervals: 1,000-draw cluster bootstrap over underlying languages "
       "(languages resampled with replacement; all pairs among drawn languages kept).")
    rng = np.random.default_rng(C.SEED + 9)
    res = {}
    for scope, conds in [("native11", C.NATIVE), ("all18", list(C.CONDITIONS))]:
        res[scope] = {}
        for yv in ["auroc", "ratio", "cos"]:
            rows_md = []
            res[scope][yv] = {}
            for m in MODELS:
                rows = pair_table(m, conds)
                beta, r2 = ols(rows, yv)
                langs = sorted({r["ls"] for r in rows})
                boots = []
                for _ in range(1000):
                    pick = rng.choice(langs, len(langs), replace=True)
                    cnt = {l: int((pick == l).sum()) for l in langs}
                    rr = [r for r in rows for _ in range(cnt[r["ls"]] * cnt[r["lt"]])]
                    if len({r["same_script"] for r in rr}) < 2:
                        continue
                    try:
                        boots.append(ols(rr, yv)[0])
                    except Exception:
                        pass
                boots = np.array(boots)
                lo, hi = np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)
                res[scope][yv][m] = {p: [float(beta[i]), float(lo[i]), float(hi[i])] for i, p in enumerate(PRED)}
                res[scope][yv][m]["R2"] = float(r2); res[scope][yv][m]["n_pairs"] = len(rows)
                rows_md.append([LAB[m]] + [fmt(res[scope][yv][m][p], True) for p in PRED] + [f"{r2:.2f}", len(rows)])
            md(f"**{scope}, outcome = {yv}**")
            table(["model"] + PRED + ["R2", "n"], rows_md)
    save("rq2_regression", res)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, yv in zip(axes, ["auroc", "ratio", "cos"]):
        for k, m in enumerate(MODELS):
            v = np.array([res["all18"][yv][m][p] for p in PRED])
            yy = np.arange(len(PRED)) + (k - 1.5) * 0.18
            ax.errorbar(v[:, 0], yy, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4, label=LAB[m])
        ax.axvline(0, color="k", lw=0.8); ax.set_title(f"all 18 conditions: {yv}")
        ax.set_yticks(range(len(PRED))); ax.set_yticklabels(PRED); ax.invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rq2_regression.png", dpi=130); plt.close(fig)


# =============================================================== RQ4: fertility
def rq4():
    md("## RQ4. Tokenizer fertility and transfer")
    res = {}
    rows = []
    for m in MODELS:
        d = D[m]; f = d["feat"]["cond"]; conds = d["conds"]
        fert = np.array([f[c]["fertility"] / f["en"]["fertility"] for c in conds])
        bpc = np.array([f[c]["bpc"] for c in conds])
        within = np.diag(d["auroc"])
        src_in = np.array([np.mean([d["auroc"][d["ix"][s], d["ix"][c]] for s in C.ORIGINAL6 if s != c]) for c in conds])
        en_in = d["auroc"][d["ix"]["en"]]
        ratio_in = np.array([np.mean([d["ratio"][d["ix"][s], d["ix"][c]] for s in C.ORIGINAL6 if s != c]) for c in conds])
        def pc(a, b, z):  # partial Spearman controlling z
            from scipy.stats import rankdata
            ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
            ea = ra - np.polyval(np.polyfit(rz, ra, 1), rz); eb = rb - np.polyval(np.polyfit(rz, rb, 1), rz)
            return float(np.corrcoef(ea, eb)[0, 1])
        res[m] = dict(rho_fert_within=float(spearmanr(fert, within).statistic),
                      rho_fert_in=float(spearmanr(fert, src_in).statistic),
                      rho_fert_en=float(spearmanr(fert, en_in).statistic),
                      rho_fert_ratio=float(spearmanr(fert, ratio_in).statistic),
                      rho_bpc_in=float(spearmanr(bpc, src_in).statistic),
                      partial_fert_in_given_bpc=pc(fert, src_in, bpc),
                      partial_bpc_in_given_fert=pc(bpc, src_in, fert),
                      fert={c: float(v) for c, v in zip(conds, fert)})
        rows.append([LAB[m]] + [f"{res[m][k]:+.2f}" for k in ["rho_fert_within", "rho_fert_in", "rho_fert_en",
                                                               "rho_fert_ratio", "rho_bpc_in", "partial_fert_in_given_bpc", "partial_bpc_in_given_fert"]])
    md("Spearman over the 18 conditions. fert = statement tokens relative to English; in = mean AUROC from the six original-language sources; "
       "partial = rank correlation after removing the other variable.")
    table(["model", "fert~within", "fert~in", "fert~EN->t", "fert~in ratio", "bpc~in", "fert~in | bpc", "bpc~in | fert"], rows)
    save("rq4_fertility", res)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in MODELS:
        d = D[m]; f = d["feat"]["cond"]; conds = d["conds"]
        fert = [f[c]["fertility"] / f["en"]["fertility"] for c in conds]
        bpc = [f[c]["bpc"] for c in conds]
        src_in = [np.mean([d["auroc"][d["ix"][s], d["ix"][c]] for s in C.ORIGINAL6 if s != c]) for c in conds]
        axes[0].scatter(fert, src_in, label=LAB[m], s=18); axes[1].scatter(bpc, src_in, label=LAB[m], s=18)
    axes[0].set_xscale("log"); axes[0].set_xlabel("fertility relative to English (log)"); axes[0].set_ylabel("mean incoming AUROC from original six")
    axes[1].set_xlabel("bits per character"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rq4_fertility.png", dpi=130); plt.close(fig)


# =============================================================== RQ6: sample size
def rq6():
    md("## RQ6. How many target labels does a native probe need to match zero-shot transfer?")
    res = {}
    rows = []
    for t in NEW5 + TRANSLIT:
        r = [t]
        for m in MODELS:
            s = json.load(open(C.RESULTS / m / "samplesize.json"))
            be = s["break_even"][t]
            d = D[m]
            cand = {k: v for k, v in be.items() if C.CONDITIONS[k]["lang"] != C.CONDITIONS[t]["lang"]}
            best = max(cand, key=lambda k: d["auroc"][d["ix"][k], d["ix"][t]])
            res.setdefault(t, {})[m] = dict(en=be["en"], hi=be.get("hi"), best=best, best_n=be[best])
            fmt_n = lambda v: ">600" if v is None else f"{v:.0f}"
            r.append(f"EN {fmt_n(be['en'])}, HI {fmt_n(be.get('hi'))}, best={best} {fmt_n(be[best])}")
        rows.append(r)
    md("Break-even labels per class: smallest n (true and false each) whose native mass-mean probe reaches the zero-shot AUROC of the source "
       "(mean of 100 draws, log-interpolated):")
    table(["target"] + [LAB[m] for m in MODELS], rows)

    rows = []
    for t in NEW5:
        r = [t]
        for m in MODELS:
            s = json.load(open(C.RESULTS / m / "samplesize.json"))
            n = s["n"]; nat = s["native"][t]; mix = s["mix"]["en"][t]
            k16 = n.index(16); k64 = n.index(64)
            r.append(f"16: {nat[k16][0]:.3f} vs mix {mix[k16]:.3f}; 64: {nat[k64][0]:.3f} vs {mix[k64]:.3f}")
        rows.append(r)
    md("Native few-shot AUROC vs EN direction + the same few target labels (unit directions averaged):")
    table(["target"] + [LAB[m] for m in MODELS], rows)
    save("rq6_samplesize", res)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for ax, m in zip(axes, MODELS):
        s = json.load(open(C.RESULTS / m / "samplesize.json")); n = s["n"]; d = D[m]
        for k, t in enumerate(NEW5):
            v = np.array(s["native"][t]); col = f"C{k}"
            ax.plot(n, v[:, 0], "-o", color=col, ms=3, label=t)
            ax.fill_between(n, v[:, 1], v[:, 2], color=col, alpha=0.12)
            ax.axhline(d["auroc"][d["ix"]["en"], d["ix"][t]], color=col, ls="--", lw=0.9)
        ax.set_xscale("log"); ax.set_xlabel("labels per class (native probe)"); ax.set_title(f"{LAB[m]} (dashed: EN zero-shot)")
    axes[0].set_ylabel("test AUROC"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rq6_samplesize.png", dpi=130); plt.close(fig)


# =============================================================== RQ7: asymmetry
def rq7():
    md("## RQ7. Asymmetry: is s->t different from t->s?")
    res = {}
    rows = []
    hi_res = C.ORIGINAL6
    for m in MODELS:
        d = D[m]; f = d["feat"]["cond"]
        nat = C.NATIVE
        pairs = [(a, b) for i, a in enumerate(nat) for b in nat[i + 1:]]
        asym = np.array([d["auroc"][d["ix"][a], d["ix"][b]] - d["auroc"][d["ix"][b], d["ix"][a]] for a, b in pairs])
        dbpc = np.array([f[b]["bpc"] - f[a]["bpc"] for a, b in pairs])          # >0: b is less familiar
        dwithin = np.array([d["auroc"][d["ix"][a], d["ix"][a]] - d["auroc"][d["ix"][b], d["ix"][b]] for a, b in pairs])
        hl = est(m, "auroc", lambda M, ix: np.mean([M[ix[a], ix[b]] - M[ix[b], ix[a]] for a in hi_res for b in NEW5]))
        rho_bpc = spearmanr(dbpc, asym).statistic
        rho_w = spearmanr(dwithin, -asym).statistic
        res[m] = dict(high_to_low_minus_low_to_high=hl, rho_asym_dbpc=float(rho_bpc),
                      rho_asym_dwithin=float(spearmanr(dwithin, asym).statistic),
                      mean_abs_asym=float(np.abs(asym).mean()))
        rows.append([LAB[m], fmt(hl, True), f"{np.abs(asym).mean():.3f}", f"{rho_bpc:+.2f}", f"{res[m]['rho_asym_dwithin']:+.2f}"])
    md("High-resource (original six) -> new five minus the reverse direction (mean over 30 pairs); asym(a,b) = AUROC(a->b) - AUROC(b->a) "
       "over the 55 native pairs, correlated with the familiarity gap (bpc_b - bpc_a) and the within-AUROC gap (within_a - within_b):")
    table(["model", "high->low minus low->high", "mean |asym|", "rho(asym, bpc gap)", "rho(asym, within gap)"], rows)
    save("rq7_asymmetry", res)


# =============================================================== RQ8/9: best source, Urdu bridge
def rq8_9():
    md("## RQ8. Which source language transfers best into each new language?")
    res = {"best": {}, "urdu": {}}
    rows = []
    for t in NEW5:
        r = [t]
        for m in MODELS:
            d = D[m]
            srcs = [s for s in C.NATIVE if s != t]
            a = {s: d["auroc"][d["ix"][s], d["ix"][t]] for s in srcs}
            rank = sorted(a, key=a.get, reverse=True)
            res["best"].setdefault(t, {})[m] = dict(ranking=[(s, float(a[s])) for s in rank],
                                                     best_minus_en=est(m, "auroc", diff(cell(rank[0], t), cell("en", t))) if rank[0] != "en" else [0, 0, 0])
            r.append(", ".join(f"{s} {a[s]:.3f}" for s in rank[:3]) + f" (EN rank {rank.index('en')+1})")
        rows.append(r)
    table(["target"] + [LAB[m] for m in MODELS], rows)

    md("Mean rank of each source over the five new targets (1 = best; native sources only, excluding the target itself):")
    rows = []
    for m in MODELS:
        ranks = {s: [] for s in C.NATIVE}
        for t in NEW5:
            rk = [s for s, _ in res["best"][t][m]["ranking"]]
            for s in rk:
                ranks[s].append(rk.index(s) + 1)
        rows.append([LAB[m]] + [f"{s} {np.mean(v):.1f}" for s, v in sorted(ranks.items(), key=lambda kv: np.mean(kv[1]))])
        res.setdefault("mean_rank", {})[m] = {s: float(np.mean(v)) for s, v in ranks.items()}
    table(["model"] + [f"#{i+1}" for i in range(len(C.NATIVE))], rows)

    md("## RQ9. The Urdu bridge: EN, HI or AR as source for Urdu (in each script)")
    rows = []
    for tgt in ["ur", "ur_Deva", "ur_Latn"]:
        for a, b in [("hi", "en"), ("hi", "ar"), ("ar", "en")]:
            r = [f"{a.upper()}->{tgt} minus {b.upper()}->{tgt}"]
            for m in MODELS:
                e = est(m, "auroc", diff(cell(a, tgt), cell(b, tgt)))
                res["urdu"].setdefault(f"{a}-{b}->{tgt}", {})[m] = e
                r.append(fmt(e, True))
            rows.append(r)
    table(["contrast (AUROC)"] + [LAB[m] for m in MODELS], rows)
    rows = []
    for s in ["en", "hi", "ar", "pa"]:
        rows.append([f"{s.upper()} -> ur / ur_Deva / ur_Latn"] + [
            " / ".join(f"{D[m]['auroc'][D[m]['ix'][s], D[m]['ix'][t]]:.3f}" for t in ["ur", "ur_Deva", "ur_Latn"]) for m in MODELS])
    table(["source"] + [LAB[m] for m in MODELS], rows)
    save("rq8_9_sources", res)


# =============================================================== RQ11/12: layers
GAPS = [("hi", "ur", "ur_Deva"), ("hi", "gu", "gu_Deva"), ("ur", "pa", "pa_Arab"),
        ("hi", "pa", "pa_Deva"), ("en", "hi", "hi_Latn"), ("hi", "mr", "mr_Gujr")]


def rq11_12():
    md("## RQ11. Layerwise emergence of truth directions")
    res = {"emergence": {}, "gaps": {}}
    rows = []
    for m in MODELS:
        z = np.load(C.RESULTS / m / "layers.npz"); conds = [str(c) for c in z["conds"]]; ix = {c: i for i, c in enumerate(conds)}
        L = z["auroc"].shape[0]
        va = z["val_auroc"]
        em = {}
        for c in conds:
            w = va[:, ix[c], ix[c]]
            thr = 0.5 + 0.9 * (w.max() - 0.5)
            em[c] = dict(block=int(np.argmax(w >= thr) + 1), depth=float((np.argmax(w >= thr) + 1) / L),
                         peak_block=int(np.argmax(w) + 1), peak_val=float(w.max()))
            # cross: mean over original-six sources (val)
            xs = np.mean([va[:, ix[s], ix[c]] for s in C.ORIGINAL6 if s != c], 0)
            thr2 = 0.5 + 0.9 * (xs.max() - 0.5)
            em[c]["cross_block"] = int(np.argmax(xs >= thr2) + 1); em[c]["cross_peak_block"] = int(np.argmax(xs) + 1)
        res["emergence"][m] = em
        g_hi = np.mean([em[c]["depth"] for c in C.ORIGINAL6]); g_lo = np.mean([em[c]["depth"] for c in NEW5])
        g_tl = np.mean([em[c]["depth"] for c in TRANSLIT])
        x_hi = np.mean([em[c]["cross_block"] / L for c in C.ORIGINAL6]); x_lo = np.mean([em[c]["cross_block"] / L for c in NEW5])
        x_tl = np.mean([em[c]["cross_block"] / L for c in TRANSLIT])
        res["emergence"][m]["_summary"] = dict(within_depth=[g_hi, g_lo, g_tl], cross_depth=[x_hi, x_lo, x_tl], L=L)
        rows.append([LAB[m], L, f"{g_hi:.2f} / {g_lo:.2f} / {g_tl:.2f}", f"{x_hi:.2f} / {x_lo:.2f} / {x_tl:.2f}"])
    md("Relative depth (block/L) at which validation AUROC first reaches 90% of its peak above chance; "
       "means for original six / new five / transliterations. 'within' = own direction; 'cross' = mean of the six original-language sources.")
    table(["model", "L", "within depth", "cross depth"], rows)

    md("## RQ12. Script gap by depth")
    md("Test AUROC(source -> transliterated target in/out of source script) minus AUROC(source -> native target), "
       "at every block; 200-draw paired test bootstrap. Table: value at the frozen block, peak gap and its block.")
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    for ax, m in zip(axes, MODELS):
        z = np.load(C.RESULTS / m / "layers.npz"); conds = [str(c) for c in z["conds"]]; ix = {c: i for i, c in enumerate(conds)}
        L = z["auroc"].shape[0]; xs = np.arange(1, L + 1) / L
        for k, (s, t0, t1) in enumerate(GAPS):
            g = z["auroc"][:, ix[s], ix[t1]] - z["auroc"][:, ix[s], ix[t0]]
            gb = z["b_auroc"][:, :, ix[s], ix[t1]] - z["b_auroc"][:, :, ix[s], ix[t0]]
            lo, hi = np.percentile(gb, 2.5, 0), np.percentile(gb, 97.5, 0)
            gc = (z["cos_zero"][:, ix[s], ix[t1]] / np.sqrt(z["ceil"][:, ix[s]] * z["ceil"][:, ix[t1]])
                  - z["cos_zero"][:, ix[s], ix[t0]] / np.sqrt(z["ceil"][:, ix[s]] * z["ceil"][:, ix[t0]]))
            key = f"{s}->{t1} minus {s}->{t0}"
            fb = int(D[m]["block"]) - 1
            res["gaps"].setdefault(key, {})[m] = dict(auroc=g.tolist(), lo=lo.tolist(), hi=hi.tolist(), cos_dis=gc.tolist(),
                                                      at_frozen=float(g[fb]), peak=float(g[np.argmax(np.abs(g))]),
                                                      peak_block=int(np.argmax(np.abs(g)) + 1), first_half_mean=float(g[:L // 2].mean()),
                                                      second_half_mean=float(g[L // 2:].mean()))
            ax.plot(xs, g, color=f"C{k}", label=key); ax.fill_between(xs, lo, hi, color=f"C{k}", alpha=0.12)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(int(D[m]["block"]) / L, color="grey", ls=":")
        ax.set_title(LAB[m]); ax.set_xlabel("relative depth (block / L)")
    axes[0].set_ylabel("Delta AUROC (translit - native)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rq12_script_gap_depth.png", dpi=130); plt.close(fig)
    for key in res["gaps"]:
        rows.append([key] + [f"{v['at_frozen']:+.3f} (early {v['first_half_mean']:+.3f}, late {v['second_half_mean']:+.3f}; peak {v['peak']:+.3f}@{v['peak_block']})"
                             for v in (res["gaps"][key][m] for m in MODELS)])
    table(["gap"] + [LAB[m] for m in MODELS], rows)
    save("rq11_12_layers", res)

    # emergence figure: within val AUROC by depth for groups
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), sharey=True)
    for ax, m in zip(axes, MODELS):
        z = np.load(C.RESULTS / m / "layers.npz"); conds = [str(c) for c in z["conds"]]; ix = {c: i for i, c in enumerate(conds)}
        L = z["auroc"].shape[0]; xs = np.arange(1, L + 1) / L
        for grp, col, name in [(C.ORIGINAL6, "C0", "original six"), (NEW5, "C1", "new five"), (TRANSLIT, "C2", "transliterated")]:
            w = np.mean([z["val_auroc"][:, ix[c], ix[c]] for c in grp], 0)
            x = np.mean([np.mean([z["val_auroc"][:, ix[s], ix[c]] for s in C.ORIGINAL6 if s != c], 0) for c in grp], 0)
            ax.plot(xs, w, color=col, label=f"{name}: within"); ax.plot(xs, x, color=col, ls="--", label=f"{name}: from original six")
        ax.axvline(int(D[m]["block"]) / L, color="grey", ls=":"); ax.set_title(LAB[m]); ax.set_xlabel("relative depth")
    axes[0].set_ylabel("validation AUROC"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rq11_emergence.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["overview", "rq1", "rq3", "rq5_rq14", "rq2", "rq4", "rq6", "rq7", "rq8_9", "rq11_12"]
    for s in steps:
        globals()[s]()
    (OUT / "tables.md").write_text("\n".join(MD))
    print("wrote", OUT / "tables.md")
