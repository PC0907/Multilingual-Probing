"""Tables + figure for the RQ-A (fact overlap) replication on the five low-resource languages."""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = {"gemma-7b": ("+0.054 [+0.035, +0.073]", "-0.0007 [-0.0157, +0.0162]"),
         "qwen3-8b-base": ("+0.038 [+0.020, +0.054]", "-0.0004 [-0.0147, +0.0135]"),
         "apertus-8b-2509": ("+0.022 [-0.225, +0.271]", "+0.0013 [-0.0141, +0.0178]"),
         "mistral-7b-v0.3": ("+0.059 [+0.041, +0.077]", "-0.0006 [-0.0168, +0.0156]")}
R = {(s, m): json.load(open(C.RESULTS / m / f"overlap_{s}.json")) for s in ["new5", "orig6"] for m in C.MODELS}
lines = ["# RQ-A on the five low-resource languages: shared facts vs cosine vs transfer", ""]


def ci(v, d=3):
    return f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]"


for s, title in [("new5", "Five low-resource languages (ur, ne, gu, pa, mr): 10 unordered pairs, 20 ordered pairs"),
                 ("orig6", "Original six (calibration against the paper's Table 2): 15 / 30 pairs")]:
    lines += [f"## {title}", "",
              "| model | block | cosine slope [95% CI] | cosine 0% -> 100% | sig. cos pairs (Holm) | AUROC slope [paired CI] | AUROC slope [unpaired CI] | AUROC 0% -> 100% | sig. AUROC pairs (paired / unpaired) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in C.MODELS:
        r = R[(s, m)]
        lines.append(f"| {C.MODEL_LABEL[m]} | {r['block']} | {r['cos_slope']:+.3f} {ci(r['cos_slope_ci'])} | "
                     f"{r['cos_by_level'][0]:.3f} -> {r['cos_by_level'][-1]:.3f} | {r['n_cos_sig_pos']}/{r['n_cos_pairs']} | "
                     f"{r['auroc_slope']:+.4f} {ci(r['auroc_slope_ci'], 4)} | {r['auroc_slope']:+.4f} {ci(r['auroc_slope_ci_unpaired'], 4)} | "
                     f"{r['auroc_by_level'][0]:.3f} -> {r['auroc_by_level'][-1]:.3f} | {r['n_auroc_sig']} / {r['n_auroc_sig_unpaired']} of {r['n_auroc_pairs']} |")
    lines.append("")
lines += ["Paper (original six, Table 2): " + "; ".join(f"{C.MODEL_LABEL[m]} cos {a}, AUROC {b}" for m, (a, b) in PAPER.items()), ""]

lines += ["## Per-pair slopes, five low-resource languages", ""]
for m in C.MODELS:
    r = R[("new5", m)]
    lines += [f"**{C.MODEL_LABEL[m]}** cosine slope per unordered pair (Holm p):", "",
              " ".join(f"{k} {v['cos_slope']:+.3f} (p={v['p_holm']:.3f});" for k, v in r["pairs"].items()), "",
              "AUROC slope per ordered pair:", "",
              " ".join(f"{k} {v['auroc_slope']:+.4f};" for k, v in r["directed"].items()), ""]
out = C.RESULTS / "summary" / "rq_a_overlap.md"
out.write_text("\n".join(lines)); print(out)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.array(R[("new5", list(C.MODELS)[0])]["levels"]) * 100
for k, m in enumerate(C.MODELS):
    for s, ls in [("new5", "-"), ("orig6", ":")]:
        r = R[(s, m)]
        for ax, key in [(axes[0], "cos_by_level"), (axes[1], "auroc_by_level")]:
            v = np.array(r[key]); c = np.array(r[key + "_ci"])
            ax.plot(x, v, ls, marker="o", ms=3.5, color=f"C{k}",
                    label=f"{C.MODEL_LABEL[m]} ({'5 low-resource' if s == 'new5' else 'original six'})")
            if s == "new5":
                ax.fill_between(x, c[:, 0], c[:, 1], color=f"C{k}", alpha=0.12)
axes[0].set_title("Direction cosine"); axes[1].set_title("Held-out transfer AUROC")
for ax in axes:
    ax.set_xlabel("shared training facts (%)"); ax.set_xticks(x)
axes[0].set_ylabel("mean over pairs"); axes[0].legend(fontsize=7, ncol=1)
fig.suptitle("RQ-A dose response: 400 groups per side, only the share of matched facts changes (50 allocations per level)", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGURES / "rq_a_overlap.png", dpi=130); plt.close(fig)
print(C.FIGURES / "rq_a_overlap.png")
