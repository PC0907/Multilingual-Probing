#!/usr/bin/env python3
"""Create figures for the post-core decisive extensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CORE = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]
ALL = CORE + ["mistral-7b-v0.3"]
X7_MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base",
             "qwen3-8b", "olmo-2-1124-7b", "gemma-7b", "apertus-8b-2509", "mistral-7b-v0.3"]
QWEN_SCALE = [("qwen3-0.6b-base", 0.6), ("qwen3-1.7b-base", 1.7), ("qwen3-4b-base", 4.0),
              ("qwen3-8b-base", 8.0), ("qwen3-14b-base", 14.0)]
DISPLAY = {
    "gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B",
    "apertus-8b-2509": "Apertus-8B", "mistral-7b-v0.3": "Mistral-7B",
    "qwen3-0.6b-base": "Qwen3-0.6B", "qwen3-1.7b-base": "Qwen3-1.7B", "qwen3-4b-base": "Qwen3-4B",
    "qwen3-14b-base": "Qwen3-14B", "qwen3-8b": "Qwen3-8B (post-trained)", "olmo-2-1124-7b": "OLMo-2-7B",
}
COLORS = {"raw_source_mass_mean": "#2457C5", "centroid_shift": "#0B8585",
          "rosh_fixed_layer": "#E07A3F", "ridge_unconstrained": "#8E5AB5",
          "lsi_raw_pca1": "#768395", "lsi_latent": "#D24A75"}
LANGUAGES = ["en", "de", "ar", "hi", "fr", "es"]
MODEL_COLORS = {"gemma-7b": "#2457C5", "qwen3-8b-base": "#0B8585",
                "apertus-8b-2509": "#E07A3F", "mistral-7b-v0.3": "#8E5AB5"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def available(root: Path, filename: str) -> list[str]:
    return [model for model in ALL if (root / model / "extensions" / filename).exists()]


def alignment(root: Path, output: Path) -> None:
    models = available(root, "alignment_baselines.json")
    methods = ["raw_source_mass_mean", "centroid_shift", "rosh_fixed_layer",
               "ridge_unconstrained", "lsi_raw_pca1"]
    labels = ["Raw", "Centroid", "RoSh\nfixed layer", "Ridge", "PCA-1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.35))
    width = .8 / max(1, len(models)); x = np.arange(len(methods))
    for index, model in enumerate(models):
        data = read(root / model / "extensions" / "alignment_baselines.json")
        summary = data["summary_off_diagonal_means"]
        offset = (index - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, [summary[m]["auroc"] for m in methods], width,
                    label=DISPLAY[model], color=MODEL_COLORS[model])
        axes[1].bar(x + offset, [summary[m]["balanced_accuracy"] for m in methods], width,
                    color=MODEL_COLORS[model])
    for ax, title, ylabel in [(axes[0], "Ranking", "Mean cross-language AUROC"),
                              (axes[1], "Threshold transport", "Mean balanced accuracy")]:
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(x, labels)
        ax.axhline(.5, color="#667085", ls="--", lw=1); ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(models), loc="lower center")
    fig.suptitle("Post-core X1: label-blind alignment baselines on the frozen test split",
                 fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, output)


def latent(root: Path, output: Path) -> None:
    models = [m for m in ALL if (root / m / "extensions" / "lsi_latent" / "results.json").exists()]
    raw, ae, raw_cos, ae_cos = [], [], [], []
    for model in models:
        base = read(root / model / "extensions" / "alignment_baselines.json")
        latent_data = read(root / model / "extensions" / "lsi_latent" / "results.json")
        raw.append(base["summary_off_diagonal_means"]["raw_source_mass_mean"]["auroc"])
        core_cells = [row for row in read(root / model / "rq_c.json")["cells"]
                      if row["source"] != row["target"]]
        raw_cos.append(float(np.mean([row["direction_cosine"] for row in core_cells])))
        ae.append(latent_data["off_diagonal_means"]["auroc"])
        ae_cos.append(latent_data["off_diagonal_means"]["direction_cosine"])
    x = np.arange(len(models)); width = .34
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    axes[0].bar(x-width/2, raw, width, label="Raw mass mean", color="#2457C5")
    axes[0].bar(x+width/2, ae, width, label="Shared-AE latent", color="#D24A75")
    axes[1].bar(x-width/2, raw_cos, width, color="#2457C5")
    axes[1].bar(x+width/2, ae_cos, width, color="#D24A75")
    for ax, title, ylabel in [(axes[0], "Transfer", "Off-diagonal AUROC"),
                              (axes[1], "Geometry", "Off-diagonal direction cosine")]:
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(x, [DISPLAY[m] for m in models])
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="lower center")
    fig.suptitle("LSI-style shared latent space: geometry and transfer need not move together",
                 fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, .08, 1, 1)); save(fig, output)


def mmmlu(root: Path, output: Path) -> None:
    models = [m for m in ALL if (root / m / "extensions" / "mmmlu_transfer.json").exists()
              and (root / m / "extensions" / "include_transfer.json").exists()]
    x = np.arange(len(models)); width = .2
    series = {("mmmlu_transfer.json", "paired_candidate_accuracy"): ("MMMLU (translated) paired acc.", "#2457C5"),
              ("include_transfer.json", "paired_candidate_accuracy"): ("INCLUDE (native) paired acc.", "#0B8585"),
              ("mmmlu_transfer.json", "micro_auroc"): ("MMMLU micro AUROC", "#8FB0E8"),
              ("include_transfer.json", "micro_auroc"): ("INCLUDE micro AUROC", "#7CC7C0")}
    fig, ax = plt.subplots(figsize=(8.8, 3.3))
    for k, ((name, key), (label, color)) in enumerate(series.items()):
        values = [read(root / m / "extensions" / name)["off_diagonal_means"][key] for m in models]
        ax.bar(x + (k - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(.5, color="#667085", ls="--", lw=1)
    ax.set_xticks(x, [DISPLAY[m] for m in models]); ax.set_ylim(.4, .85)
    ax.set_ylabel("Cross-language zero-shot score"); ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("Generic-claim probes on external questions: translated MMMLU vs native INCLUDE")
    fig.tight_layout(); save(fig, output)


def damage(root: Path, output: Path) -> None:
    models = [m for m in ALL if (root / m / "extensions" / "damage_budgeted_steering.json").exists()]
    x = np.arange(len(models)); width = .34
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for k, (direction, label, color) in enumerate([("zero_overlap_source", "Zero-overlap source", "#2457C5"),
                                                    ("target_native", "Target-native", "#E07A3F")]):
        values, labels = [], []
        for m in models:
            agg = read(root / m / "extensions" / "damage_budgeted_steering.json")["aggregate"][direction]
            values.append(agg["mean_symmetric_slope"] or 0.0)
            labels.append(f'{agg["pairs_with_safe_nonzero_alpha"]}/{agg["pairs"]}')
        bars = ax.bar(x + (k - .5) * width, values, width, label=label, color=color)
        for bar, text in zip(bars, labels):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + (.015 if y >= 0 else -.05), text,
                    ha="center", fontsize=7, color="#17243A")
    ax.axhline(0, color="#667085", lw=1)
    low, high = ax.get_ylim(); ax.set_ylim(low - .08, high + .05)
    ax.set_xticks(x, [DISPLAY[m] for m in models]); ax.grid(axis="y", alpha=.2)
    ax.set_ylabel("Safe symmetric margin slope"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("X2: steering at the largest neutral-safe magnitude (labels: pairs with a safe alpha)")
    fig.tight_layout(); save(fig, output)


def fourth_family(root: Path, output: Path) -> None:
    models = [m for m in ALL if (root / m / "rq_a" / "summary.json").exists()]
    zero_cos, full_cos, cross_auc = [], [], []
    for model in models:
        a = read(root / model / "rq_a" / "summary.json")["cells"]
        c = [row for row in read(root / model / "rq_c.json")["cells"] if row["source"] != row["target"]]
        zero_cos.append(np.mean([row["metrics"]["raw_cosine"]["mean"] for row in a
                                 if row["overlap_fraction"] == 0]))
        full_cos.append(np.mean([row["metrics"]["raw_cosine"]["mean"] for row in a
                                 if row["overlap_fraction"] == 1]))
        cross_auc.append(np.mean([row["auroc"] for row in c]))
    x = np.arange(len(models)); width=.25
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.bar(x-width, zero_cos, width, label="Zero-overlap cosine", color="#2457C5")
    ax.bar(x, full_cos, width, label="Full-overlap cosine", color="#E07A3F")
    ax.bar(x+width, cross_auc, width, label="Cross-language AUROC", color="#0B8585")
    ax.set_xticks(x, [DISPLAY[m] for m in models]); ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title("Core replication across model families")
    fig.tight_layout(); save(fig, output)


def transport_mosaic(root: Path, output: Path) -> None:
    models = [m for m in ALL if (root / m / "rq_c.json").exists()]
    fig, axes = plt.subplots(2, len(models), figsize=(10.4, 5.9), squeeze=False)
    for column, model in enumerate(models):
        cells = read(root / model / "rq_c.json")["cells"]
        lookup = {(row["source"], row["target"]): row for row in cells}
        for row_index, (metric, title, limits) in enumerate([
            ("auroc", "AUROC", (.45, .95)),
            ("balanced_accuracy", "Balanced accuracy", (.45, .85)),
        ]):
            values = np.asarray([[lookup[(s, t)][metric] for t in LANGUAGES]
                                 for s in LANGUAGES])
            ax = axes[row_index, column]
            image = ax.imshow(values, cmap="viridis", vmin=limits[0], vmax=limits[1])
            ax.set_xticks(range(6), [x.upper() for x in LANGUAGES], fontsize=6.3)
            ax.set_yticks(range(6), [x.upper() for x in LANGUAGES], fontsize=6.3)
            for i in range(6):
                for j in range(6):
                    shade = image.norm(values[i, j])
                    r, g, b_, _ = image.cmap(shade)
                    dark = (.2126 * r + .7152 * g + .0722 * b_) < .5
                    ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.6, color="white" if dark else "#17243A")
            if row_index == 0:
                ax.set_title(DISPLAY[model], fontsize=9, fontweight="bold")
            if column == 0:
                ax.set_ylabel(title + "\nSource", fontsize=7)
            ax.set_xlabel("Target", fontsize=7)
    fig.suptitle("Complete source-to-target truth-score transport",
                 fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, 0, 1, .95)); save(fig, output)


def trajectory(root: Path, output: Path) -> None:
    data = read(root / "rq_f_pythia_trajectory.json")["checkpoints"]
    labels = [f'{row["step"] // 1000}k' if row["step"] else "0" for row in data]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25))
    axes[0].plot(x, [row["zero_overlap_cosine"] for row in data], marker="o", color="#2457C5", label="Zero overlap")
    axes[0].plot(x, [row["full_overlap_cosine"] for row in data], marker="o", color="#E07A3F", label="Full overlap")
    axes[1].plot(x, [row["zero_overlap_auroc"] for row in data], marker="o", color="#2457C5", label="Zero overlap")
    axes[1].plot(x, [row["full_overlap_auroc"] for row in data], marker="s", ms=4, ls="--", color="#E07A3F",
                 label="Full overlap (coincides)")
    for ax, title, ylabel in [(axes[0], "Direction geometry", "Mean cosine"),
                              (axes[1], "Held-out transfer", "Mean AUROC")]:
        ax.set_xticks(x, labels); ax.set_xlabel("Pythia training step (checkpoint tag)")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Exploratory RQ-F: one deduplicated Pythia-1.4B training trajectory",
                 fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(); save(fig, output)


def x7_predictors(root: Path, output: Path) -> None:
    models = [m for m in X7_MODELS if (root / m / "extensions" / "transfer_predictors.json").exists()]
    if not models:
        return
    keys = [("p1_euclidean_cosine", "P1 Euclidean cosine", "#2457C5"),
            ("p2_translation_consistency", "P2 translation consistency (label-free)", "#0B8585"),
            ("p3_fisher_efficiency", "P3 Fisher efficiency", "#E07A3F")]
    data = {m: read(root / m / "extensions" / "transfer_predictors.json") for m in models}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(models)); width = .27
    for k, (key, label, color) in enumerate(keys):
        axes[0].bar(x + (k - 1) * width, [data[m]["spearman_with_auroc_all_blocks"][key] for m in models],
                    width, label=label, color=color)
    axes[0].set_xticks(x, [DISPLAY[m] for m in models], rotation=35, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Spearman ρ with held-out AUROC"); axes[0].axhline(0, color="#667085", lw=.8)
    axes[0].set_title("All blocks × 30 pairs, per model"); axes[0].grid(axis="y", alpha=.2)
    pooled = [c for m in models for c in data[m]["configurations"]]
    for key, label, color in (keys[0], keys[2]):
        axes[1].scatter([c[key] for c in pooled], [c["auroc"] for c in pooled], s=3, alpha=.25,
                        color=color, label=label.split(" (")[0], rasterized=True)
    axes[1].set_xlabel("Predictor value"); axes[1].set_ylabel("Held-out cross-language AUROC")
    axes[1].set_title(f"Pooled configurations (n={len(pooled):,})"); axes[1].grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=7.5, markerscale=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle("X7: which statistic tracks cross-lingual truth transfer?", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, output)


def x8_scale(root: Path, output: Path) -> None:
    sizes = [(m, b) for m, b in QWEN_SCALE if (root / m / "rq_c.json").exists() and (root / m / "rq_a" / "inference.json").exists()]
    if len(sizes) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    xs = [b for _, b in sizes]
    def series(fn):
        return [fn(m) for m, _ in sizes]
    cross = series(lambda m: float(np.mean([c["auroc"] for c in read(root / m / "rq_c.json")["cells"] if c["source"] != c["target"]])))
    cosine = series(lambda m: float(np.mean([c["direction_cosine"] for c in read(root / m / "rq_c.json")["cells"] if c["source"] != c["target"]])))
    slope = series(lambda m: read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["cosine_slope"]["mean"])
    aslope = series(lambda m: read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["auroc_slope"]["mean"])
    axes[0].plot(xs, cross, marker="o", color="#0B8585", label="Cross AUROC")
    axes[0].plot(xs, cosine, marker="o", color="#2457C5", label="Direction cosine")
    axes[0].set_title("Transfer and geometry")
    axes[1].plot(xs, slope, marker="o", color="#2457C5", label="Cosine slope")
    axes[1].plot(xs, aslope, marker="o", color="#E07A3F", label="AUROC slope")
    axes[1].axhline(0, color="#667085", lw=.8); axes[1].set_title("RQ-A overlap slopes")
    ext = [(m, b) for m, b in sizes if (root / m / "extensions" / "include_transfer.json").exists()]
    if ext:
        axes[2].plot([b for _, b in ext], [read(root / m / "extensions" / "mmmlu_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"] for m, _ in ext],
                     marker="o", color="#2457C5", label="MMMLU paired acc.")
        axes[2].plot([b for _, b in ext], [read(root / m / "extensions" / "include_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"] for m, _ in ext],
                     marker="o", color="#0B8585", label="INCLUDE paired acc.")
        axes[2].axhline(.5, color="#667085", lw=.8, ls="--")
    axes[2].set_title("External transfer")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(xs, [f"{b:g}B" for b in xs]); ax.minorticks_off()
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("X8: Qwen3 base models from 0.6B to the largest feasible size", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(); save(fig, output)


def x10_x13(root: Path, output: Path) -> None:
    models = [m for m in X7_MODELS if (root / m / "extensions" / "x13_probes.json").exists()]
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for m in models:
        pairs = read(root / m / "extensions" / "x10_x12.json")["x10"]["pairs"]
        apertus = m == "apertus-8b-2509"
        axes[0].scatter([q["predicted_slope"] for q in pairs], [q["observed_slope"] for q in pairs], s=12,
                        color="#E07A3F" if apertus else "#2457C5", alpha=.9 if apertus else .45,
                        label="Apertus-8B" if apertus else ("Other nine models" if m == models[0] else None))
    lim = axes[0].get_xlim()
    axes[0].plot(lim, lim, color="#667085", lw=.8, ls="--", label="y = x")
    axes[0].set_xlabel("Predicted cosine slope (pool statistics only)"); axes[0].set_ylabel("Observed RQ-A cosine slope")
    axes[0].set_title("X10: estimator-noise prediction (15 pairs × model)"); axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7.5)
    fams = [("mm", "Mass mean", "#2457C5"), ("lda", "Shrinkage LDA", "#0B8585"), ("lr", "Logistic regression", "#E07A3F")]
    data = {m: read(root / m / "extensions" / "x13_probes.json")["families"] for m in models}
    x = np.arange(len(models)); width = .27
    for k, (key, label, color) in enumerate(fams):
        mean = [data[m][key]["overlap_slopes"]["cosine"]["mean"] for m in models]
        lo = [mean[i] - data[m][key]["overlap_slopes"]["cosine"]["ci95"][0] for i, m in enumerate(models)]
        hi = [data[m][key]["overlap_slopes"]["cosine"]["ci95"][1] - mean[i] for i, m in enumerate(models)]
        axes[1].bar(x + (k - 1) * width, mean, width, yerr=[lo, hi], color=color, label=f"{label}: cosine slope",
                    error_kw={"lw": .6, "capsize": 1.5})
        axes[1].plot(x + (k - 1) * width, [data[m][key]["overlap_slopes"]["auroc"]["mean"] for m in models], "k_", ms=6)
    axes[1].plot([], [], "k_", ms=6, label="AUROC slope (all families)")
    axes[1].axhline(0, color="#667085", lw=.8)
    axes[1].set_xticks(x, [DISPLAY[m] for m in models], rotation=35, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Slope per unit overlap"); axes[1].set_title("X13: overlap artifact by probe family"); axes[1].grid(axis="y", alpha=.2)
    axes[1].set_ylim(-.12, .45); axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("X10 and X13: the overlap artifact is predicted, and grows with probe flexibility", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(); save(fig, output)


def _rs():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from study_truth_transport_v2.reporting import replication_stats
    return replication_stats


def x14(output: Path) -> None:
    """Within-model Spearman rho of each X14 predictor with held-out AUROC, v3 (primary) and v5 (replication)."""
    rs = _rs()
    marks = [("p1_euclidean_cosine", "Cosine", "#768395", "o"), ("p3_fisher_efficiency", "Fisher efficiency", "#2457C5", "s"),
             ("p4a_mcs_total_vs_target_probe", "Mahalanobis cos. vs target probe", "#E07A3F", "v"),
             ("p4b_mcs_total_vs_target_optimal", "Mahalanobis cos. vs target optimum", "#0B8585", "D"),
             ("p5_prop1_predicted_auroc", "Prop. 1 prediction (no fit)", "#D24A75", "*")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
    for ax, (label, root) in zip(axes, (("v3 (primary claims)", rs.V3), ("v5 (replication dataset)", rs.V5))):
        within = rs.x14(root)["within"]
        models = [m for m in rs.MODELS if m in within]
        x = np.arange(len(models))
        for k, (key, name, color, marker) in enumerate(marks):
            ax.scatter(x + (k - 2) * .12, [within[m][key] for m in models], color=color, marker=marker, s=26 if marker != "*" else 60,
                       label=name, zorder=3)
        ax.axhline(0, color="#667085", lw=.8)
        ax.set_xticks(x, [DISPLAY[m] for m in models], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(label); ax.grid(axis="y", alpha=.2); ax.set_ylim(-.25, 1.05)
    axes[0].set_ylabel("Spearman ρ with held-out AUROC\n(all blocks × 30 pairs)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("X14: which geometry statistic tracks transfer", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(); save(fig, output)


def v5_replication(output: Path) -> None:
    """Cross-language AUROC and the overlap cosine slope: v3 versus v5 (full and without the name-overlap shortcut)."""
    rs = _rs()
    rows = rs.replication()["rows"]
    models = rs.MODELS
    x = np.arange(len(models)); width = .27
    variants = [("v3", "v3 (Opus translations)", "#2457C5"), ("v5", "v5, all families", "#E07A3F"),
                ("v5_sf", "v5, shortcut families removed", "#0B8585")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for k, (key, label, color) in enumerate(variants):
        axes[0].bar(x + (k - 1) * width, [rows[m][key]["cross_auroc"] for m in models], width, color=color, label=label)
        mean = np.array([rows[m][key]["cos_slope"] for m in models])
        lo = mean - np.array([rows[m][key]["cos_lo"] for m in models]); hi = np.array([rows[m][key]["cos_hi"] for m in models]) - mean
        axes[1].errorbar(x + (k - 1) * width, mean, yerr=[lo, hi], fmt="o", ms=3.5, color=color, capsize=2, lw=.8, label=label)
    axes[0].set_ylim(.5, 1.0); axes[0].axhline(.5, color="#667085", lw=.8)
    axes[0].set_ylabel("Mean cross-language AUROC"); axes[0].set_title("Accuracy: the shortcut inflates v5")
    axes[1].axhline(0, color="#667085", lw=.8); axes[1].set_ylim(-.06, .12)
    axes[1].set_ylabel("Cosine slope per unit overlap (95% CI)"); axes[1].set_title("Overlap raises cosine much less on v5")
    for ax in axes:
        ax.set_xticks(x, [DISPLAY[m] for m in models], rotation=35, ha="right", fontsize=7.5); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3)
    fig.suptitle("v5 replication on an independently built dataset", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(); save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); setup()
    alignment(args.results_root, args.output_dir / "x1_alignment.png")
    latent(args.results_root, args.output_dir / "x1_lsi_latent.png")
    mmmlu(args.results_root, args.output_dir / "x3_x6_external.png")
    damage(args.results_root, args.output_dir / "x2_damage_budget.png")
    fourth_family(args.results_root, args.output_dir / "x4_fourth_family.png")
    transport_mosaic(args.results_root, args.output_dir / "rq_c_transport_mosaic.png")
    x7_predictors(args.results_root, args.output_dir / "x7_predictors.png")
    x8_scale(args.results_root, args.output_dir / "x8_scale.png")
    x10_x13(args.results_root, args.output_dir / "x10_x13.png")
    if (args.results_root / "rq_f_pythia_trajectory.json").exists():
        trajectory(args.results_root, args.output_dir / "x5_rqf_trajectory.png")
    x14(args.output_dir / "x14_predictors.png")
    v5_replication(args.output_dir / "v5_replication.png")


if __name__ == "__main__":
    main()
