#!/usr/bin/env python3
"""Create report figures from the frozen JSON result artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CORE_MODELS = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]
# Mistral (X4) adds RQ-A/B/C replication; RQ-D and the original RQ-E grid are core-only.
MODELS = CORE_MODELS + ["mistral-7b-v0.3"]
DISPLAY = {"gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B-Base", "apertus-8b-2509": "Apertus-8B",
           "mistral-7b-v0.3": "Mistral-7B"}
SHORT = {"gemma-7b": "Gemma", "qwen3-8b-base": "Qwen", "apertus-8b-2509": "Apertus", "mistral-7b-v0.3": "Mistral"}
COLORS = {"gemma-7b": "#2457C5", "qwen3-8b-base": "#0B8585", "apertus-8b-2509": "#E07A3F",
          "mistral-7b-v0.3": "#8E5AB5"}
LANGUAGES = ["en", "de", "ar", "hi", "fr", "es"]
LANG_LABELS = ["EN", "DE", "AR", "HI", "FR", "ES"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlap_figure(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.15))
    for model in MODELS:
        payload = read(root / model / "rq_a" / "summary.json")
        grouped = defaultdict(list)
        for cell in payload["cells"]:
            grouped[float(cell["overlap_fraction"])].append(cell)
        x = sorted(grouped)
        cosine = [np.mean([row["metrics"]["raw_cosine"]["mean"] for row in grouped[value]]) for value in x]
        auroc = [np.mean([row["metrics"]["auroc"]["mean"] for row in grouped[value]]) for value in x]
        axes[0].plot(x, cosine, marker="o", lw=2, label=DISPLAY[model], color=COLORS[model])
        axes[1].plot(x, auroc, marker="o", lw=2, label=DISPLAY[model], color=COLORS[model])
    axes[0].set(title="Probe-direction alignment", xlabel="Shared training facts", ylabel="Mean cosine")
    axes[1].set(title="Cross-language discrimination", xlabel="Shared training facts", ylabel="Mean test AUROC")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(MODELS), loc="lower center")
    fig.suptitle("Fixed-n fact-overlap dose response", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, output)


def difficulty_figure(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25))
    x = np.arange(len(MODELS)); width = .34
    original_cos, residual_cos, original_auc, residual_auc = [], [], [], []
    for model in MODELS:
        payload = read(root / model / "rq_b.json")
        pairs = [row for row in payload["truth_pair_results"] if row["source"] != row["target"]]
        original_cos.append(np.mean([row["truth_cosine_original"] for row in pairs]))
        residual_cos.append(np.mean([row["truth_cosine_residual"] for row in pairs]))
        original_auc.append(np.mean([row["original_transfer"]["auroc"] for row in pairs]))
        residual_auc.append(np.mean([row["residual_transfer"]["auroc"] for row in pairs]))
    for ax, before, after, title, ylabel in [
        (axes[0], original_cos, residual_cos, "Geometry", "Mean cross-language cosine"),
        (axes[1], original_auc, residual_auc, "Transfer", "Mean cross-language AUROC"),
    ]:
        ax.bar(x - width/2, before, width, label="Original", color="#2457C5")
        ax.bar(x + width/2, after, width, label="Difficulty removed", color="#E07A3F")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
        ax.set_xticks(x, [SHORT[m] for m in MODELS])
        ax.set_ylim(0, 1.0)
        for index, value in enumerate(after):
            ax.text(index + width/2, value + .008, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="lower center")
    fig.suptitle("RQ-B: truth geometry after cross-fitted difficulty removal", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, output)


def transport_summary_figure(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    metrics = [
        ("auroc", "Cross-language AUROC"),
        ("balanced_accuracy", "Source-threshold balanced accuracy"),
        ("recenter_gain", "Unlabeled recentering gain"),
    ]
    x = np.arange(len(MODELS))
    for ax, (metric, title) in zip(axes, metrics):
        values = []
        for model in MODELS:
            cells = [row for row in read(root / model / "rq_c.json")["cells"] if row["source"] != row["target"]]
            if metric == "recenter_gain":
                values.append(np.mean([row["unlabeled_adaptation"]["balanced_accuracy_change"] for row in cells]))
            else:
                values.append(np.mean([row[metric] for row in cells]))
        ax.bar(x, values, color=[COLORS[m] for m in MODELS], width=.62)
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
        ax.set_xticks(x, [SHORT[m] for m in MODELS], rotation=18)
        for i, value in enumerate(values):
            ax.text(i, value + (.01 if metric != "recenter_gain" else .0015), f"{value:.3f}", ha="center", fontsize=7.2)
    fig.suptitle("RQ-C: ranking transfer and threshold transport", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout()
    save(fig, output)


def matrix(cells: list[dict], value_fn) -> np.ndarray:
    lookup = {(row["source"], row["target"]): value_fn(row) for row in cells}
    return np.asarray([[lookup[(source, target)] for target in LANGUAGES] for source in LANGUAGES], dtype=float)


def annotated_heatmap(ax, values: np.ndarray, title: str, cmap: str, vmin=None, vmax=None, fmt=".2f") -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(6), LANG_LABELS, fontsize=7)
    ax.set_yticks(range(6), LANG_LABELS, fontsize=7)
    for row in range(6):
        for col in range(6):
            value = values[row, col]
            color = "white" if image.norm(value) > .68 or image.norm(value) < .18 else "#17243A"
            ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=6.3, color=color)
    ax.set_xlabel("Target", fontsize=7); ax.set_ylabel("Source", fontsize=7)


def transfer_matrix_figure(root: Path, model: str, output: Path) -> None:
    cells = read(root / model / "rq_c.json")["cells"]
    specifications = [
        ("Direction cosine", lambda row: row["direction_cosine"], "viridis", -0.1, 1.0, ".2f"),
        ("Test AUROC", lambda row: row["auroc"], "viridis", .45, .95, ".2f"),
        ("Balanced accuracy", lambda row: row["balanced_accuracy"], "viridis", .45, .85, ".2f"),
        ("Standardized separation d'", lambda row: row["standardized_separation"], "viridis", -0.2, 1.8, ".2f"),
        ("Offset / pooled SD", lambda row: row["global_offset"] / row["pooled_sd"] if row["pooled_sd"] else np.nan, "coolwarm", -1.5, 1.5, ".2f"),
        ("Recenter BA gain", lambda row: row["unlabeled_adaptation"]["balanced_accuracy_change"], "coolwarm", -.08, .15, "+.2f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 6.5))
    for ax, spec in zip(axes.flat, specifications):
        title, fn, cmap, vmin, vmax, fmt = spec
        annotated_heatmap(ax, matrix(cells, fn), title, cmap, vmin, vmax, fmt)
    fig.suptitle(f"{DISPLAY[model]}: complete 6 x 6 source-to-target transport", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, output)


def de_models(root: Path) -> list[str]:
    """Models with RQ-D and RQ-E outputs (v3 also ran them for Mistral)."""
    return [m for m in MODELS if (root / m / "rq_d.json").exists() and (root / m / "steering_summary.json").exists()]


def quadrant_figure(root: Path, output: Path) -> None:
    quadrants = ["K/K", "K/U", "U/K", "U/U"]
    fig, ax = plt.subplots(figsize=(8.7, 3.25))
    models = de_models(root)
    x = np.arange(4); width = .8 / len(models)
    for model_index, model in enumerate(models):
        payload = read(root / model / "rq_d.json")
        values, valid_counts = [], []
        for quadrant in quadrants:
            valid = [q["zero_overlap_transfer"]["auroc"] for pair in payload["pairs"] for q in pair["quadrants"]
                     if q["quadrant"] == quadrant and q["zero_overlap_transfer"]["auroc"] is not None]
            values.append(np.mean(valid) if valid else np.nan); valid_counts.append(len(valid))
        positions = x + (model_index - (len(models) - 1) / 2) * width
        bars = ax.bar(positions, values, width, color=COLORS[model], label=DISPLAY[model])
        for bar, value, count in zip(bars, values, valid_counts):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width()/2, value + .018, f"{value:.2f}\n({count})", ha="center", fontsize=6.5)
    ax.axhline(.5, color="#667085", lw=1, ls="--")
    ax.set_xticks(x, quadrants); ax.set_ylim(0, 1.3)
    ax.set_ylabel("Mean zero-overlap AUROC")
    ax.set_title("RQ-D behavioral quadrants (number of valid two-class cells in parentheses)")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center")
    fig.tight_layout()
    save(fig, output)


def steering_figure(root: Path, output: Path) -> None:
    directions = ["zero_overlap_source", "full_overlap_source", "target_native", "difficulty", "language_identity"]
    labels = ["Zero overlap", "Full overlap", "Target native", "Difficulty", "Language ID"]
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.25))
    models = de_models(root)
    x = np.arange(len(directions)); width = .8 / len(models)
    neutral_values = []
    for model_index, model in enumerate(models):
        offset = model_index - (len(models) - 1) / 2
        payload = read(root / model / "steering_summary.json")
        summary = payload["named_direction_pair_distribution"]["margin"]["all"]
        values = [summary[name]["mean"] for name in directions]
        axes[0].bar(x + offset * width, values, width, color=COLORS[model], label=DISPLAY[model])
        random = payload["random_direction_pair_draw_distribution"]["margin"]["all"]
        axes[0].errorbar(
            [len(directions) + .25 + offset * .09], [random["mean"]],
            yerr=[[random["mean"] - random["p025"]], [random["p975"] - random["mean"]]],
            fmt="o", color=COLORS[model], capsize=3,
        )
        neutral_path = root / model / "steering_neutral_flores" / "summary.json"
        neutral = read(neutral_path)
        zero = [row["mean_delta_nll"] for row in neutral if row["direction"] == "zero_overlap_source"]
        target = [row["mean_delta_nll"] for row in neutral if row["direction"] == "target_native"]
        zero_value, target_value = np.mean(zero), np.mean(target)
        neutral_values.extend([zero_value, target_value])
        left_bar = axes[1].bar(model_index - width/2, zero_value, width, color=COLORS[model], alpha=.75)
        right_bar = axes[1].bar(model_index + width/2, target_value, width, color=COLORS[model], alpha=.35, hatch="//")
        for bars, value in ((left_bar, zero_value), (right_bar, target_value)):
            axes[1].text(bars[0].get_x() + bars[0].get_width()/2, value, f"{value:.2g}",
                         ha="center", va="bottom", fontsize=6.5)
    axes[0].axhline(0, color="#667085", lw=1)
    axes[0].set_xticks(list(x) + [len(directions) + .25], labels + ["Random 95%"], rotation=24, ha="right")
    axes[0].set_ylabel("Margin slope per alpha")
    axes[0].set_title("Causal effect on true-vs-false margin")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    axes[0].grid(axis="y", alpha=.2)
    axes[1].axhline(0, color="#667085", lw=1)
    axes[1].set_xticks(range(len(models)), [SHORT[m] for m in models])
    axes[1].set_ylabel("Mean neutral-text delta NLL (symlog)")
    axes[1].set_title("Off-target cost at alpha = +/-4")
    axes[1].set_yscale("symlog", linthresh=.05, linscale=1.0)
    axes[1].set_ylim(0, max(neutral_values) * 1.25)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(["Zero-overlap", "Target-native"], frameon=False, fontsize=8)
    fig.suptitle("RQ-E: zero-overlap steering and neutral-text cost", fontsize=12, fontweight="bold", color="#17243A")
    fig.tight_layout()
    save(fig, output)


def layers_figure(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(10.0, 3.0))
    for ax, model in zip(axes, MODELS):
        payload = read(root / model / "all_layers.json")
        layers = payload["layers"]
        block = [row["block_number"] for row in layers]
        ax.plot(block, [row["mean_native_test_auroc"] for row in layers], label="Native AUROC", color="#2457C5")
        ax.plot(block, [row["mean_cross_language_auroc"] for row in layers], label="Cross AUROC", color="#0B8585")
        ax.plot(block, [row["mean_cross_language_cosine"] for row in layers], label="Cross cosine", color="#E07A3F")
        ax.axvline(payload["selected_block_number"], color="#17243A", lw=1, ls="--")
        ax.set_title(DISPLAY[model]); ax.set_xlabel("Decoder block"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean metric")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Layer sensitivity: the selected middle block is not interchangeable with the final block", fontsize=11.5, fontweight="bold", color="#17243A")
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    setup()
    overlap_figure(args.results_root, args.output_dir / "rq_a_overlap.png")
    difficulty_figure(args.results_root, args.output_dir / "rq_b_difficulty.png")
    transport_summary_figure(args.results_root, args.output_dir / "rq_c_transport_summary.png")
    for model in MODELS:
        transfer_matrix_figure(args.results_root, model, args.output_dir / f"rq_c_matrix_{model}.png")
    quadrant_figure(args.results_root, args.output_dir / "rq_d_quadrants.png")
    steering_figure(args.results_root, args.output_dir / "rq_e_steering.png")
    layers_figure(args.results_root, args.output_dir / "layer_sensitivity.png")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "figures": 9}))


if __name__ == "__main__":
    main()
