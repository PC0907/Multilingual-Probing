#!/usr/bin/env python3
"""Render the living NAACL-format results page directly from JSON artifacts.

Every number on the page is read from the results JSON at build time (primary: ``results/remote_v3``); the
only hand-maintained input is ``run_status.json`` (queue state on the server).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
from pathlib import Path

import numpy as np

from study_truth_transport_v2.reporting import replication_stats as rs
from study_truth_transport_v2.reporting.compose_report import (
    rq_a_stats, rq_b_stats, rq_c_stats, rq_d_stats, rq_e_stats)

LANGS = ["en", "de", "ar", "hi", "fr", "es"]
CORE = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]
ALL = CORE + ["mistral-7b-v0.3"]
NAME = {"gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B-Base",
        "apertus-8b-2509": "Apertus-8B-2509", "mistral-7b-v0.3": "Mistral-7B-v0.3",
        "qwen3-0.6b-base": "Qwen3-0.6B-Base", "qwen3-1.7b-base": "Qwen3-1.7B-Base", "qwen3-4b-base": "Qwen3-4B-Base",
        "qwen3-14b-base": "Qwen3-14B-Base", "qwen3-8b": "Qwen3-8B (post-trained)", "olmo-2-1124-7b": "OLMo-2-7B"}
X7_MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base",
             "qwen3-8b", "olmo-2-1124-7b", "gemma-7b", "apertus-8b-2509", "mistral-7b-v0.3"]
X8_MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base",
             "olmo-2-1124-7b", "qwen3-8b"]
SERIES = {"gemma-7b": "s1", "qwen3-8b-base": "s2", "apertus-8b-2509": "s3", "mistral-7b-v0.3": "s4"}
REVISION = {"gemma-7b": "google/gemma-7b @ ff6768d",
            "qwen3-8b-base": "Qwen/Qwen3-8B-Base @ 49e3418",
            "apertus-8b-2509": "swiss-ai/Apertus-8B-2509 @ 3162c99",
            "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3 @ caa1feb"}
METHODS = [("raw_source_mass_mean", "Raw source mass mean"), ("centroid_shift", "Centroid shift"),
           ("rosh_fixed_layer", "RoSh, fixed layer"), ("rosh_scrambled_correspondence", "RoSh, scrambled pairs"),
           ("ridge_unconstrained", "Ridge map"), ("lsi_raw_pca1", "LSI raw PCA-1 removal"),
           ("lsi_raw_mean_shift", "LSI raw mean shift"), ("target_native_mass_mean", "Target-native mass mean")]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def e(text) -> str:
    return html.escape(str(text))


def f(value, digits=3, sign=False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def ci(item: dict, digits=3) -> str:
    low, high = item["ci95"]
    return f'{item["mean_change"]:+.{digits}f} <span class="ci">[{low:+.{digits}f}, {high:+.{digits}f}]</span>'


def holm_mark(item: dict) -> str:
    return "<sup>*</sup>" if item.get("p_holm", 1) < .05 else ""


def table(head: list[str], rows: list[list[str]], caption: str, number: int, num_cols=None) -> str:
    num_cols = num_cols if num_cols is not None else set(range(1, len(head)))
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<figure class="tbl"><figcaption><b>Table {number}:</b> {caption}</figcaption>'
            f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></figure>')


def fig_png(path: Path, caption: str, number: int) -> str:
    import base64
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure class="fig wide"><img class="png" src="data:image/png;base64,{data}" alt="{e(caption)}">'
            f'<figcaption><b>Figure {number}:</b> {caption}</figcaption></figure>')


def fig(svg: str, caption: str, number: int, wide=False) -> str:
    return (f'<figure class="fig{" wide" if wide else ""}"><div class="scroll">{svg}</div>'
            f'<figcaption><b>Figure {number}:</b> {caption}</figcaption></figure>')


# ---------------------------------------------------------------- SVG charts
def heat_color(value: float, low: float, high: float) -> tuple[str, str]:
    t = float(np.clip((value - low) / (high - low), 0, 1))
    stops = [(0, (247, 244, 236)), (.5, (126, 172, 181)), (1, (22, 64, 96))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            rgb = [round(a + (b - a) * u) for a, b in zip(c0, c1)]
            break
    lum = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
    return "#%02x%02x%02x" % tuple(rgb), ("#101820" if lum > 140 else "#f7f4ec")


def heatmaps(panels: list[tuple[str, np.ndarray]], low: float, high: float, label: str) -> str:
    cell, pad_left, pad_top, gap = 34, 30, 34, 26
    width = len(panels) * (pad_left + 6 * cell) + (len(panels) - 1) * gap + 4
    height = pad_top + 6 * cell + 34
    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" role="img" aria-label="{e(label)}" class="chart">']
    for p, (title, values) in enumerate(panels):
        x0 = p * (pad_left + 6 * cell + gap) + pad_left
        out.append(f'<text x="{x0 + 3 * cell}" y="14" class="ttl" text-anchor="middle">{e(title)}</text>')
        for j, t in enumerate(LANGS):
            out.append(f'<text x="{x0 + j * cell + cell / 2}" y="{pad_top - 6}" class="ax" text-anchor="middle">{t.upper()}</text>')
        for i, s in enumerate(LANGS):
            out.append(f'<text x="{x0 - 5}" y="{pad_top + i * cell + cell / 2 + 4}" class="ax" text-anchor="end">{s.upper()}</text>')
            for j in range(6):
                v = values[i, j]
                fill, ink = heat_color(v, low, high)
                stroke = ' stroke="var(--ink)" stroke-width="1.4"' if i == j else ""
                out.append(f'<rect x="{x0 + j * cell}" y="{pad_top + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{fill}"{stroke}/>'
                           f'<text x="{x0 + j * cell + cell / 2 - .5}" y="{pad_top + i * cell + cell / 2 + 3.5}" font-size="9.5" text-anchor="middle" fill="{ink}" class="mono">{v:.2f}</text>')
        out.append(f'<text x="{x0 + 3 * cell}" y="{pad_top + 6 * cell + 16}" class="ax" text-anchor="middle">target language →</text>')
    out.append(f'<text x="2" y="{pad_top + 3 * cell}" class="ax" transform="rotate(-90 9 {pad_top + 3 * cell})" text-anchor="middle">source ↓</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series: dict[str, list[float]], xs: list[float], ylim, ylabel: str, title: str) -> str:
    w, h, l, r, t, b = 300, 210, 46, 12, 26, 36
    sx = lambda x: l + (x - xs[0]) / (xs[-1] - xs[0]) * (w - l - r)
    sy = lambda y: t + (ylim[1] - y) / (ylim[1] - ylim[0]) * (h - t - b)
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" class="chart" role="img" aria-label="{e(title)}">',
           f'<text x="{l}" y="14" class="ttl">{e(title)}</text>']
    for tick in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" class="grid"/>'
                   f'<text x="{l - 5}" y="{sy(tick) + 3:.1f}" class="ax" text-anchor="end">{tick:.2f}</text>')
    for x in xs:
        out.append(f'<text x="{sx(x):.1f}" y="{h - b + 14}" class="ax" text-anchor="middle">{int(x * 100)}%</text>')
    out.append(f'<text x="{(l + w - r) / 2}" y="{h - 4}" class="ax" text-anchor="middle">shared training facts</text>')
    out.append(f'<text x="11" y="{(t + h - b) / 2}" class="ax" text-anchor="middle" transform="rotate(-90 11 {(t + h - b) / 2})">{e(ylabel)}</text>')
    for model, ys in series.items():
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" class="ln {SERIES[model]}"/>')
        out.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.8" class="pt {SERIES[model]}"/>' for x, y in zip(xs, ys))
    out.append("</svg>")
    return "".join(out)


def bar_chart(groups: list[str], series: dict[str, list[float | None]], ylim, ylabel: str, title: str,
              ref: float | None = None, width=560) -> str:
    w, h, l, r, t, b = width, 220, 46, 10, 26, 46
    n = len(series)
    gw = (w - l - r) / len(groups)
    bw = min(18, gw * .8 / max(n, 1))
    sy = lambda y: t + (ylim[1] - y) / (ylim[1] - ylim[0]) * (h - t - b)
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" class="chart" role="img" aria-label="{e(title)}">',
           f'<text x="{l}" y="14" class="ttl">{e(title)}</text>']
    for tick in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" class="grid"/>'
                   f'<text x="{l - 5}" y="{sy(tick) + 3:.1f}" class="ax" text-anchor="end">{tick:.2f}</text>')
    if ref is not None:
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" class="ref"/>')
    for g, group in enumerate(groups):
        cx = l + gw * (g + .5)
        for k, (key, values) in enumerate(series.items()):
            v = values[g]
            if v is None:
                continue
            x = cx + (k - (n - 1) / 2) * bw - bw / 2
            y0 = sy(max(min(v, ylim[1]), ylim[0]))
            base = sy(ylim[0])
            out.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw - 1.5:.1f}" height="{base - y0:.1f}" class="bar {key}"><title>{v:.3f}</title></rect>')
        for li, part in enumerate(group.split("\n")):
            out.append(f'<text x="{cx:.1f}" y="{h - b + 14 + li * 11}" class="ax" text-anchor="middle">{e(part)}</text>')
    out.append(f'<text x="11" y="{(t + h - b) / 2}" class="ax" text-anchor="middle" transform="rotate(-90 11 {(t + h - b) / 2})">{e(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(models: list[str]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="sw {SERIES[m]}"></i>{NAME[m]}</span>' for m in models) + "</div>"


def custom_legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i class="sw {k}"></i>{e(v)}</span>' for k, v in items) + "</div>"


# ------------------------------------------------------------------ helpers
def matrix(cells: list[dict], key: str, sub: str | None = None) -> np.ndarray:
    look = {(c["source"], c["target"]): c for c in cells}
    return np.asarray([[look[(s, t)][key] if sub is None else look[(s, t)][sub][key]
                        for t in LANGS] for s in LANGS], dtype=float)


def deviations_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    out = []
    for section in sections:
        title, _, rest = section.partition("\n")
        items = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"^- ", rest, flags=re.M)[1:]]
        lis = "".join(f"<li>{inline_md(i)}</li>" for i in items)
        out.append(f'<details><summary>{e(title.strip())}</summary><ul>{lis}</ul></details>')
    return "".join(out)


def inline_md(text: str) -> str:
    text = e(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


STATUS_CLASS = {"complete": "ok", "running": "run", "queued": "wait", "unavailable": "na", "failed": "bad", "not done": "na"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.results_root
    status = read(args.status)
    have = lambda m, rel: (root / m / rel).exists()
    core_models = [m for m in ALL if have(m, "rq_c.json") and have(m, "rq_a/inference.json")]
    a = {m: rq_a_stats(root, m) for m in core_models}
    b = {m: rq_b_stats(root, m) for m in core_models if have(m, "rq_b.json")}
    c = {m: rq_c_stats(root, m) for m in core_models}
    d = {m: rq_d_stats(root, m) for m in core_models if have(m, "rq_d.json")}
    st = {m: rq_e_stats(root, m) for m in core_models if have(m, "steering_summary.json")
          and have(m, "steering_sensitivity_layer_plus_2_summary.json")}
    x1 = {m: read(root / m / "extensions/alignment_baselines.json") for m in ALL
          if have(m, "extensions/alignment_baselines.json")}
    lsi = {m: (read(root / m / "extensions/lsi_latent/results.json"),
               read(root / m / "extensions/lsi_latent/inference.json")) for m in ALL
           if have(m, "extensions/lsi_latent/inference.json")}
    x3 = {m: read(root / m / "extensions/mmmlu_transfer.json") for m in ALL
          if have(m, "extensions/mmmlu_transfer.json")}
    x2 = {m: read(root / m / "extensions/damage_budgeted_steering.json") for m in ALL
          if have(m, "extensions/damage_budgeted_steering.json")}
    rqf = read(root / "rq_f_pythia_trajectory.json") if (root / "rq_f_pythia_trajectory.json").exists() else None
    layer = {m: read(root / m / "layer_selection.json")["selected_block_number"] for m in core_models}
    sources = sorted({str(p.relative_to(root)) for m in ALL for p in (root / m).rglob("*.json")
                      if (root / m).exists() and "grid" not in str(p) and "rq_a/" not in str(p)
                      and "steering_full" not in str(p) and "steering_sensitivity_" not in p.parent.name
                      and "neutral_flores" not in str(p)})

    fig_n = iter(range(1, 50))
    tab_n = iter(range(1, 50))
    parts = []

    # ---- ledger
    ledger_rows = []
    for row in status["ledger"]:
        cls = STATUS_CLASS.get(row["status"], "wait")
        ledger_rows.append(f'<tr><td>{e(row["item"])}</td><td>{e(row["model"])}</td>'
                           f'<td><span class="pill {cls}">{e(row["status"])}</span></td><td>{inline_md(row.get("note", ""))}</td></tr>')
    ledger = (f'<div class="scroll"><table class="ledger"><thead><tr><th>Component</th><th>Model</th><th>Status</th><th>Note</th></tr></thead>'
              f'<tbody>{"".join(ledger_rows)}</tbody></table></div>')

    # ---- numbers for prose
    def cross_mean(m, key):
        return float(np.mean([r[key] for r in read(root / m / "rq_c.json")["cells"] if r["source"] != r["target"]]))

    # ---- abstract
    _rep = rs.replication()
    qwen_l = lsi.get("qwen3-8b-base")
    gem_l = lsi.get("gemma-7b")
    abstract = (
        "Multilingual probing work often treats a high cosine between truth directions learned in two languages "
        "as evidence that the model shares one language-independent notion of truth. We test that inference with "
        f"mass-mean probes on 2,000 parallel true/false claims in six languages (English, German, Arabic, Hindi, French, Spanish), "
        f"grouped into 1,972 dependency units, in {len(core_models)} base models. Holding training size fixed while changing only the share of "
        "fact identities seen in both languages raises direction cosine "
        + ", ".join(f'{NAME[m].split("-")[0]} {f(a[m]["cos_slope"]["mean"], 3, True)}' for m in core_models)
        + " per unit overlap ("
        + ", ".join(NAME[m].split("-")[0] for m in core_models if a[m]["cos_slope"]["p025"] <= 0)
        + " imprecise; interval includes zero), while held-out transfer AUROC barely moves (slopes "
        + ", ".join(f'{f(a[m]["auc_slope"]["mean"], 4, True)}' for m in core_models) + "). "
        "Label-blind alignment shows the same dissociation: an LSI-style shared autoencoder raises cross-language cosine "
        + (f'from {f(float(np.mean([r["direction_cosine"] for r in read(root / "qwen3-8b-base/rq_c.json")["cells"] if r["source"] != r["target"]])))} to {f(qwen_l[0]["off_diagonal_means"]["direction_cosine"])} in Qwen with an AUROC change of {f(qwen_l[1]["comparison"]["auroc"]["mean_change"], 4, True)}, '
           if qwen_l else "")
        + (f'and in Gemma lowers AUROC from {f(x1["gemma-7b"]["summary_off_diagonal_means"]["raw_source_mass_mean"]["auroc"])} to {f(gem_l[0]["off_diagonal_means"]["auroc"])} despite higher cosine. ' if gem_l else "")
        + "Ranking transfers better than source thresholds, and unlabeled recentering repairs part of the threshold gap. "
        + (f'Directions trained on generic claims transfer zero-shot to professionally translated MMMLU (off-diagonal paired candidate accuracy '
           + ", ".join(f'{NAME[m].split("-")[0]} {f(x3[m]["off_diagonal_means"]["paired_candidate_accuracy"])}' for m in x3) + "). " if x3 else "")
        + "Zero-overlap directions causally shift truth judgments in some models but not uniformly. "
        "Geometric alignment, score transport, and causal usefulness are separate quantities and should be reported separately. "
        + f"Across {len(rs.MODELS)} models, Fisher efficiency, an established diagnostic, tracks transfer far better than cosine, and a total-covariance Mahalanobis cosine "
        f"is as good, and calibrated across models, only when computed against the target's optimal discriminant. The conclusions hold with a second translation system "
        f"and replicate on an independently built claim set, where transfer is again flat under overlap in {_rep['agg']['v5_sf']['auc_flat']}/10 models but the cosine "
        f"artifact is smaller and clearly present in only {_rep['agg']['v5_sf']['cos_up']}/10.")

    # ---- section: setup tables
    model_rows = [[NAME[m], e(REVISION[m]), str(layer.get(m, "—")),
                   "core + X1" + (" + X2/X3" if m in x3 else "") if m in core_models else "pending"] for m in ALL]
    t_models = table(["Model", "Exact revision", "Block", "Analyses with results"], model_rows,
                     "Models. The frozen block is chosen by mean within-language validation AUROC over six languages; test data never enter selection.", next(tab_n), num_cols={2})

    # ---- RQ-A figure
    xs = [0, .25, .5, .75, 1.0]
    cos_series, auc_series = {}, {}
    for m in core_models:
        cells = read(root / m / "rq_a/summary.json")["cells"]
        cos_series[m] = [float(np.mean([x["metrics"]["raw_cosine"]["mean"] for x in cells if x["overlap_fraction"] == o])) for o in xs]
        auc_series[m] = [float(np.mean([x["metrics"]["auroc"]["mean"] for x in cells if x["overlap_fraction"] == o])) for o in xs]
    lo = min(min(v) for v in cos_series.values()) - .03
    hi_ = max(max(v) for v in cos_series.values()) + .03
    alo = min(min(v) for v in auc_series.values()) - .03
    ahi = max(max(v) for v in auc_series.values()) + .03
    fig_rqa = fig('<div class="pair">' + line_chart(cos_series, xs, (lo, hi_), "mean cosine", "Direction cosine")
                  + line_chart(auc_series, xs, (alo, ahi), "mean AUROC", "Held-out transfer AUROC") + "</div>" + legend(core_models),
                  "RQ-A dose response. Each fit uses 400 dependency groups; only the share of fact identities present in both training sets changes (50 allocations per level, all 30 ordered pairs averaged).", next(fig_n), wide=True)
    t_rqa = table(["Model", "Cosine slope [95% CI]", "AUROC slope [95% CI]", "Cosine 0%→100%", "AUROC 0%→100%", "Holm-sig. cosine / AUROC tests"],
                  [[NAME[m], f'{f(a[m]["cos_slope"]["mean"], 3, True)} <span class="ci">[{f(a[m]["cos_slope"]["p025"], 3, True)}, {f(a[m]["cos_slope"]["p975"], 3, True)}]</span>',
                    f'{f(a[m]["auc_slope"]["mean"], 4, True)} <span class="ci">[{f(a[m]["auc_slope"]["p025"], 4, True)}, {f(a[m]["auc_slope"]["p975"], 4, True)}]</span>',
                    f'{f(a[m]["zero_cos"]["mean"])}→{f(a[m]["full_cos"]["mean"])}', f'{f(a[m]["zero_auc"]["mean"])}→{f(a[m]["full_auc"]["mean"])}',
                    f'{a[m]["cos_sig"]}/15 · {a[m]["auc_sig"]}/30'] for m in core_models],
                  "RQ-A pooled hierarchical bootstrap (1,000 draws over dependency groups and allocations). Slopes are per unit of overlap fraction.", next(tab_n))

    # ---- RQ-B
    t_rqb = table(["Model", "Cosine raw→residual", "AUROC raw→residual", "BA raw→residual", "Difficulty-direction cosine", "Difficulty-direction AUROC"],
                  [[NAME[m], f'{f(b[m]["cos_original"])}→{f(b[m]["cos_residual"])}', f'{f(b[m]["auc_original"])}→{f(b[m]["auc_residual"])}',
                    f'{f(b[m]["ba_original"])}→{f(b[m]["ba_residual"])}', f(b[m]["difficulty_cos"]), f(b[m]["difficulty_auc"])] for m in b],
                  "RQ-B. Cross-fitted (five-fold, group-disjoint) removal of a behavioral-difficulty direction estimated on training data only. Means over 30 ordered pairs.", next(tab_n))

    # ---- RQ-C
    t_rqc = table(["Model", "Direction cosine", "Cross AUROC", "Source-threshold BA", "d′", "Recentering ΔBA", "ΔBA at 30% / 70% prior"],
                  [[NAME[m], f(c[m]["cos"]), f(c[m]["auc"]), f(c[m]["ba"]), f(c[m]["dprime"], 2), f(c[m]["recenter_gain"], 3, True),
                    f'{f(c[m]["prior30_gain"], 3, True)} / {f(c[m]["prior70_gain"], 3, True)}'] for m in core_models],
                  "RQ-C score transport, means over the 30 off-diagonal cells. Unlabeled target recentering cannot change AUROC; it moves only the threshold.", next(tab_n))
    fig_rqc = fig(heatmaps([(NAME[m], matrix(read(root / m / "rq_c.json")["cells"], "auroc")) for m in core_models], .45, .95, "RQ-C AUROC matrices"),
                  "Complete 6×6 AUROC transport matrices (rows: probe source; columns: target; outlined diagonal = native probe). Color scale 0.45–0.95.", next(fig_n), wide=True)
    fig_rqc_ba = fig(heatmaps([(NAME[m], matrix(read(root / m / "rq_c.json")["cells"], "balanced_accuracy")) for m in core_models], .45, .85, "RQ-C balanced accuracy matrices"),
                     "Same matrices for balanced accuracy at the frozen source-language threshold. Color scale 0.45–0.85.", next(fig_n), wide=True)

    # ---- RQ-D
    def dq(m, q):
        return "—" if d[m][q]["mean_auc"] is None else f'{f(d[m][q]["mean_auc"])} <span class="ci">({d[m][q]["valid"]})</span>'
    t_rqd = table(["Model", "K/K", "K/U", "U/K", "U/U"], [[NAME[m], *[dq(m, q) for q in ("K/K", "K/U", "U/K", "U/U")]] for m in d],
                  "RQ-D zero-overlap AUROC by behavioral knowledge quadrant (source/target; K = known, U = unknown). Parentheses: number of pair cells with both labels present.", next(tab_n))

    # ---- RQ-E
    t_rqe = table(["Model", "Zero-overlap slope", "Full-overlap", "Target-native", "Difficulty", "Language identity", "Random mean", "Neutral ΔNLL", "Layer −2 / +2", "All tokens"],
                  [[NAME[m], f(st[m]["zero"]["mean"], 4), f(st[m]["full"]["mean"], 4), f(st[m]["native"]["mean"], 4), f(st[m]["difficulty"]["mean"], 4),
                    f(st[m]["language"]["mean"], 4), f(st[m]["random"]["mean"], 4), f(st[m]["neutral_zero"], 4),
                    f'{f(st[m]["layer_minus_2"], 4)} / {f(st[m]["layer_plus_2"], 4)}', f(st[m]["all_tokens"], 4)] for m in st],
                  "RQ-E steering: mean judgment-margin slope per unit alpha across the six prespecified pairs, with control directions and neutral-text cost (FLORES).", next(tab_n))

    # ---- X1
    x1_models = list(x1)
    x1_rows = []
    for key, label in METHODS:
        row = [label]
        for m in x1_models:
            s = x1[m]["summary_off_diagonal_means"][key]
            if key == "raw_source_mass_mean":
                row.append(f'{f(s["auroc"])} / {f(s["balanced_accuracy"])}')
            else:
                inf = x1[m]["paired_macro_bootstrap_vs_raw"][key]
                row.append(f'{f(inf["auroc"]["mean_change"], 3, True)}{holm_mark(inf["auroc"])} / {f(inf["balanced_accuracy"]["mean_change"], 3, True)}{holm_mark(inf["balanced_accuracy"])}')
        x1_rows.append(row)
    t_x1 = table(["Method", *[NAME[m] for m in x1_models]], x1_rows,
                 "X1 label-blind alignment baselines. First row: raw AUROC / balanced accuracy. Other rows: paired change in AUROC / BA vs raw; * = Holm-adjusted p < 0.05 within model and outcome (1,000 synchronized group bootstraps).", next(tab_n))
    fig_x1 = fig(bar_chart(["Raw", "Centroid", "RoSh\nfixed", "RoSh\nscrambled", "Ridge", "PCA-1", "Mean\nshift", "Target\nnative"],
                           {SERIES[m]: [x1[m]["summary_off_diagonal_means"][k]["auroc"] for k, _ in METHODS] for m in x1_models},
                           (.4, .9), "mean cross AUROC", "Off-diagonal AUROC by alignment method", ref=.5, width=620) + legend(x1_models),
                 "X1 AUROC by method; the dashed line is chance. Transforms use parallel training-group identities but never truth labels.", next(fig_n), wide=True)
    lsi_rows = []
    for m, (res, inf) in lsi.items():
        raw_cos = float(np.mean([r["direction_cosine"] for r in read(root / m / "rq_c.json")["cells"] if r["source"] != r["target"]]))
        lsi_rows.append([NAME[m], f'{f(raw_cos)}→{f(res["off_diagonal_means"]["direction_cosine"])}',
                         f'{f(x1[m]["summary_off_diagonal_means"]["raw_source_mass_mean"]["auroc"])}→{f(res["off_diagonal_means"]["auroc"])}',
                         ci(inf["comparison"]["auroc"], 4), ci(inf["comparison"]["balanced_accuracy"], 4)])
    t_lsi = table(["Model", "Cosine raw→latent", "AUROC raw→latent", "ΔAUROC [95% CI]", "ΔBA [95% CI]"], lsi_rows,
                  "LSI-style shared autoencoder (256-d latent, released architecture and loss weights, trained label-blind on parallel training pairs at the frozen layer). Same-dataset adaptation, not the authors' TED-trained model.", next(tab_n))
    lsi_svg = bar_chart([NAME[m] for m in lsi], {"s1": [float(np.mean([r["direction_cosine"] for r in read(root / m / "rq_c.json")["cells"] if r["source"] != r["target"]])) for m in lsi],
                                                  "s2": [lsi[m][0]["off_diagonal_means"]["direction_cosine"] for m in lsi]}, (0, 1), "cosine", "Direction cosine", width=330) + \
        bar_chart([NAME[m] for m in lsi], {"s1": [x1[m]["summary_off_diagonal_means"]["raw_source_mass_mean"]["auroc"] for m in lsi],
                                           "s2": [lsi[m][0]["off_diagonal_means"]["auroc"] for m in lsi]}, (.4, .9), "AUROC", "Transfer AUROC", ref=.5, width=330)
    def lsi_phrase(m):
        raw_cos = float(np.mean([r["direction_cosine"] for r in read(root / m / "rq_c.json")["cells"] if r["source"] != r["target"]]))
        d_cos = lsi[m][0]["off_diagonal_means"]["direction_cosine"] - raw_cos
        d_auc = lsi[m][1]["comparison"]["auroc"]
        sig = d_auc["ci95"][0] > 0 or d_auc["ci95"][1] < 0
        return (f'{NAME[m]}: cosine {f(d_cos, 3, True)}, AUROC {f(d_auc["mean_change"], 3, True)}'
                + ("" if sig else " (CI includes 0)"))
    fig_lsi = fig('<div class="pair">' + lsi_svg + "</div>" + custom_legend([("s1", "Raw residual stream"), ("s2", "Shared-AE latent")]),
                  "Latent minus raw, off-diagonal means. " + "; ".join(lsi_phrase(m) for m in lsi)
                  + ". Higher latent cosine never comes with a significant AUROC gain.", next(fig_n), wide=True)

    # ---- X2
    if x2:
        x2_rows = []
        for m, data in x2.items():
            for direction in ("zero_overlap_source", "target_native"):
                agg = data["aggregate"][direction]
                curves = [cell for cell in data["cells"] if cell["direction"] == direction]
                max_dmg = max(max(abs(p["symmetric_mean_delta_nll"]) for p in cell["damage_curve"]) for cell in curves)
                x2_rows.append([NAME[m], direction.replace("_", " "), f'{agg["pairs_with_safe_nonzero_alpha"]}/{agg["pairs"]}',
                                f(agg["mean_safe_alpha"], 2), f(agg["mean_symmetric_slope"], 3), f(max_dmg, 4)])
            rnd = data["random_safe_slope_distribution"]
            x2_rows.append([NAME[m], "random (5 per pair)", f'{rnd["n"]}', "—", f'{f(rnd["p50"], 4)} <span class="ci">[{f(rnd["p025"], 4)}, {f(rnd["p975"], 4)}]</span>', "—"])
        t_x2 = table(["Model", "Direction", "Pairs with safe α", "Mean safe |α|", "Safe symmetric slope", "Max |neutral ΔNLL| on grid"], x2_rows,
                     "X2 damage-budgeted steering. α is chosen without truth labels as the largest symmetric magnitude with mean neutral-continuation ΔNLL ≤ 0.05; grid |α| ∈ {0.125, …, 4}.", next(tab_n))
    else:
        t_x2 = '<p class="pending">X2 results are pending.</p>'

    # ---- X3
    if x3:
        x3_rows = []
        for m, data in x3.items():
            o, u = data["off_diagonal_means"], data["off_diagonal_means_uncompacted_subset"]
            n_q = data["cells"][0]["n_questions"]
            n_u = data["cells"][0]["uncompacted_subset"]["n_questions"]
            x3_rows.append([NAME[m], str(n_q), f(o["micro_auroc"]), f(o["paired_candidate_accuracy"]), f(o["macro_subject_auroc"]),
                            f(o["source_threshold_balanced_accuracy"]), f'{f(u["micro_auroc"])} / {f(u["paired_candidate_accuracy"])} <span class="ci">(n={n_u})</span>'])
        t_x3 = table(["Model", "Questions", "Micro AUROC", "Paired accuracy", "Macro-subject AUROC", "Source-threshold BA", "Uncompacted subset AUROC / paired"],
                     x3_rows, "X3 zero-shot transfer to MMMLU (English MMLU plus OpenAI's professional translations), off-diagonal means over 30 ordered pairs. No MMMLU item is used for fitting or selection.", next(tab_n))
        fig_x3 = fig(heatmaps([(NAME[m], matrix(x3[m]["cells"], "paired_candidate_accuracy")) for m in x3], .45, .95, "MMMLU paired accuracy"),
                     "X3 paired candidate accuracy (does the true candidate outscore its matched false candidate?) for every source probe and MMMLU target language. Chance is 0.5.", next(fig_n), wide=True)
    else:
        t_x3, fig_x3 = '<p class="pending">X3 results are pending.</p>', ""

    # ---- X6 native-authored INCLUDE
    x6 = {m: read(root / m / "extensions/include_transfer.json") for m in ALL
          if have(m, "extensions/include_transfer.json")}
    if x6:
        x6_rows = []
        for m, data in x6.items():
            o = data["off_diagonal_means"]
            per_target = {t: float(np.mean([c["paired_candidate_accuracy"] for c in data["cells"] if c["target"] == t]))
                          for t in data["target_languages"]}
            x6_rows.append([NAME[m], f(o["micro_auroc"]), f(o["paired_candidate_accuracy"]),
                            f(o["source_threshold_balanced_accuracy"]), *[f(per_target[t]) for t in data["target_languages"]]])
        t_x6 = table(["Model", "Cross-lang. AUROC", "Cross-lang. paired acc.", "Source-threshold BA", "AR", "DE", "HI", "FR", "ES"], x6_rows,
                     "X6 zero-shot transfer to native-authored INCLUDE exam questions. First three columns: means over cells whose probe-source language differs from the target; last five: paired accuracy per target averaged over all six source probes.", next(tab_n))
        reg = {}
        for m, data in x6.items():
            for cell in data["cells"]:
                for item in cell["breakdown"]["values"]:
                    reg.setdefault(m, {}).setdefault(item["value"], []).append(item["paired_accuracy"])
        values = sorted({v for d in reg.values() for v in d})
        t_x6b = table(["Model", *values], [[NAME[m], *[f(float(np.mean(reg[m][v]))) if v in reg[m] else "—" for v in values]] for m in reg],
                      "X6 paired accuracy by INCLUDE regional feature, averaged over all 30 source-target cells (descriptive).", next(tab_n))
        t_x6 = t_x6 + t_x6b
    else:
        t_x6 = '<p class="pending">X6 is frozen and queued: Mistral, then Qwen, then Apertus, each with an exact-revision re-download and a reproduction check of the regenerated training cache.</p>'

    # ---- X4 / X5
    if "mistral-7b-v0.3" in core_models:
        mi = "mistral-7b-v0.3"
        x4 = (f'<p>Mistral-7B-v0.3 (block {layer[mi]}) replicates the central dissociation in a fourth family: the overlap slope for cosine is '
              f'{f(a[mi]["cos_slope"]["mean"], 3, True)} (95% CI {f(a[mi]["cos_slope"]["p025"], 3, True)} to {f(a[mi]["cos_slope"]["p975"], 3, True)}; '
              f'{a[mi]["cos_sig"]}/15 Holm-significant pair tests), while the AUROC slope is {f(a[mi]["auc_slope"]["mean"], 4, True)} '
              f'(95% CI {f(a[mi]["auc_slope"]["p025"], 4, True)} to {f(a[mi]["auc_slope"]["p975"], 4, True)}; {a[mi]["auc_sig"]}/30 significant). '
              'Its results appear in every table above' + ('.</p>' if mi in d and mi in st else
              '; RQ-D and the original RQ-E steering grid were secondary in the X4 protocol and were not run.</p>'))
    else:
        x4 = '<p class="pending">Mistral-7B-v0.3 replication (layer selection, RQ-A, RQ-B with behavioral scoring, RQ-C, X1, X2, X3) is queued on the server and will fill the tables above when it completes.</p>'
    if rqf:
        rows = rqf["checkpoints"]
        t_x5 = table(["Step", "Block", "Zero-overlap cosine", "Full-overlap cosine", "Zero-overlap AUROC", "Full-overlap AUROC"],
                     [[f'{r["step"]:,}', str(r.get("selected_block_number", "—")), f(r["zero_overlap_cosine"]), f(r["full_overlap_cosine"]),
                       f(r["zero_overlap_auroc"]), f(r["full_overlap_auroc"])] for r in rows],
                     "X5 exploratory Pythia-1.4B-deduped trajectory; layer re-selected on validation data at each checkpoint.", next(tab_n))
    else:
        t_x5 = '<p class="pending">The Pythia-1.4B-deduped trajectory (steps 0, 1k, 4k, 16k, 64k, 143k) runs after Mistral.</p>'

    # ---- X7 label-free transfer prediction
    x7 = {m: read(root / m / "extensions/transfer_predictors.json") for m in X7_MODELS
          if have(m, "extensions/transfer_predictors.json")}
    if x7:
        rows7 = []
        for m, d7 in x7.items():
            r, h = d7["spearman_with_auroc_all_blocks"], d7["h1_delta_rho_p2_minus_p1"]
            rows7.append([NAME[m], str(d7["n_configurations"]), f(r["p1_euclidean_cosine"]), f(r["p2_translation_consistency"]),
                          f(r["p3_fisher_efficiency"]),
                          f'{f(h["observed"], 3, True)} <span class="ci">[{f(h["ci95"][0], 3, True)}, {f(h["ci95"][1], 3, True)}]</span>'])
        t_x7 = table(["Model", "Configs", "ρ P1 cosine", "ρ P2 consistency", "ρ P3 Fisher", "Δρ P2−P1 [95% CI]"], rows7,
                     "X7: Spearman correlation of each predictor with held-out cross-language AUROC over every decoder block × 30 ordered pairs (500-draw group bootstrap for Δρ).", next(tab_n))
        rows7b = []
        for m, d7 in x7.items():
            ov, xm = d7.get("overlap_secondary"), d7.get("x1_methods_secondary")
            rows7b.append([NAME[m],
                           "—" if not ov else f'{f(ov["slopes_per_full_overlap"]["p1"], 3, True)} / {f(ov["slopes_per_full_overlap"]["p2"], 3, True)} / {f(ov["slopes_per_full_overlap"]["auroc"], 4, True)}',
                           "—" if not xm else f'{f(xm["spearman_over_methods_and_pairs"]["p1"])} / {f(xm["spearman_over_methods_and_pairs"]["p2"])}'])
        t_x7 = t_x7 + table(["Model", "Overlap slope: P1 / P2 / AUROC", "ρ over X1 methods: P1 / P2"], rows7b,
                            "X7 secondary checks at the frozen layer: does each statistic move when alignment is manipulated (shared facts; RoSh, ridge, PCA-1, scrambled RoSh) without a matching change in transfer?", next(tab_n))
        meta_path = root / "x7_meta.json"
        if meta_path.exists():
            meta = read(meta_path)
            h1, h2 = meta["h1"], meta["h2_leave_one_model_out_r2"]
            t_x7 += (f'<p><b>Pooled tests.</b> H1: Δρ interval above zero in {h1["models_with_delta_ci_above_zero"]}/{h1["n_models"]} models '
                     f'and below zero in {h1["models_with_delta_ci_below_zero"]}; Δρ positive in {h1["models_with_positive_delta"]}/{h1["n_models"]} (sign test p = {h1["sign_test_p"]:.3f}, favoring cosine); '
                     f'H1 {"supported" if h1["supported"] else "not supported"}. H2 leave-one-model-out R²: '
                     f'P1 {h2["p1_euclidean_cosine"]["mean"]:.3f}, P2 {h2["p2_translation_consistency"]["mean"]:.3f}, '
                     f'P3 {h2["p3_fisher_efficiency"]["mean"]:.3f}.</p>')
        fig_x7 = fig_png(args.study_root / "reporting/output/figures/x7_predictors.png",
                         "X7 predictors: per-model correlations and pooled configurations.", next(fig_n))
    else:
        t_x7, fig_x7 = '<p class="pending">X7 is running.</p>', ""

    # ---- X8 scale / X9 post-training
    x8_rows = []
    for m in X8_MODELS:
        if not (have(m, "rq_c.json") and have(m, "rq_a/inference.json")):
            continue
        ai = read(root / m / "rq_a/inference.json")["pooled_hierarchical_bootstrap"]
        cc = [c for c in read(root / m / "rq_c.json")["cells"] if c["source"] != c["target"]]
        ext = lambda name, key: (f(read(root / m / "extensions" / name)["off_diagonal_means"][key]) if have(m, "extensions/" + name) else "—")
        lsi_d = read(root / m / "extensions/lsi_latent/inference.json")["comparison"]["auroc"]["mean_change"] if have(m, "extensions/lsi_latent/inference.json") else None
        x8_rows.append([NAME[m], f(float(np.mean([c["auroc"] for c in cc]))), f(float(np.mean([c["direction_cosine"] for c in cc]))),
                        f'{f(ai["cosine_slope"]["mean"], 3, True)} <span class="ci">[{f(ai["cosine_slope"]["p025"], 3, True)}, {f(ai["cosine_slope"]["p975"], 3, True)}]</span>',
                        f'{f(ai["auroc_slope"]["mean"], 4, True)} <span class="ci">[{f(ai["auroc_slope"]["p025"], 4, True)}, {f(ai["auroc_slope"]["p975"], 4, True)}]</span>',
                        "—" if lsi_d is None else f(lsi_d, 3, True),
                        ext("mmmlu_transfer.json", "paired_candidate_accuracy"), ext("include_transfer.json", "paired_candidate_accuracy")])
    t_x8 = (table(["Model", "Cross AUROC", "Cosine", "Overlap cosine slope", "Overlap AUROC slope", "Latent ΔAUROC", "MMMLU paired", "INCLUDE paired"],
                  x8_rows, "X8/X9: frozen core and extensions for the Qwen3 scale series, OLMo-2-7B, and post-trained Qwen3-8B.", next(tab_n))
            + fig_png(args.study_root / "reporting/output/figures/x8_scale.png", "X8 Qwen3 scale trends (descriptive; at most five sizes).", next(fig_n))
            if x8_rows else '<p class="pending">X8/X9 models are running.</p>')

    # ---- X10-X13 (protocol/X10_X11_PREREGISTRATION.md)
    meta10 = root / "x10_x13_meta.json"
    if meta10.exists():
        mx = read(meta10)
        x10m, x11m, x12m, fam = mx["x10"], mx["x11"], mx["x12"], mx["x13"]["families"]
        rows10 = []
        for m in mx["models"]:
            cells = [NAME.get(m, m), f(x10m["per_model_slope_r"][m], 2)]
            for k in ("mm", "lda", "lr"):
                v = fam[k]["per_model"][m]["overlap_slopes"]
                cells += [f(v["cosine"]["mean"], 3, True), f(v["auroc"]["mean"], 4, True)]
            cells.append(f(x11m["per_model"][m]["delta_cross_auroc"], 3, True))
            cells.append(f(x12m["k16_fisher_minus_direct"][m]["mean"], 3, True))
            rows10.append(cells)
        t_x10 = (f'<p><b>Preregistered verdicts.</b> X10 pooled test: <b>{e(x10m["verdict"])}</b> (r = {f(x10m["slope_r"], 2)} over {x10m["n_units"]} units; '
                 f'Apertus within-model r = {f(x10m["per_model_slope_r"]["apertus-8b-2509"], 2)}, the other nine models '
                 f'{f(min(v for k, v in x10m["per_model_slope_r"].items() if not k.startswith("apertus")), 2)} to '
                 f'{f(max(v for k, v in x10m["per_model_slope_r"].items() if not k.startswith("apertus")), 2)}). '
                 f'X11: <b>{e(x11m["verdict"])}</b> (max |ΔAUROC| {f(x11m["max_abs_delta_cross_auroc"], 3)}). '
                 f'X12: <b>{e(x12m["verdict"])}</b> (few-shot Fisher beat direct few-shot AUROC at k = 16 in {x12m["k16_models_positive"]}/{len(mx["models"])} models). '
                 f'X13: the overlap dissociation replicates in {fam["mm"]["n_models_dissociation"]}/10 (mass mean), {fam["lda"]["n_models_dissociation"]}/10 (shrinkage LDA) and {fam["lr"]["n_models_dissociation"]}/10 (logistic regression) models; '
                 f'mean cosine slopes {f(fam["mm"]["mean_cosine_slope"], 3, True)}, {f(fam["lda"]["mean_cosine_slope"], 3, True)} and {f(fam["lr"]["mean_cosine_slope"], 3, True)}.</p>'
                 + table(["Model", "X10 r", "MM cos slope", "MM AUROC slope", "LDA cos slope", "LDA AUROC slope", "LR cos slope", "LR AUROC slope", "X11 ΔAUROC", "X12 Fisher−direct (k=16)"],
                         rows10, "X10 to X13 per model. Slopes per unit overlap fraction (10 allocations per level). X11: change in mean cross AUROC after dropping the lowest-LaBSE 10% of test claims per language.", next(tab_n))
                 + fig_png(args.study_root / "reporting/output/figures/x10_x13.png", "X10 predicted versus observed overlap slopes, and X13 cosine slope by probe family (black ticks: AUROC slopes).", next(fig_n)))
    else:
        t_x10 = '<p class="pending">X10 to X13 are running.</p>'

    # ---- X14, translation robustness, v5 replication (shared numbers: reporting/replication_stats.py)
    figdir = args.study_root / "reporting/output/figures"
    x3k, x5k = rs.x14(rs.V3), rs.x14(rs.V5)
    def rng(xr, key):
        vals = [v[key] for v in xr["within"].values()]
        return f'{f(float(np.median(vals)), 2)} <span class="ci">[{f(min(vals), 2)}, {f(max(vals), 2)}]</span>'
    rows14 = [[label, rng(x3k, key), f(x3k["lomo_r2"][key], 2) if key in x3k["lomo_r2"] else "—",
               rng(x5k, key), f(x5k["lomo_r2"][key], 2) if key in x5k["lomo_r2"] else "—"] for key, label in rs.X14_KEYS.items()]
    h14d = read(rs.V3 / "x14_meta.json")["h14d"]["per_model"]; ap = h14d["apertus-8b-2509"]
    t_x14 = (table(["Statistic", "Median ρ [range], primary", "LOMO R², primary", "Median ρ [range], v5", "LOMO R², v5"], rows14,
                   "X14: within-model Spearman ρ with held-out AUROC over all blocks × 30 pairs (10 models), and the preregistered pooled leave-one-model-out R² of a linear calibration (calibrated if ≥ 0.5). Proposition 1 has no fitted parameter.", next(tab_n))
             + f'<p>Against the target\'s trained probe the Mahalanobis cosine is weak; against the optimal discriminant it is at least as good as Fisher efficiency in {x3k["h14a"]["mcs_optimal_ge_fisher"]}/10 models and the only statistic calibrated across models '
               f'(LOMO R² {f(x3k["lomo_r2"]["p4b_mcs_total_vs_target_optimal"], 2)} primary, {f(x5k["lomo_r2"]["p4b_mcs_total_vs_target_optimal"], 2)} on v5). '
               f'Proposition 1 ranks configurations as well as any statistic but overestimates AUROC by {f(x3k["prop1_bias"], 3, True)} on the primary claims (R² {f(x3k["prop1"]["r2_no_fit"], 2)}); on v5 it is calibrated (R² {f(x5k["prop1"]["r2_no_fit"], 2)}, MAE {f(x5k["prop1"]["mae"], 3)}). '
               f'Apertus is the failure case: {ap["n_outlier_dims"]} outlier dimensions carry {ap["outlier_variance_share"]:.0%} of its activation variance, but removing them does not rescue the X10 prediction '
               f'(r {f(ap["x10_slope_r_raw"], 2)} → {f(ap["x10_slope_r_reduced"], 2)}; preregistered H14d not supported).</p>'
             + fig_png(figdir / "x14_predictors.png", "X14 statistics per model, primary claims (left) and v5 (right).", next(fig_n)))
    tr, rep_ = rs.translation(), rs.replication()
    rows_tr = [[l.upper(), *(f(tr["labse"][l][s]) for s in ("NLLB", "LLM", "Google")),
                " / ".join(str(tr["fidelity"][l][s][0]) for s in ("NLLB", "Opus", "Google")),
                " / ".join(str(tr["fidelity"][l][s][1]) for s in ("NLLB", "Opus", "Google"))] for l in ("de", "ar", "hi", "fr", "es")]
    g = rep_["agg"]
    t_trans = ('<p>Every experiment was run on both translation systems. Automatic checks do not settle which is better: LaBSE similarity to the English is marginally higher for NLLB, '
               'but NLLB changes numbers and swaps imperial units for metric ones while keeping the number; the Opus translations have no digit or unit errors. '
               'LaBSE measures surface closeness, not fidelity. Google Translate output (obtained by the first author) was used only for these checks.</p>'
               + table(["Language", "LaBSE NLLB", "LaBSE Opus", "LaBSE Google", "Digit errors N / O / G", f"Unit swaps N / O / G (of {tr['n_imperial']})"], rows_tr,
                       "Automatic translation checks on the 2,000 claims (results/translation_qe_three_way.json, results/translation_fidelity.json).", next(tab_n))
               + table(["Across 10 models", "NLLB (v2)", "Opus (v3, primary)"],
                       [["Overlap dissociation (cosine up, AUROC flat)", f'{g["v2"]["dissociation"]}/10', f'{g["v3"]["dissociation"]}/10'],
                        ["Fisher efficiency beats cosine", f'{g["v2"]["fisher_beats_cos"]}/10', f'{g["v3"]["fisher_beats_cos"]}/10'],
                        ["Mean cross-language AUROC", f(g["v2"]["mean_cross_auroc"]), f(g["v3"]["mean_cross_auroc"])],
                        ["Mean overlap cosine slope", f(g["v2"]["mean_cos_slope"], 3, True), f(g["v3"]["mean_cos_slope"], 3, True)]],
                       "Translation-system robustness: no qualitative conclusion changes.", next(tab_n))
               + '<p><b>Disclosure.</b> The first author chose the Opus translations as primary for translation fidelity after results for three core models and the automatic checks had been seen, and without the planned blind human comparison; the analyst had recommended keeping NLLB primary unless a human comparison preferred Opus.</p>')
    rows_v5 = []
    for m in rs.MODELS:
        r = rep_["rows"][m]; s = r["v5_sf"]
        rows_v5.append([rs.NAME[m], f(r["v3"]["cross_auroc"]), f(r["v5"]["cross_auroc"]), f(s["cross_auroc"]),
                        f'{f(s["cos_slope"], 3, True)} <span class="ci">[{f(s["cos_lo"], 3, True)}, {f(s["cos_hi"], 3, True)}]</span>',
                        f'{f(s["auc_slope"], 4, True)} <span class="ci">[{f(s["auc_lo"], 4, True)}, {f(s["auc_hi"], 4, True)}]</span>',
                        f'{f(s["rho_fisher"], 2)} / {f(s["rho_cos"], 2)}'])
    mb = rs.mistral_v5_behavior()
    t_v5 = ('<p>v5 is the first author\'s MTruth release: 2,000 mLAMA-derived fact families, each with one true claim and one false claim made by substituting an object of the same relation, rendered with short entity templates in the same six languages, with its own split and overlap allocations. '
            'An automatic audit removed 11 families with wrong-entity labels (3,978 claims per language remain). A second audit found a label shortcut: in 343 families the true object\'s name occurs inside the subject\'s name (e.g. "Audi Q7" made by "Audi"), against 2 for the false object. '
            'Every analysis was therefore also run without those families (3,296 claims per language), a control fixed before any v5 result.</p>'
            + table(["Model", "AUROC primary", "AUROC v5", "AUROC v5 shortcut-free", "Cosine slope, shortcut-free [95% CI]", "AUROC slope, shortcut-free [95% CI]", "ρ Fisher / cosine, shortcut-free"],
                    rows_v5, "v5 replication versus the primary claims.", next(tab_n))
            + f'<p>The shortcut inflates cross-language AUROC by {f(rep_["shortcut_gain"]["min"])} to {f(rep_["shortcut_gain"]["max"])} in every model; without it, accuracy is close to the primary claims (mean difference {f(rep_["v5_sf_minus_v3"]["mean"], 3, True)}). '
              f'Replicated: AUROC flat under overlap in {g["v5_sf"]["auc_flat"]}/10 models, Fisher efficiency beats cosine in {g["v5_sf"]["fisher_beats_cos"]}/10, and the X14 results. '
              f'Not fully replicated: overlap raises cosine a median {f(rep_["v3_over_v5sf_cos_slope"]["median"], 1)}× less, with an interval above zero in only {g["v5_sf"]["cos_up"]}/10 models '
              f'({", ".join(rs.NAME[m] for m in rep_["v5_sf_dissociation_models"])}); we read this as a boundary condition of the overlap artifact on short entity templates. '
              f'Mistral answers the v5 judgment prompts degenerately (Hindi {mb["hi"]["share_true"]:.0%} "true", French {1 - mb["fr"]["share_true"]:.0%} "false"; its Arabic templates never agree), so one constant difficulty component was dropped for Arabic. v5 labels are not fact-checked.</p>'
            + fig_png(figdir / "v5_replication.png", "v5 replication: cross-language AUROC and overlap cosine slopes, primary claims versus v5 with and without the shortcut families.", next(fig_n)))

    deviations = deviations_html(args.study_root / "protocol/DEVIATIONS.md")
    updated = dt.datetime.now(dt.timezone.utc).strftime("%-d %B %Y, %H:%M UTC")
    srcs = "".join(f"<li><code>{e(s)}</code></li>" for s in sources)

    rqa_text = " ".join(
        f'{NAME[m]}: cosine slope {f(a[m]["cos_slope"]["mean"], 3, True)}, AUROC slope {f(a[m]["auc_slope"]["mean"], 4, True)}.' for m in core_models)

    page = TEMPLATE.format(
        updated=e(updated), queue=e(status["headline"]), abstract=e(abstract), ledger=ledger, t_models=t_models,
        fig_rqa=fig_rqa, t_rqa=t_rqa, rqa_text=e(rqa_text), t_rqb=t_rqb, t_rqc=t_rqc, fig_rqc=fig_rqc, fig_rqc_ba=fig_rqc_ba,
        t_rqd=t_rqd, t_rqe=t_rqe, t_x1=t_x1, fig_x1=fig_x1, t_lsi=t_lsi, fig_lsi=fig_lsi, t_x2=t_x2, t_x3=t_x3, fig_x3=fig_x3,
        x4=x4, t_x5=t_x5, t_x6=t_x6, t_x7=t_x7, fig_x7=fig_x7, t_x8=t_x8, t_x10=t_x10, t_x14=t_x14, t_trans=t_trans, t_v5=t_v5, deviations=deviations, sources=srcs, n_models=len(core_models))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": len(page), "core_models": core_models,
                      "x2": list(x2), "x3": list(x3), "rqf": rqf is not None}))


TEMPLATE = (Path(__file__).with_name("blog_template.html")).read_text(encoding="utf-8") if Path(__file__).with_name("blog_template.html").exists() else ""

if __name__ == "__main__":
    main()
