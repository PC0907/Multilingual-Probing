import json
import subprocess
import sys
from pathlib import Path

import pytest

import numpy as np

from study_truth_transport_v2.runners.analyze_x10_x12 import UnlabeledGeometry, pool_statistics, predicted_cosine, N_PER_CLASS
from study_truth_transport_v2.runners.analyze_x13_probes import lda_direction, logistic
from study_truth_transport_v2.runners.analyze_transfer_predictors import auroc_matrix


def test_x10_prediction_matches_monte_carlo_sampling():
    rng = np.random.default_rng(0)
    big_n, d = 600, 30  # pool per class
    fact = rng.normal(size=(2 * big_n, d))  # shared fact content -> cross-language residual covariance
    y = np.repeat([0, 1], big_n)
    truth = rng.normal(size=d)
    xs = fact + 0.8 * rng.normal(size=fact.shape) + 0.4 * y[:, None] * truth
    xt = 0.7 * fact + rng.normal(size=fact.shape) + 0.4 * y[:, None] * (truth + 0.5 * rng.normal(size=d))
    stats = pool_statistics(xs, xt, y)
    for f in (0.0, 0.5, 1.0):
        sims = []
        for _ in range(400):
            idx = []
            for cls in (0, 1):
                pool = np.flatnonzero(y == cls)
                k = int(f * N_PER_CLASS)
                perm = rng.permutation(pool)
                shared, rest = perm[:k], perm[k:]
                s_ix = np.concatenate([shared, rest[:N_PER_CLASS - k]])
                t_ix = np.concatenate([shared, rest[N_PER_CLASS - k:2 * (N_PER_CLASS - k)]])
                idx.append((s_ix, t_ix))
            ws = xs[idx[1][0]].mean(0) - xs[idx[0][0]].mean(0)
            wt = xt[idx[1][1]].mean(0) - xt[idx[0][1]].mean(0)
            sims.append(ws @ wt / np.linalg.norm(ws) / np.linalg.norm(wt))
        assert abs(np.mean(sims) - predicted_cosine(stats, f)) < 0.02


def test_unlabeled_inverse_quadratic_matches_dense():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 60))
    geo = UnlabeledGeometry(x)
    z = x - x.mean(0)
    sigma = geo.a * z.T @ z / len(x) + geo.b * np.eye(60)
    v = rng.normal(size=60)
    assert np.isclose(geo.inverse_quadratic(v), v @ np.linalg.solve(sigma, v))
    assert np.isclose(geo.quadratic(v), v @ sigma @ v)


def test_lda_direction_matches_dense_and_logistic_separates():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 30)
    x = rng.normal(size=(60, 80)) + y[:, None] * rng.normal(size=80)
    w, _ = lda_direction(x, y)
    mu1, mu0 = x[y == 1].mean(0), x[y == 0].mean(0)
    from study_truth_transport_v2.runners.analyze_transfer_predictors import TargetGeometry
    geo = TargetGeometry(x, y)
    sigma = geo.a * geo.z.T @ geo.z / geo.n + geo.b * np.eye(80)
    assert np.allclose(w, np.linalg.solve(sigma, mu1 - mu0), atol=1e-6)
    wl, thr = logistic(x, y, 1.0)
    assert auroc_matrix((x @ wl)[None], y)[0] > 0.95


def _synthetic_inputs(tmp_path):
    import hashlib
    study = Path(__file__).resolve().parents[1]
    prompt_root = study / "data" / "prompts_v3"
    if not (prompt_root / "en.activation.json").exists():
        pytest.skip("build data/prompts_v3 first (README, step 2)")
    rng = np.random.default_rng(3)
    shared = rng.normal(size=16)
    for lang in ("en", "de", "ar", "hi", "fr", "es"):
        prompt = prompt_root / f"{lang}.activation.json"
        rows = json.loads(prompt.read_text(encoding="utf-8"))
        labels = np.asarray([r["label"] for r in rows])
        cache = rng.normal(size=(len(rows), 2, 16)).astype(np.float32)
        cache[:, 1] += (labels[:, None] * (shared + 0.5 * rng.normal(size=16))).astype(np.float32)
        out = tmp_path / "cache" / lang
        out.mkdir(parents=True)
        np.save(out / "last.npy", cache)
        (out / "metadata.json").write_text(json.dumps({"complete": True, "status": "complete", "shape": list(cache.shape),
                                                       "input_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                                                       "source_ids": [r["id"] for r in rows]}))
        (out / "COMPLETE").write_text("ok\n")
    selection = tmp_path / "layer_selection.json"
    selection.write_text(json.dumps({"model_id": "synthetic", "selected_cache_index": 1}))
    cells = [{"source": s, "target": t, "overlap_fraction": f, "metrics": {"raw_cosine": {"mean": 0.5 + 0.05 * f}}}
             for s in ("en", "de", "ar", "hi", "fr", "es") for t in ("en", "de", "ar", "hi", "fr", "es") if s != t
             for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"cells": cells}))
    en = json.loads((prompt_root / "en.activation.json").read_text(encoding="utf-8"))
    groups = sorted({r["group_id"] for r in en})
    qe = tmp_path / "qe.json"
    qe.write_text(json.dumps({"languages": {l: {"flagged_group_ids": groups[:50]} for l in ("de", "ar", "hi", "fr", "es")}}))
    return study, prompt_root, selection, summary, qe


def test_end_to_end_x10_x13_on_synthetic_caches(tmp_path):
    study, prompt_root, selection, summary, qe = _synthetic_inputs(tmp_path)
    common = ["--model-id", "synthetic", "--prompt-root", str(prompt_root), "--cache-root", str(tmp_path / "cache"),
              "--layer-selection", str(selection), "--allocation-root", str(study / "data" / "rq_a_allocations")]
    subprocess.run([sys.executable, "-m", "study_truth_transport_v2.runners.analyze_x10_x12", *common,
                    "--rqa-summary", str(summary), "--qe", str(qe), "--output", str(tmp_path / "x10.json")],
                   check=True, cwd=study.parent, capture_output=True)
    r = json.loads((tmp_path / "x10.json").read_text())
    assert len(r["x10"]["pairs"]) == 15
    assert set(r["x12"]["by_k"]) == {"8", "16", "32", "64"} and r["x12"]["n_configurations"] == 2 * 30
    assert r["x11"]["filtered"]["n_test_groups"]["de"] <= r["x11"]["unfiltered"]["n_test_groups"]["de"]
    subprocess.run([sys.executable, "-m", "study_truth_transport_v2.runners.analyze_x13_probes", *common,
                    "--output", str(tmp_path / "x13.json"), "--reps", "1"],
                   check=True, cwd=study.parent, capture_output=True)
    p = json.loads((tmp_path / "x13.json").read_text())
    assert set(p["families"]) == {"mm", "lda", "lr"}
    assert all(0.5 < v["mean_cross_auroc"] <= 1.0 for v in p["families"].values())
