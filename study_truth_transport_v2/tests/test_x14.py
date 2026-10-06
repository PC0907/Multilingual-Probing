import json
import math
import subprocess
import sys

import numpy as np

from study_truth_transport_v2.runners.analyze_transfer_predictors import TargetGeometry, auroc_matrix
from study_truth_transport_v2.runners.analyze_x14_mcs import TotalCovariance, lda_direction, phi
from study_truth_transport_v2.tests.test_x10_x13 import _synthetic_inputs


def test_total_covariance_cosine_matches_dense():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 30)) @ rng.normal(size=(30, 30))
    u, v = rng.normal(size=30), rng.normal(size=30)
    sigma = np.cov(x, rowvar=False, bias=True)
    dense = u @ sigma @ v / math.sqrt((u @ sigma @ u) * (v @ sigma @ v))
    assert np.isclose(TotalCovariance(x).cosine(u, v), dense)


def test_lda_direction_matches_dense_high_dim():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 25)
    x = rng.normal(size=(50, 120)) + y[:, None] * rng.normal(size=120)
    geo = TargetGeometry(x, y)
    sigma = geo.a * geo.z.T @ geo.z / geo.n + geo.b * np.eye(120)
    assert np.allclose(lda_direction(geo), np.linalg.solve(sigma, geo.delta), atol=1e-6)


def test_prop1_prediction_matches_gaussian_auroc():
    rng = np.random.default_rng(2)
    d, n = 10, 4000
    a = rng.normal(size=(d, d)); sigma = a @ a.T / d + np.eye(d)
    delta = rng.normal(size=d) * 0.3
    y = np.repeat([0, 1], n // 2)
    x = rng.multivariate_normal(np.zeros(d), sigma, size=n) + y[:, None] * delta
    w = delta + rng.normal(size=d) * 0.3
    geo = TargetGeometry(x, y)
    f = geo.fisher_efficiency(w)
    pred = phi(f * math.sqrt(geo.q) / math.sqrt(2))
    sub = np.arange(n)
    emp = auroc_matrix((x[sub] @ w)[None], y[sub])[0]
    assert abs(pred - emp) < 0.02


def test_end_to_end_x14_on_synthetic_caches(tmp_path):
    study, prompt_root, selection, summary, qe = _synthetic_inputs(tmp_path)
    out = tmp_path / "x14.json"
    subprocess.run([sys.executable, "-m", "study_truth_transport_v2.runners.analyze_x14_mcs", "--model-id", "synthetic",
                    "--prompt-root", str(prompt_root), "--cache-root", str(tmp_path / "cache"),
                    "--layer-selection", str(selection), "--allocation-root", str(study / "data" / "rq_a_allocations"),
                    "--output", str(out), "--reps", "1"], check=True, cwd=study.parent, capture_output=True)
    r = json.loads(out.read_text())
    assert r["n_configurations"] == 2 * 30
    assert set(r["spearman_all_blocks"]) >= {"p4b_mcs_total_vs_target_optimal", "p5_prop1_predicted_auroc"}
    assert len(r["outlier_analysis"]["pairs_raw"]) == 15
    assert all(0 <= c["p5_prop1_predicted_auroc"] <= 1 for c in r["configurations"])
