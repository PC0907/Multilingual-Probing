import numpy as np
import pytest

from study_truth_transport_v2.runners.analyze_transfer_predictors import (
    TargetGeometry, auroc_matrix, average_ranks, pearson_rows, spearman)


def dense_ledoit_wolf(z):
    n, d = z.shape
    s = z.T @ z / n
    nu = np.trace(s) / d
    delta2 = np.sum((s - nu * np.eye(d)) ** 2)
    beta2 = sum(np.sum((np.outer(v, v) - s) ** 2) for v in z) / n**2
    rho = min(beta2, delta2) / delta2
    return (1 - rho) * s + rho * nu * np.eye(d), rho


def test_fisher_efficiency_matches_dense_computation():
    rng = np.random.default_rng(0)
    n, d = 40, 60  # fewer samples than dimensions, as in the real data
    y = np.repeat([0, 1], n // 2)
    x = rng.normal(size=(n, d)) + y[:, None] * rng.normal(size=d)
    geometry = TargetGeometry(x, y)
    mu1, mu0 = x[y == 1].mean(0), x[y == 0].mean(0)
    z = x - np.where(y[:, None] == 1, mu1, mu0)
    sigma, rho = dense_ledoit_wolf(z)
    assert np.isclose(geometry.rho, rho)
    delta = mu1 - mu0
    optimal = np.linalg.solve(sigma, delta)
    w = rng.normal(size=d)
    expected = (w @ delta) / np.sqrt((w @ sigma @ w) * (delta @ optimal))
    assert np.isclose(geometry.fisher_efficiency(w), expected)
    # The optimal discriminant has efficiency one; efficiency is scale invariant.
    assert np.isclose(geometry.fisher_efficiency(optimal), 1.0)
    assert np.isclose(geometry.fisher_efficiency(3 * w), geometry.fisher_efficiency(w))


def test_auroc_matrix_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(1)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    scores = rng.integers(0, 3, size=(5, len(y))).astype(float)
    got = auroc_matrix(scores, y)
    for row, value in zip(scores, got):
        pos, neg = row[y == 1], row[y == 0]
        expected = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        assert np.isclose(value, expected)


def test_ranks_spearman_and_pearson():
    assert np.allclose(average_ranks(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1.0, 3.5, 2.0])
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(spearman(a, a**3), 1.0)
    assert np.isclose(spearman(a, -a), -1.0)
    assert np.isclose(pearson_rows(a[None], (2 * a + 1)[None])[0], 1.0)


def test_end_to_end_on_synthetic_caches(tmp_path):
    """Run the full X7 analysis on tiny synthetic caches in the frozen on-disk format."""
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

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
        own = rng.normal(size=16)
        cache = rng.normal(size=(len(rows), 3, 16)).astype(np.float32)
        for block, strength in enumerate((0.0, 0.8, 1.5)):
            cache[:, block] += (labels[:, None] * strength * (shared + 0.5 * own)).astype(np.float32)
        out = tmp_path / "cache" / lang
        out.mkdir(parents=True)
        np.save(out / "last.npy", cache)
        (out / "metadata.json").write_text(json.dumps({
            "complete": True, "status": "complete", "shape": list(cache.shape),
            "input_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
            "source_ids": [r["id"] for r in rows]}))
        (out / "COMPLETE").write_text("ok\n")
    selection = tmp_path / "layer_selection.json"
    selection.write_text(json.dumps({"model_id": "synthetic", "selected_cache_index": 1, "selected_block_number": 2}))
    output = tmp_path / "x7.json"
    subprocess.run([sys.executable, "-m", "study_truth_transport_v2.runners.analyze_transfer_predictors",
                    "--model-id", "synthetic", "--prompt-root", str(prompt_root),
                    "--cache-root", str(tmp_path / "cache"), "--layer-selection", str(selection),
                    "--allocation-root", str(study / "data" / "rq_a_allocations"),
                    "--output", str(output), "--bootstrap-draws", "5", "--with-x1-methods"],
                   check=True, cwd=study.parent, capture_output=True)
    result = json.loads(output.read_text())
    assert result["n_configurations"] == 3 * 30
    assert sum(c["frozen_block"] for c in result["configurations"]) == 30
    assert np.isfinite(result["h1_delta_rho_p2_minus_p1"]["observed"])
    assert result["overlap_secondary"]["overlap_levels"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert set(result["x1_methods_secondary"]["method_means"]) == {"raw", "rosh_fixed_layer", "rosh_scrambled", "ridge", "pca1"}
    # Signal grows with block, so transfer AUROC should too (sanity of the plumbing).
    mean_auc = [np.mean([c["auroc"] for c in result["configurations"] if c["block_number"] == b]) for b in (1, 2, 3)]
    assert mean_auc[0] < mean_auc[1] < mean_auc[2]
