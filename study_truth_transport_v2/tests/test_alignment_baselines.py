import numpy as np

from study_truth_transport_v2.runners.analyze_alignment_baselines import (
    derangement,
    holm_adjust,
    paired_macro_bootstrap,
    ridge_score_vector,
    top_difference_pc,
)


def test_derangement_has_no_fixed_points_and_is_permutation():
    rng = np.random.default_rng(7)
    order = derangement(50, rng)
    assert sorted(order.tolist()) == list(range(50))
    assert np.all(order != np.arange(50))


def test_ridge_score_vector_matches_explicit_primal_map():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(30, 6))
    y = rng.normal(size=(30, 6))
    w = rng.normal(size=6)
    fraction = 0.01
    gram = x @ x.T
    lam = fraction * np.trace(gram) / len(gram)
    expected = np.linalg.solve(x.T @ x + lam * np.eye(6), x.T @ y) @ w
    observed = ridge_score_vector(x, y, w, fraction)
    assert np.allclose(observed, expected, atol=1e-8)


def test_top_difference_pc_recovers_rank_one_direction():
    rng = np.random.default_rng(9)
    v = rng.normal(size=12)
    v /= np.linalg.norm(v)
    coefficient = rng.normal(size=(80, 1))
    a = coefficient * v + 0.001 * rng.normal(size=(80, 12))
    b = np.zeros_like(a)
    recovered = top_difference_pc(a, b)
    assert abs(float(recovered @ v)) > 0.999


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"small": 0.01, "large": 0.04})
    assert adjusted == {"small": 0.02, "large": 0.04}


def test_paired_macro_bootstrap_reports_all_outcomes():
    rng = np.random.default_rng(10)
    y = np.asarray([0] * 20 + [1] * 20)
    records = []
    for _ in range(3):
        raw = rng.normal(size=len(y)) + 0.2 * y
        records.append({
            "ids": [f"g{i}" for i in range(len(y))],
            "y": y,
            "scores": {"raw_source_mass_mean": raw, "candidate": raw + 0.5 * y},
        })
    result = paired_macro_bootstrap(records, draws=50, seed=11)
    assert set(result["candidate"]) == {
        "auroc", "balanced_accuracy", "standardized_separation"}
    assert result["candidate"]["auroc"]["mean_change"] >= 0
    assert "p_holm" in result["candidate"]["balanced_accuracy"]
