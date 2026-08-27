"""Can this pipeline SEE a dissociation, or only assert one?

The paper's claim is that geometric alignment need not predict probe transfer. If
the measurements are run on real models and the two turn out to correlate, that
result is only meaningful if the pipeline was capable of showing otherwise. A null
result from an instrument that cannot detect the effect is not evidence of
absence — it is no evidence at all.

So this file constructs activation spaces where the answer is known by
construction, and asserts the pipeline recovers it. Run before touching a real
model. If `test_low_alignment_high_transfer` fails, no null result from
scripts/04_transfer.py can be interpreted.

THE MECHANISM
-------------
A dissociation is not exotic. It follows from the raw and whitened directions
answering different questions in an anisotropic space.

Suppose in language B the difference between true and false activations is::

    mu_true - mu_false  =  0.5 * t  +  4.0 * v

where `t` is the truth axis shared with language A, and `v` is a
language-specific direction that ALSO happens to carry enormous variance in B —
a "this is language B" direction, say, that shifts slightly between the classes.

Then:

  * The RAW mass-mean direction in B is dominated by the 4.0 * v term, because it
    is eight times larger. Its cosine with A's direction (which is close to `t`)
    is low. RQ1 reports weak alignment, and correctly so: the directions really
    do point different ways.

  * The WHITENED direction divides each component by its variance. `v` has
    variance ~100, so its contribution shrinks by 100x to 0.04; `t` has variance
    ~1 and survives at 0.5. The whitened direction in B is therefore close to
    `t`, as is A's — so A's probe transfers to B well. RQ2 reports strong
    transfer.

Both measurements are correct. They disagree because "which way does the class
difference point" and "which way best separates the classes" are different
questions whenever the space is anisotropic. That is the whole content of the
dissociation, and it is why RQ1 must use raw directions and RQ2 whitened ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.baselines import cosine                      # noqa: E402
from src.probes.mass_mean import mass_mean_direction           # noqa: E402
from src.transfer.crosslingual import evaluate_transfer        # noqa: E402


DIM = 256
N = 4000


def _orthonormal_pair(rng: np.random.Generator, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Two orthogonal unit vectors, so the mechanism is not an artefact of overlap."""
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(dim)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    return a, b


def make_dissociated_pair(
    truth_weight_b: float = 0.5,
    nuisance_weight_b: float = 4.0,
    nuisance_std: float = 10.0,
    seed: int = 0,
) -> dict:
    """Language A with a clean truth axis; language B with the same axis buried.

    B's class difference is mostly along a high-variance nuisance direction.
    Raw geometry should therefore show the languages misaligned, while whitened
    probes should transfer.
    """
    rng = np.random.default_rng(seed)
    truth_axis, nuisance_axis = _orthonormal_pair(rng, DIM)

    y_a = rng.integers(0, 2, N)
    acts_a = rng.standard_normal((N, DIM)) + np.outer(2 * y_a - 1, truth_axis) * 0.5

    y_b = rng.integers(0, 2, N)
    acts_b = (
        rng.standard_normal((N, DIM))
        + np.outer(rng.standard_normal(N) * nuisance_std, nuisance_axis)
        + np.outer(2 * y_b - 1, truth_axis) * truth_weight_b
        + np.outer(2 * y_b - 1, nuisance_axis) * nuisance_weight_b
    )
    return {
        "acts_a": acts_a, "y_a": y_a,
        "acts_b": acts_b, "y_b": y_b,
        "truth_axis": truth_axis, "nuisance_axis": nuisance_axis,
    }


def make_concordant_pair(seed: int = 1) -> dict:
    """Control: two languages sharing an axis in an isotropic space.

    Both alignment and transfer should be high. Without this the dissociation
    test is unfalsifiable — a pipeline that always reports low alignment and high
    transfer would pass the first test for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    truth_axis = rng.standard_normal(DIM)
    truth_axis /= np.linalg.norm(truth_axis)

    out = {}
    for key in ("a", "b"):
        y = rng.integers(0, 2, N)
        out[f"acts_{key}"] = (
            rng.standard_normal((N, DIM)) + np.outer(2 * y - 1, truth_axis) * 0.5
        )
        out[f"y_{key}"] = y
    return out


def _measure(fixture: dict) -> dict:
    """Run both measurements exactly as the drivers do."""
    raw_a = mass_mean_direction(fixture["acts_a"], fixture["y_a"])
    raw_b = mass_mean_direction(fixture["acts_b"], fixture["y_b"])

    half = N // 2
    transfer = evaluate_transfer(
        source_acts=fixture["acts_a"][:half],
        source_labels=fixture["y_a"][:half],
        target_acts=fixture["acts_b"][half:],
        target_labels=fixture["y_b"][half:],
        source="a", target="b", layer=0, whiten=True,
    )
    unwhitened = evaluate_transfer(
        source_acts=fixture["acts_a"][:half],
        source_labels=fixture["y_a"][:half],
        target_acts=fixture["acts_b"][half:],
        target_labels=fixture["y_b"][half:],
        source="a", target="b", layer=0, whiten=False,
    )
    return {
        "alignment": abs(cosine(raw_a, raw_b)),
        "transfer_auroc": transfer.auroc,
        "transfer_auroc_unwhitened": unwhitened.auroc,
    }


def test_low_alignment_high_transfer() -> dict:
    """The case the paper is about. Raw directions diverge; probes still transfer."""
    result = _measure(make_dissociated_pair())
    assert result["alignment"] < 0.25, (
        f"expected weak raw alignment, got {result['alignment']:.3f} — the "
        "nuisance term is not dominating the raw direction, so this fixture does "
        "not construct a dissociation"
    )
    assert result["transfer_auroc"] > 0.65, (
        f"expected strong whitened transfer, got {result['transfer_auroc']:.3f} — "
        "the pipeline CANNOT detect a dissociation, so a null result from "
        "04_transfer.py would be uninterpretable"
    )
    return result


def test_concordant_control() -> dict:
    """Control: shared axis, isotropic space. Both measurements should be high."""
    result = _measure(make_concordant_pair())
    assert result["alignment"] > 0.5, (
        f"control alignment is {result['alignment']:.3f}; the geometry "
        "measurement is broken independently of any dissociation"
    )
    assert result["transfer_auroc"] > 0.65, (
        f"control transfer is {result['transfer_auroc']:.3f}; the transfer "
        "measurement is broken"
    )
    return result


def test_dissociation_needs_whitening() -> dict:
    """Confirm WHERE the raw/whitened split does its work.

    A first attempt asserted that whitening improves A -> B transfer. It does not,
    and the reason is worth recording: A -> B transfer uses only language A's
    direction, and A's space is isotropic, so whitening leaves it essentially
    unchanged. The contamination lives in B's OWN direction.

    That is precisely the asymmetry the pipeline depends on:

        B's RAW direction      is pulled off the truth axis by the nuisance term
                               -> RQ1 correctly reports the languages misaligned
        B's WHITENED direction recovers the truth axis
                               -> RQ2 correctly reports that probes transfer

    Same activations, two directions, two valid answers. If whitening did NOT
    recover the truth axis in B, then the raw/whitened split would be decorative
    and the dissociation reported by the pipeline would be an artefact of using a
    contaminated direction in one place and not the other.
    """
    from src.probes.mass_mean import mass_mean_iid_direction

    fixture = make_dissociated_pair()
    truth_axis = fixture["truth_axis"]

    raw_b = mass_mean_direction(fixture["acts_b"], fixture["y_b"])
    whitened_b = mass_mean_iid_direction(fixture["acts_b"], fixture["y_b"])

    raw_recovery = abs(cosine(raw_b, truth_axis))
    whitened_recovery = abs(cosine(whitened_b, truth_axis))

    assert raw_recovery < 0.3, (
        f"B's raw direction already recovers the truth axis ({raw_recovery:.3f}); "
        "the nuisance term is not dominating and this fixture is not dissociated"
    )
    assert whitened_recovery > 0.6, (
        f"whitening fails to recover the truth axis in B ({whitened_recovery:.3f}); "
        "the raw/whitened split is not doing what the module docstrings claim, so "
        "the pipeline's dissociation would be an artefact of probe choice"
    )

    result = _measure(fixture)
    result["raw_recovery_b"] = raw_recovery
    result["whitened_recovery_b"] = whitened_recovery
    return result


if __name__ == "__main__":
    print(f"{'test':<34} {'align':>7} {'transfer':>9} {'unwhit':>8}")
    failures = 0
    for name, test in [
        ("low_alignment_high_transfer", test_low_alignment_high_transfer),
        ("concordant_control", test_concordant_control),
        ("dissociation_needs_whitening", test_dissociation_needs_whitening),
    ]:
        try:
            r = test()
            print(f"{name:<34} {r['alignment']:>7.3f} {r['transfer_auroc']:>9.3f} "
                  f"{r['transfer_auroc_unwhitened']:>8.3f}  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"{name:<34} {'':>7} {'':>9} {'':>8}  FAIL")
            print(f"    {exc}")

    print()
    if failures:
        print(f"{failures} test(s) failed. Do NOT interpret a null dissociation "
              "result until these pass.")
    else:
        print("The pipeline can distinguish a dissociation from a concordance. "
              "A null result on real data is now interpretable as evidence.")
    raise SystemExit(1 if failures else 0)
