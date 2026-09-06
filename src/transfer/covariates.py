"""RQ2 covariates: what might predict transfer, and how much to trust each one.

Computes the per-language predictors the transfer regression uses. The
computation is easy. The honest reporting is not, so most of this module is
about the second part.

THE STATISTICAL SITUATION, STATED PLAINLY
------------------------------------------
Five collinear regressors over five or six languages is not a regression anyone
should believe. With six languages there are fifteen pairs, and pairs are not
independent observations — they share languages, so the effective sample size is
closer to six than fifteen. Any p-value computed here is decorative.

This module therefore refuses to be a regression fitter. It produces the
covariate table and a collinearity report, and `describe_relationships` returns
rank correlations with explicit language counts attached. The framing in the
paper must be descriptive correlation with leave-one-family-out
cross-validation, not causal inference, and the limitation belongs in the text
rather than in a reviewer's response.

The one comparison that IS interpretable with this design is the planned
contrast: Hindi and Urdu are near-identical in typology and differ in script, so
a transfer difference between them isolates script and tokenization in a way no
regression over six languages can. Prefer reporting that contrast over a fitted
coefficient. `planned_contrasts` computes it.

FERTILITY IS PER (LANGUAGE, MODEL)
-----------------------------------
Tokenizer fertility is not a property of a language. It depends on the
tokenizer, so it must be recomputed for every model in the study, over a
parallel corpus so that content is held constant. Measuring fertility on each
language's own dataset would confound fragmentation with topic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "LanguageCovariates",
    "compute_fertility",
    "typological_distance",
    "build_covariate_table",
    "collinearity_report",
    "describe_relationships",
    "planned_contrasts",
]


@dataclass
class LanguageCovariates:
    """Per-language predictors. Every field carries a known weakness."""

    language: str
    fertility: float | None = None          # tokens per word, relative to English
    typological_distance: float | None = None
    script: str = ""
    resource_tier: int | None = None
    pretraining_share_proxy: float | None = None
    in_language_auroc: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def compute_fertility(
    texts_by_language: dict[str, list[str]],
    tokenizer,
    reference: str = "en",
) -> dict[str, float]:
    """Tokens per whitespace-word, normalised so the reference language is 1.0.

    Args:
        texts_by_language: PARALLEL text — the same content in every language.
            FLORES-200 devtest is the standard choice. Using each language's own
            dataset instead would mix fragmentation with topic and sentence
            length, and the resulting number would not be a tokenizer property.
        tokenizer: the model's tokenizer. Fertility differs between models, so
            this must be recomputed per model rather than looked up.
        reference: language whose fertility is defined as 1.0.

    Whitespace word counts are a poor unit for languages that do not delimit
    words with spaces (Chinese, Japanese, Thai). For the Indic languages and
    Spanish in this study it is adequate; if the language set expands, switch to
    characters per token and say so.
    """
    if reference not in texts_by_language:
        raise ValueError(f"reference language {reference!r} not in the corpus")

    lengths = {}
    for language, texts in texts_by_language.items():
        if not texts:
            raise ValueError(f"{language}: empty corpus")
        n_tokens = sum(
            len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        )
        n_words = sum(len(text.split()) for text in texts)
        if n_words == 0:
            raise ValueError(
                f"{language}: no whitespace-delimited words. Use a "
                "characters-per-token measure for this script."
            )
        lengths[language] = n_tokens / n_words

    base = lengths[reference]
    return {language: value / base for language, value in lengths.items()}


def typological_distance(
    language_a: str,
    language_b: str,
    feature_set: str = "syntax_knn",
) -> float:
    """Distance from lang2vec / URIEL.

    Kept as a thin wrapper so the feature set is an explicit, recorded choice.
    Results shift between `syntax_knn`, `genetic` and `geographic`, and picking
    whichever gives the cleanest correlation after seeing the data is exactly the
    thing that makes a descriptive analysis dishonest. Fix the feature set before
    looking at transfer, and report the others in an appendix.

    URIEL coverage is uneven and many entries are imputed from related languages,
    so distances involving lower-resource languages are less reliable than their
    precision suggests.
    """
    try:
        import lang2vec.lang2vec as l2v
    except ImportError as exc:
        raise ImportError(
            "pip install lang2vec. If it is unavailable, substitute a documented "
            "hand-coded distance and say so — do not silently drop the regressor."
        ) from exc

    codes = {"en": "eng", "hi": "hin", "ur": "urd", "ta": "tam",
             "es": "spa", "bn": "ben", "mr": "mar", "ar": "arb"}
    iso_a = codes.get(language_a, language_a)
    iso_b = codes.get(language_b, language_b)

    distances = l2v.distance(feature_set, iso_a, iso_b)
    return float(distances if np.isscalar(distances) else distances[0][0])


def build_covariate_table(
    covariates: list[LanguageCovariates],
    transfer: pd.DataFrame,
    feature_set: str = "syntax_knn",
) -> pd.DataFrame:
    """One row per ordered language pair, with pair-level and target-level features.

    Distinguishing the two kinds matters. Typological distance is a property of
    the PAIR; fertility and resource tier are properties of the TARGET. Mixing
    them into one undifferentiated feature vector makes the resulting
    coefficients uninterpretable, because a target-level feature is repeated
    across every pair sharing that target and is therefore heavily
    pseudo-replicated.
    """
    by_language = {c.language: c for c in covariates}
    rows = []

    for _, record in transfer.iterrows():
        source, target = record["source"], record["target"]
        if source == target or source not in by_language or target not in by_language:
            continue

        target_covariates = by_language[target]
        try:
            distance = typological_distance(source, target, feature_set)
        except Exception:
            distance = np.nan

        rows.append({
            "source": source,
            "target": target,
            "auroc": record.get("auroc"),
            "relative_transfer": record.get("relative_transfer"),
            "calibration_gap": record.get("calibration_gap"),
            # pair-level
            "typological_distance": distance,
            "same_script": int(
                by_language[source].script == target_covariates.script
            ),
            # target-level (pseudo-replicated across pairs sharing this target)
            "target_fertility": target_covariates.fertility,
            "target_resource_tier": target_covariates.resource_tier,
            "target_pretraining_proxy": target_covariates.pretraining_share_proxy,
            "target_in_language_auroc": target_covariates.in_language_auroc,
        })
    return pd.DataFrame(rows)


def collinearity_report(table: pd.DataFrame, threshold: float = 0.7) -> dict:
    """Which regressors are too correlated to separate.

    Fertility and resource tier are expected to be strongly related: low-resource
    languages are both badly tokenised and rare in pretraining. When they are, no
    amount of regression will attribute effect between them, and claiming
    otherwise is the most likely error in this analysis.
    """
    numeric = table.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ("auroc", "relative_transfer", "calibration_gap")
                 if c in table.columns],
        errors="ignore",
    )
    correlation = numeric.corr(method="spearman")

    entangled = [
        (a, b, float(correlation.loc[a, b]))
        for a, b in itertools.combinations(correlation.columns, 2)
        if abs(correlation.loc[a, b]) >= threshold
    ]
    return {
        "correlation_matrix": correlation,
        "entangled_pairs": entangled,
        "message": (
            "no pair above threshold" if not entangled
            else "; ".join(
                f"{a} and {b} correlate at {r:+.2f} — their separate effects are "
                "not identifiable with this language set"
                for a, b, r in entangled
            )
        ),
    }


def describe_relationships(
    table: pd.DataFrame,
    outcome: str = "relative_transfer",
) -> pd.DataFrame:
    """Rank correlation of each covariate with transfer, with sample sizes shown.

    Spearman rather than Pearson because none of these relationships is expected
    to be linear and the sample is far too small to detect the functional form.
    No p-values: pairs share languages, so they are not independent observations
    and any nominal significance would be overstated.
    """
    predictors = [
        c for c in table.select_dtypes(include=[np.number]).columns if c != outcome
    ]
    rows = []
    for predictor in predictors:
        subset = table[[predictor, outcome]].dropna()
        rows.append({
            "covariate": predictor,
            "spearman_rho": (
                float(subset[predictor].corr(subset[outcome], method="spearman"))
                if len(subset) >= 3 else np.nan
            ),
            "n_pairs": len(subset),
            "n_distinct_targets": table.loc[subset.index, "target"].nunique(),
        })
    frame = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    frame.attrs["caution"] = (
        "Effective sample size is the number of distinct languages, not pairs. "
        "Report as descriptive correlation only."
    )
    return frame


def planned_contrasts(transfer: pd.DataFrame) -> pd.DataFrame:
    """The comparisons this language set was actually designed to support.

    Each contrast holds most factors fixed and varies one, which is what makes it
    interpretable where a six-language regression is not:

        hi vs ur   same family and word order, different script
                   -> isolates script and tokenization
        hi vs ta   both Indian, similar resource level, different family
                   -> isolates typological distance
        es vs hi   both Indo-European, very different pretraining share
                   -> isolates resource level

    Reported as the difference in transfer from English into each member of the
    pair. A difference here is evidence; a regression coefficient over six
    languages mostly is not.
    """
    contrasts = [
        ("script", "hi", "ur", "same family and word order, different script"),
        ("family", "hi", "ta", "similar region and resource level, different family"),
        ("resource", "es", "hi", "both Indo-European, different pretraining share"),
    ]

    from_english = transfer[transfer["source"] == "en"].set_index("target")
    rows = []
    for name, left, right, rationale in contrasts:
        if left not in from_english.index or right not in from_english.index:
            continue
        left_value = float(from_english.loc[left, "auroc"])
        right_value = float(from_english.loc[right, "auroc"])
        rows.append({
            "contrast": name,
            "language_a": left,
            "language_b": right,
            "auroc_a": left_value,
            "auroc_b": right_value,
            "difference": left_value - right_value,
            "isolates": rationale,
        })
    return pd.DataFrame(rows)
