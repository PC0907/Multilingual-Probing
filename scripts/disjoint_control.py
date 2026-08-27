#!/usr/bin/env python
"""Week two: the content-disjoint control.

Run this before investing in more languages, more layers, or RQ3. It can
invalidate everything downstream, and that is much cheaper to learn now.

THE QUESTION
------------
Every cross-lingual probing result in the literature is computed on translated
data. If a probe trained on English statements transfers to their Hindi
translations, two explanations fit equally well:

    (a) the model has a shared, language-agnostic truth representation
    (b) the probe is reading lexical and syntactic residue that survived
        translation — the same facts, the same entity names, the same clause
        order, wearing a different script

These are indistinguishable on parallel data, because parallel data holds content
constant by design. The control separates them: derive the direction from English
set A, evaluate against target-language set B covering ENTIRELY DIFFERENT FACTS.
No shared entities, no shared objects. If alignment survives, (a). If it
collapses, the finding was (b) and the project changes shape.

WHAT THIS SCRIPT COMPARES
-------------------------
Three conditions, same model, same layers, same sample size:

    parallel      en set A  ->  target translations of set A   (what prior work does)
    disjoint      en set A  ->  target set B, different facts   (the control)
    within        en set A  ->  en set B                        (English-only
                               content-disjointness, to separate the cost of
                               changing CONTENT from the cost of changing LANGUAGE)

The third condition is the one that is usually missing and it is what makes the
result interpretable. Content-disjointness costs something even within a single
language, because a direction estimated on geography facts is not identical to
one estimated on institutional facts. If `within` drops by 30% and `disjoint`
drops by 35%, then almost the entire drop is a content effect and language
contributed almost nothing. Reporting `disjoint` alone would attribute the whole
drop to cross-lingual failure.

    language_cost = within_drop - disjoint_drop      (reported explicitly)

Usage::

    python scripts/05_disjoint_control.py \\
        --cache-parallel  en=.../en_A  hi=.../hi_A_translated \\
        --cache-disjoint  en=.../en_A  hi=.../hi_B \\
        --cache-within-en a=.../en_A   b=.../en_B \\
        --index ... --ceiling results/tables/ceiling_en.csv \\
        --n-per-language 500 --out results/tables/disjoint_control.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.extract import ActivationCache               # noqa: E402
from src.geometry.baselines import cosine, normalised_alignment   # noqa: E402
from src.probes.mass_mean import mass_mean_direction              # noqa: E402
from src.transfer.crosslingual import evaluate_transfer           # noqa: E402


def parse_mapping(pairs: list[str], flag: str) -> dict[str, Path]:
    mapping = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"{flag} expects key=path, got {item!r}")
        key, path = item.split("=", 1)
        mapping[key] = Path(path)
    return mapping


def load_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def assert_disjoint(left: pd.DataFrame, right: pd.DataFrame, name: str) -> dict:
    """Verify two sets really share no content.

    Checked on OBJECT entities, not just facts. Two statements about different
    cities in the same state are not content-disjoint in the sense that matters:
    the object entity is the token the probe most likely pools at, so a shared
    object leaks exactly the surface information the control exists to remove.

    Raises rather than warns. A control that silently is not a control produces a
    reassuring number and an unsupportable claim.
    """
    for frame, side in ((left, "left"), (right, "right")):
        if "object_qid" not in frame.columns:
            raise ValueError(
                f"{name}/{side}: no object_qid column, so disjointness cannot be "
                "verified. Regenerate the dataset with build_statements.to_frame."
            )

    shared_objects = set(left["object_qid"]) & set(right["object_qid"])
    shared_facts = set(left["group_id"]) & set(right["group_id"])

    if shared_objects or shared_facts:
        raise ValueError(
            f"{name} is not content-disjoint: {len(shared_objects)} shared object "
            f"entities, {len(shared_facts)} shared facts. Examples: "
            f"{sorted(shared_objects)[:5]}. Use "
            "build_statements.content_disjoint_split to partition by object."
        )

    return {
        "n_left": len(left),
        "n_right": len(right),
        "objects_left": left["object_qid"].nunique(),
        "objects_right": right["object_qid"].nunique(),
        "shared_objects": 0,
    }


def balanced_rows(index: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = n // 2
    out = []
    for label in (0, 1):
        rows = index.index[index["label"] == label].to_numpy()
        if len(rows) < per_class:
            raise ValueError(
                f"need {per_class} per class, have {len(rows)} for label={label}"
            )
        out.append(rng.choice(rows, per_class, replace=False))
    return np.concatenate(out)


def measure_condition(
    source_cache: ActivationCache,
    source_index: pd.DataFrame,
    target_cache: ActivationCache,
    target_index: pd.DataFrame,
    layers: list[int],
    ceiling: pd.DataFrame,
    n_per_language: int,
    seed: int,
) -> pd.DataFrame:
    """Alignment and transfer for one source/target pairing, per layer."""
    source_cache.config.assert_compatible(target_cache.config)

    source_rows = balanced_rows(source_index, n_per_language, seed)
    target_rows = balanced_rows(target_index, n_per_language, seed)
    source_labels = source_index.loc[source_rows, "label"].to_numpy()
    target_labels = target_index.loc[target_rows, "label"].to_numpy()

    rows = []
    for layer in layers:
        source_acts = np.asarray(
            source_cache.load_layer(layer, mmap=True), dtype=np.float64
        )[source_rows]
        target_acts = np.asarray(
            target_cache.load_layer(layer, mmap=True), dtype=np.float64
        )[target_rows]

        raw = cosine(
            mass_mean_direction(source_acts, source_labels),
            mass_mean_direction(target_acts, target_labels),
        )
        floor = float(ceiling.loc[layer, "floor"])
        ceil = float(ceiling.loc[layer, "ceiling"])
        try:
            normalised = normalised_alignment(raw, floor, ceil)
        except ValueError:
            normalised = np.nan

        transfer = evaluate_transfer(
            source_acts, source_labels, target_acts, target_labels,
            source="source", target="target", layer=layer, whiten=True,
        )
        rows.append({
            "layer": layer,
            "cosine_raw": raw,
            "cosine_normalised": normalised,
            "transfer_auroc": transfer.auroc,
            "calibration_gap": transfer.calibration_gap,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-parallel", nargs=2, required=True,
                        metavar="SRC=DIR", help="en=DIR target=DIR (translations)")
    parser.add_argument("--cache-disjoint", nargs=2, required=True,
                        metavar="SRC=DIR", help="en=DIR target=DIR (different facts)")
    parser.add_argument("--cache-within-en", nargs=2, default=None,
                        metavar="A=DIR", help="en set A = DIR, en set B = DIR")
    parser.add_argument("--index", nargs="+", required=True, metavar="KEY=PATH")
    parser.add_argument("--ceiling", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-language", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    indices = {k: load_index(v) for k, v in parse_mapping(args.index, "--index").items()}
    ceiling = pd.read_csv(args.ceiling).set_index("layer")
    layers = [int(l) for l in ceiling.index[ceiling["usable"]]]
    if not layers:
        raise SystemExit("no usable layers in the ceiling table")

    meta = args.ceiling.with_suffix(".meta.json")
    if meta.exists():
        built = json.loads(meta.read_text()).get("n_per_split")
        if built is not None and built != args.n_per_language:
            raise SystemExit(
                f"ceiling built at n={built}, run uses n={args.n_per_language}; "
                "normalised alignment would be systematically wrong"
            )

    conditions = {
        "parallel": parse_mapping(args.cache_parallel, "--cache-parallel"),
        "disjoint": parse_mapping(args.cache_disjoint, "--cache-disjoint"),
    }
    if args.cache_within_en:
        conditions["within"] = parse_mapping(args.cache_within_en, "--cache-within-en")

    frames = []
    for name, mapping in conditions.items():
        source_key, target_key = list(mapping)
        for key in (source_key, target_key):
            if key not in indices:
                raise SystemExit(f"--index is missing an entry for {key!r}")

        if name in ("disjoint", "within"):
            stats = assert_disjoint(
                indices[source_key], indices[target_key], name
            )
            print(f"{name}: verified disjoint — {stats['objects_left']} vs "
                  f"{stats['objects_right']} distinct objects, no overlap")

        frame = measure_condition(
            ActivationCache(mapping[source_key]), indices[source_key],
            ActivationCache(mapping[target_key]), indices[target_key],
            layers, ceiling, args.n_per_language, args.seed,
        )
        frame.insert(0, "condition", name)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    peak = result.loc[result.groupby("condition")["cosine_normalised"].idxmax()]
    print()
    print(peak[["condition", "layer", "cosine_normalised",
                "transfer_auroc"]].round(3).to_string(index=False))

    values = peak.set_index("condition")["cosine_normalised"]
    parallel, disjoint = values["parallel"], values["disjoint"]
    content_and_language_drop = (parallel - disjoint) / parallel if parallel else np.nan

    print()
    print(f"parallel -> disjoint: {content_and_language_drop:+.1%} change in "
          "normalised alignment")

    if "within" in values:
        within = values["within"]
        content_only_drop = (parallel - within) / parallel if parallel else np.nan
        language_cost = content_and_language_drop - content_only_drop
        print(f"  of which content alone (English-only, disjoint facts): "
              f"{content_only_drop:+.1%}")
        print(f"  attributable to LANGUAGE: {language_cost:+.1%}")
        if abs(language_cost) < 0.05:
            print("  -> changing content costs as much within English as across "
                  "languages. The cross-lingual drop is a content effect, and "
                  "reporting it as a language effect would be wrong.")
    else:
        print("  NOTE: no --cache-within-en given, so this drop confounds the cost "
              "of changing CONTENT with the cost of changing LANGUAGE. Run the "
              "English-only disjoint condition before interpreting it.")

    print()
    if disjoint < 0.2 * parallel:
        print("ALIGNMENT COLLAPSES UNDER THE CONTROL. The parallel-data result was "
              "largely translation residue. This is a publishable finding about "
              "the literature's methodology, but RQ1 as framed no longer has the "
              "phenomenon it assumes — reshape the project now, not in week eight.")
    elif disjoint < 0.6 * parallel:
        print("Alignment is substantially reduced but survives. Report both "
              "conditions everywhere; the parallel number alone overstates the "
              "effect, which is itself the contribution against prior work.")
    else:
        print("Alignment survives content-disjointness. The shared-representation "
              "reading holds, and unlike prior work this is now supported by a "
              "control rather than assumed.")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
