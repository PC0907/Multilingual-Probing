#!/usr/bin/env python3
"""Freeze pair-specific metadata and exact overlap allocations for RQ-A."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from study_truth_transport_v2.src.allocation import GroupRecord, build_overlap_plan
from study_truth_transport_v2.src.data import LANGUAGES, load_parallel_dataset


NEGATION = {
    "en": re.compile(r"\b(?:not|no|never|neither|nor|without|nobody|nothing|none|cannot)\b|n['’]t\b", re.I),
    "de": re.compile(r"\b(?:nicht|nie|niemals|niemand|nichts|weder|ohne|kein\w*)\b", re.I),
    "ar": re.compile(r"(?:^|\s)(?:لا|ليس|ليست|لم|لن|بدون|دون|غير)(?:\s|$)"),
    "hi": re.compile(r"(?:^|\s)(?:नहीं|नही|मत|बिना|कभी नहीं|न तो)(?:\s|$)"),
    "fr": re.compile(r"\b(?:ne|n['’]|pas|jamais|aucun\w*|personne|rien|sans|ni)\b", re.I),
    "es": re.compile(r"\b(?:no|nunca|jamás|ningún|ninguna|nadie|nada|sin|ni)\b", re.I),
}
NUMBER = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|first|second|third|quarter|half)\b",
    re.I,
)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    dataset = ROOT / "generic_claims_2000_v2"
    study = ROOT / "study_truth_transport_v2"
    rows = load_parallel_dataset(dataset)
    risk_rows = json.loads((ROOT / "generic_claims_2000" / "quality_audit" / "automatic_risk_rows.json").read_text())
    risk = {(x["language"], x["id"]) for x in risk_rows}
    by_language_id = {language: {r["id"]: r for r in rows[language]} for language in LANGUAGES}
    members = defaultdict(list)
    for row in rows["en"]:
        members[row["group_id"]].append(row["id"])

    metadata = []
    mixed_groups = []
    for group_id, ids in sorted(members.items()):
        labels = {int(by_language_id["en"][claim_id]["label"]) for claim_id in ids}
        partition = by_language_id["en"][ids[0]]["partition"]
        if len(labels) != 1:
            mixed_groups.append({"group_id": group_id, "ids": ids, "labels": sorted(labels), "partition": partition})
            continue
        language = {}
        for lang in LANGUAGES:
            texts = [by_language_id[lang][claim_id]["sentence"] for claim_id in ids]
            language[lang] = {
                "mean_characters": float(np.mean([len(x) for x in texts])),
                "mean_whitespace_tokens": float(np.mean([len(x.split()) for x in texts])),
                "has_negation": any(bool(NEGATION[lang].search(x)) for x in texts),
                "translation_risk": lang != "en" and any((lang, claim_id) in risk for claim_id in ids),
                "researcher_corrected": lang != "en" and any(
                    by_language_id[lang][claim_id].get("translation_revision") == "v2_researcher_screened" for claim_id in ids
                ),
            }
        english_texts = [by_language_id["en"][claim_id]["sentence"] for claim_id in ids]
        metadata.append({
            "group_id": group_id,
            "member_ids": ids,
            "group_size": len(ids),
            "label": labels.pop(),
            "partition": partition,
            "contains_number": any(bool(NUMBER.search(x)) for x in english_texts),
            "topic": None,
            "topic_status": "unavailable_in_supplied_dataset",
            "language": language,
        })
    dump(study / "data" / "derived_group_metadata.json", {
        "dataset_manifest_sha256": sha256(dataset / "data" / "manifest.json"),
        "unit": "dependency group; same-label paraphrase members averaged at activation analysis",
        "n_pure_label_groups": len(metadata),
        "mixed_label_groups_excluded_from_probe_fits": mixed_groups,
        "records": metadata,
    })

    training = [x for x in metadata if x["partition"] == "train"]
    plans = {}
    for source, target in combinations(LANGUAGES, 2):
        source_length = np.asarray([x["language"][source]["mean_characters"] for x in training])
        target_length = np.asarray([x["language"][target]["mean_characters"] for x in training])
        zs = (source_length - source_length.mean()) / source_length.std(ddof=1)
        zt = (target_length - target_length.mean()) / target_length.std(ddof=1)
        pair_length = (zs + zt) / 2
        cuts = np.quantile(pair_length, [0.25, 0.5, 0.75])
        records = []
        group_annotations = {}
        for i, group in enumerate(training):
            length_bin = int(np.searchsorted(cuts, pair_length[i], side="right"))
            either_negation = group["language"][source]["has_negation"] or group["language"][target]["has_negation"]
            either_risk = group["language"][source]["translation_risk"] or group["language"][target]["translation_risk"]
            stratum = f"len{length_bin}|neg{int(either_negation)}|num{int(group['contains_number'])}|risk{int(either_risk)}"
            records.append(GroupRecord(group["group_id"], int(group["label"]), stratum))
            group_annotations[group["group_id"]] = {
                "length_bin": length_bin,
                "either_language_negation": either_negation,
                "contains_number": group["contains_number"],
                "either_language_translation_risk": either_risk,
                "stratum": stratum,
            }
        plan = build_overlap_plan(records, repetitions=50, seed=20261002)
        payload = {
            "language_pair": [source, target],
            "unordered_geometry_pair": True,
            "topic_matching": False,
            "topic_matching_reason": "No documented topic field exists in the supplied dataset.",
            "length_definition": "quartile of the mean of within-language z-scored group-mean character lengths",
            "negation_definition": "present in either language surface using frozen lexical patterns",
            "number_definition": "digit or English number word in the English source surface",
            "translation_risk_definition": "v1 automatic QA flag in either language; corrected rows remain conservatively flagged",
            "length_bin_cuts": cuts.tolist(),
            "group_annotations": group_annotations,
            **plan,
        }
        path = study / "data" / "rq_a_allocations" / f"{source}__{target}.json"
        dump(path, payload)
        plans[f"{source}__{target}"] = {
            "path": str(path.relative_to(study)),
            "sha256": sha256(path),
            "eligible_groups": payload["eligible_groups"],
            "excluded_groups": len(payload["excluded_groups"]),
            "n_strata": len(payload["fit_composition"]),
        }
    dump(study / "data" / "rq_a_allocation_manifest.json", {
        "complete": True,
        "dataset": "generic_claims_2000_v2",
        "dataset_manifest_sha256": sha256(dataset / "data" / "manifest.json"),
        "n_language_pairs": len(plans),
        "n_plans_per_pair": 250,
        "pairs": plans,
    })
    print(json.dumps({"complete": True, "pure_groups": len(metadata), "mixed_excluded": len(mixed_groups), "pairs": len(plans)}))


if __name__ == "__main__":
    main()
