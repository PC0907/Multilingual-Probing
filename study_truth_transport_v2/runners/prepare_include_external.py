#!/usr/bin/env python3
"""Build native-authored true/false candidate statements from INCLUDE (X6)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .prepare_mmmlu_external import STATEMENTS, sha256

FOLDERS = {"ar": "Arabic", "de": "German", "hi": "Hindi", "fr": "French", "es": "Spanish"}
OPTIONS = ["option_a", "option_b", "option_c", "option_d"]
REVISION = "d2e1f6015f67a43c02a9a68db98e2298e2d6a660"


def wrong_index(lang: str, row_index: int, correct: int, seed: int) -> int:
    wrong = [value for value in range(4) if value != correct]
    digest = hashlib.sha256(f"include-wrong-v1|{seed}|{lang}|{row_index}".encode()).digest()
    return wrong[int.from_bytes(digest[:8], "big") % len(wrong)]


def eligible_rows(frame: pd.DataFrame) -> tuple[list[int], dict]:
    """Label-blind exclusions in file order: non-distinct options, repeated questions."""
    keep, seen = [], set()
    excluded = {"non_distinct_options": [], "repeated_question": []}
    for index, row in frame.iterrows():
        options = [str(row[name]).strip() for name in OPTIONS]
        if len(set(options)) < 4:
            excluded["non_distinct_options"].append(int(index))
            continue
        question = str(row["question"]).strip()
        if question in seen:
            excluded["repeated_question"].append(int(index))
            continue
        seen.add(question)
        keep.append(int(index))
    return keep, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-root", required=True, type=Path)
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Refusing existing output: {args.output_dir}")
    prompts = json.loads(args.prompt_config.read_text(encoding="utf-8"))["languages"]
    args.output_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "dataset": "CohereLabs/include-base-44", "revision": REVISION,
                "split": "test", "selection_seed": args.seed,
                "negative_choice_rule": "one of three incorrect choices selected by fixed SHA-256 hash",
                "protocol": "protocol/X6_NATIVE_INCLUDE_PREREGISTRATION.md", "languages": {}}
    for lang, folder in FOLDERS.items():
        source = next((args.include_root / folder).glob("test-*.parquet"))
        frame = pd.read_parquet(source).reset_index(drop=True)
        keep, excluded = eligible_rows(frame)
        records = []
        for index in keep:
            row = frame.loc[index]
            correct = int(str(row["answer"]))
            if correct not in range(4):
                raise ValueError(f"Bad answer index at {lang}:{index}")
            choices = [str(row[name]) for name in OPTIONS]
            question = str(row["question"])
            group_id = f"include-{lang}-{index:05d}"
            for label, choice, suffix in ((1, correct, "correct"),
                                          (0, wrong_index(lang, index, correct, args.seed), "incorrect")):
                raw = STATEMENTS[lang].format(question=question, answer=choices[choice])
                records.append({
                    "id": f"{group_id}-{suffix}", "group_id": group_id, "source_row": index,
                    "subject": str(row["subject"]), "regional_feature": str(row["regional_feature"]),
                    "country": str(row["country"]), "level": str(row["level"]), "domain": str(row["domain"]),
                    "label": label, "choice_index": choice, "correct_choice_index": correct,
                    "question": question, "answer_text": choices[choice],
                    "raw_statement": raw, "sentence": prompts[lang]["activation_prefix"] + raw,
                    "provenance": "CohereLabs/include-base-44 native-language exam question",
                })
        path = args.output_dir / f"{lang}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest["languages"][lang] = {"source_file": source.name, "source_sha256": sha256(source),
                                       "rows": len(frame), "questions": len(keep), "excluded": excluded,
                                       "file_sha256": sha256(path)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({lang: item["questions"] for lang, item in manifest["languages"].items()}))


if __name__ == "__main__":
    main()
