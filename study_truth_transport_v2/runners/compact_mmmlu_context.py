#!/usr/bin/env python3
"""Fit frozen MMMLU candidate prompts into the frozen extraction context limit.

Rule (frozen in protocol/DEVIATIONS.md before any MMMLU outcome was inspected):
for each question, language, and model tokenizer, if either candidate prompt
exceeds ``min(512, max_position_embeddings)`` tokens (special tokens included),
remove whole whitespace-delimited words from the START of the shared question
text and prepend the marker ``"… "``.  The longest retained question suffix for
which BOTH candidate prompts fit is used for both candidates.  Answers, labels,
prefixes, templates, IDs, and groups are never changed.  If no suffix of at
least one word lets both candidates fit in some language, the question is
excluded for that model in all six languages so the question set stays
parallel across languages.  The rule is label-blind
(it uses only the maximum over the two candidates) and language-neutral (it uses
only whitespace and the model's own token count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .prepare_mmmlu_external import LANG_FILES, LETTERS, STATEMENTS, load_csv

FROZEN_LIMIT = 512
MARKER = "… "
LANGS = ("en", "de", "ar", "hi", "fr", "es")


def word_starts(question: str) -> list[int]:
    """Character offsets at which each whitespace-delimited word begins."""
    return [match.start() for match in re.finditer(r"\S+", question)]


def render(prefix: str, lang: str, question: str, answer: str) -> str:
    return prefix + STATEMENTS[lang].format(question=question, answer=answer)


def fit_question(count, prefix: str, lang: str, question: str, answers: list[str],
                 limit: int) -> tuple[str, int, int] | None:
    """Return (question_text, n_words, n_words_kept), or None if infeasible.

    ``count`` maps a full prompt to its token length with special tokens.
    ``answers`` holds both candidates' answers; their order is irrelevant.
    """
    def fits(text: str) -> bool:
        return max(count(render(prefix, lang, text, answer)) for answer in answers) <= limit

    starts = word_starts(question)
    if fits(question):
        return question, len(starts), len(starts)

    def suffix(kept: int) -> str:
        return MARKER + question[starts[len(starts) - kept]:]

    low, high = 0, len(starts) - 1  # largest fitting kept-word count in [1, n-1]
    while low < high:
        middle = (low + high + 1) // 2
        if fits(suffix(middle)):
            low = middle
        else:
            high = middle - 1
    if low < 1 or not fits(suffix(low)):
        return None  # answers alone exceed the limit; caller excludes the question
    return suffix(low), len(starts), low


def source_fields(english: pd.DataFrame | None, translated: dict, lang: str, row: dict) -> tuple[str, str]:
    if "question" in row and "answer_text" in row:  # self-contained records (X6 INCLUDE)
        return row["question"], row["answer_text"]
    index = int(row["source_row"])
    if lang == "en":
        question = str(english.loc[index, "question"])
        choices = [str(value) for value in english.loc[index, "choices"]]
    else:
        source = translated[lang][index]
        question = source["Question"]
        choices = [source[letter] for letter in LETTERS]
    return question, choices[int(row["choice_index"])]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--english", type=Path, help="MMLU parquet (X3 only)")
    parser.add_argument("--mmmlu-root", type=Path, help="MMMLU root (X3 only)")
    parser.add_argument("--languages", nargs="+", default=list(LANGS))
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Refusing existing output: {args.output_dir}")
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True,
                                        trust_remote_code=False)
    limit = min(FROZEN_LIMIT, int(config.max_position_embeddings))

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    english = pd.read_parquet(args.english).reset_index(drop=True) if args.english else None
    translated = ({lang: load_csv(args.mmmlu_root / "test" / name) for lang, name in LANG_FILES.items()}
                  if args.mmmlu_root else {})
    prefixes = json.loads(args.prompt_config.read_text(encoding="utf-8"))["languages"]
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "model_id": args.model_id, "rule": __doc__.split("\n\n", 1)[1].strip(),
        "context_limit_tokens": limit, "marker": MARKER, "tokenizer_class": type(tokenizer).__name__,
        "input_root": str(args.input_root), "languages": {},
    }
    loaded, fitted_by_lang = {}, {}
    for lang in args.languages:
        rows = json.loads((args.input_root / f"{lang}.json").read_text(encoding="utf-8"))
        prefix = prefixes[lang]["activation_prefix"]
        groups = defaultdict(list)
        for row in rows:
            question, answer = source_fields(english, translated, lang, row)
            if render(prefix, lang, question, answer) != row["sentence"]:
                raise ValueError(f"Cannot reproduce frozen sentence for {row['id']}")
            groups[row["group_id"]].append((row, question, answer))
        fitted = {}
        for group_id, members in groups.items():
            if len(members) != 2 or len({question for _, question, _ in members}) != 1:
                raise ValueError(f"Group {group_id} is not one question with two candidates")
            fitted[group_id] = fit_question(count, prefix, lang, members[0][1],
                                            [answer for _, _, answer in members], limit)
        loaded[lang], fitted_by_lang[lang] = rows, fitted
    excluded = sorted({group_id for fitted in fitted_by_lang.values()
                       for group_id, value in fitted.items() if value is None})
    manifest["excluded_infeasible_group_ids"] = excluded
    manifest["excluded_infeasible_by_language"] = {
        lang: sorted(g for g, value in fitted.items() if value is None)
        for lang, fitted in fitted_by_lang.items()}
    for lang in args.languages:
        rows, fitted, prefix = loaded[lang], fitted_by_lang[lang], prefixes[lang]["activation_prefix"]
        output, compacted, lengths = [], [], []
        for row in rows:  # preserve frozen row order
            if row["group_id"] in excluded:
                continue
            question, answer = source_fields(english, translated, lang, row)
            text, n_words, kept = fitted[row["group_id"]]
            raw = STATEMENTS[lang].format(question=text, answer=answer)
            sentence = prefix + raw
            length = count(sentence)
            if length > limit:
                raise AssertionError(f"Record still exceeds limit: {row['id']}")
            lengths.append(length)
            applied = kept < n_words
            output.append({**row, "raw_statement": raw, "sentence": sentence,
                           "compaction": {"applied": applied, "question_words": n_words,
                                          "question_words_kept": kept,
                                          "original_sentence_sha256": sha256_text(row["sentence"])}})
            if applied and row["group_id"] not in compacted:
                compacted.append(row["group_id"])
        path = args.output_dir / f"{lang}.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        manifest["languages"][lang] = {
            "n_records": len(output), "n_questions": len(output) // 2,
            "compacted_group_ids": sorted(compacted),
            "max_tokens_after": max(lengths),
            "min_kept_word_fraction": min(
                (fitted[g][2] / fitted[g][1] for g in compacted), default=1.0),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    union = sorted({g for item in manifest["languages"].values() for g in item["compacted_group_ids"]})
    manifest["compacted_in_any_language"] = union
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "limit": limit,
                      "compacted": {lang: len(v["compacted_group_ids"]) for lang, v in manifest["languages"].items()},
                      "union": len(union), "excluded": excluded}))


if __name__ == "__main__":
    main()
