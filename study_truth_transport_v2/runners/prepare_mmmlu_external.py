#!/usr/bin/env python3
"""Build aligned true/false candidate statements from English MMLU and MMMLU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


LANG_FILES = {
    "de": "mmlu_DE-DE.csv",
    "ar": "mmlu_AR-XY.csv",
    "hi": "mmlu_HI-IN.csv",
    "fr": "mmlu_FR-FR.csv",
    "es": "mmlu_ES-LA.csv",
}
LETTERS = "ABCD"
STATEMENTS = {
    "en": 'For the question "{question}", the correct answer is "{answer}".',
    "de": 'Für die Frage „{question}“ lautet die richtige Antwort „{answer}“.',
    "ar": 'بالنسبة إلى السؤال «{question}»، الإجابة الصحيحة هي «{answer}».',
    "hi": 'प्रश्न “{question}” के लिए सही उत्तर “{answer}” है।',
    "fr": 'Pour la question « {question} », la bonne réponse est « {answer} ».',
    "es": 'Para la pregunta «{question}», la respuesta correcta es «{answer}».',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"", "Question", "A", "B", "C", "D", "Answer", "Subject"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    # OpenAI's first CSV column is an index local to each subject and therefore
    # repeats across the 57 concatenated subjects.  Preserve released row order.
    by_subject: dict[str, list[int]] = {}
    for row in rows:
        by_subject.setdefault(row["Subject"], []).append(int(row[""]))
    for subject, indices in by_subject.items():
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous within-subject IDs in {path}:{subject}")
    return rows


def selected_indices(subjects: list[str], per_subject: int, seed: int) -> list[int]:
    by_subject: dict[str, list[int]] = {}
    for index, subject in enumerate(subjects):
        by_subject.setdefault(subject, []).append(index)
    selected = []
    for subject, indices in sorted(by_subject.items()):
        ranked = sorted(indices, key=lambda i: hashlib.sha256(
            f"mmmlu-external-v1|{seed}|{subject}|{i}".encode()).hexdigest())
        if len(ranked) < per_subject:
            raise ValueError(f"Subject {subject} has only {len(ranked)} rows")
        selected.extend(ranked[:per_subject])
    return sorted(selected)


def wrong_index(row_index: int, correct: int, seed: int) -> int:
    wrong = [value for value in range(4) if value != correct]
    digest = hashlib.sha256(f"mmmlu-wrong-v1|{seed}|{row_index}".encode()).digest()
    return wrong[int.from_bytes(digest[:8], "big") % len(wrong)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", required=True, type=Path,
                        help="cais/mmlu all test parquet")
    parser.add_argument("--mmmlu-root", required=True, type=Path)
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-subject", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Refusing existing output: {args.output_dir}")

    english = pd.read_parquet(args.english).reset_index(drop=True)
    required = {"question", "subject", "choices", "answer"}
    if set(english.columns) != required:
        raise ValueError(f"Unexpected English columns: {english.columns.tolist()}")
    translated = {lang: load_csv(args.mmmlu_root / "test" / filename)
                  for lang, filename in LANG_FILES.items()}
    if any(len(rows) != len(english) for rows in translated.values()):
        raise ValueError("English and translated row counts differ")
    for lang, rows in translated.items():
        for index, row in enumerate(rows):
            if row["Subject"] != english.loc[index, "subject"]:
                raise ValueError(f"Subject mismatch at {lang}:{index}")

    prompts = json.loads(args.prompt_config.read_text(encoding="utf-8"))["languages"]
    answer_consistent = [
        all(rows[index]["Answer"] == LETTERS[int(english.loc[index, "answer"])]
            for rows in translated.values())
        for index in range(len(english))
    ]
    eligible = english.loc[answer_consistent].copy()
    eligible_indices = eligible.index.tolist()
    local_selected = selected_indices(eligible["subject"].tolist(), args.per_subject, args.seed)
    chosen = sorted(eligible_indices[index] for index in local_selected)
    args.output_dir.mkdir(parents=True)
    file_hashes = {}
    for lang in ("en", "de", "ar", "hi", "fr", "es"):
        records = []
        for index in chosen:
            if lang == "en":
                question = str(english.loc[index, "question"])
                choices = [str(value) for value in english.loc[index, "choices"]]
                correct = int(english.loc[index, "answer"])
                subject = str(english.loc[index, "subject"])
            else:
                source = translated[lang][index]
                question = source["Question"]
                choices = [source[letter] for letter in LETTERS]
                correct = LETTERS.index(source["Answer"])
                subject = source["Subject"]
            incorrect = wrong_index(index, correct, args.seed)
            group_id = f"mmmlu-{index:05d}"
            for label, choice_index, suffix in ((1, correct, "correct"), (0, incorrect, "incorrect")):
                raw = STATEMENTS[lang].format(question=question, answer=choices[choice_index])
                records.append({
                    "id": f"{group_id}-{suffix}",
                    "group_id": group_id,
                    "source_row": index,
                    "subject": subject,
                    "label": label,
                    "choice_index": choice_index,
                    "correct_choice_index": correct,
                    "raw_statement": raw,
                    "sentence": prompts[lang]["activation_prefix"] + raw,
                    "provenance": "cais/mmlu English or OpenAI MMMLU professional translation",
                })
        path = args.output_dir / f"{lang}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        file_hashes[lang] = sha256(path)

    manifest = {
        "schema_version": 1,
        "source_english": str(args.english.resolve()),
        "source_english_sha256": sha256(args.english),
        "source_mmmlu_root": str(args.mmmlu_root.resolve()),
        "source_mmmlu_hashes": {lang: sha256(args.mmmlu_root / "test" / filename)
                                 for lang, filename in LANG_FILES.items()},
        "translation_documentation": "OpenAI MMMLU states that professional human translators produced all 14 locales",
        "languages": ["en", "de", "ar", "hi", "fr", "es"],
        "subjects": sorted(set(english.loc[chosen, "subject"].tolist())),
        "questions_per_subject": args.per_subject,
        "n_questions": len(chosen),
        "excluded_answer_disagreement_rows": int(len(english) - sum(answer_consistent)),
        "records_per_language": 2 * len(chosen),
        "selection_seed": args.seed,
        "negative_choice_rule": "one of three incorrect choices selected by fixed SHA-256 hash",
        "file_sha256": file_hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "questions": len(chosen),
                      "records_per_language": 2 * len(chosen)}))


if __name__ == "__main__":
    main()
