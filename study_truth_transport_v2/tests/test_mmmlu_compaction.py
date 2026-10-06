import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from study_truth_transport_v2.runners.compact_mmmlu_context import (
    LANGS, MARKER, fit_question, render, source_fields, word_starts)
from study_truth_transport_v2.runners.prepare_mmmlu_external import LANG_FILES, load_csv

STUDY = Path(__file__).resolve().parents[1]
FIT_ROOT = STUDY / "data" / "mmmlu_external_v1_fit512"
RAW_ROOT = STUDY / "data" / "mmmlu_external_v1"
# These tests validate the built MMMLU files, which are not distributed (README, step 4).
pytestmark = pytest.mark.skipif(not FIT_ROOT.exists(), reason="MMMLU data not built (README, step 4)")
MODELS = ("qwen3-8b-base", "apertus-8b-2509", "mistral-7b-v0.3")
PREFIXES = json.loads((STUDY / "config" / "prompts.json").read_text(encoding="utf-8"))["languages"]


def word_count(text: str) -> int:
    return len(text.split())


def test_rule_truncates_shared_question_head_for_both_candidates():
    question = " ".join(f"w{i}" for i in range(40))
    answers = ["short", "a much longer answer with many words"]
    text, n_words, kept = fit_question(word_count, "P: ", "en", question, answers, 30)
    assert text.startswith(MARKER) and text.endswith("w39")
    assert n_words == 40 and 0 < kept < 40
    assert max(word_count(render("P: ", "en", text, a)) for a in answers) <= 30
    longer = MARKER + question[word_starts(question)[40 - kept - 1]:]
    assert max(word_count(render("P: ", "en", longer, a)) for a in answers) > 30
    # Label-blind: candidate order cannot change the result.
    assert fit_question(word_count, "P: ", "en", question, answers[::-1], 30) == (text, n_words, kept)


def test_rule_leaves_fitting_questions_and_rejects_infeasible_pairs():
    assert fit_question(word_count, "", "en", "a b c", ["x", "y"], 100) == ("a b c", 3, 3)
    assert fit_question(word_count, "", "en", "a b c", ["x", " ".join(["y"] * 200)], 100) is None


def load_rows(model_id: str, lang: str) -> list[dict]:
    return json.loads((FIT_ROOT / model_id / f"{lang}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("model_id", MODELS)
def test_compacted_files_are_paired_parallel_and_conservative(model_id):
    manifest = json.loads((FIT_ROOT / model_id / "manifest.json").read_text(encoding="utf-8"))
    excluded = set(manifest["excluded_infeasible_group_ids"])
    group_sets = []
    for lang in LANGS:
        raw = {row["id"]: row for row in json.loads((RAW_ROOT / f"{lang}.json").read_text(encoding="utf-8"))}
        rows = load_rows(model_id, lang)
        groups = defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
            original = raw[row["id"]]
            for key in ("group_id", "label", "subject", "source_row", "choice_index"):
                assert row[key] == original[key]
            if not row["compaction"]["applied"]:
                assert row["sentence"] == original["sentence"]
        assert {g for g in (r["group_id"] for r in raw.values())} - set(groups) == excluded
        for members in groups.values():
            assert sorted(m["label"] for m in members) == [0, 1]
            first, second = (m["compaction"] for m in members)
            assert first["applied"] == second["applied"]
            assert first["question_words_kept"] == second["question_words_kept"]
        assert sorted(g for g, m in groups.items() if m[0]["compaction"]["applied"]) == \
            manifest["languages"][lang]["compacted_group_ids"]
        group_sets.append(set(groups))
    assert all(groups == group_sets[0] for groups in group_sets)


@pytest.mark.skipif(not (STUDY / "external" / "MMMLU" / "test").exists(), reason="MMMLU sources absent")
@pytest.mark.parametrize("model_id", MODELS)
def test_both_candidates_share_the_identical_question_suffix(model_id):
    english = pd.read_parquet(STUDY / "external/mmlu_en/all/test-00000-of-00001.parquet").reset_index(drop=True)
    translated = {lang: load_csv(STUDY / "external/MMMLU/test" / name) for lang, name in LANG_FILES.items()}
    for lang in LANGS:
        for row in load_rows(model_id, lang):
            question, answer = source_fields(english, translated, lang, row)
            info = row["compaction"]
            if info["applied"]:
                starts = word_starts(question)
                question = MARKER + question[starts[len(starts) - info["question_words_kept"]]:]
            assert row["sentence"] == render(PREFIXES[lang]["activation_prefix"], lang, question, answer)


TOKENIZER_ROOT = os.environ.get("TRUTH_TRANSPORT_TOKENIZER_ROOT")


@pytest.mark.skipif(not TOKENIZER_ROOT, reason="set TRUTH_TRANSPORT_TOKENIZER_ROOT/<model_id>")
@pytest.mark.parametrize("model_id", MODELS)
def test_every_record_fits_frozen_context_limit(model_id):
    from transformers import AutoConfig, AutoTokenizer

    path = Path(TOKENIZER_ROOT) / model_id
    if not path.exists():
        pytest.skip(f"no tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    limit = min(512, int(AutoConfig.from_pretrained(path, local_files_only=True).max_position_embeddings))
    for lang in LANGS:
        rows = load_rows(model_id, lang)
        ids = tokenizer([row["sentence"] for row in rows], add_special_tokens=True, truncation=False)["input_ids"]
        assert max(len(item) for item in ids) <= limit
