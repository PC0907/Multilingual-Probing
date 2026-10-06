import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1]
RAW = STUDY / "data" / "include_external_v1"
FIT = STUDY / "data" / "include_external_v1_fit512"
# These tests validate the built INCLUDE files, which are not distributed (README, step 4).
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="INCLUDE data not built (README, step 4)")
LANGS = ("ar", "de", "hi", "fr", "es")
MODELS = ("qwen3-8b-base", "apertus-8b-2509", "mistral-7b-v0.3")


@pytest.mark.parametrize("lang", LANGS)
def test_include_candidates_are_paired_and_distinct(lang):
    rows = json.loads((RAW / f"{lang}.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for members in groups.values():
        assert sorted(m["label"] for m in members) == [0, 1]
        assert len({m["question"] for m in members}) == 1
        assert members[0]["answer_text"].strip() != members[1]["answer_text"].strip()
    questions = [m[0]["question"].strip() for m in groups.values()]
    assert len(questions) == len(set(questions))


@pytest.mark.parametrize("model_id", MODELS)
def test_include_compacted_files_preserve_groups(model_id):
    for lang in LANGS:
        raw = {r["id"]: r for r in json.loads((RAW / f"{lang}.json").read_text(encoding="utf-8"))}
        rows = json.loads((FIT / model_id / f"{lang}.json").read_text(encoding="utf-8"))
        assert {r["id"] for r in rows} == set(raw)
        for row in rows:
            assert row["label"] == raw[row["id"]]["label"]
            if not row["compaction"]["applied"]:
                assert row["sentence"] == raw[row["id"]]["sentence"]


TOKENIZER_ROOT = os.environ.get("TRUTH_TRANSPORT_TOKENIZER_ROOT")


@pytest.mark.skipif(not TOKENIZER_ROOT, reason="set TRUTH_TRANSPORT_TOKENIZER_ROOT/<model_id>")
@pytest.mark.parametrize("model_id", MODELS)
def test_include_records_fit_frozen_limit(model_id):
    from transformers import AutoConfig, AutoTokenizer

    path = Path(TOKENIZER_ROOT) / model_id
    if not path.exists():
        pytest.skip(f"no tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    limit = min(512, int(AutoConfig.from_pretrained(path, local_files_only=True).max_position_embeddings))
    for lang in LANGS:
        rows = json.loads((FIT / model_id / f"{lang}.json").read_text(encoding="utf-8"))
        ids = tokenizer([r["sentence"] for r in rows], add_special_tokens=True)["input_ids"]
        assert max(len(i) for i in ids) <= limit
