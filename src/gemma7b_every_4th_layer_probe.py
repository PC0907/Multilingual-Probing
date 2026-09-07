#!/usr/bin/env python3
"""Extract Gemma hidden states and train a frozen-model linear truth probe.

This script is tailored to ``json_merged_4185.json`` (a JSON array with
``id``, ``sentence``, ``label``, and ``tag``), while keeping field names
configurable.  It probes every fourth *transformer block* of Gemma-7B:

    4, 8, 12, 16, 20, 24, 28

Hugging Face hidden-state index 0 is the embedding output, so transformer block
``k`` is read from ``outputs.hidden_states[k]``.  The base model is frozen and
only scikit-learn logistic-regression probes are trained.

Required Conda environment: gemma_probe (Python 3.11).

Put environment.yml, this script, and json_merged_4185.json in your project
folder. Open a Conda-enabled terminal in that folder and run:

    conda env create -f environment.yml
    conda activate gemma_probe
    hf auth login

The environment file installs dependencies into gemma_probe. If that environment
already exists, activate it and run ``conda env update -n gemma_probe -f
environment.yml`` instead of creating it again.

The script checks Conda activation and the running Python interpreter before
importing NumPy, scikit-learn, or model libraries. It exits with instructions if
the environment is missing, has another name, or uses a different Python.
The same requirement applies to --help, --validate-only, and cached-feature runs.

To run without activating your interactive shell, use:

    conda run --no-capture-output -n gemma_probe python \
        gemma7b_every_4th_layer_probe.py --data json_merged_4185.json

For NVIDIA GPUs, check that the installed PyTorch build detects CUDA:

    python -c "import torch; print(torch.cuda.is_available())"

If necessary, install the CUDA-compatible PyTorch build for your system inside
gemma_probe using https://pytorch.org/get-started/locally/ . Optional 4/8-bit
loading also requires ``python -m pip install bitsandbytes`` in this environment.

Before downloading ``google/gemma-7b``, accept its Gemma license on the model
page and authenticate once with ``hf auth login`` (or set ``HF_TOKEN``).
Never put an access token directly in this file.

Example:

    python gemma7b_every_4th_layer_probe.py \
        --data json_merged_4185.json \
        --output-dir gemma7b_probe_results \
        --model-id google/gemma-7b \
        --batch-size 4 \
        --max-length 128 \
        --pooling last

For a quick schema/split check that does not load Gemma:

    python gemma7b_every_4th_layer_probe.py \
        --data json_merged_4185.json --validate-only

The main outputs are ``metrics.csv``, ``metrics_by_tag.csv``,
``predictions.csv``, ``split_assignments.csv``, fitted ``.joblib`` probes,
activation-space directions, a layer-profile plot, and ``run_manifest.json``.
By default pooled activations are saved. Add --reuse-activation-cache to reuse
them for probe fitting without another Gemma forward pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_CONDA_ENV = "gemma_probe"


def require_conda_environment() -> dict[str, str]:
    """Reject an inactive/wrong Conda environment or a different interpreter.

    Use standard-library checks so missing ML packages cannot hide the setup
    instructions. This is a runtime configuration check, not a security boundary.
    Both `conda activate gemma_probe` and `conda run -n gemma_probe` are supported.
    """
    active_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix_value = os.environ.get("CONDA_PREFIX", "")
    problem = None
    if not active_name or not prefix_value:
        problem = "No active Conda environment was detected."
    else:
        active_prefix = Path(prefix_value).expanduser()
        resolved_prefix = active_prefix.resolve()
        # Prefix-based activation may put an absolute path in CONDA_DEFAULT_ENV.
        active_name_matches = active_name == REQUIRED_CONDA_ENV
        if not active_name_matches:
            active_name_path = Path(active_name).expanduser()
            active_name_matches = (
                active_name_path.is_absolute()
                and active_name_path.resolve() == resolved_prefix
            )
        if active_prefix.name != REQUIRED_CONDA_ENV or not active_name_matches:
            problem = f"The active Conda environment is {active_name!r}."
        elif not (resolved_prefix / "conda-meta").is_dir():
            problem = "The active environment does not contain Conda metadata."
        elif Path(sys.prefix).resolve() != resolved_prefix:
            problem = (
                "The running Python interpreter belongs to a different environment. "
                f"Interpreter: {sys.executable}"
            )
    if problem is not None:
        raise RuntimeError(
            f"This script must run inside the Conda environment '{REQUIRED_CONDA_ENV}'.\n"
            f"{problem}\n\n"
            "Create it once from the project folder:\n"
            "  conda env create -f environment.yml\n"
            "Activate it before running Python:\n"
            "  conda activate gemma_probe\n"
            "  python gemma7b_every_4th_layer_probe.py --data json_merged_4185.json\n"
            "Or run directly with:\n"
            "  conda run --no-capture-output -n gemma_probe python "
            "gemma7b_every_4th_layer_probe.py --data json_merged_4185.json"
        )
    return {
        "conda_environment": REQUIRED_CONDA_ENV,
        "conda_prefix": str(Path(prefix_value).expanduser().resolve()),
        "python_executable": sys.executable,
    }


# Colab already supplies a managed Python environment, so allow an explicit
# opt-out there (and in other notebook runtimes) without weakening the default
# local Conda check. Set this before starting Python:
#   PROBE_ALLOW_NON_CONDA=1 python gemma7b_every_4th_layer_probe.py ...
if os.environ.get("PROBE_ALLOW_NON_CONDA") == "1":
    CONDA_RUNTIME = {
        "conda_environment": "not enforced (PROBE_ALLOW_NON_CONDA=1)",
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "python_executable": sys.executable,
    }
else:
    try:
        CONDA_RUNTIME = require_conda_environment()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CACHE_SCHEMA_VERSION = 1
POSITIVE_LABEL = 1
UPLOADED_DATASET_SHA256 = "038876974110343c81e570b9399f216adb5a95af4676dd06b7bf4083b7d9142b"


@dataclass(frozen=True)
class Record:
    row_index: int
    record_id: int | str
    sentence: str
    label: int
    tag: str
    group: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run linear probes on every fourth transformer block.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Input JSON array.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gemma7b_probe_results")
    )
    parser.add_argument("--model-id", default="google/gemma-7b")
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision; use a commit hash for strict reproducibility.",
    )
    parser.add_argument("--text-field", default="sentence")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--tag-field", default="tag")
    parser.add_argument(
        "--group-field",
        default=None,
        help=(
            "Optional provenance/pair field whose equal values must stay in one "
            "split. When omitted, normalized exact sentences are grouped."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=("auto", "exact"),
        default="auto",
        help=(
            "auto groups exact duplicates and, for the uploaded dataset fingerprint, "
            "its verified sports counterfactual pairs; exact groups duplicates only."
        ),
    )
    parser.add_argument("--layer-step", type=int, default=4)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Group-stratified folds; fold 0=test, fold 1=validation, rest=train.",
    )
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        help="Inverse regularization strengths selected on validation ROC-AUC.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="auto = bf16 on supported CUDA, fp16 on older CUDA, fp32 on CPU.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Hugging Face device_map (normally 'auto'; use 'cpu' for CPU only).",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help=(
            "Optional bitsandbytes loading. It reduces memory but changes hidden "
            "representations, so do not mix precisions in one comparison."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Test-set bootstrap resamples for 95%% confidence intervals; 0 disables.",
    )
    parser.add_argument(
        "--no-activation-cache",
        action="store_true",
        help="Do not save/reuse output-dir/activations.npz.",
    )
    parser.add_argument(
        "--reuse-activation-cache",
        action="store_true",
        help=(
            "Explicitly reuse an existing compatible activation cache, including "
            "its recorded resolved checkpoint and numerical precision."
        ),
    )
    parser.add_argument(
        "--recompute-activations",
        action="store_true",
        help="Ignore and replace a compatible/incompatible activation cache.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and create the split in memory, then exit before model loading.",
    )
    return parser.parse_args(argv)


def validate_cli(args: argparse.Namespace) -> None:
    if args.layer_step < 1:
        raise ValueError("--layer-step must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.n_splits < 3:
        raise ValueError("--n-splits must be at least 3 (test, validation, train)")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    if not args.c_grid or any((not math.isfinite(c) or c <= 0) for c in args.c_grid):
        raise ValueError("Every --c-grid value must be finite and greater than zero")
    args.c_grid = sorted(set(args.c_grid))
    if args.quantization != "none" and args.device_map == "cpu":
        raise ValueError("bitsandbytes quantization requires an accelerator, not --device-map cpu")
    if args.reuse_activation_cache and args.recompute_activations:
        raise ValueError(
            "Choose only one of --reuse-activation-cache and --recompute-activations"
        )
    if args.no_activation_cache and (
        args.reuse_activation_cache or args.recompute_activations
    ):
        raise ValueError(
            "--no-activation-cache cannot be combined with cache reuse/recompute flags"
        )


def normalized_group(text: str) -> str:
    """A split-only key; model input remains byte-for-byte unchanged."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_records(args: argparse.Namespace) -> tuple[list[Record], dict[str, Any]]:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    raw_bytes = args.data.read_bytes()
    dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{args.data} is not valid UTF-8 JSON: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError("Dataset must be a non-empty top-level JSON array")

    required = {args.id_field, args.text_field, args.label_field, args.tag_field}
    if args.group_field:
        required.add(args.group_field)

    records: list[Record] = []
    seen_ids: set[int | str] = set()
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"Row {i} must be an object, got {type(item).__name__}")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Row {i} is missing fields: {sorted(missing)}")

        record_id = item[args.id_field]
        if isinstance(record_id, bool) or not isinstance(record_id, (int, str)):
            raise TypeError(f"Row {i} {args.id_field!r} must be an integer or string")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate {args.id_field!r}: {record_id!r}")
        seen_ids.add(record_id)

        text = item[args.text_field]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {i} {args.text_field!r} must be a non-empty string")

        label = item[args.label_field]
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(
                f"Row {i} {args.label_field!r} must be integer 0 or 1, got {label!r}"
            )

        tag = item[args.tag_field]
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Row {i} {args.tag_field!r} must be a non-empty string")

        if args.group_field:
            group_value = item[args.group_field]
            if isinstance(group_value, (dict, list)) or group_value is None:
                raise ValueError(
                    f"Row {i} {args.group_field!r} must be a non-null scalar value"
                )
            group = f"provided:{type(group_value).__name__}:{group_value!r}"
        elif (
            args.grouping == "auto"
            and dataset_sha256 == UPLOADED_DATASET_SHA256
            and tag == "sports"
            and isinstance(record_id, int)
            and 501 <= record_id <= 1000
        ):
            # This dataset contains five 100-row sports blocks. Within each block,
            # row j and row j+50 are a verified true/false counterfactual pair.
            # Grouping is used only for splitting; the ID is never a model feature.
            local_id = record_id - 501
            group = f"known-sports-pair:{local_id // 100}:{local_id % 50}"
        else:
            group = f"text:{normalized_group(text)}"

        records.append(Record(i, record_id, text, label, tag, group))

    labels = Counter(r.label for r in records)
    if set(labels) != {0, 1}:
        raise ValueError(f"Both binary labels 0 and 1 are required; found {dict(labels)}")

    exact_text_labels: dict[str, set[int]] = {}
    for record in records:
        exact_text_labels.setdefault(normalized_group(record.sentence), set()).add(record.label)
    conflicting_text = [text for text, values in exact_text_labels.items() if len(values) > 1]
    if conflicting_text:
        example = conflicting_text[0][:120]
        raise ValueError(
            "The same normalized sentence has conflicting labels. Fix the data before "
            f"probing. Example text: {example!r}"
        )

    # Union all applicable constraints: the selected provenance/pair key *and*
    # exact duplicate text. This prevents one grouping rule from overriding another.
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for key_fn in (
        lambda record: record.group,
        lambda record: f"exact:{normalized_group(record.sentence)}",
    ):
        first_by_key: dict[str, int] = {}
        for i, record in enumerate(records):
            key = key_fn(record)
            if key in first_by_key:
                union(first_by_key[key], i)
            else:
                first_by_key[key] = i

    component_members: dict[int, list[int]] = {}
    for i in range(len(records)):
        component_members.setdefault(find(i), []).append(i)
    canonical_group = {
        member: f"component:{min(members)}"
        for members in component_members.values()
        for member in members
    }
    records = [
        Record(
            record.row_index,
            record.record_id,
            record.sentence,
            record.label,
            record.tag,
            canonical_group[i],
        )
        for i, record in enumerate(records)
    ]

    group_labels: dict[str, set[int]] = {}
    for record in records:
        group_labels.setdefault(record.group, set()).add(record.label)

    tag_label_counts: dict[str, dict[str, int]] = {}
    for record in records:
        tag_label_counts.setdefault(record.tag, {"0": 0, "1": 0})[str(record.label)] += 1

    n_unique_groups = len(group_labels)
    if args.group_field:
        grouping_strategy = f"provided field: {args.group_field}"
        grouping_limitation = (
            "Quality of content-disjointness depends on the supplied grouping field."
        )
    elif args.grouping == "auto" and dataset_sha256 == UPLOADED_DATASET_SHA256:
        grouping_strategy = "normalized exact text + verified uploaded-dataset sports pairs"
        grouping_limitation = (
            "The source file has no global fact/pair ID. Other near-counterfactual "
            "statements may cross splits, so this is group-aware IID evaluation, not "
            "a fully content-disjoint evaluation."
        )
    else:
        grouping_strategy = "normalized exact text only"
        grouping_limitation = (
            "Only exact duplicates are grouped; semantic/counterfactual counterparts "
            "may cross splits unless --group-field is supplied."
        )
    summary = {
        "path": str(args.data.resolve()),
        "sha256": dataset_sha256,
        "n_records": len(records),
        "n_unique_ids": len(seen_ids),
        "n_unique_split_groups": n_unique_groups,
        "n_rows_consolidated_into_groups": len(records) - n_unique_groups,
        "grouping_strategy": grouping_strategy,
        "grouping_limitation": grouping_limitation,
        "label_semantics": (
            "Label 1 is the positive class and appears to mean true; the JSON file "
            "does not explicitly declare label semantics."
        ),
        "label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        "tag_label_counts": tag_label_counts,
    }
    return records, summary


def make_fixed_splits(
    records: Sequence[Record], n_splits: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return train/validation/test names using one shared group-aware split."""
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    # Joint stratification matters because the uploaded file is physically tag-blocked.
    strata = np.asarray([f"{r.tag}\x1f{r.label}" for r in records], dtype=object)
    groups = np.asarray([r.group for r in records], dtype=object)

    counts = Counter(strata.tolist())
    too_small = {key: value for key, value in counts.items() if value < n_splits}
    if too_small:
        raise ValueError(
            f"Each tag×label stratum needs at least {n_splits} rows; too small: {too_small}"
        )
    groups_by_stratum: dict[str, set[str]] = {}
    for stratum, group in zip(strata.tolist(), groups.tolist()):
        groups_by_stratum.setdefault(stratum, set()).add(group)
    too_few_groups = {
        key: len(value)
        for key, value in groups_by_stratum.items()
        if len(value) < n_splits
    }
    if too_few_groups:
        raise ValueError(
            f"Each tag×label stratum needs at least {n_splits} independent groups; "
            f"too few: {too_few_groups}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    fold = np.full(len(records), -1, dtype=np.int64)
    dummy_x = np.zeros((len(records), 1), dtype=np.float32)
    for fold_id, (_, held_out) in enumerate(splitter.split(dummy_x, strata, groups)):
        fold[held_out] = fold_id
    if np.any(fold < 0):
        raise RuntimeError("Internal error: some rows were not assigned to a fold")

    split = np.full(len(records), "train", dtype=object)
    split[fold == 0] = "test"
    split[fold == 1] = "validation"

    # A group must never occur in more than one partition.
    seen_group_split: dict[str, str] = {}
    for record, split_name in zip(records, split.tolist()):
        old = seen_group_split.setdefault(record.group, split_name)
        if old != split_name:
            raise AssertionError(f"Group leakage detected for {record.group!r}")

    split_summary: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        idx = np.flatnonzero(split == split_name)
        y = labels[idx]
        if set(y.tolist()) != {0, 1}:
            raise ValueError(f"Split {split_name!r} does not contain both labels")
        tag_label = Counter(f"{records[i].tag}|{records[i].label}" for i in idx)
        expected_tag_labels = set(groups_by_stratum)
        # The internal separator differs from the report separator.
        present_tag_labels = {
            key.replace("\x1f", "|") for key in expected_tag_labels
        }
        missing_tag_labels = present_tag_labels.difference(tag_label)
        if missing_tag_labels:
            raise ValueError(
                f"Split {split_name!r} is missing tag×label cells: "
                f"{sorted(missing_tag_labels)}"
            )
        split_summary[split_name] = {
            "n": int(idx.size),
            "label_counts": {str(k): int(v) for k, v in sorted(Counter(y).items())},
            "tag_label_counts": {k: int(v) for k, v in sorted(tag_label.items())},
        }
    return split, split_summary


def print_data_report(dataset: dict[str, Any], splits: dict[str, Any]) -> None:
    print("Dataset validation passed")
    print(f"  records: {dataset['n_records']}")
    print(f"  labels: {dataset['label_counts']}")
    print(f"  tag × label: {dataset['tag_label_counts']}")
    print(
        "  rows consolidated into split groups: "
        f"{dataset['n_rows_consolidated_into_groups']} (kept together)"
    )
    print(f"  grouping: {dataset['grouping_strategy']}")
    print(f"  split caveat: {dataset['grouping_limitation']}")
    for name in ("train", "validation", "test"):
        print(f"  {name}: {splits[name]['n']} rows, {splits[name]['label_counts']}")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def runtime_imports() -> tuple[Any, Any, Any, str]:
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model extraction requires torch, transformers, and accelerate. "
            "Install the packages shown at the top of this script."
        ) from exc
    return torch, AutoModel, AutoTokenizer, transformers.__version__


def resolve_dtype(torch: Any, requested: str, device_map: str) -> tuple[Any, str]:
    if requested != "auto":
        return getattr(torch, requested), requested
    force_cpu = str(device_map).lower() == "cpu"
    if not force_cpu and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    return torch.float32, "float32"


def language_config(config: Any) -> Any:
    """Return the decoder config for text-only or composite multimodal models."""
    return getattr(config, "text_config", config)


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    torch, AutoModel, AutoTokenizer, transformers_version = runtime_imports()
    dtype, dtype_name = resolve_dtype(torch, args.dtype, args.device_map)

    token = os.environ.get("HF_TOKEN") or None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            revision=args.revision,
            token=token,
            use_fast=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("gated", "401", "403", "access")):
            raise RuntimeError(
                f"Could not access model {args.model_id!r}. If it is gated, accept "
                "its license and run `hf auth login` (or set HF_TOKEN), then retry."
            ) from exc
        raise
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "The tokenizer has no padding token; refusing "
            "to silently replace it with EOS because that changes pooling semantics."
        )
    tokenizer.padding_side = "right"

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model_id, revision=args.revision, token=token
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("gated", "401", "403", "access")):
            raise RuntimeError(
                f"Could not access model {args.model_id!r}. If it is gated, accept "
                "its license and run `hf auth login` (or set HF_TOKEN), then retry."
            ) from exc
        raise
    text_config = language_config(config)
    model_limit = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    plausible_limits = [
        limit for limit in (model_limit, tokenizer_limit) if 0 < limit < 10_000_000
    ]
    if plausible_limits and args.max_length > min(plausible_limits):
        raise ValueError(
            f"--max-length={args.max_length} exceeds the model/tokenizer limit "
            f"of {min(plausible_limits)}"
        )

    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "token": token,
        "device_map": args.device_map,
        "config": config,
    }
    # Transformers 5 renamed torch_dtype to dtype; support both major versions.
    try:
        major = int(transformers_version.split(".", 1)[0])
    except ValueError:
        major = 5
    load_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype

    if args.quantization != "none":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Quantization requires bitsandbytes and BitsAndBytesConfig") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("4/8-bit bitsandbytes loading requires a CUDA GPU")
        if args.quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoModel.from_pretrained(args.model_id, **load_kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("gated", "401", "403", "access")):
            raise RuntimeError(
                f"Could not access model {args.model_id!r}. If it is gated, accept "
                "its license and run `hf auth login` (or set HF_TOKEN), then retry."
            ) from exc
        raise

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(language_config(model.config), "use_cache"):
        language_config(model.config).use_cache = False

    text_config = language_config(model.config)
    num_layers = int(text_config.num_hidden_layers)
    layers = list(range(args.layer_step, num_layers + 1, args.layer_step))
    if not layers:
        raise ValueError(
            f"Model has {num_layers} blocks, fewer than --layer-step={args.layer_step}"
        )
    metadata = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "num_hidden_layers": num_layers,
        "hidden_size": int(text_config.hidden_size),
        "selected_transformer_blocks": layers,
        "hidden_state_indexing": (
            "hidden_states[0] is embeddings; hidden_states[k] is transformer block k. "
            "The final entry is Hugging Face's canonical post-final-RMSNorm state."
        ),
        "resolved_dtype": dtype_name,
        "realized_model_dtype": str(getattr(model, "dtype", "unknown")),
        "quantization": args.quantization,
        "device_map": args.device_map,
        "realized_device_map": {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        },
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        ),
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
    }
    print(
        f"Loaded {args.model_id}: {num_layers} blocks; probing {layers}; "
        f"dtype={dtype_name}, quantization={args.quantization}"
    )
    return model, tokenizer, metadata


def pool_hidden(hidden: Any, attention_mask: Any, mode: str, torch: Any) -> Any:
    """Pool [batch, sequence, hidden] without depending on padding side."""
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError(
            f"Hidden/mask shape mismatch: hidden={tuple(hidden.shape)}, mask={tuple(mask.shape)}"
        )
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Tokenizer produced an all-padding example")

    if mode == "mean":
        # Accumulate in fp32 even when model weights/activations are bf16/fp16.
        hidden_fp32 = hidden.to(dtype=torch.float32)
        weights = mask.unsqueeze(-1).to(dtype=torch.float32)
        return (hidden_fp32 * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    last_position = positions.expand_as(mask).masked_fill(~mask, -1).max(dim=1).values
    batch_index = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_index, last_position]


def extract_activations(
    model: Any,
    tokenizer: Any,
    records: Sequence[Record],
    layers: Sequence[int],
    batch_size: int,
    max_length: int,
    pooling: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    torch, _, _, _ = runtime_imports()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(values: Iterable[int], **_: Any) -> Iterable[int]:  # type: ignore[misc]
            return values

    texts = [record.sentence for record in records]
    # This pass is inexpensive and tells us whether truncation actually occurred.
    lengths_encoding = tokenizer(
        texts, add_special_tokens=True, padding=False, truncation=False
    )
    token_lengths = np.asarray(
        [len(ids) for ids in lengths_encoding["input_ids"]], dtype=np.int64
    )
    tokenization = {
        "min_tokens": int(token_lengths.min()),
        "median_tokens": float(np.median(token_lengths)),
        "p95_tokens": float(np.percentile(token_lengths, 95)),
        "max_tokens": int(token_lengths.max()),
        "max_length": int(max_length),
        "n_truncated": int(np.sum(token_lengths > max_length)),
        "add_special_tokens": True,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
    }
    print(f"Tokenization: {tokenization}")
    if tokenization["n_truncated"]:
        warnings.warn(
            f"{tokenization['n_truncated']} statements exceed max_length={max_length} "
            "and will be truncated. Consider increasing --max-length.",
            stacklevel=2,
        )

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None and hasattr(model, "language_model"):
        input_embeddings = model.language_model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("Could not locate the model's text input embeddings")
    input_device = input_embeddings.weight.device
    if str(input_device) == "meta":
        raise RuntimeError("Could not determine the real input-embedding device")

    text_config = language_config(model.config)
    hidden_size = int(text_config.hidden_size)
    features = {
        int(layer): np.empty((len(records), hidden_size), dtype=np.float32)
        for layer in layers
    }
    for start in tqdm(range(0, len(texts), batch_size), desc="Model forward pass"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            key: encoded[key].to(input_device)
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }
        if "attention_mask" not in encoded:
            raise RuntimeError("Tokenizer did not return an attention_mask")
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states")
        expected = int(text_config.num_hidden_layers) + 1
        if len(hidden_states) != expected:
            raise RuntimeError(
                f"Expected {expected} hidden-state tensors (embedding + blocks), "
                f"received {len(hidden_states)}"
            )

        for layer in layers:
            pooled = pool_hidden(
                hidden_states[layer], encoded["attention_mask"], pooling, torch
            )
            array = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite activation found at layer {layer}")
            features[int(layer)][start : start + len(batch_texts)] = array

        del outputs, hidden_states, encoded

    for layer, matrix in features.items():
        expected_shape = (len(records), hidden_size)
        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"Layer {layer} activation shape {matrix.shape}; expected {expected_shape}"
            )
    return features, tokenization


def cache_expectation(args: argparse.Namespace, dataset_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "text_field": args.text_field,
        "layer_step": args.layer_step,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "quantization": args.quantization,
        "requested_dtype": args.dtype,
    }


def save_activation_cache(
    path: Path,
    features: dict[int, np.ndarray],
    expectation: dict[str, Any],
    model_metadata: dict[str, Any],
    tokenization: dict[str, Any],
) -> None:
    metadata = {
        **expectation,
        "layers": sorted(features),
        "model_metadata": model_metadata,
        "tokenization": tokenization,
    }
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    arrays.update({f"layer_{layer:03d}": matrix for layer, matrix in features.items()})
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    print(f"Saved pooled activation cache: {path}")


def load_activation_cache(
    path: Path, expectation: dict[str, Any], n_records: int
) -> tuple[dict[int, np.ndarray], dict[str, Any], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
            mismatches = {
                key: {"expected": expected, "found": metadata.get(key)}
                for key, expected in expectation.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "Activation cache does not match this run: "
                    + json.dumps(mismatches, ensure_ascii=False)
                    + ". Use --recompute-activations or a different --output-dir."
                )
            layers = [int(layer) for layer in metadata["layers"]]
            features = {
                layer: np.asarray(cache[f"layer_{layer:03d}"], dtype=np.float32)
                for layer in layers
            }
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read activation cache {path}; use --recompute-activations"
        ) from exc

    hidden_size = int(metadata["model_metadata"]["hidden_size"])
    for layer, matrix in features.items():
        if (
            matrix.shape != (n_records, hidden_size)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                f"Invalid cached layer {layer} array shape/content: {matrix.shape}; "
                f"expected {(n_records, hidden_size)}"
            )
    resolved_commit = metadata["model_metadata"].get("resolved_commit_hash")
    resolved_dtype = metadata["model_metadata"].get("resolved_dtype")
    print(
        f"Reused pooled activation cache: {path}; layers={layers}; "
        f"resolved_commit={resolved_commit}; dtype={resolved_dtype}"
    )
    return features, metadata["model_metadata"], metadata["tokenization"]


def positive_scores(probe: Pipeline, x: np.ndarray) -> np.ndarray:
    classes = probe.named_steps["probe"].classes_.tolist()
    if POSITIVE_LABEL not in classes:
        raise RuntimeError(f"Fitted probe classes do not contain {POSITIVE_LABEL}")
    column = classes.index(POSITIVE_LABEL)
    return probe.predict_proba(x)[:, column]


def metric_values(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["average_precision"] = float(average_precision_score(y_true, y_score))
        result["brier_score"] = float(brier_score_loss(y_true, y_score))
        result["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
        result["brier_score"] = float("nan")
        result["log_loss"] = float("nan")
    return result


def make_cluster_bootstrap_indices(
    groups: Sequence[str], n_bootstrap: int, seed: int
) -> list[np.ndarray]:
    """Resample independent split groups, preserving all rows inside each group."""
    if n_bootstrap == 0:
        return []
    group_to_rows: dict[str, list[int]] = {}
    for local_row, group in enumerate(groups):
        group_to_rows.setdefault(group, []).append(local_row)
    unique_groups = list(group_to_rows)
    rng = np.random.default_rng(seed)
    resamples: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(unique_groups), size=len(unique_groups))
        resamples.append(
            np.concatenate(
                [
                    np.asarray(group_to_rows[unique_groups[index]], dtype=np.int64)
                    for index in chosen
                ]
            )
        )
    return resamples


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    resample_indices: Sequence[np.ndarray],
) -> dict[str, float]:
    if not resample_indices:
        return {}
    values: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "roc_auc": [],
    }
    for idx in resample_indices:
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        values["accuracy"].append(float(accuracy_score(sample_y, y_pred[idx])))
        values["balanced_accuracy"].append(
            float(balanced_accuracy_score(sample_y, y_pred[idx]))
        )
        values["f1"].append(
            float(f1_score(sample_y, y_pred[idx], pos_label=POSITIVE_LABEL, zero_division=0))
        )
        values["roc_auc"].append(float(roc_auc_score(sample_y, y_score[idx])))

    intervals: dict[str, float] = {}
    for metric, samples in values.items():
        if not samples:
            continue
        low, high = np.percentile(samples, [2.5, 97.5])
        intervals[f"{metric}_ci_low"] = float(low)
        intervals[f"{metric}_ci_high"] = float(high)
    return intervals


def build_probe(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def direction_parameters(probe: Pipeline) -> dict[str, Any]:
    """Convert standardized logistic coefficients back to activation coordinates."""
    scaler: StandardScaler = probe.named_steps["scale"]
    classifier: LogisticRegression = probe.named_steps["probe"]
    w_standardized = classifier.coef_[0].astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    w_raw = w_standardized / scale
    intercept_raw = float(
        classifier.intercept_[0] - np.dot(w_standardized, scaler.mean_ / scale)
    )
    norm = float(np.linalg.norm(w_raw))
    if not math.isfinite(norm) or norm == 0:
        raise FloatingPointError("Probe produced a zero or non-finite direction")
    return {
        "standardized_coefficient": w_standardized.astype(np.float32),
        "standardized_intercept": float(classifier.intercept_[0]),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "raw_coefficient": w_raw.astype(np.float32),
        "raw_intercept": intercept_raw,
        "raw_norm": norm,
        "unit_direction": (w_raw / norm).astype(np.float32),
        "normalized_intercept": intercept_raw / norm,
        "classes": classifier.classes_.astype(np.int64),
        "C": float(classifier.C),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_all_layers(
    features: dict[int, np.ndarray],
    records: Sequence[Record],
    split: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Saving probes requires joblib") from exc

    labels = np.asarray([r.label for r in records], dtype=np.int64)
    train_idx = np.flatnonzero(split == "train")
    validation_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")
    train_validation_idx = np.concatenate([train_idx, validation_idx])
    bootstrap_indices = make_cluster_bootstrap_indices(
        [records[i].group for i in test_idx], args.bootstrap_samples, args.seed
    )

    metrics_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    layer_details: dict[str, Any] = {}

    for layer in sorted(features):
        x = features[layer]
        if x.shape[0] != len(records) or x.ndim != 2:
            raise ValueError(f"Unexpected feature matrix for layer {layer}: {x.shape}")

        candidates: list[tuple[float, dict[str, float | int], Pipeline]] = []
        for c_value in args.c_grid:
            candidate = build_probe(c_value, args.seed)
            candidate.fit(x[train_idx], labels[train_idx])
            val_pred = candidate.predict(x[validation_idx])
            val_score = positive_scores(candidate, x[validation_idx])
            val_metrics = metric_values(labels[validation_idx], val_pred, val_score)
            search_rows.append(
                {"layer": layer, "C": c_value, **{f"validation_{k}": v for k, v in val_metrics.items()}}
            )
            candidates.append((c_value, val_metrics, candidate))

        # Primary selection = validation ROC-AUC, secondary = balanced accuracy;
        # on an exact tie, the earlier/smaller C in the user-provided grid wins.
        selected_c, selected_val_metrics, _ = max(
            candidates,
            key=lambda item: (
                float(item[1]["roc_auc"]),
                float(item[1]["balanced_accuracy"]),
                -args.c_grid.index(item[0]),
            ),
        )
        if selected_c in (min(args.c_grid), max(args.c_grid)):
            warnings.warn(
                f"Layer {layer}: selected C={selected_c:g} is at the search-grid edge; "
                "consider widening --c-grid in a separately planned development run.",
                stacklevel=2,
            )

        final_probe = build_probe(selected_c, args.seed)
        final_probe.fit(x[train_validation_idx], labels[train_validation_idx])
        test_pred = final_probe.predict(x[test_idx]).astype(np.int64)
        test_score = positive_scores(final_probe, x[test_idx])
        test_metrics = metric_values(labels[test_idx], test_pred, test_score)
        ci = bootstrap_intervals(
            labels[test_idx],
            test_pred,
            test_score,
            bootstrap_indices,
        )

        direction_data = direction_parameters(final_probe)
        direction = direction_data["unit_direction"]
        normalized_intercept = direction_data["normalized_intercept"]
        # Verify the coordinate conversion before exporting the direction.
        check_x = x[train_validation_idx[: min(16, train_validation_idx.size)]]
        normalized_decision = check_x @ direction + normalized_intercept
        original_decision = final_probe.decision_function(check_x)
        if not np.allclose(
            normalized_decision * direction_data["raw_norm"],
            original_decision,
            rtol=2e-4,
            atol=2e-4,
        ):
            raise AssertionError("Raw-coordinate probe direction conversion failed")

        probe_path = args.output_dir / f"layer_{layer:02d}_probe.joblib"
        direction_path = args.output_dir / f"layer_{layer:02d}_direction.npz"
        probe_temporary = probe_path.with_name(probe_path.name + ".tmp")
        direction_temporary = direction_path.with_name(direction_path.name + ".tmp.npz")
        joblib.dump(final_probe, probe_temporary)
        probe_temporary.replace(probe_path)
        np.savez(
            direction_temporary,
            **{
                key: np.asarray(value)
                for key, value in direction_data.items()
            },
            positive_label=np.asarray(POSITIVE_LABEL, dtype=np.int64),
            layer=np.asarray(layer, dtype=np.int64),
        )
        direction_temporary.replace(direction_path)

        main_row: dict[str, Any] = {
            "layer": layer,
            "selected_C": selected_c,
            **{f"validation_{k}": v for k, v in selected_val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"test_{k}": v for k, v in ci.items()},
        }
        metrics_rows.append(main_row)

        test_tags = np.asarray([records[i].tag for i in test_idx], dtype=object)
        for tag in sorted(set(test_tags.tolist())):
            local = np.flatnonzero(test_tags == tag)
            tag_metrics = metric_values(
                labels[test_idx][local], test_pred[local], test_score[local]
            )
            tag_rows.append({"layer": layer, "tag": tag, **tag_metrics})

        for local_i, row_i in enumerate(test_idx.tolist()):
            record = records[row_i]
            prediction_rows.append(
                {
                    "layer": layer,
                    "id": record.record_id,
                    "tag": record.tag,
                    "true_label": record.label,
                    "predicted_label": int(test_pred[local_i]),
                    "probability_label_1": float(test_score[local_i]),
                }
            )

        layer_details[str(layer)] = {
            "selected_C": selected_c,
            "validation": selected_val_metrics,
            "test": {**test_metrics, **ci},
            "probe_file": probe_path.name,
            "direction_file": direction_path.name,
        }
        print(
            f"Layer {layer:>2}: C={selected_c:g}, "
            f"test accuracy={test_metrics['accuracy']:.4f}, "
            f"ROC-AUC={test_metrics['roc_auc']:.4f}"
        )

    write_csv(args.output_dir / "metrics.csv", metrics_rows)
    write_csv(args.output_dir / "metrics_by_tag.csv", tag_rows)
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "c_search.csv", search_rows)
    return {
        "metrics_rows": metrics_rows,
        "layer_details": layer_details,
        "prediction_count": len(prediction_rows),
    }


def save_split_assignments(
    path: Path, records: Sequence[Record], split: np.ndarray
) -> None:
    rows = [
        {
            "row_index": record.row_index,
            "id": record.record_id,
            "tag": record.tag,
            "label": record.label,
            "split": str(split[i]),
            "group_sha256": hashlib.sha256(record.group.encode("utf-8")).hexdigest(),
        }
        for i, record in enumerate(records)
    ]
    write_csv(path, rows)


def save_layer_plot(path: Path, metric_rows: Sequence[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is unavailable; skipping layerwise_metrics.png", stacklevel=2)
        return False

    layers = np.asarray([row["layer"] for row in metric_rows], dtype=np.int64)
    accuracy = np.asarray([row["test_accuracy"] for row in metric_rows], dtype=float)
    auc = np.asarray([row["test_roc_auc"] for row in metric_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(layers, accuracy, marker="o", linewidth=2, label="Test accuracy")
    ax.plot(layers, auc, marker="s", linewidth=2, label="Test ROC-AUC")
    if "test_roc_auc_ci_low" in metric_rows[0]:
        low = np.asarray([row["test_roc_auc_ci_low"] for row in metric_rows], dtype=float)
        high = np.asarray([row["test_roc_auc_ci_high"] for row in metric_rows], dtype=float)
        ax.fill_between(layers, low, high, alpha=0.15, label="ROC-AUC 95% bootstrap CI")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance ROC-AUC")
    ax.set(
        xlabel="Transformer block (1-based)",
        ylabel="Score",
        title="Linear probe by layer",
        xticks=layers,
        ylim=(0.0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    temporary.replace(path)
    return True


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli(args)
    set_reproducibility(args.seed)
    records, dataset_summary = load_records(args)
    split, split_summary = make_fixed_splits(records, args.n_splits, args.seed)
    print_data_report(dataset_summary, split_summary)
    if args.validate_only:
        print("Validation-only mode complete; the model was not loaded.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "activations.npz"
    if args.reuse_activation_cache and not cache_path.is_file():
        raise FileNotFoundError(
            f"--reuse-activation-cache was requested, but no cache exists at {cache_path}"
        )
    incomplete_marker = args.output_dir / "RUN_INCOMPLETE.txt"
    incomplete_marker.write_text(
        "A probe run started in this directory but has not completed. Ignore newly "
        "modified result files until this marker disappears.\n",
        encoding="utf-8",
    )

    expectation = cache_expectation(args, dataset_summary["sha256"])
    if cache_path.exists() and not args.no_activation_cache and not args.recompute_activations:
        if not args.reuse_activation_cache:
            raise FileExistsError(
                f"Activation cache already exists at {cache_path}. Pass "
                "--reuse-activation-cache to reuse its recorded checkpoint/precision, "
                "or --recompute-activations to replace it."
            )
        features, model_metadata, tokenization = load_activation_cache(
            cache_path, expectation, len(records)
        )
    else:
        model, tokenizer, model_metadata = load_model_and_tokenizer(args)
        layers = model_metadata["selected_transformer_blocks"]
        features, tokenization = extract_activations(
            model=model,
            tokenizer=tokenizer,
            records=records,
            layers=layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        )
        if not args.no_activation_cache:
            save_activation_cache(
                cache_path, features, expectation, model_metadata, tokenization
            )
        del model, tokenizer

    save_split_assignments(args.output_dir / "split_assignments.csv", records, split)

    expected_layers = list(
        range(
            args.layer_step,
            int(model_metadata["num_hidden_layers"]) + 1,
            args.layer_step,
        )
    )
    if sorted(features) != expected_layers:
        raise RuntimeError(
            f"Expected layer features {expected_layers}, found {sorted(features)}"
        )

    fitted = fit_all_layers(features, records, split, args)
    plot_created = save_layer_plot(
        args.output_dir / "layerwise_metrics.png", fitted["metrics_rows"]
    )

    manifest = {
        "status": "complete",
        "method": {
            "base_model_frozen": True,
            "probe": "StandardScaler + L2 logistic regression",
            "positive_label": POSITIVE_LABEL,
            "pooling": args.pooling,
            "layer_step": args.layer_step,
            "selected_transformer_blocks": expected_layers,
            "split": (
                "StratifiedGroupKFold over tag×label; fold 0 test, fold 1 "
                "validation, remaining folds train"
            ),
            "c_selection": "maximum validation ROC-AUC, then balanced accuracy",
            "final_fit": "train + validation; exactly one evaluation on test",
            "confidence_intervals": (
                "95% percentile bootstrap over test split groups, using identical "
                "resamples for every layer; conditional on the fitted probes"
            ),
            "direction_note": (
                "Saved unit vectors are logistic decision normals converted from "
                "standardized coordinates back to original activation coordinates; "
                "they are not mass-mean directions."
            ),
        },
        "dataset": dataset_summary,
        "splits": split_summary,
        "model": model_metadata,
        "tokenization": tokenization,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"validate_only"}
        },
        "environment": {
            **CONDA_RUNTIME,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "layers": fitted["layer_details"],
        "files": {
            "main_metrics": "metrics.csv",
            "tag_metrics": "metrics_by_tag.csv",
            "hyperparameter_search": "c_search.csv",
            "test_predictions": "predictions.csv",
            "split_assignments": "split_assignments.csv",
            "plot": "layerwise_metrics.png" if plot_created else None,
            "activation_cache": None if args.no_activation_cache else "activations.npz",
        },
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with manifest_temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    manifest_temporary.replace(manifest_path)
    incomplete_marker.unlink()

    print(f"\nComplete. Results are in: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
