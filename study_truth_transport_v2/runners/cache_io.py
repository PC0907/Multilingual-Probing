"""Strict cache loading shared by confirmatory analysis runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_complete_cache(cache_dir: Path, expected_input: Path, mmap_mode: str = "r") -> tuple[np.ndarray, dict]:
    metadata_path = cache_dir / "metadata.json"
    array_path = cache_dir / "last.npy"
    if not (cache_dir / "COMPLETE").is_file() or not metadata_path.is_file() or not array_path.is_file():
        raise ValueError(f"Incomplete cache: {cache_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete") or metadata.get("status") != "complete":
        raise ValueError(f"Cache metadata is not complete: {cache_dir}")
    observed_input_hash = sha256(expected_input)
    if metadata.get("input_sha256") != observed_input_hash:
        raise ValueError(f"Input hash mismatch for {cache_dir}")
    array = np.load(array_path, mmap_mode=mmap_mode)
    if list(array.shape) != metadata.get("shape"):
        raise ValueError(f"Cache shape mismatch for {cache_dir}")
    if not np.isfinite(np.asarray(array[0])).all():
        raise ValueError(f"Nonfinite cache sample in {cache_dir}")
    return array, metadata


def assert_row_alignment(rows: list[dict], metadata: dict) -> None:
    ids = [row["id"] for row in rows]
    if metadata.get("source_ids") != ids:
        raise ValueError("Cache source IDs do not match prompt rows")
