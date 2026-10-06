#!/usr/bin/env python3
"""Download one exact public Hugging Face revision with an acquisition manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--ignore-pattern", action="append", default=[],
                        help="Skip duplicate or non-inference files (e.g. optimizer states)")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing model directory: {args.output}")
    # The first sequential download may run after the preceding checkpoint's
    # parent directory has been removed.  Create the container directory before
    # asking the filesystem for free space so a clean one-model-at-a-time queue
    # can start without a pre-existing ``models/`` directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output.parent).free
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f"Only {free / 2**30:.2f} GiB free before model download")

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(args.repository, revision=args.revision)
    if info.sha != args.revision:
        raise RuntimeError(f"Resolved revision changed: {info.sha}")
    snapshot_download(repo_id=args.repository, revision=args.revision,
                      local_dir=args.output, local_dir_use_symlinks=False,
                      ignore_patterns=args.ignore_pattern or None)
    important = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = args.output / name
        if path.is_file():
            important[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    weights = [{"name": path.name, "bytes": path.stat().st_size}
               for path in sorted(args.output.glob("*.safetensors"))]
    if not weights:
        raise RuntimeError("Downloaded snapshot contains no safetensors weights")
    manifest = {
        "schema_version": 1,
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": args.repository,
        "requested_revision": args.revision,
        "resolved_revision": info.sha,
        "gated": info.gated,
        "local_directory": str(args.output.resolve()),
        "important_file_hashes": important,
        "weight_files": weights,
        "weight_bytes": sum(row["bytes"] for row in weights),
        "free_gib_after": shutil.disk_usage(args.output).free / 2**30,
    }
    (args.output / "ACQUISITION.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "revision": info.sha,
                      "weight_gib": manifest["weight_bytes"] / 2**30,
                      "free_gib_after": manifest["free_gib_after"]}))


if __name__ == "__main__":
    main()
