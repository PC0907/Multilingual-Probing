#!/usr/bin/env python3
"""Freeze an aligned neutral-text control from the official FLORES-200 dev split."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
LANGUAGE_FILES = {
    "en": "eng_Latn", "de": "deu_Latn", "ar": "arb_Arab",
    "hi": "hin_Deva", "fr": "fra_Latn", "es": "spa_Latn",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Unexpected FLORES archive hash: {digest}")
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    by_language = {}
    with tarfile.open(args.archive, "r:gz") as archive:
        for language, code in LANGUAGE_FILES.items():
            member = archive.getmember(f"./flores200_dataset/dev/{code}.dev")
            with archive.extractfile(member) as stream:
                lines = io.TextIOWrapper(stream, encoding="utf-8").read().splitlines()
            by_language[language] = lines
    lengths = {len(lines) for lines in by_language.values()}
    if len(lengths) != 1 or args.samples > next(iter(lengths)):
        raise ValueError("FLORES languages are unaligned or sample count is too large")
    indices = sorted(range(next(iter(lengths))),
                     key=lambda i: hashlib.sha256(f"flores-neutral-v1|{i}".encode()).hexdigest())[:args.samples]
    args.output.mkdir(parents=True)
    hashes = {}
    for language, lines in by_language.items():
        path = args.output / f"{language}.json"
        payload = [{"id": f"flores_dev_{index:04d}", "line_index": index,
                    "language": language, "text": lines[index]} for index in indices]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        hashes[language] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": "FLORES-200 official public archive, dev split",
        "source_url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC-BY-SA-4.0",
        "citation": "NLLB Team et al. (2022), No Language Left Behind",
        "selection": "100 aligned indices with smallest sha256('flores-neutral-v1|index')",
        "samples_per_language": args.samples,
        "file_sha256": hashes,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "languages": list(LANGUAGE_FILES),
                      "samples_per_language": args.samples}))


if __name__ == "__main__":
    main()
