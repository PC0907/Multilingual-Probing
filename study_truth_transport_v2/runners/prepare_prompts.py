#!/usr/bin/env python3
"""Render frozen activation and behavioral prompts from dataset_v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LANGUAGES = ("en", "de", "ar", "hi", "fr", "es")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(dataset_root: Path, prompt_config: Path, output: Path) -> None:
    config = json.loads(prompt_config.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest_sha256": sha256(dataset_root / "data" / "manifest.json"),
        "prompt_config": str(prompt_config.resolve()),
        "prompt_config_sha256": sha256(prompt_config),
        "prompt_review_status": config["status"],
        "languages": {},
    }
    reference = None
    for language in LANGUAGES:
        rows = json.loads((dataset_root / "data" / f"{language}.json").read_text(encoding="utf-8"))
        identity = [(r["id"], r["label"], r["group_id"], r["partition"]) for r in rows]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"Alignment mismatch in {language}")
        language_config = config["languages"][language]
        activation = []
        behavioral = []
        for row in rows:
            statement = row["sentence"].strip()
            prompt = language_config["activation_prefix"] + statement
            if not prompt.endswith(statement):
                raise AssertionError("Activation prompt must end at the statement")
            base = {k: v for k, v in row.items() if k != "sentence"}
            activation.append({
                **base,
                "sentence": prompt,
                "raw_statement": statement,
                "prompt_template_id": config["activation_template_id"],
                "target_char_end": len(prompt),
            })
            for template_index, template in enumerate(language_config["behavior_templates"]):
                behavioral.append({
                    **base,
                    "raw_statement": statement,
                    "prompt": template.format(statement=statement),
                    "template_index": template_index,
                    "answer_true": language_config["answer_true"],
                    "answer_false": language_config["answer_false"],
                })
        activation_path = output / f"{language}.activation.json"
        behavior_path = output / f"{language}.behavior.json"
        activation_path.write_text(json.dumps(activation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        behavior_path.write_text(json.dumps(behavioral, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["languages"][language] = {
            "activation_rows": len(activation),
            "behavior_rows": len(behavioral),
            "activation_sha256": sha256(activation_path),
            "behavior_sha256": sha256(behavior_path),
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.dataset_root, args.prompt_config, args.output)


if __name__ == "__main__":
    main()
