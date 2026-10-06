#!/usr/bin/env python3
"""Automatic number and unit fidelity of the three claim translations (NLLB = v2, Opus 5.5 = v3, Google Translate).

For each language and system:
  digit_mismatch   claims whose multiset of digit sequences differs from the English claim (Arabic-Indic and
                   Devanagari digits normalised to 0-9; thousands separators removed)
  imperial_to_metric  English claims with an imperial unit and no metric unit whose translation contains a metric unit
Writes study_truth_transport_v2/results/translation_fidelity.json. These are automatic checks, not human judgement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = {"NLLB": ROOT / "generic_claims_2000_v2/data", "Opus": ROOT / "generic_claims_2000_v3/data",
           "Google": ROOT / "generic_claims_2000_gt/data"}
LANGS = ["de", "ar", "hi", "fr", "es"]
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹०१२३४५६७८९", "0123456789" * 3)
IMPERIAL = re.compile(r"\b(foot|feet|inch|inches|mile|miles|pound|pounds|ounce|ounces|pint|pints|gallon|gallons|yard|yards|"
                      r"acre|acres|fahrenheit)\b|°F", re.I)
METRIC_EN = re.compile(r"\b(meter|metre|meters|metres|centimet|kilomet|kilogram|gram|grams|litre|liter|celsius)\w*|\b(cm|km|kg|mm|ml)\b|°C", re.I)
METRIC = {
    "de": r"(meter|kilogramm|gramm|liter|grad celsius)|\b(cm|km|kg|mm|ml)\b|°c",
    "fr": r"(mètre|kilogramme|gramme|litre|degrés celsius)|\b(cm|km|kg|mm|ml)\b|°c",
    "es": r"(metro|kilogramo|gramo|litro|grados celsius)|\b(cm|km|kg|mm|ml)\b|°c",
    "ar": r"(متر|كيلوغرام|كيلوجرام|غرام|جرام|لتر|مئوية)|\b(cm|km|kg)\b|°c",
    "hi": r"(मीटर|किलोग्राम|ग्राम|लीटर|सेल्सियस)|\b(cm|km|kg)\b|°c",
}


def digits(text: str) -> list[str]:
    text = text.translate(DIGITS)
    text = re.sub(r"(?<=\d)[,.٬  ](?=\d{3}\b)", "", text)
    return sorted(re.findall(r"\d+", text))


def load(path: Path, lang: str) -> dict[str, str]:
    return {row["id"]: row["sentence"] for row in json.loads((path / f"{lang}.json").read_text(encoding="utf-8"))}


def main() -> None:
    english = load(SYSTEMS["Opus"], "en")
    imperial_ids = [i for i, s in english.items() if IMPERIAL.search(s) and not METRIC_EN.search(s)]
    out = {"systems": {k: str(v.relative_to(ROOT)) for k, v in SYSTEMS.items()}, "n_claims": len(english),
           "n_imperial_english_claims": len(imperial_ids), "languages": {}}
    for lang in LANGS:
        metric = re.compile(METRIC[lang], re.I)
        row = {}
        for name, path in SYSTEMS.items():
            text = load(path, lang)
            assert text.keys() == english.keys(), (name, lang)
            mismatch = [i for i in english if digits(english[i]) != digits(text[i])]
            converted = [i for i in imperial_ids if metric.search(text[i])]
            row[name] = {"digit_mismatch": len(mismatch), "imperial_to_metric": len(converted),
                         "digit_mismatch_ids": mismatch[:20], "imperial_to_metric_ids": converted[:20]}
        out["languages"][lang] = row
    dest = ROOT / "study_truth_transport_v2/results/translation_fidelity.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({lang: {s: (v["digit_mismatch"], v["imperial_to_metric"]) for s, v in r.items()}
                      for lang, r in out["languages"].items()}), out["n_imperial_english_claims"])


if __name__ == "__main__":
    main()
