"""Assemble translation chunks into data/<cond>.json (same schema as the original six
languages: id, sentence, label, group_id, partition, surface_language, ...) and create the
deterministic Brahmic transliterations (gu_Deva, pa_Deva, mr_Gujr).

Usage: python build_data.py ur pa mr gu ne hiLatn translit
"""
import json, sys, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

WORK = C.DATA / "translation_work"
EN = json.load(open(C.DATA / "en.json"))
# chunk pass -> output conditions (column order)
PASSES = {"ur": ["ur", "ur_Deva", "ur_Latn"], "pa": ["pa", "pa_Arab"], "mr": ["mr"],
          "gu": ["gu"], "ne": ["ne"], "hiLatn": ["hi_Latn"]}
TRANSLATOR = "claude-opus-5-5 (Anthropic), in-session LLM translation, label-blind"
NOTES = {
    "ur": "native Urdu translation of the English claim",
    "ur_Deva": "the Urdu sentence (same words) written in Devanagari with nuqta",
    "ur_Latn": "the Urdu sentence (same words) in informal Roman Urdu",
    "pa": "native Punjabi (Gurmukhi) translation of the English claim",
    "pa_Arab": "the Gurmukhi Punjabi sentence (same words) in Shahmukhi script",
    "mr": "native Marathi translation", "gu": "native Gujarati translation",
    "ne": "native Nepali translation",
    "hi_Latn": "the existing Hindi sentence (same words) in informal Roman Hindi",
}


def read_pass(p):
    rows = {}
    for f in sorted(WORK.glob(f"{p}_[0-9][0-9][0-9].tsv")):
        for line in open(f, encoding="utf8"):
            line = line.rstrip("\n")
            if line.strip():
                parts = line.split("\t")
                rows[f"claim_{parts[0]}"] = parts[1:]
    return rows


def write_cond(cond, sentences, method, extra=None):
    out = []
    for r in EN:
        out.append(dict(id=r["id"], sentence=sentences[r["id"]], label=r["label"],
                        group_id=r["group_id"], partition=r["partition"],
                        surface_language=cond, source_row=r["source_row"],
                        translation_model=TRANSLATOR if method == "llm" else None,
                        derivation=NOTES.get(cond, method) if method == "llm" else method,
                        translation_human_validated=False, dataset_version="research_n-1.0",
                        **(extra or {})))
    json.dump(out, open(C.DATA / f"{cond}.json", "w"), ensure_ascii=False, indent=1)
    h = hashlib.sha256(open(C.DATA / f"{cond}.json", "rb").read()).hexdigest()
    print(f"wrote {cond}.json  n={len(out)}  sha256={h[:16]}")


# nukta / sign characters that indic_transliteration passes through unchanged
NUKTA = {"\u0A3C": "\u093C", "\u0ABC": "\u093C", "\u093C_gujr": "\u0ABC"}


def fix_translit(t, target):
    if target == "devanagari":
        t = t.replace("\u0A3C", "\u093C").replace("\u0ABC", "\u093C")
        t = t.replace("\u0A70", "\u0902").replace("\u0A71", "")  # stray tippi / addak
        t = t.replace("\u0938\u093C", "\u0936").replace("\u0932\u093C", "\u0933")  # sha, lla
        # Gujarati candra vowels: ઑ ૉ ઍ ૅ -> ऑ ॉ ॲ ॅ
        for a, b in [("\u0A91", "\u0911"), ("\u0AC9", "\u0949"), ("\u0A8D", "\u0972"),
                     ("\u0AC5", "\u0945")]:
            t = t.replace(a, b)
    else:  # gujarati
        t = t.replace("\u093C", "\u0ABC").replace("\u0AB0\u0ABC", "\u0AB0")  # eyelash ra -> ra
        # candra vowels: ऑ ॉ ॲ ॅ -> ઑ ૉ ઍ ૅ
        for a, b in [("\u0911", "\u0A91"), ("\u0949", "\u0AC9"), ("\u0972", "\u0A8D"),
                     ("\u0945", "\u0AC5"), ("\u0904", "\u0A8D")]:
            t = t.replace(a, b)
    return t


def main(args):
    for p in args:
        if p == "translit":
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            for cond, src, a, b in [("gu_Deva", "gu", sanscript.GUJARATI, sanscript.DEVANAGARI),
                                    ("pa_Deva", "pa", sanscript.GURMUKHI, sanscript.DEVANAGARI),
                                    ("mr_Gujr", "mr", sanscript.DEVANAGARI, sanscript.GUJARATI)]:
                if not (C.DATA / f"{src}.json").exists():
                    print(f"skip {cond}: {src}.json not built yet"); continue
                if (C.DATA / f"{cond}.json").exists():
                    continue
                base = {r["id"]: r["sentence"] for r in json.load(open(C.DATA / f"{src}.json"))}
                sents = {k: fix_translit(transliterate(v, a, b), b) for k, v in base.items()}
                rng = {"devanagari": (0x0900, 0x097F), "gujarati": (0x0A80, 0x0AFF)}[b]
                bad = {ch for t in sents.values() for ch in t
                       if 0x0900 <= ord(ch) <= 0x0DFF and not rng[0] <= ord(ch) <= rng[1]}
                assert not bad, f"{cond}: leftover source-script chars {sorted(bad)}"
                write_cond(cond, sents, f"deterministic transliteration of {src}.json "
                           f"({a} -> {b}, indic_transliteration)")
            continue
        rows = read_pass(p)
        missing = [r["id"] for r in EN if r["id"] not in rows]
        assert not missing, f"{p}: {len(missing)} missing, e.g. {missing[:3]}"
        for j, cond in enumerate(PASSES[p]):
            write_cond(cond, {k: v[j] for k, v in rows.items()}, "llm")


if __name__ == "__main__":
    main(sys.argv[1:])
