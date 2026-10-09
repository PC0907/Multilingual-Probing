"""Validate translation chunks: data/translation_work/<pass>_NNN.tsv

Each line: id \t col1 \t col2 ...  Columns per pass are given in PASSES. Reports missing /
duplicate ids, wrong column counts, and the share of letters outside the expected script
(letters from the Latin script are allowed only if they also occur verbatim in the English
source, e.g. "tw erk").
"""
import sys, re, unicodedata
from pathlib import Path

WORK = Path(__file__).resolve().parents[1] / "data" / "translation_work"
PASSES = {"ur": ["Arab", "Deva", "Latn"], "pa": ["Guru", "Arab"], "mr": ["Deva"],
          "gu": ["Gujr"], "ne": ["Deva"], "hiLatn": ["Latn"]}
RANGES = {"Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
          "Deva": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)], "Guru": [(0x0A00, 0x0A7F)],
          "Gujr": [(0x0A80, 0x0AFF)], "Latn": [(0x0041, 0x024F)]}


def script_of(ch):
    o = ord(ch)
    for s, rr in RANGES.items():
        if any(a <= o <= b for a, b in rr):
            return s
    return None


def main(pass_name):
    en = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1]
          for l in open(WORK / "en_ids.tsv", encoding="utf8")}
    cols = PASSES[pass_name]
    seen, problems = {}, []
    for f in sorted(WORK.glob(f"{pass_name}_[0-9][0-9][0-9].tsv")):
        for ln, line in enumerate(open(f, encoding="utf8"), 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != len(cols) + 1:
                problems.append(f"{f.name}:{ln} has {len(parts)-1} columns"); continue
            i = parts[0]
            if i in seen:
                problems.append(f"{f.name}:{ln} duplicate id {i}")
            seen[i] = parts[1:]
            src_latin = set(re.findall(r"[A-Za-z]+", en.get(i, "")))
            for c, (txt, sc) in enumerate(zip(parts[1:], cols)):
                letters = [ch for ch in txt if unicodedata.category(ch)[0] in "LM"]
                bad = [ch for ch in letters if script_of(ch) != sc]
                if sc != "Latn":
                    # Latin allowed only as verbatim English tokens
                    words = re.findall(r"[A-Za-z]+", txt)
                    bad_latin = [w for w in words if w not in src_latin]
                    bad = [ch for ch in bad if script_of(ch) != "Latn"]
                    if bad_latin:
                        problems.append(f"{i} col{c+1}({sc}) latin not in source: {bad_latin}")
                if bad:
                    problems.append(f"{i} col{c+1}({sc}) foreign letters: {''.join(sorted(set(bad)))}")
                endp = {"Arab": "۔", "Deva": "।", "Guru": "।", "Latn": ".", "Gujr": "."}
                ok_end = {endp[sc], "?", "!", "»", "."} if sc in ("Gujr",) else {endp.get(sc, "."), "?", "!", "»"}
                if pass_name in ("mr", "gu"):
                    ok_end = {".", "?", "!", "»"}
                if txt.strip() and txt.strip()[-1] not in ok_end:
                    problems.append(f"{i} col{c+1}({sc}) ends with {txt.strip()[-1]!r}")
                if any(ch in txt for ch in "۔،") and sc != "Arab":
                    problems.append(f"{i} col{c+1}({sc}) contains Arabic punctuation")
                if not txt.strip():
                    problems.append(f"{i} col{c+1} empty")
    missing = [k for k in en if k not in seen]
    print(f"{pass_name}: {len(seen)} ids done, {len(missing)} missing"
          + (f" (first missing {missing[0]})" if missing else ""))
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main(sys.argv[1])
