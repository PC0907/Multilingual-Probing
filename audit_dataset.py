#!/usr/bin/env python
"""Audit a true/false statement dataset before it reaches a GPU.

Every check here answers one question: could a probe reach its accuracy without
using the model's representation of truth at all? If yes, the probing result is
uninterpretable, and no amount of careful geometry downstream repairs it.

Runs on the JSON schemas in use in this project:
    {"id", "sentence", "label", "tag"}          # newer files
    {"statement", "label"}                      # sports files

LABEL CONVENTION IS NOT AUTO-DETECTABLE. The script cannot know whether 1 means
true or false, so it prints samples from each class and asks you to confirm.
Pass --true-label to assert what you believe; the script re-reports under that
assumption. Getting this wrong flips the sign of every direction downstream
without changing any accuracy number, which is why it is checked first and
loudly.

Usage::

    python audit_dataset.py data/raw/de_facts.json
    python audit_dataset.py data/raw/de_facts.json --true-label 1 --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# German negation and universal-quantifier markers. Both matter: "nie"/"kein"
# make a statement false-looking, and "nur"/"ausschliesslich" are hedges that
# writers reach for when constructing falsehoods.
NEG_MARKERS = [
    "nie", "kein", "keine", "keinen", "keiner", "nicht", "niemals",
    "ausschließlich", "ausschliesslich", "nur", "völlig", "voellig",
]

RULE = "=" * 72


def load(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        import pandas as pd
        return pd.read_csv(path).to_dict("records")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON array, got {type(data).__name__}")
    return data


def normalise(records: list[dict]) -> tuple[list[str], list[int], list[str], list[dict]]:
    """Map either schema onto (texts, labels, tags, problems)."""
    text_key = None
    for candidate in ("sentence", "statement", "text"):
        if candidate in records[0]:
            text_key = candidate
            break
    if text_key is None:
        raise SystemExit(
            f"no text field found; expected one of sentence/statement/text, "
            f"got keys {sorted(records[0])}"
        )

    texts, labels, tags, problems = [], [], [], []
    for i, r in enumerate(records):
        if text_key not in r or "label" not in r:
            problems.append({"row": i, "issue": "missing text or label"})
            continue
        if not isinstance(r["label"], int) or r["label"] not in (0, 1):
            problems.append({"row": i, "issue": f"label is {r['label']!r}, want 0 or 1"})
            continue
        texts.append(str(r[text_key]).strip())
        labels.append(int(r["label"]))
        tags.append(str(r.get("tag", "")) or "(untagged)")
    return texts, labels, tags, problems


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def has_negation(t: str) -> bool:
    low = t.lower()
    return any(re.search(rf"\b{m}\b", low) for m in NEG_MARKERS)


def subject_key(t: str, n_words: int = 4) -> str:
    return " ".join(t.split()[:n_words]).lower().rstrip(".,")


def report_schema(records, texts, labels, tags, problems) -> dict:
    section("1. SCHEMA")
    keys = Counter(k for r in records for k in r)
    print(f"records: {len(records)}   usable: {len(texts)}")
    print("fields present:")
    for k, v in keys.most_common():
        flag = "" if v == len(records) else f"   <-- only {v}/{len(records)} records"
        print(f"  {k:12} {v:5}{flag}")

    ids = [r["id"] for r in records if "id" in r]
    if ids:
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        print(f"ids: {len(ids)} present, {len(dupes)} duplicated"
              + (f" -> {dupes[:10]}" if dupes else ""))
    if problems:
        print(f"\n{len(problems)} malformed rows:")
        for p in problems[:10]:
            print(f"  row {p['row']}: {p['issue']}")
    return {"n_records": len(records), "n_usable": len(texts),
            "fields": dict(keys), "malformed": problems}


def report_label_convention(texts, labels, assumed) -> dict:
    section("2. LABEL CONVENTION  (verify by eye - not auto-detectable)")
    counts = Counter(labels)
    print(f"label=0: {counts[0]}    label=1: {counts[1]}")
    for lab in (0, 1):
        print(f"\n  --- three statements with label={lab} ---")
        for t in [x for x, l in zip(texts, labels) if l == lab][:3]:
            print(f"    {t}")
    if assumed is None:
        print("\n  Read those. Which class is TRUE? Re-run with --true-label 0|1.")
    else:
        print(f"\n  ASSUMING label={assumed} means TRUE.")
        print(f"  Repo convention (Statement.label) is 1 = true.")
        if assumed == 0:
            print("  MISMATCH: the loader must invert before writing the index,")
            print("  or every mass-mean direction points from true toward false.")
    return {"counts": dict(counts), "assumed_true_label": assumed}


def report_tags(texts, labels, tags, true_label) -> dict:
    section("3. TOPIC BALANCE")
    per = defaultdict(Counter)
    for l, tg in zip(labels, tags):
        per[tg][l] += 1
    print(f"{'tag':20} {'n':>6} {'true':>6} {'false':>6} {'true %':>8}")
    out = {}
    for tg in sorted(per, key=lambda k: -sum(per[k].values())):
        n = sum(per[tg].values())
        t = per[tg][true_label] if true_label is not None else per[tg][1]
        print(f"{tg:20} {n:6} {t:6} {n - t:6} {t / n:8.1%}")
        out[tg] = {"n": n, "true": t, "false": n - t}
    if len(per) > 1:
        sizes = [sum(c.values()) for c in per.values()]
        print(f"\nsmallest tag has {min(sizes)} statements. A per-topic mass-mean")
        print("direction needs enough of BOTH classes; under ~50 per class the")
        print("direction estimate is too noisy for a topic-transfer matrix.")
    return out


def report_duplicates(texts) -> dict:
    section("4. DUPLICATES AND LEAKAGE GROUPS")
    exact = [t for t, c in Counter(texts).items() if c > 1]
    print(f"exact duplicate statements: {len(exact)}")
    for t in exact[:5]:
        print(f"  {t}")

    groups = Counter(subject_key(t) for t in texts)
    shared = {k: v for k, v in groups.items() if v > 1}
    covered = sum(shared.values())
    print(f"\nstatements sharing a 4-word opening: {covered}/{len(texts)} "
          f"in {len(shared)} groups")
    for k, v in sorted(shared.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v:3}x  {k!r}")
    if covered > 0.1 * len(texts):
        print("\n  These are near-minimal pairs. A RANDOM train/test split puts")
        print("  variants of one fact on both sides and inflates accuracy and the")
        print("  split-half ceiling. Assign group_id by this key and split by group.")
    return {"exact_duplicates": len(exact), "grouped_statements": covered,
            "n_groups": len(shared)}


def report_polarity(texts, labels, true_label) -> dict:
    section("5. POLARITY x TRUTH  (the 2x2 from templates.py)")
    tl = true_label if true_label is not None else 1
    cells = Counter((1 if l == tl else 0, int(has_negation(t)))
                    for t, l in zip(texts, labels))
    total = sum(cells.values())
    expected = total / 4
    print(f"{'truth':>6} {'polarity':>9} {'n':>6} {'dev from n/4':>14}")
    for truth in (1, 0):
        for pol in (0, 1):
            n = cells[(truth, pol)]
            dev = abs(n - expected) / expected if expected else 0
            print(f"{truth:>6} {pol:>9} {n:6} {dev:13.1%}")
    empty = [k for k in [(a, b) for a in (0, 1) for b in (0, 1)] if cells[k] == 0]
    ok = not empty and all(
        abs(cells[k] - expected) / expected <= 0.1
        for k in [(a, b) for a in (0, 1) for b in (0, 1)]
    )
    print(f"\ncheck_cell_balance would return ok={ok}")
    if not ok:
        print("  Polarity and truth are correlated. The 2D truth subspace is")
        print("  ill-conditioned, so the second principal angle is noise. RQ1's")
        print("  principal-angle method needs all four cells populated.")
    return {"cells": {f"truth={k[0]},pol={k[1]}": v for k, v in cells.items()},
            "ok": ok}


def report_surface(texts, labels, true_label) -> dict:
    section("6. SURFACE-FORM CONFOUNDS")
    tl = true_label if true_label is not None else 1
    t_len = [len(t) for t, l in zip(texts, labels) if l == tl]
    f_len = [len(t) for t, l in zip(texts, labels) if l != tl]
    mt, mf = sum(t_len) / len(t_len), sum(f_len) / len(f_len)
    rel = abs(mt - mf) / ((mt + mf) / 2)
    print(f"mean chars  true={mt:.2f}  false={mf:.2f}  relative diff={rel:.2%}"
          f"   surface_asymmetry ok={rel < 0.02}")
    print(f"max chars: {max(len(t) for t in texts)}  "
          f"(max whitespace tokens: {max(len(t.split()) for t in texts)})")

    neg_t = sum(1 for t, l in zip(texts, labels) if l == tl and has_negation(t))
    neg_f = sum(1 for t, l in zip(texts, labels) if l != tl and has_negation(t))
    rule_acc = (len(t_len) - neg_t + neg_f) / len(texts)
    print(f"\nnegation present:  true={neg_t}  false={neg_f}")
    print(f"'negation => false' rule alone scores {rule_acc:.1%}")

    out = {"mean_chars_true": mt, "mean_chars_false": mf, "relative_difference": rel,
           "negation_true": neg_t, "negation_false": neg_f, "negation_rule_acc": rule_acc}

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, GroupKFold
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("\n(scikit-learn not installed; skipping lexical baselines)")
        return out

    y = np.array([1 if l == tl else 0 for l in labels])
    groups = np.array([subject_key(t) for t in texts])
    cv = GroupKFold(n_splits=5)
    print("\nlexical baselines, GROUPED 5-fold (no model, no activations):")
    for name, vec in [
        ("word 1-2gram", TfidfVectorizer(ngram_range=(1, 2))),
        ("char 3-5gram", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
    ]:
        pipe = make_pipeline(vec, LogisticRegression(max_iter=2000))
        s = cross_val_score(pipe, texts, y, cv=cv, groups=groups, scoring="accuracy")
        print(f"  {name}: {s.mean():.3f} +/- {s.std():.3f}")
        out[f"tfidf_{name.split()[0]}"] = float(s.mean())

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("\n  most FALSE-indicating tokens:", ", ".join(names[order[:10]]))
    print("  most TRUE-indicating tokens: ", ", ".join(names[order[-10:]]))
    print("\n  Any lexical baseline meaningfully above 50% is accuracy a probe can")
    print("  reach without reading the model at all. Rewrite the tokens above.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--true-label", type=int, choices=[0, 1], default=None,
                    help="which label value means TRUE (verify from section 2 first)")
    ap.add_argument("--json", type=Path, default=None, help="write summary here")
    args = ap.parse_args()

    records = load(args.path)
    if not records:
        raise SystemExit("empty file")
    texts, labels, tags, problems = normalise(records)

    print(f"\nAUDIT: {args.path}")
    summary = {"path": str(args.path)}
    summary["schema"] = report_schema(records, texts, labels, tags, problems)
    summary["labels"] = report_label_convention(texts, labels, args.true_label)
    summary["tags"] = report_tags(texts, labels, tags, args.true_label)
    summary["duplicates"] = report_duplicates(texts)
    summary["polarity"] = report_polarity(texts, labels, args.true_label)
    summary["surface"] = report_surface(texts, labels, args.true_label)

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())