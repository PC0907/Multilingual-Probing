#!/usr/bin/env python3
"""Side-by-side LaBSE source-translation similarity for several translation systems.

Each input is a QE JSON written by runners/compute_translation_qe.py (same LaBSE revision, same device).
Statistics per language over the 1,972 dependency groups (group minimum similarity): mean, 10th percentile,
minimum, and, against the reference system, the median paired difference and the share of groups scoring higher.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("de", "ar", "hi", "fr", "es")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qe", nargs="+", required=True, help="NAME=path pairs; the first is the reference")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    systems = [(s.split("=", 1)[0], json.loads(Path(s.split("=", 1)[1]).read_text())["languages"]) for s in args.qe]
    ref_name, ref = systems[0]
    out = {"reference": ref_name, "systems": [n for n, _ in systems], "languages": {}}
    for lang in LANGS:
        groups = sorted(ref[lang]["group_min_similarity"])
        r = np.array([ref[lang]["group_min_similarity"][g] for g in groups])
        out["languages"][lang] = {}
        for name, qe in systems:
            x = np.array([qe[lang]["group_min_similarity"][g] for g in groups])
            d = x - r
            out["languages"][lang][name] = {"mean": float(x.mean()), "p10": float(np.quantile(x, .1)), "min": float(x.min()),
                                            "mean_item_similarity": qe[lang]["mean_similarity"],
                                            "median_diff_vs_ref": float(np.median(d)), "share_higher_than_ref": float(np.mean(d > 1e-4)), "share_tied_with_ref": float(np.mean(np.abs(d) <= 1e-4)),
                                            "share_lower_than_ref": float(np.mean(d < -1e-4))}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    names = [n for n, _ in systems]
    print("lang | " + " | ".join(f"{n} mean" for n in names) + " | " + " | ".join(f"{n} p10" for n in names)
          + " | " + " | ".join(f"{n} higher/tied/lower vs {ref_name}" for n in names[1:]))
    for lang in LANGS:
        v = out["languages"][lang]
        print(lang + " | " + " | ".join(f"{v[n]['mean']:.3f}" for n in names) + " | " + " | ".join(f"{v[n]['p10']:.3f}" for n in names)
              + " | " + " | ".join(f"{v[n]['share_higher_than_ref']:.2f}/{v[n]['share_tied_with_ref']:.2f}/{v[n]['share_lower_than_ref']:.2f}" for n in names[1:]))


if __name__ == "__main__":
    main()
