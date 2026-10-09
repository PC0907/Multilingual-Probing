"""Label-blind translation quality checks for every condition (results/quality.json).

  labse      cosine between LaBSE embeddings of each sentence and its English source
             (mean, 5th percentile, share below 0.70); also by label, to check that
             translation quality does not differ between true and false claims
  len_ratio  characters / English characters (median, 5th-95th percentile)
  purity     share of letters in the condition's own script (Latin letters allowed only
             when they occur verbatim in the English source, e.g. 'pH', quoted English)
"""
import json, os, sys, unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K

os.environ.setdefault("HF_HOME", C.HF_HOME)
SCRIPT_PREFIX = {"Latn": "LATIN", "Arab": "ARABIC", "Deva": "DEVANAGARI", "Gujr": "GUJARATI", "Guru": "GURMUKHI"}


def purity(sent, script, en):
    ok = bad = 0
    enl = en.lower()
    for w in sent.split():
        for ch in w:
            if not ch.isalpha() and unicodedata.category(ch) not in ("Mn", "Mc"):
                continue
            name = unicodedata.name(ch, "")
            if name.startswith(SCRIPT_PREFIX[script]):
                ok += 1
            elif name.startswith("LATIN") and w.strip(".,;:!?«»\"'()").lower() in enl:
                ok += 1
            else:
                bad += 1
    return ok / max(ok + bad, 1)


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    en = [r["sentence"] for r in json.load(open(C.DATA / "en.json"))]
    E = model.encode(en, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    y = K.LABELS
    out = {}
    for c, meta in C.CONDITIONS.items():
        if c == "en":
            continue
        s = [r["sentence"] for r in json.load(open(C.DATA / f"{c}.json"))]
        V = model.encode(s, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        sim = (E * V).sum(1)
        lr = np.array([len(a) / len(b) for a, b in zip(s, en)])
        pur = np.array([purity(a, meta["script"], b) for a, b in zip(s, en)])
        out[c] = dict(labse_mean=float(sim.mean()), labse_p5=float(np.percentile(sim, 5)),
                      labse_below_070=float((sim < 0.70).mean()),
                      labse_true=float(sim[y == 1].mean()), labse_false=float(sim[y == 0].mean()),
                      len_ratio_median=float(np.median(lr)), len_ratio_p5=float(np.percentile(lr, 5)),
                      len_ratio_p95=float(np.percentile(lr, 95)), purity_mean=float(pur.mean()),
                      purity_min=float(pur.min()), lowest_ids=[int(i) + 1 for i in np.argsort(sim)[:5]])
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items()}, flush=True)
    json.dump(out, open(C.RESULTS / "quality.json", "w"), indent=1)


if __name__ == "__main__":
    main()
