"""Per-(model, condition) and per-pair predictors for RQ2/RQ4/RQ5/RQ7.

Per model x condition (results/<model>/features.json):
  fertility   mean statement tokens per sentence
  tok_per_char mean statement tokens / statement characters
  bpc         statement bits per character under the model (sum NLL / ln2 / sum chars):
              tokenizer-independent familiarity / resource proxy
  vocab       set size of distinct statement token ids
Per model x pair: token-type Jaccard overlap of statement vocabularies.
Language-level (results/lang_distances.json): lang2vec genetic (fam) and syntactic (syntax_knn)
cosine distances between the underlying languages, plus script identity.
"""
import json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import common as K

os.environ.setdefault("HF_HOME", C.HF_HOME)
ISO3 = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "ar": "arb", "hi": "hin",
        "ur": "urd", "mr": "mar", "ne": "npi", "gu": "guj", "pa": "pan"}


def model_features(model):
    from transformers import AutoTokenizer
    spec = C.MODELS[model]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    out, vocab = {}, {}
    for c in K.available(model):
        m = K.meta(model, c)
        rows = json.load(open(C.DATA / f"{c}.json"))
        chars = np.array([len(r["sentence"]) for r in rows])
        ntok = np.array(m["n_statement_tokens"])
        nll = np.array(m["statement_nll"])
        ids = set()
        for r in rows:
            ids.update(tok(r["sentence"], add_special_tokens=False)["input_ids"])
        vocab[c] = ids
        out[c] = dict(fertility=float(ntok.mean()), tok_per_char=float((ntok / chars).mean()),
                      bpc=float(nll.sum() / np.log(2) / chars.sum()),
                      nll_per_token=float(nll.sum() / ntok.sum()), vocab=len(ids),
                      chars=float(chars.mean()))
    conds = list(out)
    jac = {s: {t: len(vocab[s] & vocab[t]) / len(vocab[s] | vocab[t]) for t in conds} for s in conds}
    res = dict(cond=out, token_jaccard=jac)
    json.dump(res, open(C.RESULTS / model / "features.json", "w"), indent=1)
    print(model, {c: round(v["bpc"], 3) for c, v in out.items()})


def lang_distances():
    import lang2vec.lang2vec as l2v
    langs = list(ISO3)
    res = {}
    for kind in ["fam", "syntax_knn", "phonology_knn", "geo"]:
        f = l2v.get_features([ISO3[l] for l in langs], kind)
        V = {l: np.array([np.nan if x == "--" else float(x) for x in f[ISO3[l]]]) for l in langs}
        D = {}
        for a in langs:
            D[a] = {}
            for b in langs:
                u, v = V[a], V[b]
                ok = ~(np.isnan(u) | np.isnan(v))
                u, v = u[ok], v[ok]
                if kind == "geo":
                    D[a][b] = float(np.linalg.norm(u - v))
                else:
                    D[a][b] = float(1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        res[kind] = D
    json.dump(res, open(C.RESULTS / "lang_distances.json", "w"), indent=1)
    for k in res:
        print(k, "hi-mr %.3f hi-ur %.3f hi-gu %.3f hi-en %.3f en-de %.3f hi-ar %.3f" % tuple(
            res[k][a][b] for a, b in [("hi", "mr"), ("hi", "ur"), ("hi", "gu"), ("hi", "en"), ("en", "de"), ("hi", "ar")]))


if __name__ == "__main__":
    lang_distances()
    for m in (sys.argv[1:] or list(C.MODELS)):
        model_features(m)
