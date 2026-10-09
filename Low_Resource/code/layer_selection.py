"""Frozen-layer rule from the paper: the block with the highest mean within-language
*validation* AUROC over the six original languages (train-split mass-mean probe). Test
data and the new conditions never enter selection. Also stores the full per-layer,
per-condition validation/test AUROC table for the layer-wise analyses."""
import json, sys
import numpy as np
from common import *

model = sys.argv[1]
conds = available(model)
L = meta(model, "en")["n_layers"]
y = LABELS
out = {"model": model, "n_layers": L, "val_auroc": {}, "test_auroc": {}}
for c in conds:
    A = load_acts(model, c)
    va, te = [], []
    for l in range(1, L + 1):
        X = np.asarray(A[:, l - 1], dtype=np.float64)
        w, b = massmean(X[TRAIN], y[TRAIN])
        va.append(auroc(X[VAL] @ w, y[VAL])); te.append(auroc(X[TEST] @ w, y[TEST]))
    out["val_auroc"][c] = va; out["test_auroc"][c] = te
mean6 = np.mean([out["val_auroc"][c] for c in C.ORIGINAL6], axis=0)
out["mean_val_auroc_original6"] = mean6.tolist()
out["selected_block"] = int(np.argmax(mean6) + 1)
out["paper_block"] = C.MODELS[model]["paper_block"]
(C.RESULTS / model).mkdir(parents=True, exist_ok=True)
json.dump(out, open(C.RESULTS / model / "layer_selection.json", "w"), indent=1)
print(model, "selected", out["selected_block"], "paper", out["paper_block"],
      "mean val AUROC at selected", round(mean6.max(), 3),
      "at paper block", round(mean6[out["paper_block"] - 1], 3))
