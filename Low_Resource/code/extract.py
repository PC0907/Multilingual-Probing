"""Extract final-statement-token residual stream after every decoder block, plus
statement token counts and statement NLL, for every condition.

Output per (model, condition):
  acts/<model>/<cond>.npy        float32 [N=2000, L, d]   index j = residual after block j+1
  acts/<model>/<cond>.meta.json  ids, n_statement_tokens, statement_nll (nats), prompt
Rows follow the order of data/<cond>.json, which is identical (claim_0001..2000) for all
conditions.

Usage: python extract.py --model qwen3-8b-base --conds en,de,... --gpus 0,1,2,3
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

os.environ.setdefault("HF_HOME", C.HF_HOME)


def load_condition(cond):
    rows = json.load(open(C.DATA / f"{cond}.json"))
    assert [r["id"] for r in rows] == [f"claim_{i:04d}" for i in range(1, 2001)], cond
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gb_per_gpu", type=float, default=9.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = C.MODELS[args.model]
    out_dir = C.ACTS / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in args.conds.split(",") if c]
    todo = [c for c in conds if args.overwrite or not (out_dir / f"{c}.npy").exists()]
    if not todo:
        print("nothing to do"); return

    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    n_gpu = len(args.gpus.split(","))
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], dtype=torch.float32, device_map="auto",
        max_memory={i: f"{args.gb_per_gpu}GiB" for i in range(n_gpu)})
    model.eval()
    layers = model.model.layers
    L = len(layers)
    first_dev = next(model.parameters()).device

    captured = {}
    gather_idx = {}

    def make_hook(j):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            idx = gather_idx["idx"].to(h.device)
            captured[j] = h[torch.arange(h.shape[0], device=h.device), idx].float().cpu()
        return hook

    hooks = [layers[j].register_forward_hook(make_hook(j)) for j in range(L)]
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm

    for cond in todo:
        t0 = time.time()
        rows = load_condition(cond)
        prompt = C.PROMPTS[cond]
        texts = [prompt + r["sentence"] for r in rows]
        enc_all = [tok(t, return_offsets_mapping=True, add_special_tokens=True) for t in texts]
        order = np.argsort([len(e["input_ids"]) for e in enc_all])[::-1]
        N = len(rows)
        acts = None
        n_stmt = np.zeros(N, dtype=np.int32)
        nll = np.zeros(N, dtype=np.float64)
        n_tok_total = np.zeros(N, dtype=np.int32)
        plen = len(prompt)
        special = set(tok.all_special_ids)
        def run_batch(bidx):
            nonlocal acts
            seqs = [enc_all[i]["input_ids"] for i in bidx]
            maxlen = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            stmt_masks = []
            for k, (i, s) in enumerate(zip(bidx, seqs)):
                ids[k, :len(s)] = torch.tensor(s)
                att[k, :len(s)] = 1
                offs = enc_all[i]["offset_mapping"]
                # a token belongs to the statement if its character span ends after the
                # prompt; special tokens (BOS) are never statement tokens
                m = [(e > plen) and (tid not in special) for (s0, e), tid in zip(offs, s)]
                stmt_masks.append(m)
                n_stmt[i] = sum(m)
                n_tok_total[i] = len(s)
            last = att.sum(1) - 1
            gather_idx["idx"] = last
            with torch.no_grad():
                out = model.model(input_ids=ids.to(first_dev), attention_mask=att.to(first_dev))
                hs = out.last_hidden_state  # final-normed
                for k, i in enumerate(bidx):
                    pos = [p for p, mm in enumerate(stmt_masks[k]) if mm]
                    pred_pos = torch.tensor([p - 1 for p in pos], device=hs.device)
                    logits = lm_head(hs[k, pred_pos].to(lm_head.weight.device)).float()
                    tgt = ids[k, pos].to(logits.device)
                    nll[i] = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
            stack = torch.stack([captured[j] for j in range(L)], dim=1).numpy()  # [B,L,d]
            if acts is None:
                acts = np.zeros((N, L, stack.shape[-1]), dtype=np.float32)
            acts[bidx] = stack

        def run_safe(bidx):
            # long scripts (e.g. Gurmukhi under Qwen) can OOM at bs=16: halve and retry
            try:
                run_batch(bidx)
            except torch.OutOfMemoryError:
                if len(bidx) == 1:
                    raise
                captured.clear(); torch.cuda.empty_cache()
                h = len(bidx) // 2
                run_safe(bidx[:h]); run_safe(bidx[h:])

        for b0 in range(0, N, args.bs):
            run_safe(order[b0:b0 + args.bs])
        assert np.isfinite(acts).all(), f"non-finite activations in {cond}"
        np.save(out_dir / f"{cond}.npy", acts)
        meta = dict(model=args.model, repo=spec["repo"], revision=spec["revision"], cond=cond,
                    prompt=prompt, n_layers=L, d=int(acts.shape[-1]),
                    ids=[r["id"] for r in rows], n_statement_tokens=n_stmt.tolist(),
                    n_tokens_total=n_tok_total.tolist(), statement_nll=nll.tolist(),
                    dtype="float32", note="index j = residual stream after decoder block j+1, "
                    "at the final statement token")
        json.dump(meta, open(out_dir / f"{cond}.meta.json", "w"))
        print(f"[{args.model}] {cond}: {time.time()-t0:.0f}s  mean stmt tokens "
              f"{n_stmt.mean():.1f}  mean nll {nll.mean():.2f}", flush=True)
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
