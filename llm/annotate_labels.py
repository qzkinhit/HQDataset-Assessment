#!/usr/bin/env python3
"""The language model as annotator of the test sets of Northcutt et al. (2021). Zero-shot and five-shot prompts at temperature zero, the answer is the option token with the highest next-token probability, first 400 tokens of each text, examples cut to 150 tokens.
The merge reports the accuracy against the original and the corrected labels, and how many of the errors confirmed by the reviewers the model flags at what precision. A flag is a disagreement with the original label. Only flagged records of the cited work were reviewed, so the precision over all flags of the model is a lower bound, the precision over reviewed records is given beside it."""
import json, os
import numpy as np
from qwen import chat, ids, next_probs
UNIT = 1024; MAX_TOKENS = 400; SHOT_TOKENS = 150; SETS = ("20news", "imdb", "amazon_reviewed", "amazon")
SYSTEM = "You label texts. You follow the answer format exactly."
_c = {}
def load(root, ds):
    if ds not in _c: _c[ds] = (json.load(open(f"{root}/labels/{ds}.meta.json")), [json.loads(l) for l in open(f"{root}/labels/{ds}.jsonl")])
    return _c[ds]
def units(root):
    for ds in SETS:
        if os.path.exists(f"{root}/labels/{ds}.jsonl"):
            for shots in (0, 5):
                for b in range(0, len(load(root, ds)[1]), UNIT): yield ds, shots, b
def run(model, tok, root, ds, shots, start, size=UNIT):
    """Record numbers, probabilities of the option tokens per record and the number of tokens read."""
    M, items = load(root, ds); items = items[start:start + size]; head = M["question"] + "\n\nText:\n"; tail = ids(tok, "\n\n" + M["instruction"] + "<|im_end|>\n<|im_start|>assistant\n"); opt = [ids(tok, o)[0] for o in M["options"]]
    if len(set(opt)) != len(opt): raise ValueError(f"{ds} option tokens are not distinct")
    turns = [(head + tok.decode(ids(tok, s["text"], special=False, limit=SHOT_TOKENS)) + "\n\n" + M["instruction"], s["answer"]) for s in M["shots"][:shots]]; prefix = ids(tok, chat(SYSTEM, turns) + "<|im_start|>user\n" + head)
    sufs = [ids(tok, r["text"], special=False, limit=MAX_TOKENS) + tail for r in items]; p = np.array(next_probs(model, tok, prefix, sufs, opt)); return np.array([r["i"] for r in items]), p, np.array([len(s) - len(tail) for s in sufs])
def merge(root, out):
    """One line per set and prompt, for the sets whose units are all finished, rewritten as a whole."""
    rows = []
    for ds in SETS:
        if not os.path.exists(f"{root}/labels/{ds}.jsonl"): continue
        M, items = load(root, ds); codes = np.array(M["codes"]); N = len(items)
        for shots in (0, 5):
            files = [f"{root}/units/labels/{ds}.{shots}.{b}.npz" for b in range(0, N, UNIT)]
            if not all(os.path.exists(f) for f in files): continue
            Z = [np.load(f) for f in files]; I = np.concatenate([z["i"] for z in Z]); P = np.concatenate([z["p"] for z in Z]); T = np.concatenate([z["tokens"] for z in Z]); at = {int(a): k for k, a in enumerate(I)}; o = np.array([at[r["i"]] for r in items]); P, T = P[o], T[o]
            g = lambda k: np.array([r[k] for r in items]); y, c, rev, wrong, rej = g("y"), g("c"), g("rev") == 1, g("wrong") == 1, g("rej") == 1; pr = codes[P.argmax(1)]; flag = pr != y
            py = P[np.arange(N), np.array([list(codes).index(v) for v in y])] / np.maximum(P.sum(1), 1e-12); rk = np.argsort(py, kind="stable")     # lowest probability of the original label first
            def hits(s, name): return {f"{name}": int(s.sum()), f"{name}_flagged": int((flag & s).sum()), f"{name}_recall": float((flag & s).sum() / s.sum()) if s.any() else None, f"{name}_precision_all": float((flag & s).sum() / flag.sum()) if flag.any() else None, f"{name}_precision_reviewed": float((flag & s).sum() / (flag & rev).sum()) if (flag & rev).any() else None, f"{name}_precision_at_k": float(s[rk[:max(1, int(s.sum()))]].mean()) if s.any() else None}
            rows.append(dict(ds=ds, shots=shots, n=N, acc_original=float((pr == y).mean()), acc_corrected=float((pr == c).mean()), flagged=int(flag.sum()), reviewed=int(rev.sum()), flagged_reviewed=int((flag & rev).sum()), **hits(wrong, "confirmed"), confirmed_suggested_label=int(((pr == c) & wrong).sum()), **hits(rej, "rejected"), option_mass_mean=float(P.sum(1).mean()), tokens_mean=float(T.mean()), align=M["align"]))
    tmp = f"{out}.{os.getpid()}"; open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in rows)); os.replace(tmp, out); return len(rows)
