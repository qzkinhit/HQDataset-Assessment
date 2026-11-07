#!/usr/bin/env python3
"""Prompts of the table repair baseline in the style of Narayan et al. (2022), a foundation model with a few examples per table. The candidate rows are the training records of the ten splits of run_real_tables.py, at most 3000 per table, drawn with seed 0, and each goes to the model as it stands in the dirty table, whether it holds an error or not.
The examples are rows that are no test record in any of the ten splits, their clean values appear as corrections. Each seed reads only the replies for its own training records, and the model keeps nothing between prompts, so no prompt read by a seed holds a test record of that seed. Wide tables go in blocks of columns. No other clean value leaves this script."""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--meta"); ap.add_argument("--out"); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--code", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study")); A = ap.parse_args()
sys.path.insert(0, A.code); import run_real_tables as R
WIDE, BLOCK = 40, 30; os.makedirs(A.out, exist_ok=True)
def pick(E, rows, k, rng):
    """k rows, each next row adds the most columns with an error that no example has shown yet"""
    rows = [int(i) for i in rng.permutation(rows)]; out = []; seen = np.zeros(E.shape[1], bool)
    for _ in range(min(k, len(rows))):
        b = max(rows, key=lambda i: int((E[i] & ~seen).sum())); rows.remove(b); out.append(b); seen |= E[b]
    return out
for ds, m in json.load(open(A.meta)).items():
    D = R.load(A.root, ds, m); n = len(D["a"]); cols = D["feat"] + [D["y"]]; E = np.c_[D["E"], D["yE"]]; rng = np.random.default_rng(0); tr, te = np.zeros(n, bool), np.zeros(n, bool)
    for s in range(A.seeds):
        p = np.random.default_rng(s + 500).permutation(n); cut = int(0.7 * n); tr[p[:cut][:R.CAP_TRAIN]] = True; te[p[cut:][:R.CAP_TEST]] = True
    wide = len(cols) > WIDE; kd, kc = (2, 1) if wide else (3, 2); ok = np.where(~te)[0]; dirty, clean = ok[E[ok].any(1)], ok[~E[ok].any(1)]
    kd, kc = kd + max(0, kc - len(clean)), kc + max(0, kd - len(dirty)); shots = pick(E, dirty, kd, rng) + [int(i) for i in rng.permutation(clean)[:kc]]; shots = [shots[i] for i in rng.permutation(len(shots))]
    if te[shots].any(): raise SystemExit(f"{ds} example in a test split")
    rows = rng.permutation(np.setdiff1d(np.where(tr)[0], shots))[:A.cap]; dv = D["a"][cols].apply(lambda s: s.str.strip()).values; cv = D["b"][cols].apply(lambda s: s.str.strip()).values
    blocks = [cols[i:i + BLOCK] for i in range(0, len(cols), BLOCK)] if wide else [cols]
    plan = dict(ds=ds, n=n, task=D["task"], columns=cols, blocks=blocks, shots=[dict(row=i, cells=dv[i].tolist(), fix={c: cv[i, j] for j, c in enumerate(cols) if E[i, j]}) for i in shots], rows=[int(i) for i in rows], cells=dv[rows].tolist())
    json.dump(plan, open(f"{A.out}/{ds}.json", "w"), ensure_ascii=False)
    print(ds, "n", n, "columns", len(cols), "blocks", len(blocks), "rows", len(rows), "dirty among them", int(E[rows].any(1).sum()), "examples", shots, "wrong cells per example", [int(E[i].sum()) for i in shots], flush=True)
