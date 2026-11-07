#!/usr/bin/env python3
"""Applies the replies of the language model to the dirty tables and measures cell precision, recall and F1 against the clean version on the rows the model has read. Rows without a reply stay as they are.
A cell counts as changed when its new value differs from the dirty one, and as repaired when it was wrong and now equals the clean one, with the same comparison of values as run_real_tables.py. A reply that would add a class to the label of a classification table is dropped and counted, the classes of a task are fixed, and one class with a single training record makes gradient boosting collapse whatever the data quality. Writes one table per dataset in the layout that the runners read."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--units"); ap.add_argument("--prompts"); ap.add_argument("--root"); ap.add_argument("--meta"); ap.add_argument("--out"); ap.add_argument("--cells"); ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--code", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study")); A = ap.parse_args()
sys.path.insert(0, A.code); import run_real_tables as R
def same(x, y, numeric):
    x, y = x.strip(), y.strip()
    if x == y: return True
    if not numeric: return False
    if x in R.NA and y in R.NA: return True
    try: a, b = float(x), float(y); return bool(abs(a - b) <= 1e-9 * (1 + abs(b)))
    except ValueError: return False
def counts(before, after, changed):
    tp, ch, wr = int((before & ~after).sum()), int(changed.sum()), int(before.sum()); pr, rc = tp / ch if ch else None, tp / wr if wr else None
    return dict(wrong_before=wr, changed=ch, repaired=tp, damaged=int((~before & after).sum()), wrong_left=int(after.sum()), precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr and rc else (0.0 if ch and wr else None))
meta = json.load(open(A.meta)); res = {}
for ds, m in meta.items():
    if not os.path.exists(f"{A.prompts}/{ds}.json"): continue
    P = json.load(open(f"{A.prompts}/{ds}.json")); cols = P["columns"]; pos = {c: j for j, c in enumerate(cols)}; D0 = R.load(A.root, ds, m); a = pd.read_csv(f"{A.root}/{ds}/dirty_index.csv", dtype=str, keep_default_na=False); n = len(a); changed = np.zeros((n, len(cols)), bool); seen = np.zeros(n, bool); bad = raw_cut = replies = 0; fixes = []
    for fn in sorted(glob.glob(f"{A.units}/{ds}.*.jsonl")):
        for l in open(fn):
            r = json.loads(l); seen[r["row"]] = True; bad += r["bad"]; replies += len(r["raw"]); fixes.append((r["row"], r["fix"]))
    if not seen.any(): continue
    numeric = {c: c in D0["num"] or (c == D0["y"] and m["task"] == "reg") for c in cols}; vals = {c: a[c].values.copy() for c in cols}; classes = set(D0["a"][D0["y"]].str.strip()); dropped = 0
    for i, fix in fixes:
        for c, v in fix.items():
            if m["task"] == "cls" and c == D0["y"] and v not in classes: dropped += 1
            elif not same(v, vals[c][i], numeric[c]): changed[i, pos[c]] = True; vals[c][i] = v
    for c in cols: a[c] = vals[c]
    os.makedirs(f"{A.out}/{ds}", exist_ok=True); a.to_csv(f"{A.out}/{ds}/dirty_index.csv.part", index=False); os.replace(f"{A.out}/{ds}/dirty_index.csv.part", f"{A.out}/{ds}/dirty_index.csv")
    link = f"{A.out}/{ds}/clean_index.csv"
    if not os.path.exists(link): os.symlink(os.path.abspath(f"{A.root}/{ds}/clean_index.csv"), link)
    D1 = R.load(A.out, ds, m); b0, b1 = np.c_[D0["E"], D0["yE"]], np.c_[D1["E"], D1["yE"]]; share = []; tr0 = None
    for s in range(A.seeds):
        p = np.random.default_rng(s + 500).permutation(n); cut = int(0.7 * n); tr = p[:cut][:R.CAP_TRAIN]; share.append(float(seen[tr].mean()))
        if s == 0: tr0 = np.zeros(n, bool); tr0[tr] = True
    y = len(cols) - 1; S = seen & tr0
    res[ds] = dict(ds=ds, task=m["task"], rows=n, rows_read=int(seen.sum()), rows_planned=len(P["rows"]), share_read=float(seen.mean()), share_train_mean=float(np.mean(share)), share_train_seed=share, blocks=len(P["blocks"]), replies=replies, replies_not_parsed=bad, label_replies_dropped=dropped,
        cells=counts(b0[seen], b1[seen], changed[seen]), cells_features=counts(b0[seen][:, :y], b1[seen][:, :y], changed[seen][:, :y]), cells_label=counts(b0[seen][:, y], b1[seen][:, y], changed[seen][:, y]), cells_train_seed0=counts(b0[S], b1[S], changed[S]))
    print(ds, res[ds]["rows_read"], res[ds]["cells"], flush=True)
json.dump(res, open(A.cells, "w"), indent=0)
