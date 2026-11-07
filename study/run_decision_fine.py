#!/usr/bin/env python3
"""Finer validation sizes for the decision experiment. Same runner, keys and splits as run_decision.py, the test scores are identical, only more nested validation samples are stored."""
import argparse, json, os
from multiprocessing import Pool
import run_decision as D
D.SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200)
if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(REPO, "data", "tables")); ap.add_argument("--meta", default=os.path.join(REPO, "data", "tables", "table_meta.json")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_decision_fine.jsonl")); ap.add_argument("--procs", type=int, default=40); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    meta = json.load(open(a.meta)); done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add((r["ds"], r["model"], r["seed"], r["action"], r["f"]))
            except Exception: pass
    jobs = [dict(ds=ds, model=m, seed=s, action=x, f=f) for s in range(a.seeds) for ds in meta for m in ("linear", "rf", "gb", "knn", "mlp") for x, f in D.ARMS if (ds, m, s, x, f) not in done]
    print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(a.procs, initializer=D.init, initargs=(a.root, meta)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(D.safe, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 200 == 0: print(i, len(jobs), flush=True)
    print("done", flush=True)
