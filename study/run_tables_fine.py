#!/usr/bin/env python3
"""Finer shares of repair and deletion on the tabular track, for the response curves. Same runner, same splits and models as run_real_tables.py, only more values of the share."""
import argparse, json, os
from multiprocessing import Pool
from run_real_tables import init, safe
SHARES = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(REPO, "data", "tables")); ap.add_argument("--meta", default=os.path.join(REPO, "data", "tables", "table_meta.json")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_tables_fine.jsonl")); ap.add_argument("--procs", type=int, default=40); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    meta = json.load(open(a.meta)); done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add((r["ds"], r["model"], r["seed"], r["action"], r["f"]))
            except Exception: pass
    jobs = [dict(ds=ds, model=m, seed=s, action=x, f=f) for s in range(a.seeds) for ds in meta for m in ("linear", "rf", "gb", "knn", "mlp") for x in ("repair_records", "delete_records") for f in SHARES if (ds, m, s, x, f) not in done]
    print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.root, meta)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 200 == 0: print(i, len(jobs), flush=True)
    print("done", flush=True)
