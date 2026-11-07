#!/usr/bin/env python3
"""The value of a task test for a decision. On the tables with benchmark errors, the assessor chooses one version of the training data among candidates.
The score of the standard picks the version with the fewest errors. A task test picks by a small clean validation sample of size n, and the pick is scored on separate clean test records.
The script stores, per run, the scores on nested validation samples and on the test part, so that every decision rule can be replayed afterwards. Nothing is injected."""
import argparse, json, os, time, zlib, warnings
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import run_real_tables as R
warnings.filterwarnings("ignore")
SIZES = (25, 50, 100, 200, 400, 800, 1600, 3200)
ARMS = [("keep", 0.0), ("repair_records", 0.25), ("repair_records", 0.5), ("repair_records", 0.75), ("repair_records", 1.0), ("delete_records", 0.25), ("delete_records", 0.5), ("delete_records", 0.75), ("delete_records", 1.0), ("repair_cols_high", 0.5), ("repair_cols_low", 0.5), ("repair_type_missing", 1.0), ("repair_type_domain", 1.0), ("repair_type_value", 1.0), ("repair_labels", 1.0)]
def init(root, meta): R.init(root, meta)
def metric(task, y, p): return float(f1_score(y, p, average="macro")) if task == "cls" else float(r2_score(y, p))
def run(key):
    # One fitted model supplies full-test, validation, and held-out scores.
    return R.run(key, validation_sizes=SIZES)

def safe(k):
    try: return run(k)
    except Exception as e: return dict(k, error=repr(e)[:300])
if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(REPO, "data", "tables")); ap.add_argument("--meta", default=os.path.join(REPO, "data", "tables", "table_meta.json")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_decision.jsonl")); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--jobs", help="Optional JSONL of job keys inside the grid"); a = ap.parse_args()
    meta = json.load(open(a.meta)); done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add((r["ds"], r["model"], r["seed"], r["action"], r["f"]))
            except Exception: pass
    jobs = [dict(ds=ds, model=mdl, seed=s, action=x, f=f) for s in range(a.seeds) for ds in meta for mdl in ("linear", "rf", "gb", "knn", "mlp") for x, f in ARMS if (ds, mdl, s, x, f) not in done]; print("jobs", len(jobs), flush=True)
    if a.jobs:
        requested = [json.loads(l) for l in open(a.jobs) if l.strip()]
        identity = lambda j: (j["ds"], j["model"], j["seed"], j["action"], j["f"])
        valid = {identity(j) for j in jobs} | done
        wanted = {identity(j) for j in requested}
        if len(wanted) != len(requested) or not wanted <= valid: raise ValueError("Job keys must be unique and inside the grid")
        jobs = [j for j in jobs if identity(j) in wanted]
        meta = {ds: m for ds, m in meta.items() if any(j["ds"] == ds for j in jobs)}
        print("selected jobs", len(jobs), flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.root, meta)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1 if a.jobs else 2)):
            f.write(json.dumps(r) + "\n")
            if a.jobs or i % 100 == 0: f.flush(); print(i, len(jobs), r.get("ds"), r.get("action"), r.get("sec"), r.get("error", "")[:90], flush=True)
    print("finished", flush=True)
