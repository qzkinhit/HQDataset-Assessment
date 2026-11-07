#!/usr/bin/env python3
"""Scores the tables repaired by the language model with the unchanged runners of the tabular track. Same 5 model families, same 10 splits, same clean test records. The repaired table takes the place of the dirty one, so the error rates are measured on it as on any other version.
Each line holds the score of run_real_tables.py, the validation and test scores of run_decision.py and the rates behind the static score, so that the version can join the choice among versions. f is the mean share of training records the model has read."""
import argparse, json, os, sys
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--meta"); ap.add_argument("--cells"); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--only", default=""); ap.add_argument("--partial", action="store_true", help="also score tables whose rows are not all read yet")
ap.add_argument("--code", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study")); A = ap.parse_args()
sys.path.insert(0, A.code); import run_real_tables as R, run_decision as D
D.SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200)    # the sizes of run_decision_fine.py
CELLS = json.load(open(A.cells))
def run(k):
    try:
        key = dict(ds=k["ds"], model=k["model"], seed=k["seed"], action="keep", f=0.0); a = R.run(key, validation_sizes=D.SIZES); c = CELLS[k["ds"]]
        fields = ("rate_missing", "rate_domain", "rate_value", "rate_label", "rate_clean_records", "cell_error", "eval_protocol", "drop_cols", "time_cols", "n_features", "train_index_sha256", "test_index_sha256") + tuple(x for x in ("gb_l2", "gb_early_stopping", "fit_max_iter", "fit_n_iter", "max_abs_train_logit") if x in a)
        return dict(ds=k["ds"], model=k["model"], seed=k["seed"], action="llm_repair", f=round(c["share_train_mean"], 4), covered=c["share_train_seed"][k["seed"]], task=a["task"], score=a["score"], test=a["test"], val=a["val"], n_val=a["n_val"], n_train=a["n_train"], n0=a["n0"], n_test=a["n_test"], **{x: a[x] for x in fields}, sec=a["sec"])
    except Exception as e: return dict(k, action="llm_repair", error=repr(e)[:300])
if __name__ == "__main__":
    meta = {ds: m for ds, m in json.load(open(A.meta)).items() if ds in CELLS and (not A.only or ds in A.only.split(",")) and (A.partial or CELLS[ds].get("rows_read") == CELLS[ds].get("rows_planned"))}; done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            r = json.loads(l)
            if "error" not in r:
                if r.get("eval_protocol") != R.PROTOCOL: raise ValueError("Existing LLM scores use another protocol; choose a fresh output file")
                done.add((r["ds"], r["model"], r["seed"]))
    jobs = [dict(ds=ds, model=m, seed=s) for s in range(A.seeds) for ds in meta for m in ("linear", "rf", "gb", "knn", "mlp") if (ds, m, s) not in done]; print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(A.procs, initializer=D.init, initargs=(A.root, meta)) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 50 == 0: print(i, len(jobs), r.get("ds"), r.get("sec"), r.get("error", "")[:90], flush=True)
    print("finished", flush=True)
