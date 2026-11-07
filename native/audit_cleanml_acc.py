#!/usr/bin/env python3
"""The CleanML runs again with accuracy beside Macro-F1, same keys, splits and seeds as run_cleanml.py. It checks that a stored score reproduces and compares the mean accuracy per table, error type, method and model with the accuracies the benchmark published (result.zip of its repository, 20 splits with a hyperparameter search). Linear, knn, rf and gb are compared with logistic regression, KNN, random forest and XGBoost, the benchmark has no MLP."""
import argparse, collections, glob, json, os, sys
import run_cleanml as M     # sets the thread limits
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import f1_score
def both(y, p): return [float(f1_score(y, p, average="macro")), float((y == p).mean())]
M.f1 = both
NAME = dict(linear="logistic_regression", knn="knn_classification", rf="random_forest_classification", gb="XGBoost")
def one(k):
    r = M.safe(k)
    if "error" in r: return r
    return dict({a: r[a] for a in ("id", "ds", "error_type", "method", "model", "seed", "n_test")}, score_f1=r["score"][0], acc=r["score"][1], test_f1=r["test"][0], test_acc=r["test"][1])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--results"); ap.add_argument("--published"); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=60); A = ap.parse_args(); A.repo, A.results, A.published, A.out = [os.path.abspath(x) for x in (A.repo, A.results, A.published, A.out)]
    M.init(A.repo); stored = {}
    for l in open(A.results): r = json.loads(l); stored[r["id"]] = r
    done = {json.loads(l)["id"] for l in open(A.out)} if os.path.exists(A.out) else set(); jobs = [dict(id=i, ds=r["ds"], error_type=r["error_type"], method=r["method"], model=r["model"], seed=r["seed"], root=os.path.join(A.repo, "data")) for i, r in stored.items() if i not in done]; print("jobs", len(jobs), flush=True)
    with Pool(A.procs, initializer=M.init, initargs=(A.repo,)) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 200 == 0: print(i, len(jobs), r["id"], flush=True)
    R = [json.loads(l) for l in open(A.out)]; bad = [r["id"] for r in R if "error" in r]; print("errors", len(bad), bad[:5]); same = [abs(r["score_f1"] - stored[r["id"]]["score"]) for r in R if "error" not in r]; print("runs", len(R), "stored Macro-F1 reproduced within 1e-9", sum(x < 1e-9 for x in same), "largest gap", max(same))
    mine = collections.defaultdict(list)
    for r in R:
        if r["model"] in NAME: mine[(r["ds"], r["error_type"], r["method"], r["model"])].append(r["acc"])
    pub = collections.defaultdict(list)
    for fn in glob.glob(f"{A.published}/*_result.json"):
        ds = os.path.basename(fn)[:-len("_result.json")]
        if ds not in M.TABLES: continue
        for k, v in json.load(open(fn)).items():
            _, _, err, me, mo, _ = k.split("/")
            if err != "mislabel" and f"{'dirty' if me == 'dirty' else me}_test_acc" in v: pub[(ds, err, me, mo)].append(v[f"{me}_test_acc"])
    rows = []
    for (ds, err, me, mo), a in sorted(mine.items()):
        p = pub.get((ds, err, me, NAME[mo]))
        if p: rows.append(dict(ds=ds, error_type=err, method=me, model=mo, acc=float(np.mean(a)), published_acc=float(np.mean(p)), published_sd=float(np.std(p)), gap=float(np.mean(a) - np.mean(p))))
    json.dump(rows, open(A.out + ".published.json", "w"), indent=0); g = collections.defaultdict(list)
    for x in rows: g[x["ds"]].append(x)
    for ds, v in sorted(g.items()): print(ds, "arms", len(v), "mean acc", round(np.mean([x["acc"] for x in v]), 3), "published", round(np.mean([x["published_acc"] for x in v]), 3), "largest gap", round(max(v, key=lambda x: abs(x["gap"]))["gap"], 3), max(v, key=lambda x: abs(x["gap"]))["method"], max(v, key=lambda x: abs(x["gap"]))["model"], flush=True)
    print("no published counterpart", sorted({k[0] for k in mine if not pub.get((k[0], k[1], k[2], NAME[k[3]]))}), flush=True); print("finished", flush=True)
