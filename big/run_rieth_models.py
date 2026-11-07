#!/usr/bin/env python3
"""Large Tennessee Eastman benchmark, other model families on the same design as run_rieth.py: the same number of records per class drawn from k independent runs, and more records from few runs.
No defect is injected. Scored on a fixed sample of the published test records."""
import argparse, json, os, time, fcntl
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(REPO, "downloads", "rieth")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_rieth_models.jsonl")); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--seeds", type=int, default=5); A = ap.parse_args()
D = None
def init():
    global D
    z = np.load(f"{A.dir}/arrays.npz"); out = {}
    for tag, on in (("tr", 20), ("te", 160)):
        X, y, run, t = z[f"f_{tag}_X"], z[f"f_{tag}_y"], z[f"f_{tag}_run"], z[f"f_{tag}_t"]; m = t > on
        out[tag] = (np.vstack([z[f"ff_{tag}_X"], X[m]]), np.r_[z[f"ff_{tag}_y"], y[m]].astype(int), np.r_[z[f"ff_{tag}_run"], run[m]])
    rng = np.random.default_rng(0); idx = rng.choice(len(out["te"][1]), 400000, replace=False); D = dict(Xa=out["tr"][0], ya=out["tr"][1], ra=out["tr"][2], Xb=out["te"][0][idx], yb=out["te"][1][idx])
    D["cls"] = {c: (D["ya"] == c).nonzero()[0] for c in np.unique(D["ya"])}
def model(name, seed):
    return {"rf": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1), "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=seed)), "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), max_iter=80, random_state=seed)), "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=1))}[name]()
def job(j):
    t0 = time.time(); rng = np.random.default_rng([j["seed"], j["k"], j["n"], 7]); ids = []
    for c, pool in D["cls"].items():
        rs = np.unique(D["ra"][pool]); ch = rng.choice(rs, min(j["k"], len(rs)), replace=False); p = pool[np.isin(D["ra"][pool], ch)]; ids.append(rng.choice(p, min(j["n"], len(p)), replace=False))
    ids = np.concatenate(ids); m = model(j["model"], j["seed"]).fit(D["Xa"][ids], D["ya"][ids]); s = f1_score(D["yb"], m.predict(D["Xb"]), average="macro")
    return dict(j, n_train=int(len(ids)), macro_f1=float(s), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            r = json.loads(l); done.add((r["model"], r["seed"], r["k"], r["n"]))
    jobs = []
    for seed in range(A.seeds):
        for name in ("rf", "linear", "mlp", "knn"):
            for k in (1, 2, 5, 10, 20, 50, 100, 200, 500): jobs.append(dict(model=name, seed=seed, k=k, n=480))
            for k, n in ((5, 960), (5, 1920), (500, 960), (500, 1920), (500, 3840)): jobs.append(dict(model=name, seed=seed, k=k, n=n))
    jobs = [j for j in jobs if (j["model"], j["seed"], j["k"], j["n"]) not in done]; print("jobs", len(jobs), flush=True)
    with Pool(A.procs, initializer=init) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 10 == 0: print(i, len(jobs), r, flush=True)
    print("done", flush=True)
