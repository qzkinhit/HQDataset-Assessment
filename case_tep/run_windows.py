#!/usr/bin/env python3
"""Window length on the classic Tennessee Eastman benchmark. For each length the windows are split by run or at random, with the labels from the fault onset and the same models as run_case.py."""
import argparse, json, os, time
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ONSET = 160; LENGTHS = (5, 10, 15, 20, 30, 40, 60, 80)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(REPO, "data", "tep")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_tep_windows.jsonl")); ap.add_argument("--procs", type=int, default=12); ap.add_argument("--seeds", type=int, default=5); A = ap.parse_args()
def read(fn):
    a = np.loadtxt(os.path.join(A.data, fn)); return a.T if a.shape[1] != 52 and a.shape[0] == 52 else a
TR = [read(f"d{k:02d}.dat") for k in range(22)]; TE = [read(f"d{k:02d}_te.dat") for k in range(22)]; POST = [TE[0]] + [TE[k][ONSET:] for k in range(1, 22)]
def mdl(name, seed):
    return {"rf": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1), "gb": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=150, random_state=seed)), "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=seed)),
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=1))}[name]()
def windows(view, L):
    Xs, ys = [], []
    for k, a in enumerate(view):
        for s in range(0, len(a) - L + 1): Xs.append(a[s:s + L].ravel()); ys.append(k)
    return np.asarray(Xs, np.float32), np.asarray(ys)
def job(key):
    name, seed, L, arm = key; rng = np.random.default_rng(seed + 7 + 1000 * L); t0 = time.time(); m = mdl(name, seed); Xa, ya = windows(TR, L); Xb, yb = windows(POST, L)
    if arm == "run":
        s = rng.permutation(len(ya))[:8000]; t = rng.permutation(len(yb))[:6000]; m.fit(Xa[s], ya[s]); sc = f1_score(yb[t], m.predict(Xb[t]), average="macro")
    else:
        X = np.vstack([Xa, Xb]); y = np.r_[ya, yb]; p = rng.permutation(len(y)); cut = int(0.7 * len(y)); tri, tei = p[:cut], p[cut:]; s = rng.permutation(len(tri))[:8000]; t = rng.permutation(len(tei))[:6000]; m.fit(X[tri[s]], y[tri[s]]); sc = f1_score(y[tei[t]], m.predict(X[tei[t]]), average="macro")
    return dict(model=name, seed=seed, length=L, split=arm, macro_f1=float(sc), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    done = set()
    if os.path.exists(A.out):
        for l in open(A.out): r = json.loads(l); done.add((r["model"], r["seed"], r["length"], r["split"]))
    keys = [(n, s, L, arm) for s in range(A.seeds) for L in LENGTHS for n in ("linear", "knn", "rf", "mlp", "gb") for arm in ("run", "random") if (n, s, L, arm) not in done]; print("jobs", len(keys), flush=True)
    with Pool(A.procs) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job, keys)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 20 == 0: print(i, len(keys), r["model"], r["length"], r["sec"], flush=True)
    print("done", flush=True)
