#!/usr/bin/env python3
"""Trial assessment of the classic Tennessee Eastman fault-diagnosis benchmark (Braatz distribution, files d00 to d21, _te for test).
Every number of the survey about the running example comes from the JSON file written here.
The models are off the shelf, the question concerns the data and the protocol.
"""
import argparse, hashlib, json, os, time
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
ONSET, HARD, L = 160, [3, 9, 15], 20      # test fault files contain the fault from sample 161, window length 20 samples, one hour of plant time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(REPO, "data", "tep")); ap.add_argument("--out", default=os.path.join(REPO, "results", "case_tep.json")); ap.add_argument("--procs", type=int, default=60); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--only-arm", default=None, help="run one arm and merge it into an existing output file"); A = ap.parse_args()
def load():
    F = {}; fmt = []
    for k in range(22):
        for suf in ("", "_te"):
            fn = f"d{k:02d}{suf}.dat"; a = np.loadtxt(os.path.join(A.data, fn)); fmt.append(dict(file=fn, shape=list(a.shape), conforms=bool(a.shape[1] == 52)))
            if a.shape[1] != 52 and a.shape[0] == 52: a = a.T
            F[fn] = a
    return F, fmt
F, FMT = load()
TR = [F[f"d{k:02d}.dat"] for k in range(22)]; TE = [F[f"d{k:02d}_te.dat"] for k in range(22)]
def mdl(name, seed):
    return {"rf": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1), "gb": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=150, random_state=seed)), "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=seed)),
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=1))}[name]()
def mf1(y, p): return float(f1_score(y, p, average="macro"))
def rows(view):
    """view: list of per-class arrays. Returns X, y, run id, row position inside the run."""
    X = np.vstack(view); y = np.concatenate([np.full(len(a), k) for k, a in enumerate(view)]); r = y.copy(); pos = np.concatenate([np.arange(len(a)) for a in view]); return X, y, r, pos
def windows(view, run_base=0):
    Xs, ys, rs, ss = [], [], [], []
    for k, a in enumerate(view):
        for s in range(0, len(a) - L + 1):
            Xs.append(a[s:s + L].ravel()); ys.append(k); rs.append(k + run_base); ss.append(s)
    return np.asarray(Xs, np.float32), np.asarray(ys), np.asarray(rs), np.asarray(ss)
POST = [TE[0]] + [TE[k][ONSET:] for k in range(1, 22)]
def job(k):
    name, seed, arm = k; rng = np.random.default_rng(seed + 7); t0 = time.time(); r = dict(model=name, seed=seed, arm=arm)
    m = mdl(name, seed)
    if arm == "official":
        Xa, ya, _, _ = rows(TR); Xb, yb, _, _ = rows(POST); m.fit(Xa, ya); p = m.predict(Xb); r["macro_f1"] = mf1(yb, p)
        f = f1_score(yb, p, average=None, labels=list(range(22))); r["per_class"] = [float(x) for x in f]; r["hard"] = float(np.mean(f[HARD])); r["other_faults"] = float(np.mean([f[i] for i in range(1, 22) if i not in HARD]))
        # bootstrap over test rows, for the resolution of this measurement
        B = []
        for _ in range(200):
            idx = rng.integers(0, len(yb), len(yb)); B.append(mf1(yb[idx], p[idx]))
        r["boot_lo"], r["boot_hi"] = float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))
        # the same predictions scored with file-name labels on all rows, the label defect in evaluation
        Xc = np.vstack(TE); yn = np.concatenate([np.full(len(a), k) for k, a in enumerate(TE)]); yt = yn.copy(); o = 0
        for k2, a in enumerate(TE):
            if k2 > 0: yt[o:o + ONSET] = 0
            o += len(a)
        pc = m.predict(Xc); r["eval_file_name_labels"] = mf1(yn, pc); r["eval_onset_labels"] = mf1(yt, pc)
    elif arm == "row_random":
        Xt, yt, _, _ = rows(TR); Xp, yp, _, _ = rows(POST); Xa = np.vstack([Xt, Xp]); ya = np.r_[yt, yp]; p = rng.permutation(len(ya)); cut = int(0.7 * len(ya)); m.fit(Xa[p[:cut]], ya[p[:cut]]); r["macro_f1"] = mf1(ya[p[cut:]], m.predict(Xa[p[cut:]]))
    elif arm in ("window_run", "window_random"):
        Xa, ya, ra, sa = windows(TR, 0); Xb, yb, rb, sb = windows(POST, 100)
        if arm == "window_run":
            s = rng.permutation(len(ya))[:8000]; t = rng.permutation(len(yb))[:6000]; m.fit(Xa[s], ya[s]); r["macro_f1"] = mf1(yb[t], m.predict(Xb[t]))
        else:
            X = np.vstack([Xa, Xb]); y = np.r_[ya, yb]; run = np.r_[ra, rb]; st = np.r_[sa, sb]; p = rng.permutation(len(y)); cut = int(0.7 * len(y)); tri, tei = p[:cut], p[cut:]
            cov = {}
            for i in tri:
                cov.setdefault(int(run[i]), np.zeros(1200, bool))[st[i]:st[i] + L] = True
            ov = np.array([cov[int(run[i])][st[i]:st[i] + L].mean() if int(run[i]) in cov else 0.0 for i in tei])
            r["window_rows_also_in_train"] = float(ov.mean()); r["windows_with_any_overlap"] = float((ov > 0).mean())
            s = rng.permutation(len(tri))[:8000]; t = rng.permutation(len(tei))[:6000]; m.fit(X[tri[s]], y[tri[s]]); r["macro_f1"] = mf1(y[tei[t]], m.predict(X[tei[t]]))
    elif arm == "repair":
        # intervention on the label defect: train on the test files, score on the training files, change only the labels of the pre-onset rows
        Xc = np.vstack(TE); yn = np.concatenate([np.full(len(a), k) for k, a in enumerate(TE)]); pre = np.zeros(len(yn), bool); o = 0
        for k2, a in enumerate(TE):
            if k2 > 0: pre[o:o + ONSET] = True
            o += len(a)
        Xe, ye, _, _ = rows(TR)
        m1 = mdl(name, seed).fit(Xc, yn); r["keep_defect"] = mf1(ye, m1.predict(Xe))
        yr = yn.copy(); yr[pre] = 0; m2 = mdl(name, seed).fit(Xc, yr); r["relabel_normal"] = mf1(ye, m2.predict(Xe))
        m3 = mdl(name, seed).fit(Xc[~pre], yn[~pre]); r["drop_pre_onset"] = mf1(ye, m3.predict(Xe))
    r["sec"] = round(time.time() - t0, 1); return r
if __name__ == "__main__":
    allr = np.vstack(list(F.values()))
    res = dict(format=dict(files=len(FMT), conforming=sum(f["conforms"] for f in FMT), nonconforming=[f for f in FMT if not f["conforms"]]),
               scan=dict(records=int(allr.shape[0]), variables=int(allr.shape[1]), missing=int(np.isnan(allr).sum()), non_finite=int((~np.isfinite(allr)).sum()),
                         duplicate_records=int(allr.shape[0] - np.unique(allr.round(8), axis=0).shape[0]), constant_variables=[int(j) for j in range(52) if np.ptp(allr[:, j]) == 0]),
               shapes=dict(train_normal=list(F["d00.dat"].shape), train_fault=list(F["d01.dat"].shape), test_normal=list(F["d00_te.dat"].shape), test_fault=list(F["d01_te.dat"].shape)),
               pre_onset=dict(rows_per_test_fault_file=ONSET, of=F["d01_te.dat"].shape[0], share=ONSET / F["d01_te.dat"].shape[0], share_of_all_test_records=21 * ONSET / sum(len(a) for a in TE)),
               data_sha256=hashlib.sha256(b"".join(open(os.path.join(A.data, f"d{k:02d}{s}.dat"), "rb").read() for k in range(22) for s in ("", "_te"))).hexdigest())
    jobs = [(n, s, arm) for s in range(A.seeds) for n in ("rf", "gb", "mlp", "linear", "knn") for arm in ("official", "row_random", "repair")] + [(n, s, arm) for s in range(min(A.seeds, 10)) for n in ("rf", "gb", "mlp", "linear", "knn") for arm in ("window_run", "window_random")]
    if A.only_arm: jobs = [j for j in jobs if j[2] == A.only_arm]
    out = []
    with Pool(A.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            out.append(r)
            if i % 20 == 0: print(i, len(jobs), r["model"], r["arm"], r["sec"], flush=True); json.dump(dict(res, runs=out), open(A.out + ".part", "w"))
    if A.only_arm and os.path.exists(A.out): out = [r for r in json.load(open(A.out))["runs"] if r["arm"] != A.only_arm] + out
    res["runs"] = out; res["hard_faults"] = HARD; res["onset_rows"] = ONSET; res["window"] = L
    json.dump(res, open(A.out, "w"), indent=0); print("done", flush=True)
