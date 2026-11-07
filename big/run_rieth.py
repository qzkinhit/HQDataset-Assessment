#!/usr/bin/env python3
"""Large Tennessee Eastman benchmark (Rieth et al., Harvard Dataverse doi:10.7910/DVN/6C3JR1): 500 independent runs per class.
Question: with the same number of records, does the number of independent runs change task utility? And how far does a record-level split inflate the score at this scale?
No defect is injected. Records, labels and runs are the published ones."""
import argparse, json, os, time
import numpy as np, rdata, rdata.parser, rdata.conversion, lightgbm as lgb
from sklearn.metrics import f1_score
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(REPO, "downloads", "rieth")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_rieth.json")); ap.add_argument("--threads", type=int, default=20); ap.add_argument("--seeds", type=int, default=3); A = ap.parse_args()
CACHE = f"{A.dir}/arrays.npz"
def load():
    if os.path.exists(CACHE): z = np.load(CACHE); return {k: z[k] for k in z.files}
    out = {}
    for tag, fn in (("ff_tr", "TEP_FaultFree_Training"), ("ff_te", "TEP_FaultFree_Testing"), ("f_tr", "TEP_Faulty_Training"), ("f_te", "TEP_Faulty_Testing")):
        t0 = time.time(); d = rdata.conversion.convert(rdata.parser.parse_file(f"{A.dir}/{fn}.RData")); df = list(d.values())[0]; print(fn, df.shape, list(df.columns)[:5], round(time.time() - t0), flush=True)
        X = df.iloc[:, 3:].to_numpy(np.float32); out[tag + "_X"] = X; out[tag + "_y"] = df.iloc[:, 0].to_numpy(np.int16); out[tag + "_run"] = df.iloc[:, 1].to_numpy(np.int32); out[tag + "_t"] = df.iloc[:, 2].to_numpy(np.int16)
    np.savez(CACHE, **out); return out
D = load()
for k in ("ff_tr", "ff_te", "f_tr", "f_te"): print(k, D[k + "_X"].shape, flush=True)
# fault-present records only, following the published protocol: training faults start after sample 20, test faults after sample 160
def table(tag):
    X, y, run, t = D[f"f_{tag}_X"], D[f"f_{tag}_y"], D[f"f_{tag}_run"], D[f"f_{tag}_t"]; on = 20 if tag == "tr" else 160; m = t > on
    Xn, yn, rn = D[f"ff_{tag}_X"], D[f"ff_{tag}_y"], D[f"ff_{tag}_run"]
    return np.vstack([Xn, X[m]]), np.r_[yn, y[m]].astype(int), np.r_[rn, run[m]]
Xa, ya, ra = table("tr"); Xb, yb, rb = table("te"); print("train", Xa.shape, "test", Xb.shape, "classes", len(np.unique(ya)), flush=True)
res = dict(records_total=int(sum(D[k + "_X"].shape[0] for k in ("ff_tr", "ff_te", "f_tr", "f_te"))), train_records=int(len(ya)), test_records=int(len(yb)), runs=[]); rng0 = np.random.default_rng(0)
cases = np.unique(np.c_[ya, ra], axis=0)                # (class, run) pairs
te_cases = np.unique(np.c_[yb, rb], axis=0)
def fit(X, y, seed):
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, min_child_samples=50, reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=A.threads, random_state=seed, verbose=-1).fit(X, y)
te_mask = rng0.random(len(yb)) < 0.25                    # one quarter of the test records, all runs, enough for tight estimates
res["test_records_used"] = int(te_mask.sum())
def save(): json.dump(res, open(A.out, "w"), indent=0)
def per_class(k, n, rng):
    """n records of every class drawn from k simulation runs of that class"""
    ids = []
    for c in np.unique(ya):
        rs = np.unique(ra[ya == c]); chosen = rng.choice(rs, min(k, len(rs)), replace=False); pool = np.where((ya == c) & np.isin(ra, chosen))[0]; ids.append(rng.choice(pool, min(n, len(pool)), replace=False))
    return np.concatenate(ids)
for seed in range(A.seeds):
    rng = np.random.default_rng(seed + 100)
    # same records per class, 480 is the length of one faulty run, drawn from k runs: only the number of independent runs changes
    for k in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        t0 = time.time(); ids = per_class(k, 480, rng); m = fit(Xa[ids], ya[ids], seed); s = f1_score(yb[te_mask], m.predict(Xb[te_mask]), average="macro"); res["runs"].append(dict(arm="runs_fixed_records", k=k, seed=seed, n=int(len(ids)), macro_f1=float(s), sec=round(time.time() - t0))); print(res["runs"][-1], flush=True); save()
    # more records from a fixed number of runs, the record count a scale indicator would report
    for k in (5, 500):
        for n in (480, 960, 1920, 3840, 7680):
            if n > k * 480: continue
            t0 = time.time(); ids = per_class(k, n, rng); m = fit(Xa[ids], ya[ids], seed); s = f1_score(yb[te_mask], m.predict(Xb[te_mask]), average="macro"); res["runs"].append(dict(arm="records_fixed_runs", k=k, seed=seed, n=int(len(ids)), per_class=n, macro_f1=float(s), sec=round(time.time() - t0))); print(res["runs"][-1], flush=True); save()
    # record-level random split inside one pool against a split by run, same pool and same sizes, one fault label at a time keeps the classes balanced in both test sets
    t0 = time.time(); pool = rng.choice(len(ya), 1_500_000, replace=False); p = rng.permutation(pool); cut = int(0.8 * len(p)); m = fit(Xa[p[:cut][:400000]], ya[p[:cut][:400000]], seed); s1 = f1_score(ya[p[cut:]], m.predict(Xa[p[cut:]]), average="macro")
    uruns = np.unique(ra); held = rng.choice(uruns, int(0.2 * len(uruns)), replace=False); tr_idx = np.where(~np.isin(ra, held))[0]; te_idx = np.where(np.isin(ra, held))[0]; sub = rng.choice(tr_idx, 400000, replace=False); m2 = fit(Xa[sub], ya[sub], seed); tes = rng.choice(te_idx, 300000, replace=False); s2 = f1_score(ya[tes], m2.predict(Xa[tes]), average="macro")
    res["runs"].append(dict(arm="split_protocol", seed=seed, record_split=float(s1), run_split=float(s2), sec=round(time.time() - t0))); print(res["runs"][-1], flush=True); save()
print("done", flush=True)
