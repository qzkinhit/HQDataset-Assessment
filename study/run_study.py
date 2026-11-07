#!/usr/bin/env python3
"""Controlled study: does a static quality score track task utility?

For each dataset, downstream model, split seed, defect type and defect rate, the training records receive one defect at a
known record-level rate r, the test records stay clean, and the script records the task score next to the score that the
national standard GB/T 48562-2026 would give.
Because r is the share of affected records, the affected sub-indicator equals 1 - r for every defect type, so that equal
score losses can be compared across defect types.

Results append to JSONL, finished keys are skipped, seeds run in order.
"""
import argparse, itertools, json, os, sys, time, warnings, hashlib, zlib
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")

RATES = [0.05, 0.10, 0.20, 0.30, 0.40]
DEFECTS = ["missing", "format", "value", "duplicate", "label", "coverage", "scale"]
PAIRS = list(itertools.combinations(["missing", "value", "duplicate", "label", "coverage", "scale"], 2))
PAIR_RATES = [0.10, 0.20]
CAP_TRAIN, CAP_TEST = 8000, 4000
# ------------------------------------------------------------------ data
def load_all(root):
    """Tables keep their raw columns so that defects act before encoding. Feature matrices come as float arrays."""
    D = {}
    for fn in sorted(os.listdir(f"{root}/tables")) if os.path.isdir(f"{root}/tables") else []:
        if not fn.endswith(".json"): continue
        name = fn[:-5]; m = json.load(open(f"{root}/tables/{fn}")); df = pd.read_csv(f"{root}/tables/{name}.csv")
        D[name] = dict(kind="table", df=df, target=m["label_col"], task=m["task"], cats=[c for c in m["categorical_cols"] if c in df.columns and c != m["label_col"]], group="tabular")
    for name, task, group in [("tep", "cls", "process"), ("cifar10_clip", "cls", "image"), ("agnews_tfidf", "cls", "text")]:
        p = f"{root}/matrices/{name}.npz"
        if not os.path.exists(p): continue
        z = np.load(p)
        D[name] = dict(kind="matrix", X=z["X"].astype(np.float32), y=z["y"], task=task, group=group, is_test=z["is_test"], X_alt=(z["X_alt"].astype(np.float32) if "X_alt" in z.files else None))
    return D

def models(task, seed):
    if task == "cls":
        return dict(linear=LogisticRegression(max_iter=300, random_state=seed), rf=RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
                    gb=HistGradientBoostingClassifier(max_iter=100, random_state=seed), knn=KNeighborsClassifier(n_neighbors=15, n_jobs=1),
                    mlp=MLPClassifier((64,), max_iter=60, random_state=seed))
    return dict(linear=Ridge(alpha=1.0), rf=RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                gb=HistGradientBoostingRegressor(max_iter=100, random_state=seed), knn=KNeighborsRegressor(n_neighbors=15, n_jobs=1),
                mlp=MLPRegressor((64,), max_iter=60, random_state=seed))

def score(task, y, p): return float(f1_score(y, p, average="macro")) if task == "cls" else float(r2_score(y, p))

# ------------------------------------------------------------------ defects, applied to the training records only
def pick(rng, n, r): return rng.choice(n, size=int(round(r * n)), replace=False)

def apply_table(df, target, cats, defect, r, rng, task):
    d = df.copy().reset_index(drop=True); n = len(d); feat = [c for c in d.columns if c != target]; idx = pick(rng, n, r)
    num = [c for c in feat if c not in cats]
    k = max(1, int(round(0.3 * len(feat))))
    if defect == "missing":
        for i in idx:
            d.loc[i, list(rng.choice(feat, size=k, replace=False))] = np.nan
    elif defect == "format":
        for i in idx:
            c = rng.choice(feat)
            if c in num:
                v = pd.to_numeric(pd.Series([d.loc[i, c]]), errors="coerce").iloc[0]; d.loc[i, c] = v * 1000.0 + 1.0 if pd.notna(v) else 1e6
            else: d.loc[i, c] = "###"
    elif defect == "value":
        for i in idx:
            cols = list(rng.choice(feat, size=k, replace=False)); j = int(rng.integers(n))
            d.loc[i, cols] = df.iloc[j][cols].values
    elif defect == "duplicate":
        rest = np.setdiff1d(np.arange(n), idx); src = rng.choice(rest, size=len(idx), replace=True)
        d.iloc[idx] = df.iloc[src].values
    elif defect == "label":
        y = d[target].values.copy(); labs = pd.unique(df[target])
        for i in idx:
            if task == "reg": y[i] = df[target].values[int(rng.integers(n))]
            else: y[i] = rng.choice(labs[labs != y[i]])
        d[target] = y
    elif defect == "coverage":
        keep = coverage_mask(encode_numeric(df[num], rng) if num else None, n, r, rng, d, cats)
        rest = np.where(keep)[0]; fill = rng.choice(rest, size=n - len(rest), replace=True)
        d = df.iloc[np.concatenate([rest, fill])].reset_index(drop=True)
    elif defect == "scale":
        d = df.drop(index=df.index[idx]).reset_index(drop=True)
    return d

def encode_numeric(frame, rng):
    x = frame.apply(pd.to_numeric, errors="coerce").values.astype(float); x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)

def coverage_mask(z, n, r, rng, d, cats):
    """Remove one contiguous region of the data: the top r share along a random direction, the stand-in for an unseen operating condition."""
    if z is None:
        z = np.column_stack([pd.factorize(d[c])[0] for c in cats]).astype(float); z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)
    w = rng.normal(size=z.shape[1]); s = z @ w; cut = np.quantile(s, 1 - r)
    return s <= cut

def apply_matrix(X, y, defect, r, rng, task, X_alt=None):
    n, dim = X.shape; idx = pick(rng, n, r); A, b = X.copy(), y.copy(); k = max(1, int(round(0.3 * dim)))
    if defect == "missing":
        for i in idx: A[i, rng.choice(dim, size=k, replace=False)] = np.nan
    elif defect == "format":
        for i in idx:
            c = int(rng.integers(dim)); A[i, c] = A[i, c] * 1000.0 + 1.0
    elif defect == "value":
        if X_alt is not None: A[idx] = X_alt[idx]                 # corrupted content, re-encoded
        else:
            for i in idx:
                cols = rng.choice(dim, size=k, replace=False); A[i, cols] = X[int(rng.integers(n)), cols]
    elif defect == "duplicate":
        rest = np.setdiff1d(np.arange(n), idx); src = rng.choice(rest, size=len(idx), replace=True); A[idx], b[idx] = X[src], y[src]
    elif defect == "label":
        labs = np.unique(y)
        for i in idx: b[i] = rng.choice(labs[labs != b[i]])
    elif defect == "coverage":
        keep = coverage_mask((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9), n, r, rng, None, None)
        rest = np.where(keep)[0]; fill = rng.choice(rest, size=n - len(rest), replace=True); ids = np.concatenate([rest, fill]); A, b = X[ids], y[ids]
    elif defect == "scale":
        keep = np.setdiff1d(np.arange(n), idx); A, b = X[keep], y[keep]
    return A, b

# ------------------------------------------------------------------ one run
DATA = None
def init(root):
    global DATA; DATA = load_all(root)

def split(n, seed, is_test=None):
    """Random 70/30 split for tables. Matrices keep their official train and test parts, the seed only draws the capped subsets."""
    rng = np.random.default_rng(seed + 1000)
    if is_test is None:
        p = rng.permutation(n); cut = int(0.7 * n); tr, te = p[:cut], p[cut:]
    else:
        tr, te = rng.permutation(np.where(~is_test)[0]), rng.permutation(np.where(is_test)[0])
    return tr[:CAP_TRAIN], te[:CAP_TEST]

def run(key):
    ds, model_name, seed, defects = key["ds"], key["model"], key["seed"], key["defects"]
    t0 = time.time(); D = DATA[ds]; rng = np.random.default_rng(zlib.crc32(json.dumps(key, sort_keys=True).encode()) % (2 ** 32))
    leak_test = None
    if D["kind"] == "table":
        df = D["df"]; tr, te = split(len(df), seed); dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True); tgt, cats = D["target"], D["cats"]
        for d, r in defects.items():
            if d == "contamination":
                n_leak = int(round(r * len(dtr))); li = rng.choice(len(dte), size=min(n_leak, len(dte)), replace=False); ti = rng.choice(len(dtr), size=len(li), replace=False)
                dtr.iloc[ti] = dte.iloc[li].values; leak_test = li
            else: dtr = apply_table(dtr, tgt, cats, d, r, rng, D["task"])
        num = [c for c in dtr.columns if c != tgt and c not in cats]
        for c in num: dtr[c] = pd.to_numeric(dtr[c], errors="coerce"); dte[c] = pd.to_numeric(dte[c], errors="coerce")
        for c in cats: dtr[c] = dtr[c].astype(object).where(dtr[c].notna(), None); dte[c] = dte[c].astype(object).where(dte[c].notna(), None)
        ct = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)] + ([("c", make_pipeline(SimpleImputer(strategy="most_frequent", missing_values=None), OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)), cats)] if cats else []))
        Xtr = ct.fit_transform(dtr.drop(columns=[tgt])); Xte = ct.transform(dte.drop(columns=[tgt])); ytr, yte = dtr[tgt].values, dte[tgt].values
    else:
        X, y = D["X"], D["y"]; tr, te = split(len(y), seed, D["is_test"]); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]; alt = D["X_alt"][tr] if D["X_alt"] is not None else None
        for d, r in defects.items():
            if d == "contamination":
                n_leak = int(round(r * len(ytr))); li = rng.choice(len(yte), size=min(n_leak, len(yte)), replace=False); ti = rng.choice(len(ytr), size=len(li), replace=False)
                Xtr = Xtr.copy(); ytr = ytr.copy(); Xtr[ti], ytr[ti] = Xte[li], yte[li]; leak_test = li
            else: Xtr, ytr = apply_matrix(Xtr, ytr, d, r, rng, D["task"], alt)
        imp = SimpleImputer(strategy="median"); sc = StandardScaler(); Xtr = sc.fit_transform(imp.fit_transform(Xtr)); Xte = sc.transform(imp.transform(Xte))
    m = models(D["task"], seed)[model_name]
    y1 = np.asarray(ytr); y2 = np.asarray(yte)
    if D["task"] == "cls":
        lab = {v: i for i, v in enumerate(sorted(set(map(str, y1)) | set(map(str, y2))))}
        y1 = np.array([lab[str(v)] for v in y1]); y2 = np.array([lab[str(v)] for v in y2])
    else:
        y1, y2 = y1.astype(float), y2.astype(float)
    m.fit(Xtr, y1); p = m.predict(Xte)
    out = dict(key, task=D["task"], group=D["group"], n_train=int(len(y1)), n_test=int(len(y2)), score=score(D["task"], y2, p), sec=round(time.time() - t0, 2))
    if leak_test is not None:
        mask = np.ones(len(y2), bool); mask[leak_test] = False; out["score_unseen"] = score(D["task"], y2[mask], p[mask])
    return out

def tasks(D, seeds):
    for seed in seeds:
        for ds, d in D.items():
            for mdl in ("linear", "rf", "gb", "knn", "mlp"):
                yield dict(ds=ds, model=mdl, seed=seed, defects={})
                for df in DEFECTS:
                    if d["kind"] == "matrix" and d["group"] in ("image", "text") and df in ("missing", "format"): continue   # no cell structure in these encodings
                    for r in RATES: yield dict(ds=ds, model=mdl, seed=seed, defects={df: r})
                for r in RATES: yield dict(ds=ds, model=mdl, seed=seed, defects=dict(contamination=r))
                for a, b in PAIRS:
                    if d["kind"] == "matrix" and d["group"] in ("image", "text") and "missing" in (a, b): continue
                    for r in PAIR_RATES: yield dict(ds=ds, model=mdl, seed=seed, defects={a: r, b: r})

def safe(key):
    try: return run(key)
    except Exception as e: return dict(key, error=repr(e)[:300])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=60); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--only", default="")
    a = ap.parse_args(); D = load_all(a.root)
    if a.only: D = {k: v for k, v in D.items() if k in a.only.split(",")}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add(json.dumps(dict(ds=r["ds"], model=r["model"], seed=r["seed"], defects=r["defects"]), sort_keys=True))
            except Exception: pass
    pending = [k for k in tasks(D, range(a.seeds)) if json.dumps(k, sort_keys=True) not in done]
    print("datasets", {k: (v["task"], v["group"]) for k, v in D.items()}, "pending", len(pending), "done", len(done), flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.root,)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, pending, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if i % 200 == 0: f.flush(); print(i, len(pending), r.get("ds"), r.get("sec"), r.get("error", "")[:80], flush=True)
    print("finished", flush=True)
