#!/usr/bin/env python3
"""CleanML tables. Thirteen real tables with their native errors and the cleaning methods of the benchmark (Li et al. 2021), nothing is injected and the mislabel variants are left out. For every table, error type and cleaning method, plus the dirty version, a model is trained on that version of the training part and scored by Macro-F1 on the same version of the test part.
The script stores nested validation scores, so that every decision rule can be replayed, and the records and cells the method changed against the dirty version. The rows of a seed fall into validation or test by one permutation of the dirty test part, so that a row keeps its side in every version. Text columns are encoded as in the benchmark, class imbalance is treated as in the benchmark by down-sampling the training part."""
import argparse, json, os, sys, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "study"))
import run_real_tables as R     # sets the thread limits, models and caps
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")
SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200)
MODELS = ("linear", "rf", "gb", "knn", "mlp")
def f1(y, p): return float(f1_score(y, p, average="macro"))
def read(root, ds, err, method, part):
    df = pd.read_csv(f"{root}/{ds['data_dir']}/{err}/{method}_{part}.csv")
    for c in ds.get("categorical_variables", []): df[c] = df[c].astype(str).replace("nan", np.nan)
    return df
def label(s):
    """one spelling per class, the versions store a label as 1, 1.0 or a string"""
    t = s.astype(str).str.strip(); v = pd.to_numeric(t, errors="coerce"); t = t.values.astype(object); ok = v.notna().values
    t[ok] = [str(int(x)) if float(x).is_integer() else str(x) for x in v.values[ok]]; return t
def kept(root, ds, err, method, part, n, v):
    """rows of the dirty table that remain in this version, read from the indicator of the benchmark"""
    if v == n: return np.ones(n, bool)
    k = ~pd.read_csv(f"{root}/{ds['data_dir']}/{err}/indicator_{method}_{part}.csv").values.astype(bool).any(axis=1)
    if len(k) != n or k.sum() != v: raise ValueError(f"indicator does not match {method} {part} {len(k)} {n} {k.sum()} {v}")
    return k
def changed(a, b):
    """cells whose value differs, a number compared as a number and two missing values as equal"""
    tot = 0
    for c in a.columns:
        sa, sb = a[c].astype(str).str.strip(), b[c].astype(str).str.strip(); xn = pd.to_numeric(sa, errors="coerce").values.astype(float); yn = pd.to_numeric(sb, errors="coerce").values.astype(float); num = np.isfinite(xn) & np.isfinite(yn); both = a[c].isna().values & b[c].isna().values
        d = np.where(num, np.abs(xn - yn) > 1e-9 * (1 + np.abs(yn)), sa.values != sb.values); tot += int((d & ~both).sum())
    return tot
def features(ds, Tr, Te, ytr):
    drop = set(ds.get("drop_variables", [])) | {ds["label"]}; text = [c for c in ds.get("text_variables", []) if c not in drop]; feat = [c for c in Tr.columns if c not in drop and c not in text]; declared = set(ds.get("categorical_variables", []))
    num = [c for c in feat if c not in declared and pd.api.types.is_numeric_dtype(Tr[c]) and not pd.api.types.is_bool_dtype(Tr[c])]; cat = [c for c in feat if c not in num]; A, B = Tr[feat].copy(), Te[feat].copy()
    for c in num: A[c] = pd.to_numeric(A[c], errors="coerce"); B[c] = pd.to_numeric(B[c], errors="coerce")
    for c in cat: A[c] = A[c].map(lambda v: np.nan if pd.isna(v) else str(v).strip()); B[c] = B[c].map(lambda v: np.nan if pd.isna(v) else str(v).strip())
    trf = [("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)] if num else []
    if cat: trf.append(("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", max_categories=R.MAX_CATEGORIES, sparse_output=False)), cat))
    P, Q = [], []
    if trf: ct = ColumnTransformer(trf).fit(A); P.append(ct.transform(A)); Q.append(ct.transform(B))
    for c in text:     # as text_embedding of the benchmark, TF-IDF then the 200 best words by chi square
        v = TfidfVectorizer(stop_words="english"); a = v.fit_transform(Tr[c].fillna("").astype(str)); b = v.transform(Te[c].fillna("").astype(str)); s = SelectKBest(chi2, k=min(200, a.shape[1])).fit(a, ytr); P.append(s.transform(a).toarray()); Q.append(s.transform(b).toarray())
    return np.hstack(P), np.hstack(Q)
TABLES = {}
def init(repo):
    global TABLES
    os.chdir(repo); sys.path.insert(0, repo)
    import clean_cleanml as C
    for n in C.NAMES: TABLES[n] = C.table(n)
def run(key):
    t0 = time.time(); root = key["root"]; ds = TABLES[key["ds"]]; err, method, seed = key["error_type"], key["method"], key["seed"]; y = ds["label"]
    Dtr, Dte = read(root, ds, err, "dirty", "train"), read(root, ds, err, "dirty", "test"); Vtr, Vte = (Dtr, Dte) if method == "dirty" else (read(root, ds, err, method, "train"), read(root, ds, err, method, "test"))
    ktr, kte = kept(root, ds, err, method, "train", len(Dtr), len(Vtr)), kept(root, ds, err, method, "test", len(Dte), len(Vte)); ntr, nte = min(len(Dtr), R.CAP_TRAIN), min(len(Dte), R.CAP_TEST)     # the first rows in dirty order, the split of the archive is random
    rtr, rte = np.where(ktr[:ntr])[0], np.where(kte[:nte])[0]; Atr, Ate = Vtr.iloc[(np.cumsum(ktr) - 1)[rtr]].reset_index(drop=True), Vte.iloc[(np.cumsum(kte) - 1)[rte]].reset_index(drop=True)
    cells, cells_te = (0, 0) if method == "dirty" else (changed(Dtr.iloc[rtr][Atr.columns].reset_index(drop=True), Atr), changed(Dte.iloc[rte][Ate.columns].reset_index(drop=True), Ate))
    Atr = Atr[Atr[y].notna().values].reset_index(drop=True); ya, yb = label(Atr[y]), label(Ate[y]); loc = np.full(nte, -1); loc[rte] = np.where(Ate[y].notna().values, np.arange(len(rte)), -1)     # a dirty test row to its row in this version
    if ds.get("class_imbalance"):     # down_sample of the benchmark, per seed here
        r = np.random.default_rng(seed + 500); u, c = np.unique(ya, return_counts=True); ids = np.concatenate([r.permutation(np.where(ya == v)[0])[:c.min()] for v in u]); Atr, ya = Atr.iloc[ids].reset_index(drop=True), ya[ids]
    lab = {v: i for i, v in enumerate(sorted(set(ya) | set(yb)))}; y1 = np.array([lab[v] for v in ya]); y2 = np.array([lab[v] for v in yb]); Ztr, Zte = features(ds, Atr, Ate, y1); m = R.models("cls", seed)[key["model"]]; m.fit(Ztr, y1); pr = m.predict(Zte)
    info = R.protocol_metadata({}, rtr, rte, m, Ztr)
    perm = np.random.default_rng(seed + 900).permutation(nte); half = nte // 2; val, test = loc[perm[:half]], loc[perm[half:]]; val, test = val[val >= 0], test[test >= 0]; every = loc[loc >= 0]     # the same rows for every version of a seed
    vs = {str(s): f1(y2[val[:s]], pr[val[:s]]) for s in SIZES if s <= len(val)}
    return dict({k: v for k, v in key.items() if k != "root"}, group="cleanml", task="cls", score=f1(y2[every], pr[every]), test=f1(y2[test], pr[test]), val=vs, n_val=int(len(val)), n_test=int(len(test)), n_train=int(len(y1)), records_dirty=int(ntr), records=int(len(rtr)), records_removed=int(ntr - len(rtr)), cells_changed=int(cells), records_dirty_test=int(nte), records_test=int(len(rte)), records_removed_test=int(nte - len(rte)), cells_changed_test=int(cells_te), balanced=bool(ds.get("class_imbalance")), sec=round(time.time() - t0, 2), **info)
def safe(k):
    try: return run(k)
    except Exception as e: return dict({a: b for a, b in k.items() if a != "root"}, error=repr(e)[:300])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=18); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--only", default=""); a = ap.parse_args(); a.repo = os.path.abspath(a.repo); a.out = os.path.abspath(a.out)
    init(a.repo); import clean_cleanml as C; done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "error" not in r: done.add(r["id"])
            except Exception: pass
    jobs = []
    for n in sorted(C.NAMES, key=lambda x: (x == "KDD", x)):
        if a.only and n not in a.only.split(","): continue
        for err in TABLES[n]["error_types"]:
            for me in C.methods(TABLES[n], err):
                if not os.path.exists(f"data/{n}/{err}/{me}_train.csv"): print("no version", n, err, me, flush=True); continue
                for s in range(a.seeds):
                    for mdl in MODELS:
                        i = f"cm_{n}_{err}_{me}_{mdl}_s{s}"
                        if i not in done: jobs.append(dict(id=i, ds=n, error_type=err, method=me, model=mdl, seed=s, root=os.path.join(a.repo, "data")))
    print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.repo,)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 50 == 0 or "error" in r: print(i, len(jobs), r["id"], r.get("score"), r.get("sec"), r.get("error", "")[:200], flush=True)
    print("finished", flush=True)
