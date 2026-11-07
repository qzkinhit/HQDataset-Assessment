#!/usr/bin/env python3
"""Tabular track on real or benchmark-provided errors, nothing injected here. Each dataset comes as a dirty table and its clean reference.
Starting from the dirty training records, an action repairs or deletes part of the errors, a model is trained, and it is scored on clean test records.
The script records the error rates that remain, so the score of the national standard can be computed for every state of the training set."""
import argparse, hashlib, json, os, time, zlib, warnings
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")
NA = {"", "nan", "NaN", "NA", "N/A", "null", "NULL", "None", "?", "empty", "-"}
CAP_TRAIN, CAP_TEST = 40000, 15000
FR = [0.1, 0.25, 0.5, 0.75]
PROTOCOL = "tabular_v2_20261107"
MAX_CATEGORIES = 100
def load(root, name, m):
    a = pd.read_csv(f"{root}/{name}/dirty_index.csv", dtype=str, keep_default_na=False); b = pd.read_csv(f"{root}/{name}/clean_index.csv", dtype=str, keep_default_na=False)
    if "index" in a.columns: a = a.set_index("index"); b = b.set_index("index").loc[a.index]
    a = a.reset_index(drop=True); b = b.reset_index(drop=True); y = m["label_col"]; feat = [c for c in a.columns if c != y]; cats = [c for c in m["categorical_cols"] if c in feat]; num = [c for c in feat if c not in cats]
    cleanv = b[feat].apply(lambda s: s.str.strip()).values; dirtyv = a[feat].apply(lambda s: s.str.strip()).values; E = cleanv != dirtyv
    for j, c in enumerate(feat):
        if c in num:      # "5" and "5.0" are the same value, and so are two different missing tokens
            x = pd.to_numeric(pd.Series(dirtyv[:, j]), errors="coerce").values; v = pd.to_numeric(pd.Series(cleanv[:, j]), errors="coerce").values
            same = (np.isfinite(x) & np.isfinite(v) & (np.abs(x - v) <= 1e-9 * (1 + np.abs(v)))) | (np.isin(dirtyv[:, j], list(NA)) & np.isin(cleanv[:, j], list(NA)))
            E[same, j] = False
    # error type of each wrong cell, read from the data: missing token, out of domain, or a wrong value inside the domain
    T = np.zeros(E.shape, dtype=np.int8)
    for j, c in enumerate(feat):
        d = dirtyv[:, j]; miss = np.isin(d, list(NA))
        if c in cats: dom = np.isin(d, np.unique(cleanv[:, j]))
        else:
            x = pd.to_numeric(pd.Series(d), errors="coerce").values; cv = pd.to_numeric(pd.Series(cleanv[:, j]), errors="coerce").values; lo, hi = np.nanmin(cv) if np.isfinite(cv).any() else -np.inf, np.nanmax(cv) if np.isfinite(cv).any() else np.inf; dom = np.isfinite(x) & (x >= lo) & (x <= hi)
        T[:, j] = np.where(~E[:, j], 0, np.where(miss, 1, np.where(~dom, 2, 3)))
    yE = (a[y].str.strip().values != b[y].str.strip().values)
    if m["task"] == "reg":
        x = pd.to_numeric(a[y], errors="coerce").values; v = pd.to_numeric(b[y], errors="coerce").values; yE = yE & ~(np.isfinite(x) & np.isfinite(v) & (np.abs(x - v) <= 1e-9 * (1 + np.abs(v))))
    drop = set(m.get("drop_cols", [])); times = set(m.get("time_cols", []))
    if not drop.issubset(feat) or not times.issubset(set(feat) - drop): raise ValueError("Invalid feature policy in metadata")
    model_feat = [c for c in feat if c not in drop]
    if not model_feat: raise ValueError("No model features remain")
    return dict(a=a, b=b, y=y, feat=feat, cats=cats, num=num, E=E, T=T, yE=yE, task=m["task"], model_feat=model_feat, time_cols=sorted(times), drop_cols=sorted(drop))
DATA = {}
def init(root, meta):
    global DATA
    for name, m in meta.items(): DATA[name] = load(root, name, m)
def models(task, seed):
    if task == "cls": return dict(linear=LogisticRegression(max_iter=300, random_state=seed), rf=RandomForestClassifier(n_estimators=100, random_state=seed), gb=HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0, early_stopping=False, random_state=seed), knn=KNeighborsClassifier(n_neighbors=15), mlp=MLPClassifier((64,), max_iter=60, random_state=seed))
    return dict(linear=Ridge(), rf=RandomForestRegressor(n_estimators=100, random_state=seed), gb=HistGradientBoostingRegressor(max_iter=100, l2_regularization=1.0, early_stopping=False, random_state=seed), knn=KNeighborsRegressor(n_neighbors=15), mlp=TransformedTargetRegressor(MLPRegressor((64,), max_iter=200, random_state=seed), transformer=StandardScaler()))
def classifier_compatibility(model, y):
    """Retained for callers; the v2 model policy no longer changes by class counts."""
    return {}
def time_minutes(values):
    """Parse 24-hour or AM/PM clock strings; invalid clock values remain missing."""
    s = values.astype(str).str.strip().str.lower().str.replace(".", "", regex=False)
    g = s.str.extract(r"^(\d{1,2})\s*:\s*(\d{2})(?::(\d{2}))?\s*(am|pm)?$")
    h, m = pd.to_numeric(g[0], errors="coerce"), pd.to_numeric(g[1], errors="coerce")
    sec = pd.to_numeric(g[2], errors="coerce").fillna(0); ap = g[3].notna()
    valid = m.between(0, 59) & sec.between(0, 59) & ((~ap & h.between(0, 23)) | (ap & h.between(1, 12)))
    h = h.where(~ap, h % 12 + (g[3] == "pm").astype(int) * 12)
    return (h * 60 + m + sec / 60).where(valid, np.nan)
def feature_matrices(D, train, test):
    """Fit preprocessing on training features, keeping measurement columns separate."""
    feat = D.get("model_feat", D["feat"]); times = set(D.get("time_cols", []))
    cats = [c for c in D["cats"] if c in feat and c not in times]; num = [c for c in feat if c not in cats]
    a, b = train[feat].copy(), test[feat].copy()
    for c in num:
        a[c] = time_minutes(a[c]) if c in times else pd.to_numeric(a[c], errors="coerce")
        b[c] = time_minutes(b[c]) if c in times else pd.to_numeric(b[c], errors="coerce")
    for c in cats:
        a[c] = a[c].astype(str).str.strip().replace(list(NA), np.nan); b[c] = b[c].astype(str).str.strip().replace(list(NA), np.nan)
    trf = [("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)] if num else []
    if cats: trf.append(("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", max_categories=MAX_CATEGORIES, sparse_output=False)), cats))
    ct = ColumnTransformer(trf); return ct.fit_transform(a), ct.transform(b)
def protocol_metadata(D, train_ids, test_ids, model, Xtrain):
    out = dict(eval_protocol=PROTOCOL, drop_cols=D.get("drop_cols", []), time_cols=D.get("time_cols", []), n_features=int(Xtrain.shape[1]), train_index_sha256=hashlib.sha256(np.asarray(train_ids, dtype="<i8").tobytes()).hexdigest(), test_index_sha256=hashlib.sha256(np.asarray(test_ids, dtype="<i8").tobytes()).hexdigest())
    if isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        out.update(gb_l2=float(model.l2_regularization), gb_early_stopping=bool(model.early_stopping), fit_max_iter=int(model.max_iter), fit_n_iter=int(model.n_iter_))
        if isinstance(model, HistGradientBoostingClassifier):
            z = model._raw_predict(Xtrain); out["max_abs_train_logit"] = float(np.max(np.abs(z)))
            if not np.isfinite(z).all(): raise ValueError("Non-finite fitted class logits")
    return out
def column_order(D, tr, seed):
    """Order the wrong columns by importance on the clean training records, from a small forest, the order is fixed per seed."""
    feat = D.get("model_feat", D["feat"]); times = set(D.get("time_cols", [])); b = D["b"].iloc[tr]; X = pd.DataFrame({c: (time_minutes(b[c]).fillna(-1) if c in times else pd.to_numeric(b[c], errors="coerce").fillna(-1) if c in D["num"] else pd.factorize(b[c])[0]) for c in feat}); y = pd.factorize(b[D["y"]])[0] if D["task"] == "cls" else pd.to_numeric(b[D["y"]], errors="coerce").fillna(0)
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    m = (ExtraTreesClassifier if D["task"] == "cls" else ExtraTreesRegressor)(n_estimators=30, random_state=seed, n_jobs=1).fit(X.values[:5000], np.asarray(y)[:5000]); imp = np.zeros(len(D["feat"]))
    for c, value in zip(feat, m.feature_importances_): imp[D["feat"].index(c)] = value
    return imp
def repair_column_order(D, tr, seed, E, fraction, high=True):
    """The column-repair budget and ordering use only the retained model features."""
    imp = column_order(D, tr, seed); active = np.array([D["feat"].index(c) for c in D.get("model_feat", D["feat"])])
    cols = active[np.argsort(-imp[active] if high else imp[active])]
    return cols, int(round(fraction * E[:, active].sum()))
def run(key, validation_sizes=None):
    t0 = time.time(); D = DATA[key["ds"]]; seed = key["seed"]; rng = np.random.default_rng(zlib.crc32(json.dumps(key, sort_keys=True).encode()) % (2 ** 32)); r0 = np.random.default_rng(seed + 500)
    n = len(D["a"]); p = r0.permutation(n); cut = int(0.7 * n); tr, te = p[:cut][:CAP_TRAIN], p[cut:][:CAP_TEST]; act, f = key["action"], key["f"]
    E, T = D["E"][tr].copy(), D["T"][tr].copy(); yE = D["yE"][tr].copy(); feat = D["feat"]; rec = E.any(axis=1) | yE; dirty_ids = np.where(rec)[0]; keep = np.ones(len(tr), bool); fix = np.zeros(E.shape, bool); yfix = np.zeros(len(tr), bool); order = rng.permutation(dirty_ids); k = int(round(f * len(dirty_ids)))
    if act == "repair_records": fix[order[:k]] = True; yfix[order[:k]] = True
    elif act == "delete_records":
        keep[order[:k]] = False
        if keep.sum() < 30: raise ValueError("fewer than 30 records left")
    elif act in ("repair_cols_high", "repair_cols_low"):
        cols, budget = repair_column_order(D, tr, seed, E, f, act == "repair_cols_high"); done = 0
        for j in cols:
            ids = np.where(E[:, j])[0]; take = ids[:max(0, budget - done)]; fix[take, j] = True; done += len(take)
            if done >= budget: break
    elif act.startswith("repair_type_"):
        t = dict(missing=1, domain=2, value=3)[act.split("_")[2]]; ids = np.argwhere(T == t); take = ids[rng.permutation(len(ids))[:int(round(f * len(ids)))]]; fix[take[:, 0], take[:, 1]] = True
    elif act == "repair_labels": ids = np.where(yE)[0]; yfix[rng.permutation(ids)[:int(round(f * len(ids)))]] = True
    # build the training table
    A = D["a"].iloc[tr].reset_index(drop=True); B = D["b"].iloc[tr].reset_index(drop=True); Vals = A[feat].values.copy(); CV = B[feat].values; Vals[fix] = CV[fix]; Xd = pd.DataFrame(Vals, columns=feat); ya = A[D["y"]].values.copy(); ya[yfix] = B[D["y"]].values[yfix]
    E2 = E & ~fix; T2 = np.where(E2, T, 0); yE2 = yE & ~yfix; Xd = Xd[keep]; ya = ya[keep]
    Xte = D["b"].iloc[te][feat].reset_index(drop=True); yte = D["b"].iloc[te][D["y"]].values
    Ztr, Zte = feature_matrices(D, Xd, Xte); m = models(D["task"], seed)[key["model"]]
    if D["task"] == "cls":
        lab = {v: i for i, v in enumerate(sorted(set(map(str, ya)) | set(map(str, yte))))}; y1 = np.array([lab[str(v)] for v in ya]); y2 = np.array([lab[str(v)] for v in yte]); m.fit(Ztr, y1); pred = m.predict(Zte); target = y2; s = float(f1_score(target, pred, average="macro"))
    else:
        y1 = pd.to_numeric(pd.Series(ya), errors="coerce"); ok = y1.notna().values; y2 = pd.to_numeric(pd.Series(yte), errors="coerce"); ok2 = y2.notna().values; m.fit(Ztr[ok], y1[ok]); pred = m.predict(Zte[ok2]); target = y2[ok2].values; s = float(r2_score(target, pred))
    validation = {}
    if validation_sizes is not None:
        perm = np.random.default_rng(seed + 900).permutation(len(target)); half = len(perm) // 2; val, test = perm[:half], perm[half:]
        def metric(indices):
            return float(f1_score(target[indices], pred[indices], average="macro")) if D["task"] == "cls" else float(r2_score(target[indices], pred[indices]))
        validation = dict(test=metric(test), val={str(n): metric(val[:n]) for n in validation_sizes if n <= half}, n_val=int(half))
    kept = keep.sum(); Tk = T2[keep]; yk = yE2[keep]
    return dict(key, task=D["task"], score=s, n_train=int(kept), n0=int(len(tr)), n_test=int(len(te)), rate_missing=float((Tk == 1).any(axis=1).mean()), rate_domain=float((Tk == 2).any(axis=1).mean()), rate_value=float((Tk == 3).any(axis=1).mean()), rate_label=float(yk.mean()), rate_clean_records=float(1 - ((Tk > 0).any(axis=1) | yk).mean()), cell_error=float((Tk > 0).mean()), sec=round(time.time() - t0, 2), **protocol_metadata(D, tr, te, m, Ztr), **validation)
def safe(k):
    try: return run(k)
    except Exception as e: return dict(k, error=repr(e)[:300])
if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(REPO, "data", "tables")); ap.add_argument("--meta", default=os.path.join(REPO, "data", "tables", "table_meta.json")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_real_tables.jsonl")); ap.add_argument("--procs", type=int, default=44); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--jobs", help="Optional JSONL of job keys inside the grid"); a = ap.parse_args()
    meta = json.load(open(a.meta)); done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add((r["ds"], r["model"], r["seed"], r["action"], r["f"]))
            except Exception: pass
    jobs = []
    for seed in range(a.seeds):
        for ds in meta:
            for mdl in ("linear", "rf", "gb", "knn", "mlp"):
                arms = [("keep", 0.0), ("repair_records", 1.0), ("delete_records", 1.0)] + [(x, f) for x in ("repair_records", "delete_records", "repair_cols_high", "repair_cols_low") for f in FR] + [("repair_type_missing", 1.0), ("repair_type_domain", 1.0), ("repair_type_value", 1.0), ("repair_labels", 1.0)]
                for act, f in arms:
                    if (ds, mdl, seed, act, f) not in done: jobs.append(dict(ds=ds, model=mdl, seed=seed, action=act, f=f))
    if a.jobs:
        requested = [json.loads(l) for l in open(a.jobs) if l.strip()]
        identity = lambda j: (j["ds"], j["model"], j["seed"], j["action"], j["f"])
        valid = {identity(j) for j in jobs} | done
        wanted = {identity(j) for j in requested}
        if len(wanted) != len(requested) or not wanted <= valid: raise ValueError("Job keys must be unique and inside the grid")
        jobs = [j for j in jobs if identity(j) in wanted]
        meta = {ds: m for ds, m in meta.items() if any(j["ds"] == ds for j in jobs)}
    print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.root, meta)) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1 if a.jobs else 2)):
            f.write(json.dumps(r) + "\n")
            if a.jobs or i % 100 == 0: f.flush(); print(i, len(jobs), r.get("ds"), r.get("action"), r.get("sec"), r.get("error", "")[:90], flush=True)
    print("finished", flush=True)
