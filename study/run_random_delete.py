#!/usr/bin/env python3
"""Control for the deletion action of the tabular track. Deleting the records that contain errors changes two things at once, the number of training
records and which records remain. This script separates the two. For each table, model family, seed and split of run_real_tables.py it deletes
either the records with errors (action delete_records, the same records as in run_real_tables.py) or the same number of training records drawn at
random from all training records (action random_delete). The fraction is the share of the records with errors whose count is deleted.
Everything else, the split, the models, the preprocessing and the metric, is taken from run_real_tables.py.
For every run it also records how far the retained records drift from all training records on the clean values, for the label and for the
features, so that the analysis can relate the loss of a deletion to the change of the population the retained records represent."""
import argparse, json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics import f1_score, r2_score
import run_real_tables as rt

FRACTIONS = (0.5, 0.9, 1.0)
ACTIONS = ("delete_records", "random_delete")
MODELS = ("linear", "rf", "gb", "knn", "mlp")


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def retained_shift(D, b, keep):
    """Shift between the retained records and all training records, measured on the clean values.
    Label: total variation distance of the class shares, or the absolute standardized mean difference for regression.
    Features: maximum over the model features of the standardized mean difference for numeric columns with more than 10 values,
    and of the total variation distance of the value shares otherwise."""
    out = {}
    y = b[D["y"]].astype(str).str.strip()
    if D["task"] == "cls":
        pa = y.value_counts(normalize=True); pk = y[keep].value_counts(normalize=True).reindex(pa.index).fillna(0)
        out["shift_label"] = float(0.5 * (pa - pk).abs().sum())
        counts = y[keep].value_counts().reindex(pa.index).fillna(0)
        out["classes_lost"] = int((counts == 0).sum()); out["classes"] = int(len(pa))
    else:
        v = _numeric(y); sd = float(v.std())
        out["shift_label"] = float(abs(v[keep].mean() - v.mean()) / sd) if sd > 0 else 0.0
    best, col = 0.0, None
    for c in D.get("model_feat", D["feat"]):
        s = b[c].astype(str).str.strip(); x = _numeric(s)
        if x.notna().mean() > 0.95 and s.nunique() > 10:
            sd = float(x.std()); d = float(abs(x[keep].mean() - x.mean()) / sd) if sd > 0 else 0.0
        else:
            pa = s.value_counts(normalize=True); pk = s[keep].value_counts(normalize=True).reindex(pa.index).fillna(0); d = float(0.5 * (pa - pk).abs().sum())
        if np.isfinite(d) and d > best: best, col = d, c
    out["shift_feature"] = best; out["shift_feature_col"] = col
    out["shift"] = max(out["shift_label"], best)
    return out


def run(key):
    """key holds ds, model, seed, action and f. For delete_records the random generator and the deleted records are those of run_real_tables.run."""
    t0 = time.time(); D = rt.DATA[key["ds"]]; seed = key["seed"]
    ident = {k: key[k] for k in ("ds", "model", "seed", "action", "f")}
    rng = np.random.default_rng(zlib.crc32(json.dumps(ident, sort_keys=True).encode()) % (2 ** 32)); r0 = np.random.default_rng(seed + 500)
    n = len(D["a"]); p = r0.permutation(n); cut = int(0.7 * n); tr, te = p[:cut][:rt.CAP_TRAIN], p[cut:][:rt.CAP_TEST]; act, f = key["action"], key["f"]
    E, T = D["E"][tr].copy(), D["T"][tr].copy(); yE = D["yE"][tr].copy(); feat = D["feat"]
    rec = E.any(axis=1) | yE; dirty_ids = np.where(rec)[0]; keep = np.ones(len(tr), bool)
    order = rng.permutation(dirty_ids); k = int(round(f * len(dirty_ids)))
    if act == "delete_records": keep[order[:k]] = False
    elif act == "random_delete": keep[rng.permutation(len(tr))[:k]] = False
    else: raise ValueError(f"unknown action {act}")
    if keep.sum() < 30: raise ValueError("fewer than 30 records left")
    A = D["a"].iloc[tr].reset_index(drop=True); B = D["b"].iloc[tr].reset_index(drop=True)
    Xd = A[feat][keep]; ya = A[D["y"]].values[keep]
    Xte = D["b"].iloc[te][feat].reset_index(drop=True); yte = D["b"].iloc[te][D["y"]].values
    Ztr, Zte = rt.feature_matrices(D, Xd, Xte); m = rt.models(D["task"], seed)[key["model"]]
    if D["task"] == "cls":
        lab = {v: i for i, v in enumerate(sorted(set(map(str, ya)) | set(map(str, yte))))}; y1 = np.array([lab[str(v)] for v in ya]); y2 = np.array([lab[str(v)] for v in yte])
        m.fit(Ztr, y1); s = float(f1_score(y2, m.predict(Zte), average="macro"))
    else:
        y1 = pd.to_numeric(pd.Series(ya), errors="coerce"); ok = y1.notna().values; y2 = pd.to_numeric(pd.Series(yte), errors="coerce"); ok2 = y2.notna().values
        m.fit(Ztr[ok], y1[ok]); s = float(r2_score(y2[ok2].values, m.predict(Zte[ok2])))
    Tk = T[keep]; yk = yE[keep]
    out = dict(ident, fraction=f, task=D["task"], score=s, n_train=int(keep.sum()), n0=int(len(tr)), n_test=int(len(te)), n_dirty=int(len(dirty_ids)), n_deleted=int(k),
               n_deleted_dirty=int((~keep & rec).sum()), rate_missing=float((Tk == 1).any(axis=1).mean()), rate_domain=float((Tk == 2).any(axis=1).mean()),
               rate_value=float((Tk == 3).any(axis=1).mean()), rate_label=float(yk.mean()), rate_clean_records=float(1 - ((Tk > 0).any(axis=1) | yk).mean()),
               cell_error=float((Tk > 0).mean()), sec=round(time.time() - t0, 2))
    out.update(retained_shift(D, B, keep)); out.update(rt.protocol_metadata(D, tr, te, m, Ztr))
    return out


def safe(k):
    try: return run(k)
    except Exception as e: return dict(k, fraction=k["f"], error=repr(e)[:300])


def jobs_for(meta, seeds, done):
    out = []
    for seed in seeds:
        for ds in meta:
            for mdl in MODELS:
                for act in ACTIONS:
                    for f in FRACTIONS:
                        if (ds, mdl, seed, act, f) not in done: out.append(dict(ds=ds, model=mdl, seed=seed, action=act, f=f))
    return out


if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "tables")); ap.add_argument("--meta", default=os.path.join(REPO, "data", "tables", "table_meta.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "results_random_delete.jsonl")); ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--tables", default="", help="comma separated table names, all tables of the metadata if empty")
    ap.add_argument("--dry", action="store_true", help="print the job count and exit"); a = ap.parse_args()
    meta = json.load(open(a.meta))
    if a.tables: meta = {k: meta[k] for k in a.tables.split(",")}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: r = json.loads(l); done.add((r["ds"], r["model"], r["seed"], r["action"], r["f"]))
            except Exception: pass
    # soccer_full is by far the slowest table, its jobs are queued first so that the pool does not end on a few long fits
    jobs = sorted(jobs_for(meta, range(a.seeds), done), key=lambda j: j["ds"] != "soccer_full")
    print("jobs", len(jobs), "done", len(done), flush=True)
    if a.dry: raise SystemExit
    with Pool(a.procs, initializer=rt.init, initargs=(a.root, meta)) as pool, open(a.out, "a") as fo:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1)):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if i % 50 == 0: print(i, len(jobs), r.get("ds"), r.get("action"), r.get("f"), r.get("sec"), r.get("error", "")[:90], flush=True)
    print("finished", flush=True)
