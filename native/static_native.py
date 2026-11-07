#!/usr/bin/env python3
"""Static scores and record counts of the native tracks. The score is P = 100 (8 - sum of the measurable rates) / 8, the formula of GB/T 48562-2026 on the data quality dimension, and every rate is measured on the data as published.
CleanML tables: one rate per error type of the benchmark, records with a missing value, records removed by the deduplicated version, records with a cell changed by the inconsistency repair, and records with a cell changed by an outlier detector, the median over the detectors. DCLM: the rule and near-duplicate rates computed for C4 (corpus/prep_corpus.py, corpus/run_corpus.py) on the unfiltered pool. DataComp and MIRFLICKR: one minus the consistency of the pool. Synthetic images: one minus precision. TEP large, Camelyon17 and CivilComments: a scan for format errors, missing values and duplicate records."""
import argparse, glob, hashlib, io, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "corpus")); sys.path.insert(0, HERE)
_argv = sys.argv; sys.argv = ["x", "--dir", ".", "--out", "."]
from prep_corpus import feats     # the static signals of the corpus track
sys.argv = _argv
import numpy as np, pandas as pd, zstandard
from multiprocessing import Pool
import run_cleanml as M
ap = argparse.ArgumentParser(); ap.add_argument("--native", default="."); ap.add_argument("--rieth"); ap.add_argument("--procs", type=int, default=48); ap.add_argument("--out", default="."); A = ap.parse_args(); A.native = os.path.abspath(A.native); A.out = os.path.abspath(A.out); A.rieth = os.path.abspath(A.rieth)
P = lambda *rates: 100 * (8 - sum(rates)) / 8
def changed_rows(a, b):
    """records with at least one changed cell, a number compared as a number and two missing values as equal"""
    out = np.zeros(len(a), bool)
    for c in a.columns:
        sa, sb = a[c].astype(str).str.strip(), b[c].astype(str).str.strip(); xn = pd.to_numeric(sa, errors="coerce").values.astype(float); yn = pd.to_numeric(sb, errors="coerce").values.astype(float); num = np.isfinite(xn) & np.isfinite(yn); both = a[c].isna().values & b[c].isna().values
        out |= np.where(num, np.abs(xn - yn) > 1e-9 * (1 + np.abs(yn)), sa.values != sb.values) & ~both
    return out
def table(name):
    """the rates of one CleanML table, on all its records"""
    import clean_cleanml as C
    t0 = time.time(); repo = f"{A.native}/cleanml/repo"; root = f"{repo}/data"; ds = M.TABLES[name]; both = lambda err, m: pd.concat([M.read(root, ds, err, m, p) for p in ("train", "test")], ignore_index=True); d = {}; rates = {}
    raw = pd.read_csv(f"{root}/{name}/raw/raw.csv", low_memory=False); d["records_raw"] = int(len(raw))
    for err in ds["error_types"]:
        D = both(err, "dirty"); d[f"records_{err}"] = int(len(D))
        if err == "missing_values": rates[err] = float(D.isna().any(axis=1).mean())
        elif err == "duplicates":
            rates[err] = 1 - len(both(err, "clean")) / len(D); d["duplicates_autoer"] = 1 - len(both(err, "AutoER")) / len(D)
        elif err == "inconsistency": rates[err] = float(changed_rows(D, both(err, "clean")[D.columns]).mean())
        elif err == "outliers":
            per = {}
            for m in C.methods(ds, err):
                if m == "dirty": continue
                V = both(err, m); assert len(V) == len(D), (name, m, len(V), len(D)); per.setdefault(m.split("_")[1], []).append(float(changed_rows(D, V[D.columns]).mean()))
            det = {k: float(np.median(v)) for k, v in per.items()}; d["outlier_detectors"] = det; rates[err] = float(np.median(list(det.values())))
    return name, dict(d, rates=rates, P=P(*rates.values()), sec=round(time.time() - t0, 1))
def docs(pat):
    for fn in sorted(glob.glob(f"{A.native}/{pat}")):
        with open(fn, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
            for l in io.TextIOWrapper(r, encoding="utf-8"):
                t = json.loads(l).get("text", "")
                if len(t) >= 200: yield t     # the documents kept by prep_dclm.py
def corpus(pat, pool):
    """rule failures and near-duplicate share as for C4: MinHash signatures, 8 bands of 4 rows, confirmed at agreement 0.75, a document counts when its group has two or more members"""
    FA, FS, FR, FN, SG = [], [], [], [], []
    for a, s, rp, n, sig in pool.imap(feats, docs(pat), chunksize=256): FA.append(a); FS.append(s); FR.append(rp); FN.append(n); SG.append(sig)
    alpha, stop, rep, words, S = np.array(FA), np.array(FS), np.array(FR), np.array(FN), np.vstack(SG); n = len(S); rule = (alpha >= .6) & (rep <= .3) & (stop >= .1) & (words >= 50); par = np.arange(n)
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for b in range(8):
        seen = {}; key = S[:, b * 4:(b + 1) * 4]
        for i in range(n):
            k = key[i].tobytes(); j = seen.setdefault(k, i)
            if j != i and (S[i] == S[j]).mean() >= .75:
                x, y = find(i), find(j)
                if x != y: par[max(x, y)] = min(x, y)
    root = np.array([find(i) for i in range(n)]); u, c = np.unique(root, return_counts=True); return dict(docs=int(n), rule_fail=float(1 - rule.mean()), near_duplicate=float(c[c > 1].sum() / n), alpha_fail=float((alpha < .6).mean()), rep_fail=float((rep > .3).mean()), stop_fail=float((stop < .1).mean()), words_fail=float((words < 50).mean()))
def row_duplicates(X, chunk=1000000):
    """records minus distinct records, by a 64-bit hash of the bits of every row"""
    R = np.random.default_rng(0).integers(1, 2 ** 63, X.shape[1], dtype=np.uint64) | 1; h = np.concatenate([((X[i:i + chunk] + 0.0).view(np.uint32).astype(np.uint64) * R).sum(1) for i in range(0, len(X), chunk)]); return int(len(h) - len(np.unique(h)))
def tep_large():
    z = np.load(f"{A.rieth}/arrays.npz"); X = np.concatenate([z[k + "_X"] for k in ("ff_tr", "ff_te", "f_tr", "f_te")]); n = len(X); bad = ~np.isfinite(X); dup = row_duplicates(X); missing = float(np.isnan(X).any(1).mean()); fmt = float((bad & ~np.isnan(X)).any(1).mean()); rates = dict(missing=missing, format=fmt, duplicates=dup / n)
    return dict(records=int(n), variables=int(X.shape[1]), duplicate_records=dup, constant_variables=[int(j) for j in range(X.shape[1]) if np.ptp(X[:, j]) == 0], rates=rates, P=P(*rates.values()))
def camelyon():
    X = np.load(f"{A.native}/camelyon_x.npy"); m = np.load(f"{A.native}/camelyon_meta.npz"); y = m["y"]; dup = len(X) - len({hashlib.blake2b(x.tobytes(), digest_size=8).digest() for x in X}); rates = dict(missing=float(np.mean(y < 0)), format=float(np.mean(~np.isin(y, [0, 1]))), duplicates=dup / len(X))
    return dict(records=int(len(X)), official_records=455954, shape=list(X.shape[1:]), duplicate_records=int(dup), rates=rates, P=P(*rates.values()))
def civil():
    ident = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]; d = pd.read_parquet(f"{A.native}/wilds/civilcomments_raw.parquet", columns=["comment_text", "split", "toxicity"] + ident); n = len(d); t = d["comment_text"]
    miss = (d[["comment_text", "toxicity"] + ident].isna().any(axis=1)).values; fmt = ((t.fillna("").astype(str).str.strip().str.len() == 0) | ~d["toxicity"].between(0, 1) | ~d["split"].isin(["train", "val", "test"])).values & ~miss; dup = int(t.duplicated().sum()); rates = dict(missing=float(miss.mean()), format=float(fmt.mean()), duplicates=dup / n); same = t.duplicated(keep=False) & (d.groupby("comment_text")["split"].transform("nunique") > 1)
    return dict(records=int(n), duplicate_records=dup, records_in_several_splits=int(same.sum()), rates=rates, P=P(*rates.values()))
if __name__ == "__main__":
    M.init(f"{A.native}/cleanml/repo"); import clean_cleanml as C; out = {}; t0 = time.time()
    with Pool(13) as pool:
        for name, v in pool.imap_unordered(table, C.NAMES): out[name] = v; print(name, v["records_raw"], v["rates"], round(v["P"], 2), v["sec"], flush=True)
    native = {}
    for l in open(f"{A.native}/results_datacomp.jsonl"): native["datacomp"] = json.loads(l)["consistency_pool"]; break
    for l in open(f"{A.native}/results_mirflickr.jsonl"): native["mirflickr"] = json.loads(l)["consistency_pool"]; break
    for k, c in native.items(): out[k] = dict(consistency_pool=c, rates=dict(consistency=1 - c), P=P(1 - c))
    F = json.load(open(f"{A.native}/results_fidelity.json"))
    for k in ("edm", "ddpm", "cifake"): out[k] = dict(precision=F[k]["precision"], rates=dict(precision=1 - F[k]["precision"]), P=P(1 - F[k]["precision"]))
    out["civil"] = civil(); print("civil", out["civil"], flush=True); out["camelyon"] = camelyon(); print("camelyon", out["camelyon"], flush=True); out["tepl"] = tep_large(); print("tepl", out["tepl"], flush=True)
    os.chdir(A.native)
    with Pool(A.procs) as pool: a = corpus("dclm_pool/*.zst", pool); b = corpus("dclm_base/*.zst", pool)
    out["dclm"] = dict(pool=a, baseline=b, baseline_P=P(b["rule_fail"], b["near_duplicate"]), rates=dict(rule_fail=a["rule_fail"], near_duplicate=a["near_duplicate"]), P=P(a["rule_fail"], a["near_duplicate"])); print("dclm", out["dclm"], flush=True)
    sizes = {k: out[k]["records_raw"] for k in C.NAMES}; sizes.update(datacomp=int(json.loads(open(f"{A.native}/results_datacomp.jsonl").readline())["n_pool"]), dclm=a["docs"] + b["docs"], camelyon=out["camelyon"]["records"], civil=out["civil"]["records"])
    json.dump(sizes, open(f"{A.out}/results_sizes_native.json", "w"), indent=0); json.dump({k: round(v["P"], 3) for k, v in out.items()}, open(f"{A.out}/static_native.json", "w"), indent=0); json.dump(out, open(f"{A.out}/static_native_detail.json", "w"), indent=0); print("finished", round(time.time() - t0), flush=True)
