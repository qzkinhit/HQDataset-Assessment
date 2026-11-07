#!/usr/bin/env python3
"""Build the inputs of run_study.py. Tables come from the clean versions of public cleaning benchmarks, TEP from the Braatz files,
AG News is encoded with TF-IDF and truncated SVD, the CIFAR-10 CLIP matrix comes from embed_cifar.py."""
import argparse, json, os, re
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--tables"); ap.add_argument("--meta"); ap.add_argument("--tep"); ap.add_argument("--agnews"); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(f"{a.out}/tables", exist_ok=True); os.makedirs(f"{a.out}/matrices", exist_ok=True)
if a.tables:
    meta = json.load(open(a.meta))
    for name, m in meta.items():
        df = pd.read_csv(f"{a.tables}/{name}/clean_index.csv")
        for c in ("index", "Unnamed: 0"):
            if c in df.columns: df = df.drop(columns=[c])
        df.to_csv(f"{a.out}/tables/{name}.csv", index=False); json.dump(m, open(f"{a.out}/tables/{name}.json", "w")); print(name, df.shape, m["task"], m["label_col"])
if a.tep:
    Xs, ys, ts = [], [], []
    for k in range(22):
        tr = np.loadtxt(f"{a.tep}/d{k:02d}.dat")
        if tr.shape[1] != 52: tr = tr.T
        te = np.loadtxt(f"{a.tep}/d{k:02d}_te.dat")
        if k > 0: te = te[160:]                      # fault present from sample 161
        Xs += [tr, te]; ys += [np.full(len(tr), k), np.full(len(te), k)]; ts += [np.zeros(len(tr), bool), np.ones(len(te), bool)]
    np.savez_compressed(f"{a.out}/matrices/tep.npz", X=np.vstack(Xs), y=np.concatenate(ys), is_test=np.concatenate(ts)); print("tep", np.vstack(Xs).shape)
if a.agnews:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr, te = pd.read_csv(f"{a.agnews}/train.csv"), pd.read_csv(f"{a.agnews}/test.csv")
    rng = np.random.default_rng(0); keep = rng.permutation(len(tr))[:40000]; tr = tr.iloc[keep]; texts = list(tr.text) + list(te.text)
    def corrupt(t):
        w = t.split(); m = rng.random(len(w)) > 0.5                         # half of the tokens removed, the rest keep their order
        return " ".join(x for x, k in zip(w, m) if k) or "x"
    v = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english"); M = v.fit_transform(texts)
    svd = TruncatedSVD(300, random_state=0); X = svd.fit_transform(M); Xa = svd.transform(v.transform([corrupt(t) for t in texts]))
    np.savez_compressed(f"{a.out}/matrices/agnews_tfidf.npz", X=X.astype(np.float32), X_alt=Xa.astype(np.float32), y=np.concatenate([tr.label.values, te.label.values]), is_test=np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)])
    print("agnews", X.shape, "variance", float(svd.explained_variance_ratio_.sum()))
