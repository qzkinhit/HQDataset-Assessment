#!/usr/bin/env python3
"""WILDS CivilComments on the official split (Koh et al. 2021), real comments with real identity annotations, nothing is injected. TF-IDF features with logistic regression and with a gradient-boosted model. The score is average accuracy, worst-group accuracy over the sixteen groups of WILDS (one of eight identities present, label toxic or not), and Macro-F1. Mode "features" builds the features once, every job reads them."""
import argparse, fcntl, json, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="job"); ap.add_argument("--native", default="."); ap.add_argument("--job"); ap.add_argument("--out", default="results_civilcomments.jsonl"); ap.add_argument("--threads", type=int, default=2); A = ap.parse_args()
os.environ.update(OMP_NUM_THREADS=str(A.threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import f1_score
IDENT = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]; F = f"{A.native}/wilds/civil_features"
if A.mode == "features":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    d = pd.read_parquet(f"{A.native}/wilds/civilcomments_raw.parquet", columns=["comment_text", "split", "toxicity"] + IDENT); y = (d["toxicity"].values >= .5).astype(np.int8); G = (d[IDENT].values >= .5); sp_ = d["split"].values; tr = sp_ == "train"
    v = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), max_features=300000, strip_accents="unicode", dtype=np.float32); X = v.fit(d["comment_text"][tr].astype(str)).transform(d["comment_text"].astype(str)); s = SelectKBest(chi2, k=5000).fit(X[tr], y[tr])
    sp.save_npz(f"{F}.npz", X.tocsr()); np.savez(f"{F}_meta.npz", y=y, G=G, split=sp_.astype(str), top=s.get_support(indices=True)); print("features", X.shape, {k: int((sp_ == k).sum()) for k in ("train", "val", "test")}, "toxic", y[tr].mean(), flush=True); raise SystemExit
J = json.loads(A.job); t0 = time.time(); seed, model = J["seed"], J["model"]; X = sp.load_npz(f"{F}.npz"); m = np.load(f"{F}_meta.npz"); y, G, split = m["y"], m["G"], m["split"]; tr = split == "train"
if model == "tfidf_lr":
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver="saga", max_iter=30, tol=1e-3, random_state=seed).fit(X[tr], y[tr]); pred = lambda x: clf.predict(x)
else:     # gradient boosting on the 5000 words with the highest chi square on the training part
    import xgboost as xgb
    Z = X[:, m["top"]]; clf = xgb.XGBClassifier(tree_method="hist", n_estimators=200, max_depth=6, learning_rate=.2, subsample=.8, colsample_bytree=.5, n_jobs=A.threads, random_state=seed).fit(Z[tr], y[tr]); X = Z; pred = lambda x: clf.predict(x)
r = dict(J, ds="civilcomments", group="text", task="cls", n_train=int(tr.sum()))
for part in ("val", "test"):
    k = split == part; p = pred(X[k]); yy, gg = y[k], G[k]; groups = {f"{IDENT[i]}_y{v}": float((p[gg[:, i] & (yy == v)] == v).mean()) for i in range(8) for v in (0, 1)}
    v = dict(avg_acc=float((p == yy).mean()), worst_group_acc=min(groups.values()), macro_f1=float(f1_score(yy, p, average="macro")), n=int(k.sum())); r.update(v if part == "test" else {f"{a}_val": b for a, b in v.items()})
    if part == "test": r["groups_test"] = groups
r.update(score=r["avg_acc"], sec=round(time.time() - t0, 1))
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
