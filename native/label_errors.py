#!/usr/bin/env python3
"""Test-set label errors of the ten datasets of Northcutt et al. (2021). A flagged label counts as wrong when at least three of the five reviewers chose the suggested label and fewer than three kept the given one, MNIST reviewers could mark both. A label is rejected when fewer than three reviewers kept it, these are the validated counts of the published table. The score is the accuracy of the cross-validated model of the same work on the original and on the corrected labels, and for CIFAR also of the models trained here.
IMDB ids are file names and ImageNet ids are numbers of validation files, both are mapped with the file lists of the archive. IMDB and Amazon give label names, the integer coding of the arrays is read off the flagged records and its agreement with the arrays is stored. For Amazon the corrected label is the majority among positive, negative and neutral when it differs from the given label and has at least three votes. AudioSet is multi-label, a record counts as wrong when one suggested label has at least three votes, and only the flagged and wrong counts are reported. QuickDraw and Amazon reviewed a sample of the flagged labels, so their rate is a lower bound."""
import ast, collections, glob, json, os, sys
import numpy as np
D, PREDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; out = {}
SAMPLED = ("quickdraw", "amazon")
def votes(r): return ast.literal_eval(r["mturk"]) if isinstance(r["mturk"], str) else r["mturk"]
def coding(pairs):
    """name to integer by majority over (name, array value) pairs, with the share of pairs that agree"""
    c = collections.defaultdict(collections.Counter)
    for name, v in pairs: c[name][int(v)] += 1
    mp = {k: v.most_common(1)[0][0] for k, v in c.items()}
    if len(set(mp.values())) != len(mp): raise SystemExit(f"coding is not one to one {dict(c)}")
    return mp, sum(c[k][mp[k]] for k in c) / max(1, sum(sum(v.values()) for v in c.values())), {k: dict(v) for k, v in c.items()}
for d in ("mnist", "cifar10", "cifar100", "caltech256", "imagenet", "quickdraw", "20news", "imdb", "amazon", "audioset"):
    pf, of, mf = f"{D}/{d}_cv_pred.npy", f"{D}/{d}_orig.npy", f"{D}/{d}_mturk.json"
    if not (os.path.exists(pf) and os.path.exists(of) and os.path.exists(mf)): print("missing", d); continue
    m = json.load(open(mf)); mm = "r" if d == "quickdraw" else None; p = np.load(pf, allow_pickle=mm is None, mmap_mode=mm); y = np.load(of, allow_pickle=mm is None, mmap_mode=mm)
    if p.dtype == object or y.dtype == object:     # multi-label, no accuracy
        wrong = sum(1 for r in m if any(votes(r).get(g, 0) >= 3 for g in r["our_guessed_labels"])); rejected = sum(1 for r in m if any(votes(r).get(g, 0) >= 3 for g in r["our_guessed_labels"]) or any(votes(r).get(g, 0) < 3 for g in r["given_original_labels"]))
        out[d] = dict(n=int(len(y)), flagged=len(m), wrong=wrong, given_rejected=rejected, multi_label=True, rate=wrong / len(y), rate_rejected=rejected / len(y)); print(d, out[d], flush=True); continue
    if d == "imdb":
        names = {x.replace(".txt", ""): i for i, x in enumerate(json.load(open(f"{D}/imdb_index.json")))}; pos = lambda r: names[r["id"].replace("test/", "", 1)]
    elif d == "imagenet":
        names = {x: i for i, x in enumerate(json.load(open(f"{D}/imagenet_index.json")))}; pos = lambda r: names["val/" + r["url"].split("/val/", 1)[1]]
    else: pos = lambda r: int(r["id"])
    info = {}
    if isinstance(m[0]["given_original_label"], str):
        cd, agree, seen = coding((r["given_original_label"], y[pos(r)]) for r in m); left = sorted(set(np.unique(y[:200000]).tolist()) - set(cd.values())); miss = sorted({r["our_guessed_label"] for r in m} - set(cd))
        if len(miss) == 1 and len(left) == 1: cd[miss[0]] = left[0]     # a name that is never given takes the only code left
        elif miss: raise SystemExit(f"cannot code {miss} {left}")
        info = dict(coding=cd, coding_agreement=agree, coding_seen=seen)
    else:
        cd = None; info = dict(id_agreement=float(np.mean([int(y[pos(r)]) == int(r["given_original_label"]) for r in m])))
    code = (lambda s: cd[s]) if cd else (lambda s: int(s))
    c = np.array(y, copy=True); wrong = rejected = 0
    for r in m:
        v = votes(r); given = r["given_original_label"]
        if "guessed" in v: fix = r["our_guessed_label"] if v["guessed"] >= 3 and v["given"] < 3 else None; kept = v["given"]
        else:     # Amazon, votes per class name
            t = {k: v.get(k, 0) for k in ("positive", "negative", "neutral")}; top = max(t, key=t.get); gl = given.lower(); kept = t[gl]
            fix = top.capitalize() if top != gl and t[top] >= 3 else None
        rejected += kept < 3
        if fix is not None: c[pos(r)] = code(fix); wrong += 1
    out[d] = dict(n=int(len(y)), flagged=len(m), wrong=wrong, given_rejected=int(rejected), rate=wrong / len(y), rate_rejected=int(rejected) / len(y), rate_lower_bound=d in SAMPLED, cv_disagree=float((p != y).mean()), acc_original=float((p == y).mean()), acc_corrected=float((p == c).mean()), **info)
    if d in ("cifar10", "cifar100"):
        fs = sorted(glob.glob(f"{PREDS}/v_{d}_*_clean.npy")); a = [np.load(x) for x in fs]
        if a: out[d].update(own_original=float(np.mean([(q == y).mean() for q in a])), own_corrected=float(np.mean([(q == c).mean() for q in a])), own_runs=len(a))
    print(d, {k: v for k, v in out[d].items() if k != "coding_seen"}, flush=True)
json.dump(out, open(OUT, "w"), indent=0)
