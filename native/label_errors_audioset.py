#!/usr/bin/env python3
"""Scores of the AudioSet row of the test-set labels (Northcutt et al. 2021). AudioSet is multi-label, so the score is the example-based accuracy, the mean over records of the size of the intersection of predicted and given labels over the size of their union (Zhang and Zhou 2014). The corrected label set of a record with a suggested label of at least three votes keeps the given labels with at least three votes and adds the suggested labels with at least three votes, as in the count of label_errors.py. The label names of the review file are mapped to the integer labels of the arrays with the class list of AudioSet (class_labels_indices.csv).
Usage: label_errors_audioset.py NATIVE_DIR class_labels_indices.csv OUT_JSON"""
import ast, csv, json, re, sys
import numpy as np
D, CLS, OUT = sys.argv[1:4]
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
code = {norm(r["display_name"]): int(r["index"]) for r in csv.DictReader(open(CLS, encoding="utf-8"))}
p = np.load(f"{D}/audioset_cv_pred.npy", allow_pickle=True); y = np.load(f"{D}/audioset_orig.npy", allow_pickle=True); m = json.load(open(f"{D}/audioset_mturk.json"))
votes = lambda r: ast.literal_eval(r["mturk"]) if isinstance(r["mturk"], str) else r["mturk"]
def acc(P, Y):
    v = [len(a & b) / max(1, len(a | b)) for a, b in zip(P, Y)]; return float(np.mean(v))
P = [set(a.tolist()) for a in p]; Y = [set(a.tolist()) for a in y]; C = list(Y); agree = changed = 0
for r in m:
    i = int(r["id"]); v = votes(r); given = {code[norm(g)] for g in r["given_original_labels"]}; agree += given == Y[i]
    if any(v.get(g, 0) >= 3 for g in r["our_guessed_labels"]):
        C[i] = {code[norm(g)] for g in r["given_original_labels"] if v.get(g, 0) >= 3} | {code[norm(g)] for g in r["our_guessed_labels"] if v.get(g, 0) >= 3}; changed += 1
out = dict(audioset=dict(n=len(Y), flagged=len(m), changed=changed, name_agreement=agree / len(m), acc_original=acc(P, Y), acc_corrected=acc(P, C)))
json.dump(out, open(OUT, "w"), indent=0); print(out)
