#!/usr/bin/env python3
"""Static rates of the C4 shards from the arrays of prep_corpus.py. rule_fail is the share of documents that fail the rule filter of run_corpus.py, near_duplicate is the share of documents in a near-duplicate group of two or more. The dataset table and the main table read the result.
Usage: static_corpus.py CORPUS_DIR OUT_JSON"""
import json, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]; z = np.load(f"{D}/c4.npz"); group = np.load(f"{D}/c4_group.npy"); n = len(group)
rule = (z["alpha"] >= .6) & (z["rep"] <= .3) & (z["stop"] >= .1) & (z["words"] >= 50); _, count = np.unique(group, return_counts=True)
out = dict(rule_fail=1 - float(rule.mean()), near_duplicate=float(count[count > 1].sum() / n), docs=int(n)); json.dump(out, open(OUT, "w")); print(out)
