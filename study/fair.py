#!/usr/bin/env python3
"""Accuracy on the test images with a near duplicate in the training set against the other test images, from the saved predictions of the runs without any data change. Lists from Barz and Denzler (ciFAIR)."""
import csv, glob, json, os, sys
import numpy as np
ROOT = sys.argv[1]; out = {}
for ds, fn in (("cifar10", "duplicates_cifar10.csv"), ("cifar100", "duplicates_cifar100.csv")):
    z = np.load(f"{ROOT}/{ds}_real.npz"); y = z["yte"]; ids = sorted({int(r["TestID"]) for r in csv.DictReader(open(f"{ROOT}/cifair/{fn}"))}); dup = np.zeros(len(y), bool); dup[ids] = True; rows = []
    for arch in ("resnet18", "vgg11"):
        for f in sorted(glob.glob(f"{ROOT}/preds/v_{ds}_{arch}_s*_clean.npy")):
            p = np.load(f); rows.append(dict(arch=arch, all=float((p == y).mean()), dup=float((p[dup] == y[dup]).mean()), rest=float((p[~dup] == y[~dup]).mean())))
    out[ds] = dict(share=float(dup.mean()), n=int(dup.sum()), runs=rows)
json.dump(out, open(sys.argv[2], "w"), indent=0); print({k: (v["share"], len(v["runs"])) for k, v in out.items()})
