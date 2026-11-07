#!/usr/bin/env python3
"""Replays the choice among data versions from the saved test predictions of the image runs. The test images are split once per seed into a validation part and a test part, each version gets scores on nested validation samples and on the test part, in the same form as run_decision.py."""
import glob, json, os, re, sys
import numpy as np
ROOT, OUT = sys.argv[1], sys.argv[2]; SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200); rows = []
def add(ds, y, arch, seed, action, f, fn):
    p = np.load(fn); perm = np.random.default_rng(seed + 900).permutation(len(y)); h = len(y) // 2; v, t = perm[:h], perm[h:]; ok = p == y
    rows.append(dict(ds=ds, model=arch, seed=seed, action=action, f=f, test=float(ok[t].mean()), val={str(n): float(ok[v[:n]].mean()) for n in SIZES if n <= h}))
for ds, base, noise in (("cifar10n", "cifar10", "worse_label"), ("cifar100n", "cifar100", "noisy_label")):
    y = np.load(f"{ROOT}/{base}_real.npz")["yte"]
    for arch in ("resnet18", "vgg11"):
        for fn in glob.glob(f"{ROOT}/preds/v_{base}_{noise}_{arch}_s*_*.npy"):
            m = re.search(r"_s(\d+)_(keep|repair|delete)([\d.]+)\.npy$", fn)
            if m: add(ds, y, arch, int(m[1]), m[2], float(m[3]), fn)
        for fn in glob.glob(f"{ROOT}/preds/v_{base}_{arch}_s*_clean.npy"):
            add(ds, y, arch, int(re.search(r"_s(\d+)_clean", fn)[1]), "repair", 1.0, fn)
y = np.load(f"{ROOT}/cifar10_real.npz")["yte"]
for g in ("edm", "ddpm", "cifake"):
    for arch in ("resnet18", "vgg11"):
        for fn in glob.glob(f"{ROOT}/preds/sy_real_{arch}_s*.npy"):
            add(g, y, arch, int(re.search(r"_s(\d+)\.npy", fn)[1]), "real", 0.0, fn)
        for fn in glob.glob(f"{ROOT}/preds/sy_{g}_{arch}_s*_*.npy"):
            m = re.search(r"_s(\d+)_(share|add)([\d.]+)\.npy$", fn); add(g, y, arch, int(m[1]), m[2], float(m[3]), fn)
open(OUT, "w").write("".join(json.dumps(r) + "\n" for r in rows)); print(len(rows))
