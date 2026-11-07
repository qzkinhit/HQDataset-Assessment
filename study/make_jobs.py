#!/usr/bin/env python3
"""Job lists for the GPU tracks. One JSON per line, in the order of the seeds."""
import itertools, json, sys
out = sys.argv[1]; RATES = [0.05, 0.10, 0.20, 0.30, 0.40]
def w(name, jobs):
    with open(f"{out}/{name}.jsonl", "w") as f:
        for j in jobs: f.write(json.dumps(j) + "\n")
    print(name, len(jobs))
V = []
NOISE = [("cifar10", "aggre_label"), ("cifar10", "random_label1"), ("cifar10", "worse_label"), ("cifar100", "noisy_label")]
for seed in range(10):
    for arch in ["resnet18", "vgg11"]:
        for ds in ["cifar10", "cifar100"]:
            V.append(dict(id=f"v_{ds}_{arch}_s{seed}_clean", ds=ds, arch=arch, seed=seed))
        for ds, nz in NOISE:
            for act, f in [("keep", 0.0), ("repair", 0.25), ("repair", 0.5), ("repair", 0.75), ("delete", 0.5), ("delete", 1.0)]:
                V.append(dict(id=f"v_{ds}_{nz}_{arch}_s{seed}_{act}{f}", ds=ds, arch=arch, seed=seed, noise=nz, action=act, f=f))
        for k in [5, 4, 3, 2, 1]:
            V.append(dict(id=f"v_super_{arch}_s{seed}_cov{k}", ds="cifar100", arch=arch, seed=seed, coverage=k))
        for f in [0.8, 0.6, 0.4, 0.2]:
            V.append(dict(id=f"v_super_{arch}_s{seed}_scale{f}", ds="cifar100", arch=arch, seed=seed, scale=f))
w("vision", V)
T = []
DT = ["truncation", "template", "crossdomain", "lowtier", "duplicate", "coverage", "scale"]
for seed in range(5):
    T.append(dict(id=f"t_s{seed}_clean", seed=seed, defects={}))
    for d, r in itertools.product(DT + ["contamination"], RATES):
        T.append(dict(id=f"t_s{seed}_{d}{r}", seed=seed, defects={d: r}))
    for a, b in itertools.combinations(["truncation", "template", "duplicate", "coverage", "scale"], 2):
        T.append(dict(id=f"t_s{seed}_{a}+{b}0.2", seed=seed, defects={a: 0.2, b: 0.2}))
    for low, f in itertools.product([0.25, 0.67], ["none", "random", "ppl", "ppl_mid", "oracle"]):
        T.append(dict(id=f"t_s{seed}_filter_{f}_{low}", seed=seed, defects={}, filter=f, low=low))
w("text", T)
M = []
DM = ["mismatch", "missing", "duplicate", "coverage", "scale"]
for seed in range(20):
    M.append(dict(id=f"m_s{seed}_clean", seed=seed, defects={}))
    for d, r in itertools.product(DM + ["contamination"], RATES):
        M.append(dict(id=f"m_s{seed}_{d}{r}", seed=seed, defects={d: r}))
    for a, b in itertools.combinations(DM, 2):
        M.append(dict(id=f"m_s{seed}_{a}+{b}0.2", seed=seed, defects={a: 0.2, b: 0.2}))
w("multimodal", M)
# finer grids for the response curves
F = []
for seed in range(20):
    for d, r in itertools.product(["mismatch", "duplicate", "coverage", "scale"], [0.01, 0.02, 0.15, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8]):
        F.append(dict(id=f"m_s{seed}_{d}{r}", seed=seed, defects={d: r}))
    for r in [0.01, 0.02, 0.15, 0.25]:
        F.append(dict(id=f"m_s{seed}_contamination{r}", seed=seed, defects={"contamination": r}))
w("multimodal_fine", F)
w("text_fine", [dict(id=f"t_s{seed}_contamination{r}", seed=seed, defects={"contamination": r}) for seed in range(5) for r in [0.01, 0.02, 0.15, 0.25, 0.35]])
w("vision_fine", [dict(id=f"v_super_{arch}_s{seed}_scale{f}", ds="cifar100", arch=arch, seed=seed, scale=f) for seed in range(5) for arch in ["resnet18", "vgg11"] for f in [0.1, 0.3, 0.5, 0.7, 0.9]])
