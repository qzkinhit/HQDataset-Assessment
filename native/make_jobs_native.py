#!/usr/bin/env python3
"""Job lists of the synthetic track, of the DCLM corpora and of Camelyon17, in the order of the pool."""
import json, sys
out = []
for s in range(5):
    for arch in ("resnet18", "vgg11"): out.append(dict(id=f"sy_real_{arch}_s{s}", ds="cifar10", arch=arch, seed=s, gen="real"))
for s in range(3):
    for g in ("edm", "ddpm", "cifake"):
        for x in (1.0, .5, .9, .75, .25, .1): out.append(dict(id=f"sy_{g}_resnet18_s{s}_share{x}", ds="cifar10", arch="resnet18", seed=s, gen=g, share=x))
        for x in (50000, 10000, 25000, 100000): out.append(dict(id=f"sy_{g}_resnet18_s{s}_add{x}", ds="cifar10", arch="resnet18", seed=s, gen=g, add=x))
        for x in (1.0, .5): out.append(dict(id=f"sy_{g}_vgg11_s{s}_share{x}", ds="cifar10", arch="vgg11", seed=s, gen=g, share=x))
        out.append(dict(id=f"sy_{g}_vgg11_s{s}_add50000", ds="cifar10", arch="vgg11", seed=s, gen=g, add=50000))
open(sys.argv[1], "w").write("".join(json.dumps(j) + "\n" for j in out)); print(len(out))
out = []
for s in range(3):
    for size in ("M", "S", "L", "XS"):
        for arm in ("dclm_pool", "dclm_baseline"): out.append(dict(id=f"c_s{s}_{size}_{arm}_30M", seed=s, size=size, arm=arm, budget=30000000))
open(sys.argv[2], "w").write("".join(json.dumps(j) + "\n" for j in out)); print(len(out))
if len(sys.argv) > 3: open(sys.argv[3], "w").write("".join(json.dumps(dict(id=f"cam_resnet18_s{s}", arch="resnet18", seed=s)) + "\n" for s in range(5)))
