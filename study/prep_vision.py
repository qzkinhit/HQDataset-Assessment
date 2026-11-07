#!/usr/bin/env python3
"""Arrays for the vision track, in the image order of the CIFAR-N label files (Wei et al., ICLR 2022), checked against their clean labels."""
import numpy as np, os, sys, torch
from datasets import load_dataset
ROOT, CN = sys.argv[1], sys.argv[2]
import pickle, tarfile
def original_cifar10(path):
    """The batch order of the original release, the order CIFAR-10N refers to."""
    t = tarfile.open(path); out = {}
    for m in t.getmembers():
        if "_batch" in m.name: out[m.name.split("/")[-1]] = pickle.load(t.extractfile(m), encoding="latin1")
    def imgs(d): return d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d["labels"])
    tr = [imgs(out[f"data_batch_{i}"]) for i in range(1, 6)]; te = imgs(out["test_batch"])
    return np.concatenate([a for a, _ in tr]), np.concatenate([b for _, b in tr]), te[0], te[1]
for name, repo, fn, lab in [("cifar10", "uoft-cs/cifar10", "CIFAR-10_human.pt", "label"), ("cifar100", "uoft-cs/cifar100", "CIFAR-100_human.pt", "fine_label")]:
    if name == "cifar10":
        if len(sys.argv) < 4 or not os.path.exists(sys.argv[3]): continue
        Xtr, ytr, Xte, yte = original_cifar10(sys.argv[3])
    else:
        d = load_dataset(repo); key = "img" if "img" in d["train"].column_names else "image"
        def arr(s): return np.stack([np.asarray(im.convert("RGB")) for im in d[s][key]]), np.array(d[s][lab])
        Xtr, ytr = arr("train"); Xte, yte = arr("test")
    H = torch.load(f"{CN}/{fn}", weights_only=False); clean = np.asarray(H["clean_label"])
    print(name, "order match with CIFAR-N clean labels:", float((ytr == clean).mean()))
    assert (ytr == clean).mean() > 0.999
    extra = {k: np.asarray(v) for k, v in H.items()}
    if name == "cifar100": extra["coarse_test"] = np.array(d["test"]["coarse_label"]); extra["coarse_train"] = np.array(d["train"]["coarse_label"])
    np.savez(f"{ROOT}/{name}_real.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, **extra)
