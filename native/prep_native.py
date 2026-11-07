#!/usr/bin/env python3
"""Published synthetic CIFAR-10 images as flat arrays. EDM (Wang et al. 2023), DDPM (Gowal et al. 2021) and CIFAKE (Bird and Lotfi 2024). The class of a CIFAKE image is in its file name, the mapping from name to class is read off the real images of the same archive against the CIFAR-10 labels."""
import io, os, re, struct, sys, zipfile, collections
import numpy as np
from PIL import Image
D, REAL = sys.argv[1], sys.argv[2]; N = 100000
def stored(path):
    """a member of an uncompressed npz fetched as a byte range: the open file at the start of the array, its shape and type"""
    f = open(path, "rb"); b = f.read(4096); k = b.find(b"PK\x03\x04"); n, m = struct.unpack("<HH", b[k + 26:k + 30]); f.seek(k + 30 + n + m); v = np.lib.format.read_magic(f); sh, _, dt = (np.lib.format.read_array_header_1_0 if v[0] == 1 else np.lib.format.read_array_header_2_0)(f); return f, sh, dt
def save(name, x, y): np.savez(f"{D}/synth_{name}.npz", image=x, label=y.astype(np.int64)); print(name, x.shape, np.bincount(y.astype(np.int64)).tolist(), flush=True)
if os.path.exists(f"{D}/edm_200k.npz") and not os.path.exists(f"{D}/synth_edm.npz"):     # this archive is sorted by class, the same number per class is drawn at random
    z = np.load(f"{D}/edm_200k.npz"); y = z["label"]; r_ = np.random.default_rng(0); ids = r_.permutation(np.concatenate([r_.permutation(np.where(y == c)[0])[:N // 10] for c in range(10)])); save("edm", z["image"][ids], y[ids])
if os.path.exists(f"{D}/cifar10_ddpm.npz") and not os.path.exists(f"{D}/synth_ddpm.npz"):
    z = np.load(f"{D}/cifar10_ddpm.npz"); save("ddpm", z["image"][:N], z["label"][:N])
if os.path.exists(f"{D}/ddpm_head.bin") and not os.path.exists(f"{D}/synth_ddpm.npz"):     # the archive holds 1M images in 3 GB, only its first images and its label member are fetched (fetch_ranges.py)
    f, sh, dt = stored(f"{D}/ddpm_head.bin"); x = np.fromfile(f, dt, N * int(np.prod(sh[1:]))).reshape((N,) + tuple(sh[1:])); f, sh, dt = stored(f"{D}/ddpm_tail.bin"); y = np.fromfile(f, dt, sh[0]); print("ddpm archive labels", sh, dt, np.bincount(y.astype(np.int64)).tolist(), flush=True); save("ddpm", x, y[:N])
if os.path.exists(f"{D}/cifake_train.zip") and not os.path.exists(f"{D}/synth_cifake.npz"):
    zf = zipfile.ZipFile(f"{D}/cifake_train.zip"); names = zf.namelist(); r = np.load(REAL); key = {x.tobytes(): int(y) for x, y in zip(r["Xtr"], r["ytr"])}
    def cls(n):
        m = re.search(r"\((\d+)\)", n); return int(m.group(1)) if m else 1
    def load(n): return np.array(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
    votes = collections.defaultdict(collections.Counter); small = r["Xtr"][:, ::4, ::4].astype(np.float32).reshape(len(r["Xtr"]), -1); by = collections.defaultdict(list)
    for x in names:
        if x.startswith("real/") and x.endswith(".jpg"): by[cls(x)].append(x)
    for k, v in by.items():
        for n in v[:200]:
            q = load(n)[::4, ::4].astype(np.float32).reshape(-1); votes[k][int(r["ytr"][np.argmin(((small - q) ** 2).sum(1))])] += 1
    print("votes", {k: v.most_common(2) for k, v in sorted(votes.items())}, flush=True); mp = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    fake = [x for x in names if x.startswith("fake/")]; X = np.stack([load(n) for n in fake]); y = np.array([mp[cls(n)] for n in fake]); save("cifake", X, y)
np.savez(f"{D}/cifar10_1.npz", X=np.load(f"{D}/cifar10.1_v6_data.npy"), y=np.load(f"{D}/cifar10.1_v6_labels.npy")); print("done")
