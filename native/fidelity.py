#!/usr/bin/env python3
"""Fidelity of the published synthetic CIFAR-10 images. CLIP ViT-B/32 image features as in study/embed_cifar.py, then precision and recall (Kynkaanniemi et al. 2019), density and coverage (Naeem et al. 2020) of 10000 synthetic images against 10000 real training images with k=5. Real test images against the same real images give the level of a generator that matches the data. A set whose result is stored is skipped, so the script can run again when a set arrives."""
import argparse, json, os
import numpy as np, torch
from transformers import CLIPImageProcessor, CLIPModel
ap = argparse.ArgumentParser(); ap.add_argument("--job", default="{}"); ap.add_argument("--out", default="results_fidelity.json"); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--k", type=int, default=5); A = ap.parse_args()
ROOT = os.environ.get("HQD_ROOT", "."); device = "cuda" if torch.cuda.is_available() else "cpu"; out = json.load(open(A.out)) if os.path.exists(A.out) else {}; n, k = A.n, A.k
model, proc = None, None
def embed(imgs):
    global model, proc
    if model is None: model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval(); proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    F = []
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            px = proc(images=list(imgs[i:i + 256]), return_tensors="pt")["pixel_values"].to(device); f = model.get_image_features(pixel_values=px); f = getattr(f, "pooler_output", f); F.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(F).astype(np.float32)
def kth(X):
    """distance of every row to its k-th nearest neighbour in the same set"""
    X = torch.from_numpy(X).to(device); out = []
    for i in range(0, len(X), 1000): out.append(torch.cdist(X[i:i + 1000], X).kthvalue(k + 1, dim=1).values)
    return torch.cat(out)
def prdc(real, fake):
    rr, ff = kth(real), kth(fake); R, Q = torch.from_numpy(real).to(device), torch.from_numpy(fake).to(device); inside_real = torch.zeros(len(Q), device=device, dtype=torch.long); cov = torch.zeros(len(R), dtype=torch.bool, device=device); rec = torch.zeros(len(R), dtype=torch.bool, device=device)
    for i in range(0, len(R), 500):     # a block of real rows against all synthetic rows
        d = torch.cdist(R[i:i + 500], Q); a = d < rr[i:i + 500, None]; inside_real += a.sum(0); cov[i:i + 500] = a.any(1); rec[i:i + 500] = (d < ff[None, :]).any(1)
    return dict(precision=float((inside_real > 0).float().mean()), recall=float(rec.float().mean()), density=float(inside_real.float().sum() / (k * len(Q))), coverage=float(cov.float().mean()), n=len(Q), k=k)
z = np.load(f"{ROOT}/cifar10_clip.npz"); X, te = z["X"].astype(np.float32), z["is_test"]; r = np.random.default_rng(0); real = X[~te][r.permutation(int((~te).sum()))[:n]]
if "real_test" not in out: out["real_test"] = prdc(real, X[te][:n]); print("real_test", out["real_test"], flush=True); json.dump(out, open(A.out, "w"), indent=0)
for g in ("edm", "ddpm", "cifake"):
    if g in out or not os.path.exists(f"{ROOT}/synth_{g}.npz"): continue
    x = np.load(f"{ROOT}/synth_{g}.npz")["image"]; ids = np.random.default_rng(1).permutation(len(x))[:n]; out[g] = prdc(real, embed(x[ids])); print(g, out[g], flush=True); json.dump(out, open(A.out, "w"), indent=0)
