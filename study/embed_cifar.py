#!/usr/bin/env python3
"""CLIP ViT-B/32 image features of CIFAR-10, for the clean images and for a corrupted copy (Gaussian noise plus box blur). Runs once on a GPU."""
import argparse, os
import numpy as np, torch
from transformers import CLIPImageProcessor, CLIPModel
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
z = np.load(a.npz); print({k: z[k].shape for k in z.files})
keys = {k.lower(): k for k in z.files}
xtr, ytr, xte, yte = z[keys.get("x_train", keys.get("train_x", z.files[0]))], z[keys.get("y_train", keys.get("train_y", z.files[1]))], z[keys.get("x_test", keys.get("test_x", z.files[2]))], z[keys.get("y_test", keys.get("test_y", z.files[3]))]
X = np.concatenate([xtr, xte]); y = np.concatenate([ytr, yte]).ravel(); is_test = np.r_[np.zeros(len(xtr), bool), np.ones(len(xte), bool)]
if X.shape[1] == 3: X = X.transpose(0, 2, 3, 1)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(a.device).eval(); proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
def corrupt(batch, rng):
    b = batch.astype(np.float32) + rng.normal(0, 40, batch.shape)
    t = torch.from_numpy(b.transpose(0, 3, 1, 2)).float(); t = torch.nn.functional.avg_pool2d(t, 3, stride=1, padding=1)
    return t.numpy().transpose(0, 2, 3, 1).clip(0, 255).astype(np.uint8)
def embed(imgs):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 512):
            px = proc(images=list(imgs[i:i + 512]), return_tensors="pt")["pixel_values"].to(a.device)
            f = model.get_image_features(pixel_values=px); f = getattr(f, "pooler_output", f); out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)
rng = np.random.default_rng(0)
F = embed(X); Fa = embed(np.concatenate([corrupt(X[i:i + 2048], rng) for i in range(0, len(X), 2048)]))
np.savez_compressed(a.out, X=F.astype(np.float32), X_alt=Fa.astype(np.float32), y=y, is_test=is_test); print("saved", a.out, F.shape)
