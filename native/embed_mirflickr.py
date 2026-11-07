#!/usr/bin/env python3
"""CLIP ViT-B/32 features of MIRFLICKR-25000 (Huiskes and Lew 2008): the images and, per image, the string of its native Flickr user tags, with the manual concept annotations of the benchmark as a binary matrix. Runs once, the archive is read directly. The r1 lists are stricter copies of the same concepts and are not used."""
import argparse, io, os, zipfile
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ap = argparse.ArgumentParser(); ap.add_argument("--native", default="."); ap.add_argument("--out"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=12); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--model", default="openai/clip-vit-base-patch32"); A = ap.parse_args(); torch.set_num_threads(A.threads)
zi = zipfile.ZipFile(f"{A.native}/mirflickr25k.zip"); za = zipfile.ZipFile(f"{A.native}/mirflickr25k_annotations_v080.zip"); N = 25000
tags = [" ".join(x.strip() for x in zi.read(f"mirflickr/meta/tags/tags{i}.txt").decode("utf-8", "ignore").split("\n") if x.strip()) for i in range(1, N + 1)]; print("images with tags", sum(1 for t in tags if t), flush=True)
names = sorted(n for n in za.namelist() if n.endswith(".txt") and n != "README.txt" and "_r1" not in n); Y = np.zeros((N, len(names)), bool)
for j, n in enumerate(names):
    for x in za.read(n).decode().split(): Y[int(x) - 1, j] = True
print("concepts", [n[:-4] for n in names], "images with a concept", int(Y.any(1).sum()), flush=True)
m = CLIPModel.from_pretrained(A.model).to(A.device).eval(); p = CLIPProcessor.from_pretrained(A.model); FI, FT = [], []
def feat(f): return torch.nn.functional.normalize(getattr(f, "pooler_output", f), dim=-1).cpu().numpy().astype(np.float32)
with torch.no_grad():
    for k in range(0, N, A.bs):
        ids = range(k, min(k + A.bs, N)); pil = [Image.open(io.BytesIO(zi.read(f"mirflickr/im{i + 1}.jpg"))).convert("RGB") for i in ids]; e = p(images=pil, text=[tags[i] for i in ids], return_tensors="pt", padding=True, truncation=True, max_length=77).to(A.device)
        FI.append(feat(m.get_image_features(pixel_values=e["pixel_values"]))); FT.append(feat(m.get_text_features(input_ids=e["input_ids"], attention_mask=e["attention_mask"])))
        if k % (A.bs * 20) == 0: print(k, N, flush=True)
np.savez(A.out, img=np.concatenate(FI), txt=np.concatenate(FT), concepts=Y, concept_names=np.array([n[:-4] for n in names]), tags=np.array(tags, dtype=object)); print("saved", A.out, flush=True)
