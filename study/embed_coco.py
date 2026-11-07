#!/usr/bin/env python3
"""CLIP features of COCO val2017 images and their first captions."""
import json, os, sys, zipfile, io
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ROOT = sys.argv[1]; device = "cuda"
ann = json.load(zipfile.ZipFile(f"{ROOT}/coco/annotations.zip").open("annotations/captions_val2017.json")); inst = json.load(zipfile.ZipFile(f"{ROOT}/coco/annotations.zip").open("annotations/instances_val2017.json"))
cap = {}
for a in ann["annotations"]: cap.setdefault(a["image_id"], a["caption"])
cat = {}
for a in inst["annotations"]: cat.setdefault(a["image_id"], a["category_id"])
zf = zipfile.ZipFile(f"{ROOT}/coco/val2017.zip"); ims = [i for i in ann["images"] if i["id"] in cap]
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval(); p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); FI, FT = [], []
with torch.no_grad():
    for k in range(0, len(ims), 256):
        b = ims[k:k + 256]; pil = [Image.open(io.BytesIO(zf.read("val2017/" + i["file_name"]))).convert("RGB") for i in b]
        e = p(images=pil, text=[cap[i["id"]] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        f = m.get_image_features(pixel_values=e["pixel_values"]); f = getattr(f, "pooler_output", f); g = m.get_text_features(input_ids=e["input_ids"], attention_mask=e["attention_mask"]); g = getattr(g, "pooler_output", g)
        FI.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy()); FT.append(torch.nn.functional.normalize(g, dim=-1).cpu().numpy())
np.savez(f"{ROOT}/coco_clip.npz", img=np.concatenate(FI), txt=np.concatenate(FT), cat=np.array([cat.get(i["id"], 0) for i in ims]), ids=np.array([i["id"] for i in ims])); print("saved", len(ims))
