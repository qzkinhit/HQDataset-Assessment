#!/usr/bin/env python3
"""WILDS Camelyon17 subset (Koh et al. 2021), real tissue patches and a real shift between hospitals, nothing is injected. ResNet-18 is trained from scratch on images of the training hospitals and scored on an in-distribution hold-out of those hospitals and on the official test hospital."""
import argparse, fcntl, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = True
ap = argparse.ArgumentParser(); ap.add_argument("--job"); ap.add_argument("--out"); A = ap.parse_args(); J = json.loads(A.job); ROOT = os.environ.get("HQD_ROOT", "."); seed = J["seed"]; EPOCHS = J.get("epochs", 10); t0 = time.time(); device = "cuda"
torch.manual_seed(seed); np.random.seed(seed); X = np.load(f"{ROOT}/camelyon_x.npy"); m = np.load(f"{ROOT}/camelyon_meta.npz"); y, part = m["y"].astype(np.int64), m["part"]; tr = np.where(part == 0)[0]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
def to_t(ids): return ((torch.from_numpy(X[ids].transpose(0, 3, 1, 2)).to(device, non_blocking=True).float() / 255.0) - MEAN) / STD
class Block(nn.Module):
    def __init__(s, a, b, st):
        super().__init__(); s.c1 = nn.Conv2d(a, b, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(b); s.c2 = nn.Conv2d(b, b, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(b)
        s.sk = nn.Sequential() if st == 1 and a == b else nn.Sequential(nn.Conv2d(a, b, 1, st, bias=False), nn.BatchNorm2d(b))
    def forward(s, x): return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))
L = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]; a = 64     # the stem of the ImageNet ResNet, the images have 96 pixels
for b, st in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]: L.append(Block(a, b, st)); a = b
net = nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 2)).to(device); BS = 128; steps = EPOCHS * (len(tr) // BS); opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps); scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(seed)
net.train(); s = 0
while s < steps:
    perm = rng.permutation(tr)
    for i in range(0, len(tr) - BS + 1, BS):
        if s >= steps: break
        ib = perm[i:i + BS]; x = to_t(np.sort(ib)); yb = torch.from_numpy(y[np.sort(ib)]).to(device)
        if np.random.random() < .5: x = x.flip(3)     # flips per batch, the tissue has no orientation
        if np.random.random() < .5: x = x.flip(2)
        with torch.amp.autocast("cuda"): loss = F.cross_entropy(net(x), yb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); s += 1
net.eval(); r = dict(J, ds="camelyon17", group="image", task="cls", model="resnet18", n_train=int(len(tr)))
for name, v in (("id", 1), ("ood", 2)):
    ids = np.where(part == v)[0]
    with torch.no_grad(), torch.amp.autocast("cuda"): p = torch.cat([net(to_t(ids[i:i + 500])).argmax(1) for i in range(0, len(ids), 500)]).cpu().numpy()
    r.update({f"score_{name}": float((p == y[ids]).mean()), f"f1_{name}": float(f1_score(y[ids], p, average="macro")), f"n_{name}": int(len(ids))})
r.update(score=r["score_ood"], sec=round(time.time() - t0, 1))
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
