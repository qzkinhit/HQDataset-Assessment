#!/usr/bin/env python3
"""Synthetic image track. Real CIFAR-10 images are replaced by, or extended with, published synthetic images of the same classes (EDM, DDPM, CIFAKE), nothing is generated or injected here. A deep network is trained from scratch and scored on the real CIFAR-10 test set and on CIFAR-10.1, a test set collected again by the original procedure."""
import argparse, json, os, time, fcntl, zlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cudnn.benchmark = True
ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ap.add_argument("--out", required=True); A = ap.parse_args(); J = json.loads(A.job)
ROOT = os.environ.get("HQD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads"))
ds, arch, seed = J["ds"], J["arch"], J["seed"]; EPOCHS = J.get("epochs", 30); t0 = time.time()
torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(zlib.crc32(A.job.encode()) % (2 ** 32))
z = np.load(f"{ROOT}/{ds}_real.npz"); Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"].astype(np.int64), z["Xte"], z["yte"].astype(np.int64); clean = ytr.copy(); info = {}
if J.get("gen") and J["gen"] != "real":
    y_ = np.load(f"{ROOT}/synth_{J['gen']}.npz"); Xs, ys = y_["image"], y_["label"].astype(np.int64); n0 = len(Xtr)
    if J.get("add", 0) > len(Xs) and os.path.exists(f"{ROOT}/synth_{J['gen']}_all.npz"): y_ = np.load(f"{ROOT}/synth_{J['gen']}_all.npz"); Xs, ys = y_["image"], y_["label"].astype(np.int64)     # EDM add200000
    if "share" in J:
        k = int(round(J["share"] * n0)); ir = rng.permutation(n0)[:n0 - k]; iss = rng.permutation(len(Xs))[:k]; Xtr, ytr = np.concatenate([Xtr[ir], Xs[iss]]), np.r_[ytr[ir], ys[iss]]; info = dict(synthetic_share=float(k / n0))
    else:
        iss = rng.permutation(len(Xs))[:J["add"]]; Xtr, ytr = np.concatenate([Xtr, Xs[iss]]), np.r_[ytr, ys[iss]]; info = dict(synthetic_share=float(len(iss) / len(ytr)))
if J.get("noise"):
    noisy = z[J["noise"]].astype(np.int64); wrong = noisy != clean; ids = rng.permutation(np.where(wrong)[0]); act, f = J["action"], J["f"]; y = noisy.copy(); k = int(round(f * len(ids))); keep = np.ones(len(y), bool)
    if act == "repair": y[ids[:k]] = clean[ids[:k]]
    elif act == "delete": keep[ids[:k]] = False
    Xtr, ytr = Xtr[keep], y[keep]; info = dict(label_accuracy=float((ytr == clean[keep]).mean()), retained=float(keep.mean()), noise_rate=float(wrong.mean()))
if J.get("coverage") is not None or J.get("scale") is not None:
    coarse, fine = z["coarse_train"].astype(np.int64), clean; Yte_coarse = z["coarse_test"].astype(np.int64); seen_fine = np.ones(100, bool)
    if J.get("coverage") is not None:
        n_keep = J["coverage"]; seen_fine[:] = False
        for c in range(20):
            sub = np.unique(fine[coarse == c]); seen_fine[rng.permutation(sub)[:n_keep]] = True
        mask = seen_fine[fine]
    else:
        mask = rng.random(len(fine)) < J["scale"]
    Xtr, ytr = Xtr[mask], coarse[mask]; yte_fine = yte; yte = Yte_coarse; info = dict(retained=float(mask.mean()), label_accuracy=1.0)
classes = int(max(ytr.max(), yte.max())) + 1
device = "cuda"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
def to_t(x): return ((torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(device) / 255.0) - MEAN) / STD
Xt, Yt = to_t(Xtr), torch.from_numpy(ytr).to(device); Xv, Yv = to_t(Xte), torch.from_numpy(yte).to(device)
class Block(nn.Module):
    def __init__(s, a, b, st):
        super().__init__(); s.c1 = nn.Conv2d(a, b, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(b); s.c2 = nn.Conv2d(b, b, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(b)
        s.sk = nn.Sequential() if st == 1 and a == b else nn.Sequential(nn.Conv2d(a, b, 1, st, bias=False), nn.BatchNorm2d(b))
    def forward(s, x): return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))
def resnet18(c):
    L = [nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]; a = 64
    for b, st in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]: L.append(Block(a, b, st)); a = b
    return nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, c))
def vgg11(c):
    L, a = [], 3
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]:
        if v == "M": L.append(nn.MaxPool2d(2))
        else: L += [nn.Conv2d(a, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU()]; a = v
    return nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, c))
net = {"resnet18": resnet18, "vgg11": vgg11}[arch](classes).to(device)
BS = 128; steps = EPOCHS * max(1, len(ytr) // BS); opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps)
scaler = torch.amp.GradScaler("cuda")
def augment(x):
    b = x.size(0); pad = F.pad(x, (4, 4, 4, 4), mode="reflect"); i = torch.randint(0, 9, (b,), device=device); j = torch.randint(0, 9, (b,), device=device)
    grid_y = (torch.arange(32, device=device).view(1, -1) + i.view(-1, 1)); grid_x = (torch.arange(32, device=device).view(1, -1) + j.view(-1, 1))
    out = pad[torch.arange(b, device=device).view(-1, 1, 1), :, grid_y.unsqueeze(2).expand(-1, -1, 32), grid_x.unsqueeze(1).expand(-1, 32, -1)].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=device) < 0.5; out[flip] = out[flip].flip(3); return out
net.train(); s = 0
while s < steps:
    perm = torch.randperm(Xt.size(0), device=device)
    for i in range(0, Xt.size(0) - BS + 1, BS):
        if s >= steps: break
        ib = perm[i:i + BS]; x = augment(Xt[ib]); y = Yt[ib]
        with torch.amp.autocast("cuda"): loss = F.cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); s += 1
net.eval(); P = []
with torch.no_grad(), torch.amp.autocast("cuda"):
    for i in range(0, Xv.size(0), 1000): P.append(net(Xv[i:i + 1000]).argmax(1))
p = torch.cat(P).cpu().numpy()
c = np.load(f"{ROOT}/cifar10_1.npz"); Xc, Yc = to_t(c["X"]), c["y"].astype(np.int64)
with torch.no_grad(), torch.amp.autocast("cuda"): pc = torch.cat([net(Xc[i:i + 1000]).argmax(1) for i in range(0, len(Yc), 1000)]).cpu().numpy()
info["score_c101"] = float((pc == Yc).mean())
r = dict(J, group="image", task="cls", n_train=int(len(ytr)), n_test=int(len(yte)), score=float((p == yte).mean()), sec=round(time.time() - t0, 1), **info)
if J.get("coverage") is not None:
    un = ~seen_fine[yte_fine]; r["score_seen_sub"] = float((p[~un] == yte[~un]).mean()); r["score_unseen_sub"] = float((p[un] == yte[un]).mean()) if un.any() else None
os.makedirs(f"{ROOT}/preds", exist_ok=True); np.save(f"{ROOT}/preds/{J['id']}.npy", p.astype(np.uint8))
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
