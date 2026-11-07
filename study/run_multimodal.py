#!/usr/bin/env python3
"""Multimodal track of the controlled study. Image-caption pairs from COCO val2017 are encoded once with CLIP. A projection head is trained on the pairs
with an InfoNCE loss under one controlled defect, and scored by image-to-text retrieval on held-out pairs. The script also measures how well the CLIP similarity of a pair,
the usual consistency proxy, detects the mismatched pairs."""
import argparse, json, os, sys, time, fcntl, zlib
import numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ap.add_argument("--out", required=True); A = ap.parse_args(); J = json.loads(A.job)
ROOT = os.environ.get("HQD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads")); device = "cuda"
z = np.load(f"{ROOT}/coco_clip.npz"); I, T, cat = z["img"].astype(np.float32), z["txt"].astype(np.float32), z["cat"]
seed, defects = J["seed"], J["defects"]; rng = np.random.default_rng(zlib.crc32(json.dumps([seed, defects], sort_keys=True).encode()) % (2 ** 32)); torch.manual_seed(seed)
p0 = np.random.default_rng(seed + 11).permutation(len(I)); te, tr = p0[:1000], p0[1000:]; t0 = time.time()
Itr, Ttr = I[tr].copy(), T[tr].copy(); n = len(tr); auc = None; est = None; leak = None; diag = {}
for d, r in defects.items():
    # the defect count follows the size of the original pool, the positions address the pairs that remain
    k = int(round(r * n)); current_n = len(Itr); ids = rng.choice(current_n, k, replace=False)
    if d == "mismatch":
        perm = rng.permutation(n); bad = np.zeros(n, bool); bad[ids] = True; Ttr[ids] = T[tr][perm[:k]]; sim = (Itr * Ttr).sum(1)
        order = np.argsort(sim); ranks = np.empty(n); ranks[order] = np.arange(n); auc = float((ranks[~bad].sum() - (n - k) * (n - k - 1) / 2) / max(1, k * (n - k)))
        # the consistency sub-indicator read off a similarity threshold fixed at the 5th percentile of clean pairs
        thr = np.percentile((I[tr] * T[tr]).sum(1), 5); est = float((sim >= thr).mean())
        # the detector error estimated from 200 pairs whose truth is known from the construction, then the corrected share (Rogan and Gladen)
        flag = sim < thr; aud = rng.choice(n, 200, replace=False); ab = bad[aud]; se_a = float(flag[aud][ab].mean()) if ab.any() else float("nan"); sp_a = float((~flag[aud][~ab]).mean())
        corr = float(1 - (flag.mean() + sp_a - 1) / (se_a + sp_a - 1)); diag = dict(detector_se=float(flag[bad].mean()), detector_sp=float((~flag[~bad]).mean()), audit_se=se_a, audit_sp=sp_a, consistency_true=float(1 - bad.mean()), consistency_corrected=corr)
    elif d == "missing":        # one modality lost, the pair cannot enter a contrastive batch
        keep = np.setdiff1d(np.arange(len(Itr)), ids[ids < len(Itr)]); Itr, Ttr = Itr[keep], Ttr[keep]
    elif d == "duplicate":
        rest = np.setdiff1d(np.arange(current_n), ids); src = rng.choice(rest, k); Itr[ids], Ttr[ids] = Itr[src], Ttr[src]
    elif d == "coverage":
        w = rng.normal(size=Itr.shape[1]); s = Itr @ w; keep = np.where(s <= np.quantile(s, 1 - r))[0]; sel = np.r_[keep, rng.choice(keep, len(Itr) - len(keep))]; Itr, Ttr = Itr[sel], Ttr[sel]
    elif d == "scale":
        keep = np.setdiff1d(np.arange(len(Itr)), ids[ids < len(Itr)]); Itr, Ttr = Itr[keep], Ttr[keep]
    elif d == "contamination":
        li = rng.choice(1000, min(k, 1000), replace=False); ti = rng.choice(len(Itr), len(li), replace=False); Itr[ti], Ttr[ti] = I[te][li], T[te][li]; leak = li
dim = I.shape[1]
class Head(torch.nn.Module):
    def __init__(s): super().__init__(); s.i = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)); s.t = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)); s.log_tau = torch.nn.Parameter(torch.tensor(2.66))
    def forward(s, a, b): return F.normalize(s.i(a) + 0, dim=-1), F.normalize(s.t(b) + 0, dim=-1)
net = Head().to(device); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
Xa, Xb = torch.from_numpy(Itr).to(device), torch.from_numpy(Ttr).to(device); steps, BS = 800, 256; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for s in range(steps):
    b = torch.randint(0, Xa.size(0), (BS,), device=device); a, c = net(Xa[b], Xb[b]); lg = a @ c.t() * net.log_tau.exp(); y = torch.arange(BS, device=device)
    loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2; opt.zero_grad(); loss.backward(); opt.step(); sch.step()
net.eval()
with torch.no_grad():
    a, c = net(torch.from_numpy(I[te]).to(device), torch.from_numpy(T[te]).to(device)); S = (a @ c.t()).cpu().numpy()
rank = (-S).argsort(1); pos = (rank == np.arange(1000)[:, None]).argmax(1)
r = dict(J, group="multimodal", task="retrieval", n_train=int(len(Itr)), n_test=1000, score=float((pos < 5).mean()), r1=float((pos < 1).mean()), mrr=float((1 / (pos + 1)).mean()), detect_auc=auc, consistency_est=est, sec=round(time.time() - t0, 1), **diag)
if leak is not None:
    m = np.ones(1000, bool); m[leak] = False; r["score_unseen"] = float((pos[m] < 5).mean()) if m.any() else None
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
