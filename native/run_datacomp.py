#!/usr/bin/env python3
"""DataComp image-text pool. One published shard of web pairs with its CLIP ViT-B/32 features (Gadre et al. 2023), nothing is injected. A filter keeps a share of the shard: random, top, middle or bottom CLIP similarity, a text rule on length and alphabetic share, or near-duplicate removal by a hash of the normalized text. The arm "all" is the raw pool, every pair without a cap.
The projection head of the multimodal track is trained on the kept pairs and scored by image-to-text retrieval on a validation part and a separate test part of COCO. The consistency sub-indicator is the share of kept pairs above the 5th percentile of the similarity of clean COCO pairs."""
import argparse, json, os, re, struct, sys, time, zlib
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import numpy as np, torch, torch.nn.functional as F
from multiprocessing import Pool
ARMS = ("random", "sim_top", "sim_mid", "sim_bottom", "text_rule", "dedup"); FRACS = (.05, .1, .2, .3, .5, .7, .9); CAP = 200000
def text_rule(texts):
    """at least 10 and at most 300 characters, at least 60% letters"""
    return np.array([10 <= len(t) <= 300 and sum(c.isalpha() for c in t) >= .6 * len(t) for t in texts])
def text_first(texts):
    """the first pair of every hash of the lower-case letters of the text"""
    seen, out = set(), np.zeros(len(texts), bool)
    for i, t in enumerate(texts):
        h = zlib.crc32(re.sub(r"[^a-z]+", " ", t.lower()).strip().encode())
        if h not in seen: seen.add(h); out[i] = True
    return out
def select(arm, frac, sim, rule, first, rng):
    n = len(sim); k = int(round(frac * n))
    if arm == "all": return np.arange(n)
    if arm == "random": return rng.permutation(n)[:k]
    if arm == "sim_top": return np.argsort(-sim)[:k]
    if arm == "sim_bottom": return np.argsort(sim)[:k]
    if arm == "sim_mid": o = np.argsort(sim); a = (n - k) // 2; return o[a:a + k]
    if arm == "text_rule": return rng.permutation(np.where(rule)[0])[:k]
    if arm == "dedup": return rng.permutation(np.where(first)[0])[:k]
    raise ValueError(arm)
class Head(torch.nn.Module):
    def __init__(s, dim=512): super().__init__(); s.i = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)); s.t = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)); s.log_tau = torch.nn.Parameter(torch.tensor(2.66))
    def forward(s, a, b): return F.normalize(s.i(a) + 0, dim=-1), F.normalize(s.t(b) + 0, dim=-1)
def train(Itr, Ttr, seed, steps=800, BS=256):
    torch.manual_seed(seed); net = Head(Itr.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); Xa, Xb = torch.from_numpy(Itr), torch.from_numpy(Ttr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        b = torch.randint(0, Xa.size(0), (BS,)); a, c = net(Xa[b], Xb[b]); lg = a @ c.t() * net.log_tau.exp(); y = torch.arange(BS); loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2; opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval()
def similarity(net, I, T):
    with torch.no_grad(): a, c = net(torch.from_numpy(I), torch.from_numpy(T)); return (a @ c.t()).numpy()
def recall(net, I, T):
    S = similarity(net, I, T); pos = ((-S).argsort(1) == np.arange(len(I))[:, None]).argmax(1); return dict(r1=float((pos < 1).mean()), r5=float((pos < 5).mean()), mrr=float((1 / (pos + 1)).mean()))
def normed(x): x = np.asarray(x, np.float32); return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
def stored(f, start):
    """shape and data offset of a member of an uncompressed npz, from its local header at byte start"""
    f.seek(start); b = f.read(30); n, m = struct.unpack("<HH", b[26:30]); f.seek(start + 30 + n + m); v = np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); return sh, dt, f.tell()
D = {}
def load(native):
    import pyarrow.parquet as pq
    t = pq.read_table(f"{native}/datacomp/0020f0cbd157d470aa56bea08e304b90.parquet", columns=["text", "clip_b32_similarity_score"]); texts = t.column("text").to_pylist(); sim = np.array(t.column("clip_b32_similarity_score").to_pylist(), np.float32); part = f"{native}/datacomp/b32_part.bin"
    with open(part, "rb") as f: a = stored(f, 0); b = stored(f, 517807317)
    I = np.memmap(part, a[1], "r", offset=a[2], shape=a[0]); T = np.memmap(part, b[1], "r", offset=b[2], shape=b[0]); assert len(I) == len(T) == len(sim)
    chk = float(np.abs((normed(I[:2000]) * normed(T[:2000])).sum(1) - sim[:2000]).max()); print("features against the published similarity, largest gap", chk, flush=True); assert chk < .02
    z = np.load(f"{native}/coco_clip.npz"); Ic, Tc = z["img"].astype(np.float32), z["txt"].astype(np.float32); tau = float(np.percentile((Ic * Tc).sum(1), 5)); D.update(I=I, T=T, sim=sim, rule=text_rule(texts), first=text_first(texts), Ic=Ic, Tc=Tc, tau=tau, consistency_all=float((sim > tau).mean()))
    print("pairs", len(sim), "tau", tau, "consistency of the shard", D["consistency_all"], "rule", D["rule"].mean(), "first", D["first"].mean(), flush=True)
def run(key):
    t0 = time.time(); torch.set_num_threads(2); seed, arm, frac = key["seed"], key["arm"], key["frac"]; rng = np.random.default_rng(zlib.crc32(json.dumps([seed, arm, frac]).encode()) % (2 ** 32)); ids = select(arm, frac, D["sim"], D["rule"], D["first"], rng); kept = len(ids); cons = float((D["sim"][ids] > D["tau"]).mean()); mean_sim = float(D["sim"][ids].mean())
    use = ids if arm == "all" else np.sort(rng.permutation(ids)[:CAP]); Itr, Ttr = normed(D["I"][use]), normed(D["T"][use]); net = train(Itr, Ttr, seed); p = np.random.default_rng(seed + 11).permutation(len(D["Ic"])); va, te = p[:1000], p[1000:2000]; a = recall(net, D["Ic"][va], D["Tc"][va]); b = recall(net, D["Ic"][te], D["Tc"][te])
    return dict(key, ds="datacomp", model="head", group="multimodal", task="retrieval", n_pool=int(len(D["sim"])), n_kept=int(kept), kept_share=kept / len(D["sim"]), n_train=int(len(use)), consistency=cons, consistency_pool=D["consistency_all"], mean_similarity=mean_sim, tau=D["tau"], val_r1=a["r1"], val_r5=a["r5"], val_mrr=a["mrr"], test_r1=b["r1"], test_r5=b["r5"], test_mrr=b["mrr"], score=b["r5"], sec=round(time.time() - t0, 1))
def safe(k):
    try: return run(k)
    except Exception as e: return dict(k, error=repr(e)[:300])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--native", default="."); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--limit", type=int); A = ap.parse_args(); load(A.native); done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            try:
                r = json.loads(l)
                if "error" not in r: done.add(r["id"])
            except Exception: pass
    jobs = [dict(id=f"dc_{arm}_f{frac}_s{s}", arm=arm, frac=frac, seed=s) for s in range(A.seeds) for arm, fr in [("all", (1.0,))] + [(a_, FRACS) for a_ in ARMS] for frac in fr if f"dc_{arm}_f{frac}_s{s}" not in done][:A.limit]; print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(A.procs) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush(); print(i, len(jobs), r["id"], r.get("test_r5"), r.get("sec"), r.get("error", "")[:200], flush=True)
    print("finished", flush=True)
