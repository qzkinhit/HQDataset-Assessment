#!/usr/bin/env python3
"""MIRFLICKR-25000 image-tag pool. Native Flickr user tags with manual concept annotations, nothing is injected. The arms of the DataComp track keep a share of 20000 pairs, the arm "all" is the raw pool of 20000 pairs, the projection head of the multimodal track is trained on the kept pairs and scored by image-to-text retrieval on COCO. On 5000 held-out MIRFLICKR images it also reports whether the retrieved tag string belongs to an image sharing a manual concept, with and without the image itself in the gallery. Consistency is the share of kept pairs above the 5th percentile of the similarity of clean COCO pairs."""
import argparse, json, os, time, zlib
import run_datacomp as C     # sets the thread limits
import numpy as np, torch
from multiprocessing import Pool
D = {}
def load(native, feats):
    z = np.load(feats, allow_pickle=True); I, T, Y, tags = z["img"], z["txt"], z["concepts"], z["tags"].tolist(); has = np.array([len(t) > 0 for t in tags]); p = np.random.default_rng(7).permutation(len(I)); ho, po = p[:5000], p[5000:]; c = np.load(f"{native}/coco_clip.npz"); Ic, Tc = c["img"].astype(np.float32), c["txt"].astype(np.float32); tau = float(np.percentile((Ic * Tc).sum(1), 5))
    own = np.where(Y[ho].any(1))[0]; q = ho[own]; share = (Y[q].astype(np.float32) @ Y[ho].T.astype(np.float32)) > 0; D.update(I=I, T=T, Y=Y, has=has, ho=ho, po=po, Ic=Ic, Tc=Tc, tau=tau, q=q, own=own, share=share, sim=(I[po] * T[po]).sum(1), rule=C.text_rule([tags[i] for i in po]), first=C.text_first([tags[i] for i in po]), chance=float(share.mean())); print("pairs", len(po), "with tags", int(has[po].sum()), "tau", tau, "consistency of the pool", float((D["sim"] > tau).mean()), "rule", D["rule"].mean(), "first", D["first"].mean(), "queries", len(q), "chance", D["chance"], flush=True)
def concept(net):
    """precision at 1 and 5 of the retrieved tag strings among the held-out images"""
    o = (-C.similarity(net, D["I"][D["q"]], D["T"][D["ho"]])).argsort(1)[:, :6]; rows = np.arange(len(o))[:, None]; own, share = D["own"], D["share"]; ns = np.array([r[r != i][:5] for r, i in zip(o, own)])
    return dict(concept_p1=float(share[rows, o[:, :1]].mean()), concept_p5=float(share[rows, o[:, :5]].mean()), concept_p1_noself=float(share[rows, ns[:, :1]].mean()), concept_p5_noself=float(share[rows, ns].mean()), concept_chance=D["chance"], self_r1=float((o[:, 0] == own).mean()))
def run(key):
    t0 = time.time(); torch.set_num_threads(2); seed, arm, frac = key["seed"], key["arm"], key["frac"]; rng = np.random.default_rng(zlib.crc32(json.dumps(["mir", seed, arm, frac]).encode()) % (2 ** 32)); ids = C.select(arm, frac, D["sim"], D["rule"], D["first"], rng); use = D["po"][ids]; net = C.train(D["I"][use], D["T"][use], seed); p = np.random.default_rng(seed + 11).permutation(len(D["Ic"])); va, te = p[:1000], p[1000:2000]; a = C.recall(net, D["Ic"][va], D["Tc"][va]); b = C.recall(net, D["Ic"][te], D["Tc"][te])
    return dict(key, ds="mirflickr25k", model="head", group="multimodal", task="retrieval", n_pool=int(len(D["po"])), n_kept=int(len(ids)), kept_share=len(ids) / len(D["po"]), n_train=int(len(use)), without_tags=float((~D["has"][use]).mean()), consistency=float((D["sim"][ids] > D["tau"]).mean()), consistency_pool=float((D["sim"] > D["tau"]).mean()), mean_similarity=float(D["sim"][ids].mean()), tau=D["tau"], val_r1=a["r1"], val_r5=a["r5"], val_mrr=a["mrr"], test_r1=b["r1"], test_r5=b["r5"], test_mrr=b["mrr"], score=b["r5"], sec=round(time.time() - t0, 1), **concept(net))
def safe(k):
    try: return run(k)
    except Exception as e: return dict(k, error=repr(e)[:300])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--native", default="."); ap.add_argument("--feats"); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--limit", type=int); A = ap.parse_args(); load(A.native, A.feats); done = set()
    if os.path.exists(A.out):
        for l in open(A.out):
            try:
                r = json.loads(l)
                if "error" not in r: done.add(r["id"])
            except Exception: pass
    jobs = [dict(id=f"mf_{arm}_f{frac}_s{s}", arm=arm, frac=frac, seed=s) for s in range(A.seeds) for arm, fr in [("all", (1.0,))] + [(a_, C.FRACS) for a_ in C.ARMS] for frac in fr if f"mf_{arm}_f{frac}_s{s}" not in done][:A.limit]; print("jobs", len(jobs), "done", len(done), flush=True)
    with Pool(A.procs) as pool, open(A.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(safe, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush(); print(i, len(jobs), r["id"], r.get("test_r5"), r.get("concept_p5"), r.get("sec"), r.get("error", "")[:200], flush=True)
    print("finished", flush=True)
