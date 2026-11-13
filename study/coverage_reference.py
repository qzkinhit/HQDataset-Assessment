#!/usr/bin/env python3
"""Reference value of the coverage track at full coverage, computed from saved test predictions without any training.
A coverage job of run_vision.py keeps k fine classes in every CIFAR-100 superclass, so 5 - k fine classes per superclass are unseen in training. The job with k = 5 has no unseen fine class and its record stores no unseen score.
For every k from 1 to 4 and every seed, the unseen fine classes of the job are replayed from its job string, whose CRC32 seeds the random generator exactly as in run_vision.py. The superclass accuracy of the k = 5 model of the same network and seed on the test images of these fine classes is the reference value of that pair. Its mean over k and seeds is the reference point of one network, and the accuracy of the same k = 5 model on all test images is reported beside it.
Before any reference value is computed, the scores of every coverage record are recomputed from the predictions of that job and compared with the stored values. Nothing is written if one of them differs."""
import argparse, collections, json, os, sys, zlib
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="folder with cifar100_real.npz and the preds/ folder written by run_vision.py")
ap.add_argument("--results", required=True, help="result file of the vision track (results_vision.jsonl)")
ap.add_argument("--out", required=True, help="output JSON file")
ap.add_argument("--tol", type=float, default=1e-12, help="largest accepted absolute difference between a recomputed and a stored score")
A = ap.parse_args()
z = np.load(os.path.join(A.root, "cifar100_real.npz"))
fine, coarse = z["ytr"].astype(np.int64), z["coarse_train"].astype(np.int64); yte_fine, yte = z["yte"].astype(np.int64), z["coarse_test"].astype(np.int64)
def replay(r):
    """the job string that run_vision.py received (the record fields before 'group'), the seen fine classes drawn from it and the training mask"""
    keys = list(r); job = json.dumps({k: r[k] for k in keys[:keys.index("group")]})
    rng = np.random.default_rng(zlib.crc32(job.encode()) % (2 ** 32)); seen = np.zeros(100, bool)
    for c in range(20): seen[rng.permutation(np.unique(fine[coarse == c]))[:r["coverage"]]] = True
    return seen, seen[fine]
def acc(p, m): return float((p[m] == yte[m]).mean()) if m.any() else None
def pred(r): return np.load(os.path.join(A.root, "preds", f"{r['id']}.npy")).astype(np.int64)
rows = {}
for line in open(A.results):
    r = json.loads(line)
    if r.get("coverage") is None: continue
    key = (r["arch"], r["seed"], r["coverage"])
    if key in rows: sys.exit(f"two coverage records for {key}")
    rows[key] = r
# every stored score is recomputed from the predictions of its own job
checked, matched, worst, bad, seen_of = collections.Counter(), collections.Counter(), 0.0, [], {}
for key, r in sorted(rows.items()):
    seen, mask = replay(r); p = pred(r); un = ~seen[yte_fine]; seen_of[key] = seen
    new = dict(score=float((p == yte).mean()), score_seen_sub=acc(p, ~un), score_unseen_sub=acc(p, un), n_train=int(mask.sum()), retained=float(mask.mean()))
    ok = True
    for f, v in new.items():
        if v is None or r[f] is None: ok = ok and v is None and r[f] is None
        else: d = abs(v - r[f]); worst = max(worst, d); ok = ok and d <= A.tol
    checked[r["arch"]] += 1; matched[r["arch"]] += int(ok)
    if not ok: bad.append((r["id"], {f: (v, r[f]) for f, v in new.items()}))
print("records", sum(checked.values()), "matched", sum(matched.values()), "max abs diff", worst, flush=True)
if bad:
    for b in bad: print("mismatch (recomputed, stored)", b)
    sys.exit(1)
# the k = 5 model of the same network and seed, scored on the fine classes that the job with k left out
out = dict(description="Superclass accuracy of the full coverage model (k = 5) on the CIFAR-100 test images whose fine class was left out of training by the coverage job with k kept fine classes per superclass, for k = 1 to 4 and every seed. "
           "The unseen fine classes of each job are replayed from its job string as in run_vision.py. reference_mean is the mean over all pairs of k and seed and serves as the unseen value at k = 5. "
           "full_coverage_all_sub is the accuracy of the same k = 5 model on all test images.",
           verification=dict(records=sum(checked.values()), matched=sum(matched.values()), records_by_network=dict(checked), matched_by_network=dict(matched), fields=list(new), max_abs_diff=worst, tolerance=A.tol), networks={})
for arch in sorted({k[0] for k in rows}):
    seeds = sorted({k[1] for k in rows if k[0] == arch}); ks = sorted({k[2] for k in rows if k[0] == arch and k[2] < 5}); by, full, n_img = {}, {}, {}
    for s in seeds:
        p5 = pred(rows[(arch, s, 5)]); full[str(s)] = float((p5 == yte).mean())
        for k in ks:
            un = ~seen_of[(arch, s, k)][yte_fine]; by.setdefault(str(k), {})[str(s)] = acc(p5, un)
            if n_img.setdefault(str(k), int(un.sum())) != int(un.sum()): sys.exit(f"unseen test images differ across seeds at k = {k}")
    vals = [v for d in by.values() for v in d.values()]
    out["networks"][arch] = dict(seeds=seeds, k_values=ks, unseen_test_images_by_k=n_img, reference_by_k_seed=by, reference_mean_by_k={k: float(np.mean(list(d.values()))) for k, d in by.items()},
                                 reference_mean=float(np.mean(vals)), n_values=len(vals), full_coverage_all_sub_by_seed=full, full_coverage_all_sub_mean=float(np.mean(list(full.values()))))
with open(A.out, "w") as f: json.dump(out, f, indent=1)
print(json.dumps({a: dict(reference_mean=v["reference_mean"], full_coverage_all_sub_mean=v["full_coverage_all_sub_mean"]) for a, v in out["networks"].items()}))
