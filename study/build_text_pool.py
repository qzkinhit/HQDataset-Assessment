#!/usr/bin/env python3
"""Build the five-domain text pool of the language-model track, qpool_train.jsonl, qpool_heldout.jsonl and pool_manifest.json.
Per domain the training file holds 3000 high-quality records and 2000 low-quality records, 500 each of truncation, template, cross-domain and low-tier text, and the held-out file holds 400 high-quality records.
Rows come from pinned revisions of five public datasets through huggingface_hub, and the manifest records revisions, shards, sizes and SHA-256 values. With the default sizes the script compares the two files with the recorded SHA-256 values.
Usage: build_text_pool.py --out DIR [--pool-hi 3000] [--held-per 400] [--max-shards 4]"""
import argparse, gzip, hashlib, json, os
LOW_FRAC, MINC, MAXC = 0.40, 150, 2000
DOMAINS = ["general", "math", "code", "image", "table"]
PREFIX = {"HuggingFaceFW/fineweb-edu": "sample/10BT/", "codeparrot/codeparrot-clean-valid": "", "yerevann/coco-karpathy": "", "mstz/adult": "income/"}
REVISIONS = {"HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "HuggingFaceTB/finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594", "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
             "yerevann/coco-karpathy": "448fdb1bc7b2d09e46881c4541a14d796a3d41e8", "mstz/adult": "f90a6fc5f6efb865382c05b75ab9037b62104172"}
POOL_SHA256 = {"train": "84e174dbb097288c6b4473af2af8d6cb46a0b00a000b6534545725e53f9939c5", "heldout": "1e4a45c9c959995a3c10c840dc2a8b84ce33bd82ba17682c8f50c3b4b3a1e785"}
def field(*names): return lambda ex: next((ex[n] for n in names if ex.get(n)), None)
def caption(ex):
    """One image-text record joins the captions of one image."""
    s = ex.get("sentences")
    return " ".join(str(x) for x in s) if isinstance(s, (list, tuple)) else s
def table_row(ex):
    """One table record writes one row as text."""
    return ". ".join(f"{k} is {v}" for k, v in ex.items() if v not in (None, ""))
HIGH = {"general": ("HuggingFaceFW/fineweb-edu", None, field("text")), "math": ("HuggingFaceTB/finemath", "finemath-4plus", field("text")), "code": ("codeparrot/codeparrot-clean-valid", None, field("content")), "image": ("yerevann/coco-karpathy", None, caption), "table": ("mstz/adult", None, table_row)}
LOWTIER = {"math": ("HuggingFaceTB/finemath", "finemath-3plus", field("text"))}
MANIFEST = {"revisions": {}, "shards": [], "builder": os.path.basename(__file__)}
MAX_SHARDS = 4
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def shard_rows(repo, config):
    from huggingface_hub import HfApi, hf_hub_download
    rev = REVISIONS[repo]; info = HfApi().repo_info(repo, repo_type="dataset", revision=rev)
    if info.sha != rev: raise RuntimeError(f"{repo} resolved {info.sha}, expected {rev}")
    MANIFEST["revisions"][repo] = rev; files = [f.rfilename for f in info.siblings]; pref = PREFIX.get(repo)
    if pref is None: pref = (config + "/") if config else ""
    cand = [f for f in files if f.startswith(pref)] if pref else list(files)
    pq_files = sorted(f for f in cand if f.endswith(".parquet")); js_files = sorted(f for f in cand if f.endswith((".jsonl", ".json.gz", ".jsonl.gz"))); csv_files = sorted(f for f in cand if f.endswith(".csv"))
    picks = (pq_files or js_files or csv_files)[:MAX_SHARDS]
    if not picks: raise RuntimeError(f"No readable shards in {repo} under {pref!r}")
    for fname in picks:
        path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev); MANIFEST["shards"].append({"repo": repo, "file": fname, "rev": rev, "bytes": os.path.getsize(path), "sha256": sha(path)})
        if fname.endswith(".csv"):
            import pyarrow.csv as pacsv
            for batch in pacsv.read_csv(path).to_batches(max_chunksize=2048): yield from batch.to_pylist()
        elif fname.endswith(".parquet"):
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048): yield from batch.to_pylist()
        else:
            op = gzip.open if fname.endswith(".gz") else open
            with op(path, "rt", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try: yield json.loads(line)
                        except Exception: continue
def texts(repo, config, extract, need, skip=0):
    """Whitespace-normalized texts of 150 to 2000 characters, in source order."""
    out, seen = [], 0
    for ex in shard_rows(repo, config):
        try: raw = extract(ex)
        except Exception: raw = None
        if not raw: continue
        t = " ".join(str(raw).split())
        if MINC <= len(t) <= MAXC:
            seen += 1
            if seen <= skip: continue
            out.append(t[:MAXC])
        if len(out) >= need: return out
    raise RuntimeError(f"{repo} yielded {len(out)} of {need} usable texts")
def rec(domain, i, text, quality, noise=""): return {"id": f"{domain[:2]}{quality[0]}{i:05d}", "modality": "text", "domain": domain, "text": text, "meta": {"quality": quality, "noise": noise}}
def held_rec(domain, i, text): return dict(rec(domain, i, text, "high", "heldout"), id=f"{domain[:2]}hH{i:05d}")
def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
def build(out, pool_hi=3000, held_per=400):
    n_low = int(round(pool_hi * LOW_FRAC / (1 - LOW_FRAC))); per = n_low // 4
    if per * 4 != n_low: raise ValueError("The low-quality records must split evenly over four kinds")
    os.makedirs(out, exist_ok=True); MANIFEST["config"] = {"POOL_HI": pool_hi, "HELD_PER": held_per, "MODALITIES": DOMAINS, "LOW_FRAC": LOW_FRAC, "MINC": MINC, "MAXC": MAXC}
    hi_all, held_all = {}, {}
    for d in DOMAINS:
        path, cfg, extract = HIGH[d]; rows = texts(path, cfg, extract, pool_hi + held_per); held_all[d] = rows[:held_per]; hi_all[d] = rows[held_per:]; print("fetched", d, len(hi_all[d]), len(held_all[d]), flush=True)
    train, held = [], []
    for d in DOMAINS:
        held += [held_rec(d, i, t) for i, t in enumerate(held_all[d])]; train += [rec(d, i, t, "high") for i, t in enumerate(hi_all[d])]
        train += [rec(d, 10000 + i, t[:80 + (i % 70)], "low", "truncation") for i, t in enumerate(hi_all[d][:per])]                 # text cut to 80 to 149 characters
        seeds = hi_all[d][per:per + max(1, (per + 11) // 12)]; c = 0                                                                # a few records repeated 12 times with a changed tail
        for s_i, seed in enumerate(seeds):
            for r in range(12):
                if c >= per: break
                train.append(rec(d, 20000 + c, f"{seed} [v{r}#{s_i}]", "low", "template")); c += 1
        if c < per: raise RuntimeError(f"{d} holds {c} of {per} template records")
        src = hi_all[DOMAINS[(DOMAINS.index(d) + 1) % len(DOMAINS)]]                                                                # texts of the next domain labelled as this domain
        train += [rec(d, 30000 + i, src[i % len(src)], "low", "crossdomain") for i in range(per)]
        if d in LOWTIER:                                                                                                            # a lower tier of the same source, or a heavy cut
            lp, lc, lx = LOWTIER[d]; train += [rec(d, 40000 + i, t, "low", "lowtier") for i, t in enumerate(texts(lp, lc, lx, per, skip=pool_hi + held_per))]
        else: train += [rec(d, 40000 + i, t[:120], "low", "lowtier") for i, t in enumerate(hi_all[d][per:2 * per])]
    train_path, held_path = os.path.join(out, "qpool_train.jsonl"), os.path.join(out, "qpool_heldout.jsonl")
    write_jsonl(train, train_path); write_jsonl(held, held_path); MANIFEST.update(train_sha256=sha(train_path), heldout_sha256=sha(held_path), counts={"train": len(train), "heldout": len(held)})
    if pool_hi == 3000 and held_per == 400 and (MANIFEST["train_sha256"], MANIFEST["heldout_sha256"]) != (POOL_SHA256["train"], POOL_SHA256["heldout"]):
        raise RuntimeError(f"The pool differs from the recorded hashes {POOL_SHA256}")
    with open(os.path.join(out, "pool_manifest.json"), "w", encoding="utf-8") as f: json.dump(MANIFEST, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("built", MANIFEST["counts"], flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", required=True); ap.add_argument("--pool-hi", type=int, default=3000); ap.add_argument("--held-per", type=int, default=400); ap.add_argument("--max-shards", type=int, default=4); a = ap.parse_args()
    MAX_SHARDS = a.max_shards; build(a.out, a.pool_hi, a.held_per)
