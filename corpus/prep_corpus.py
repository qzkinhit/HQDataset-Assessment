#!/usr/bin/env python3
"""Web corpus track, data preparation. Real web text from the C4 corpus (Dodge et al. 2021, three public shards), nothing injected, and Wikipedia as a curated reference.
Tokenizes the first 512 tokens of every document, and computes the static signals that need no model: alphabetic share, share of tokens inside repeated 5-grams,
stop-word share, and a MinHash signature for near-duplicate groups. Writes flat arrays so that the GPU side needs no text."""
import argparse, glob, gzip, json, re, zlib, os
import numpy as np, pyarrow.parquet as pq
from multiprocessing import Pool
from tokenizers import Tokenizer
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(REPO, "downloads", "corpus")); ap.add_argument("--out", default=os.path.join(REPO, "downloads", "corpus")); ap.add_argument("--procs", type=int, default=24); ap.add_argument("--max-docs", type=int, default=1200000); ap.add_argument("--wiki-docs", type=int, default=20000); A = ap.parse_args()
TOK = None; MAXLEN = 512; NPERM = 32; P = (1 << 31) - 1
r = np.random.default_rng(7); PA = r.integers(1, P, NPERM, dtype=np.int64); PB = r.integers(0, P, NPERM, dtype=np.int64)
WORD = re.compile(r"[A-Za-z0-9']+"); STOP = set("the of and to a in that is was he for it with as his on be at by i this had not are but from or have an they which one you were all her she there would their we him been has when who will no more if out so up said what its about than into them can only other time new some could these two may first then do any like my now over such our man me even most made after also did many off before must well back through years much where your way down should because long".split())
def init(): global TOK; TOK = Tokenizer.from_file(os.path.join(A.dir, "tok", "tokenizer.json")); TOK.no_truncation(); TOK.no_padding()
def feats(text):
    w = [x.lower() for x in WORD.findall(text)][:4000]; n = len(w)
    alpha = sum(c.isalpha() for c in text[:20000]) / max(1, len(text[:20000])); stop = sum(x in STOP for x in w) / max(1, n)
    if n >= 5:
        g = [zlib.crc32(" ".join(w[i:i + 5]).encode()) for i in range(n - 4)]; u, c = np.unique(g, return_counts=True); rep = float(c[c > 1].sum() / len(g)); h = (np.outer(PA, u) + PB[:, None]) % P; sig = h.min(1).astype(np.int32)
    else: rep = 0.0; sig = np.full(NPERM, zlib.crc32(text.encode()) % P, dtype=np.int32)
    return alpha, stop, rep, n, sig
def work(text):
    ids = TOK.encode(text[:6000], add_special_tokens=False).ids[:MAXLEN]; a, s, rp, n, sig = feats(text); return np.array(ids, np.uint16), a, s, rp, n, sig
def reader():
    k = 0
    for fn in sorted(glob.glob(f"{A.dir}/c4-*.json.gz")):
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            for l in f:
                yield json.loads(l)["text"]; k += 1
                if k >= A.max_docs: return
def flat(seq, out):
    T, O, FA, FS, FR, FN, SG = [], [0], [], [], [], [], []
    for i, (ids, a, s, rp, n, sig) in enumerate(seq):
        T.append(ids); O.append(O[-1] + len(ids)); FA.append(a); FS.append(s); FR.append(rp); FN.append(n); SG.append(sig)
        if i % 100000 == 0: print(out, i, flush=True)
    np.savez(out, tokens=np.concatenate(T), offsets=np.array(O, np.int64), alpha=np.array(FA, np.float32), stop=np.array(FS, np.float32), rep=np.array(FR, np.float32), words=np.array(FN, np.int32), sig=np.vstack(SG))
    print(out, "docs", len(T), "tokens", O[-1], flush=True)
if __name__ == "__main__":
    with Pool(A.procs, initializer=init) as pool:
        flat(pool.imap(work, reader(), chunksize=256), f"{A.out}/c4.npz")
        tb = pq.read_table(f"{A.dir}/wiki-00001.parquet", columns=["text"]); wiki = [t for t in tb.column("text").to_pylist() if len(t) > 500][:A.wiki_docs]
        flat(pool.imap(work, wiki, chunksize=64), f"{A.out}/wiki.npz")
    # near-duplicate groups from the MinHash signatures, 8 bands of 4 rows, confirmed at signature agreement of 0.75
    z = np.load(f"{A.out}/c4.npz"); S = z["sig"]; n = len(S); par = np.arange(n); json.dump(dict(docs=int(n), tokens=int(z["offsets"][-1])), open(f"{A.out}/corpus_stats.json", "w"))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for b in range(8):
        seen = {}; key = S[:, b * 4:(b + 1) * 4]
        for i in range(n):
            k = key[i].tobytes(); j = seen.setdefault(k, i)
            if j != i and (S[i] == S[j]).mean() >= .75:
                a, c = find(i), find(j)
                if a != c: par[max(a, c)] = min(a, c)
        print("band", b, flush=True)
    root = np.array([find(i) for i in range(n)]); np.save(f"{A.out}/c4_group.npy", root.astype(np.int32)); print("docs", n, "groups", len(np.unique(root)), "share in a group of two or more", float(1 - len(np.unique(root)) / n), flush=True)
