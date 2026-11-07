#!/usr/bin/env python3
"""Tokenizes shards of the two published DCLM corpora with the tokenizer of the corpus track, the first 512 tokens of every document, so that they enter the same training recipe as C4."""
import glob, io, json, os, sys
import numpy as np, zstandard
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(sys.argv[1]); TOK.no_truncation(); TOK.no_padding(); D, OUT = sys.argv[2], sys.argv[3]; LIMIT = 130000000
for name, pat in (("dclm_pool", "dclm_pool/*.zst"), ("dclm_baseline", "dclm_base/*.zst")):
    T, O, B = [], [0], []
    def flush():
        for e in TOK.encode_batch(B, add_special_tokens=False): ids = e.ids[:512]; T.append(np.array(ids, np.uint16)); O.append(O[-1] + len(ids))
        B.clear()
    for fn in sorted(glob.glob(f"{D}/{pat}")):
        with open(fn, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
            for l in io.TextIOWrapper(r, encoding="utf-8"):
                t = json.loads(l).get("text", "")
                if len(t) < 200: continue
                B.append(t[:6000])
                if len(B) == 4000:
                    flush()
                    if O[-1] > LIMIT: break
        if O[-1] > LIMIT: break
    flush(); np.savez(f"{OUT}/{name}.npz", tokens=np.concatenate(T), offsets=np.array(O, np.int64)); print(name, len(O) - 1, O[-1], flush=True)
