#!/usr/bin/env python3
"""Job lists of the language model baselines, the workers that hold the model on a card and the corpus training jobs of the two judge arms at 30M tokens."""
import json, os, sys
out = sys.argv[1]; docs = 300032; os.makedirs(out, exist_ok=True)     # at least 300000 documents, in whole units of 2048
open(f"{out}/workers.jsonl", "w").write("".join(json.dumps(dict(id=f"llm_w{k}", judge_docs=docs, selftest=True, wide_first=k == 0)) + "\n" for k in range(3)))
open(f"{out}/corpus_llm.jsonl", "w").write("".join(json.dumps(dict(id=f"c_s{seed}_{size}_{arm}_30M", seed=seed, size=size, arm=arm, budget=30_000_000)) + "\n" for seed in range(3) for size in ("M", "S", "L") for arm in ("llm_high", "llm_mid")))
