#!/usr/bin/env python3
"""Job list of the web corpus track, in the order of the seeds."""
import json, sys
out = []
for seed in range(5):
    for size in ("S", "M", "L"):
        out.append(dict(id=f"c_s{seed}_{size}_all", seed=seed, size=size, arm="all", budget=None))
        for budget in (30_000_000, 100_000_000):
            for arm in ("random", "ppl_low", "ppl_mid", "ppl_high", "rule", "dedup", "rule_dedup"):
                out.append(dict(id=f"c_s{seed}_{size}_{arm}_{budget // 1000000}M", seed=seed, size=size, arm=arm, budget=budget))
ARMS = ("random", "ppl_low", "ppl_mid", "ppl_high", "rule", "dedup", "rule_dedup")
for seed in range(3):
    for size in ("XS", "T", "SM", "MS", "ML"):          # more model sizes at the base budget
        for arm in ARMS: out.append(dict(id=f"c_s{seed}_{size}_{arm}_30M", seed=seed, size=size, arm=arm, budget=30_000_000))
    for budget in (3, 5, 10, 15, 20):                   # more token budgets at the middle model size
        for arm in ARMS: out.append(dict(id=f"c_s{seed}_M_{arm}_{budget}M", seed=seed, size="M", arm=arm, budget=budget * 1_000_000))
with open(sys.argv[1], "w") as f:
    for j in out: f.write(json.dumps(j) + "\n")
print(len(out))
