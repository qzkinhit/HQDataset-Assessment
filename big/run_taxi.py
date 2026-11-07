#!/usr/bin/env python3
"""New York City yellow taxi records (TLC, public): 72 months, 2019 to 2024, real errors and real change over time, nothing injected.
Static side: the share of records breaking simple validity rules from the published data dictionary, month by month, on a full scan.
Task side: trip duration from what is known at pickup, a model trained on one month and scored on the clean records of later months."""
import argparse, glob, hashlib, json, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(REPO, "downloads", "tlc")); ap.add_argument("--out", default=os.path.join(REPO, "results", "results_taxi.json")); ap.add_argument("--threads", type=int, default=24); ap.add_argument("--sizes", default="1000000"); ap.add_argument("--stride", type=int, default=6); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--jobs", help="Optional JSONL of seed, arm, requested_size, train_month"); A = ap.parse_args()
EVAL_CAP = 100000
EVAL_PROTOCOL = "tlc_clean_month_100k_seed_v1"
COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "RatecodeID", "PULocationID", "DOLocationID", "fare_amount", "VendorID"]
def month(fn):
    t = pq.read_table(fn, columns=COLS).to_pandas(); y, m = map(int, os.path.basename(fn)[16:23].split("-")); t = t.rename(columns={"tpep_pickup_datetime": "pu", "tpep_dropoff_datetime": "do"}); t["pu"] = pd.to_datetime(t["pu"]); t["do"] = pd.to_datetime(t["do"])
    dur = (t["do"] - t["pu"]).dt.total_seconds(); speed = t["trip_distance"] / (dur.clip(lower=1) / 3600)
    R = dict(r_time=(t["pu"].dt.year == y) & (t["pu"].dt.month == m), r_duration=(dur > 0) & (dur <= 6 * 3600), r_distance=(t["trip_distance"] > 0) & (t["trip_distance"] <= 200), r_fare=(t["fare_amount"] > 0) & (t["fare_amount"] <= 500), r_passengers=t["passenger_count"].between(1, 8), r_ratecode=t["RatecodeID"].isin([1, 2, 3, 4, 5, 6]), r_speed=speed <= 80, r_location=~t["PULocationID"].isin([264, 265]) & ~t["DOLocationID"].isin([264, 265]))
    ok = np.ones(len(t), bool)
    for v in R.values(): ok &= np.asarray(v.fillna(False))
    stat = {k: float(1 - np.asarray(v.fillna(False)).mean()) for k, v in R.items()}; stat.update(records=int(len(t)), any_rule=float(1 - ok.mean()), missing_passengers=float(t["passenger_count"].isna().mean()))
    t["y"] = np.log1p(dur.clip(lower=1)); t["hour"] = t["pu"].dt.hour.astype(np.int16); t["dow"] = t["pu"].dt.dayofweek.astype(np.int16); t["ok"] = ok
    F = t[["trip_distance", "passenger_count", "RatecodeID", "PULocationID", "DOLocationID", "hour", "dow", "VendorID", "y", "ok"]].copy()
    for c in ("trip_distance", "passenger_count", "RatecodeID", "y"): F[c] = F[c].astype(np.float32)
    return F, stat
FEATS = ["trip_distance", "passenger_count", "RatecodeID", "PULocationID", "DOLocationID", "hour", "dow", "VendorID"]; CAT = ["PULocationID", "DOLocationID", "RatecodeID"]
files = sorted(glob.glob(f"{A.dir}/yellow_tripdata_*.parquet")); print(len(files), "months", flush=True); M = {}; res = dict(months={}, runs=[]); t0 = time.time()
previous = json.load(open(A.out)) if os.path.exists(A.out) else {}
old = previous.get("runs", [])
if old and (previous.get("eval_protocol") != EVAL_PROTOCOL or any(r.get("eval_protocol") != EVAL_PROTOCOL for r in old)):
    raise ValueError("The output file holds runs of another evaluation protocol, choose a new output path")
res["eval_protocol"] = EVAL_PROTOCOL; res["eval_max_records"] = EVAL_CAP
res["runs"] = old; finished = {(r["seed"], r["arm"], r["size"], r["train_month"]) for r in old}
for fn in files:
    key = os.path.basename(fn)[16:23]; F, st = month(fn); M[key] = F; res["months"][key] = st
    print(key, st["records"], round(st["any_rule"], 4), round(time.time() - t0), flush=True)
res["records_total"] = int(sum(v["records"] for v in res["months"].values())); json.dump(res, open(A.out, "w"), indent=0)
keys = sorted(M); train_months = keys[0::A.stride]
selected = None
if A.jobs:
    requested = [json.loads(l) for l in open(A.jobs) if l.strip()]
    selected = {(j["seed"], j["arm"], j["requested_size"], j["train_month"]) for j in requested}
    valid = {(s, a, n, tm) for s in range(A.seeds) for a in ("raw", "rule_filtered") for n in map(int, A.sizes.split(",")) for tm in train_months}
    if len(selected) != len(requested) or not selected <= valid: raise ValueError("Job keys must be unique and inside the grid")
    print("selected jobs", len(selected), "evaluation", EVAL_PROTOCOL, flush=True)
for seed in range(A.seeds):
    if selected is not None and not any(j[0] == seed for j in selected): continue
    rng = np.random.default_rng(seed)
    te = {k: M[k][M[k]["ok"]].sample(n=min(EVAL_CAP, int(M[k]["ok"].sum())), random_state=seed) for k in keys}
    eval_hashes = {k: hashlib.sha256(np.asarray(v.index, dtype="<i8").tobytes()).hexdigest() for k, v in te.items()}
    for arm in ("raw", "rule_filtered"):
        for size in map(int, A.sizes.split(",")):
            for tm in train_months:
                if selected is not None and (seed, arm, size, tm) not in selected: continue
                F = M[tm]; pool = F if arm == "raw" else F[F["ok"]]
                if (seed, arm, min(size, len(pool)), tm) in finished: continue
                t1 = time.time(); tr = pool.sample(n=min(size, len(pool)), random_state=seed)
                m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=127, subsample=0.8, subsample_freq=1, n_jobs=A.threads, random_state=seed, verbose=-1).fit(tr[FEATS], tr["y"], categorical_feature=CAT)
                sc = {k: float(r2_score(te[k]["y"], m.predict(te[k][FEATS]))) for k in keys if k >= tm}
                res["runs"].append(dict(seed=seed, arm=arm, size=int(len(tr)), requested_size=size, train_month=tm, r2=sc, sec=round(time.time() - t1), eval_protocol=EVAL_PROTOCOL, eval_max_records=EVAL_CAP, eval_counts={k: len(te[k]) for k in sc}, eval_index_sha256={k: eval_hashes[k] for k in sc})); print(seed, arm, tm, len(tr), round(sc.get(tm, 0), 4), round(list(sc.values())[-1], 4), round(time.time() - t1), flush=True)
                json.dump(res, open(A.out, "w"), indent=0)
print("done", flush=True)
