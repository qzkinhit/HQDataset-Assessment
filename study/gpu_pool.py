#!/usr/bin/env python3
"""Run a list of jobs over several GPUs, a fixed number per card. A job starts on a card only while the memory of the processes outside this pool stays under --foreign-mib.
Finished job ids are read back from the result file."""
import argparse, json, os, subprocess, time
ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--script", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpus", default="0"); ap.add_argument("--per-gpu", type=int, default=6)
ap.add_argument("--foreign-mib", type=int, default=46000); ap.add_argument("--python", default="python"); ap.add_argument("--logdir", default="pool_logs"); A = ap.parse_args()
os.makedirs(A.logdir, exist_ok=True)
jobs = [json.loads(l) for l in open(A.jobs)]
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
# jobs that are already running are not started twice
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
active = {j["id"] for j in jobs if f'"id": "{j["id"]}"' in ps}
pending = [j for j in jobs if j["id"] not in done and j["id"] not in active]; print("jobs", len(jobs), "pending", len(pending), "active", len(active), flush=True)
gpus = [int(x) for x in A.gpus.split(",")]; run = {g: [] for g in gpus}
def foreign(g, mine):
    q = subprocess.run(["nvidia-smi", "-i", str(g), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(x.split(",")[1]) for x in q.strip().splitlines() if x.strip() and x.split(",")[1].strip().isdigit() and int(x.split(",")[0]) not in mine)
i, fails = 0, 0
while i < len(pending) or any(run.values()):
    for g in gpus:
        run[g] = [p for p in run[g] if p.poll() is None]
        mine = {p.pid for p in run[g]}
        while i < len(pending) and len(run[g]) < A.per_gpu and foreign(g, mine) < A.foreign_mib:
            j = pending[i]; i += 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log = open(f"{A.logdir}/{j['id']}.log", "w")
            p = subprocess.Popen([A.python, A.script, "--job", json.dumps(j), "--out", A.out], env=env, stdout=log, stderr=subprocess.STDOUT); run[g].append(p); mine.add(p.pid)
            print(time.strftime("%H:%M:%S"), "start", i, len(pending), j["id"], "gpu", g, flush=True)
    time.sleep(10)
print("finished", flush=True)
