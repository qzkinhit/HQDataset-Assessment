#!/usr/bin/env python3
"""Worker of the language model baselines. One process holds the model on one card and takes work units of the three tracks from a shared list until none is left, so that a card is claimed once, and the memory is released when the process ends, before any training starts.
A unit is claimed atomically, its result is written atomically, a unit of a crashed worker goes back to the list. The order is the small tables, the label sets, then the corpus and the widest table, one worker starts on the widest table and the others on the corpus. The first step is a test on a very small sample that stops the worker when a reply cannot be used."""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--job"); ap.add_argument("--out"); A = ap.parse_args(); J = json.loads(A.job)
import numpy as np, torch
import annotate_labels as L, judge_corpus as C, qwen, repair_tables as T
ROOT = os.environ.get("HQD_LLM", "llm"); N_JUDGE = int(J.get("judge_docs", 300000)); model, tok = qwen.load(); t0 = time.time(); print("loaded", time.strftime("%H:%M:%S"), torch.cuda.max_memory_reserved() // 2 ** 20 if torch.cuda.is_available() else "cpu", flush=True); src = C.source_tok()
def save_jsonl(p, rows): open(p + ".part", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); os.replace(p + ".part", p)
def save_npz(path, **a):
    with open(path + ".part", "wb") as f: np.savez(f, **a)
    os.replace(path + ".part", path)
def selftest(n=int(J.get("test_items", 4))):
    t = time.time()
    for ds in J.get("test_tables", ("flights", "hospitals", "mercedes")):
        if not os.path.exists(f"{ROOT}/prompts/{ds}.json"): continue
        P = T.plan(ROOT, ds); r = T.run(model, tok, ROOT, ds, 0, n); print("test table", ds, "reply", json.dumps(r[0]["raw"])[:400], "fix", json.dumps(r[0]["fix"])[:200], "not parsed", sum(x["bad"] for x in r), flush=True)
        if sum(x["bad"] for x in r) > n * len(P["blocks"]) / 2: raise SystemExit(f"replies of {ds} do not parse")
    for ds in L.SETS:
        if os.path.exists(f"{ROOT}/labels/{ds}.jsonl"):
            for shots in (0, 5):
                i, p, k = L.run(model, tok, ROOT, ds, shots, 0, n); print("test labels", ds, shots, "records", i.tolist(), "pick", [L.load(ROOT, ds)[0]["options"][int(x)] for x in p.argmax(1)], "truth", [r["y"] for r in L.load(ROOT, ds)[1][:n]], "mass", np.round(p.sum(1), 3).tolist(), "tokens", k.tolist(), flush=True)
                if not np.isfinite(p).all() or p.sum(1).mean() < .3: raise SystemExit(f"labels of {ds} fail")
    if os.path.exists(f"{C.ROOT}/c4.npz"):
        d, s, m, k = C.run(model, tok, src, 0, n * 2, n * 2); print("test judge docs", d.tolist(), "rating", np.round(s, 2).tolist(), "mass", np.round(m, 3).tolist(), "tokens", k.tolist(), flush=True)
        if not np.isfinite(s).all() or m.mean() < .3: raise SystemExit("judge fails")
    print("test finished", round(time.time() - t), torch.cuda.max_memory_reserved() // 2 ** 20 if torch.cuda.is_available() else "cpu", flush=True)
def one_pass():
    done = 0
    def run_tables(tables):
        nonlocal done
        for ds, b in T.units(ROOT, tables):
            p = f"{ROOT}/units/tables/{ds}.{b}.jsonl"
            if qwen.claim(p):
                t = time.time(); save_jsonl(p, T.run(model, tok, ROOT, ds, b)); qwen.release(p); done += 1; print("table", ds, b, round(time.time() - t, 1), flush=True)
    run_tables(T.ORDER)
    for ds, shots, b in L.units(ROOT):
        p = f"{ROOT}/units/labels/{ds}.{shots}.{b}.npz"
        if qwen.claim(p):
            t = time.time(); i, pr, k = L.run(model, tok, ROOT, ds, shots, b); save_npz(p, i=i, p=pr.astype(np.float32), tokens=k); qwen.release(p); done += 1; print("labels", ds, shots, b, round(time.time() - t, 1), int(k.sum()), flush=True)
    L.merge(ROOT, f"{ROOT}/results_llm_labels.jsonl")
    if J.get("wide_first"): run_tables(T.LAST)
    for start, stop in C.units(N_JUDGE):
        p = f"{ROOT}/units/judge/{start}.npz"
        if qwen.claim(p):
            t = time.time(); d, s, m, k = C.run(model, tok, src, start, stop); save_npz(p, doc=d, score=s.astype(np.float32), mass=m.astype(np.float32), tokens=k); qwen.release(p); done += 1; print("judge", start, round(time.time() - t, 1), int(k.sum()), flush=True)
    C.merge(ROOT, N_JUDGE); run_tables(T.LAST); return done
def work():
    """Passes over the list until a pass finds nothing to do, a later pass takes the units of a worker that has died."""
    total = 0
    while True:
        n = one_pass(); total += n
        if n == 0: break
    L.merge(ROOT, f"{ROOT}/results_llm_labels.jsonl"); C.merge(ROOT, N_JUDGE); return total
if J.get("selftest", True): selftest()
done = 0 if J.get("test_only") else work()
with open(A.out, "a") as f: f.write(json.dumps(dict(J, units=done, sec=round(time.time() - t0), peak_mib=torch.cuda.max_memory_reserved() // 2 ** 20 if torch.cuda.is_available() else None)) + "\n")
