#!/usr/bin/env python3
"""Downloads whole files or byte ranges of published archives with HTTP range requests, several connections per file, resumable. A failed chunk is retried and never truncates what is already on disk. Usage: fetch_ranges.py URL OUT [START END] [--threads N]. A sidecar file OUT.parts lists the finished chunks, it is deleted when the file is complete."""
import argparse, concurrent.futures as cf, os, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("out"); ap.add_argument("start", nargs="?", type=int, default=0); ap.add_argument("end", nargs="?", type=int); ap.add_argument("--threads", type=int, default=6); ap.add_argument("--chunk", type=int, default=16 * 2 ** 20); A = ap.parse_args()
P = os.environ.get("https_proxy", ""); op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": P, "https": P}) if P else urllib.request.BaseHandler()); HD = {"User-Agent": "curl/8.5"}
url = [A.url]
def resolve():
    """size of the file and the URL behind the redirect, which saves requests against the rate limit of the hub"""
    with op.open(urllib.request.Request(A.url, headers=dict(HD, Range="bytes=0-0")), timeout=120) as r: url[0] = r.geturl(); return int(r.headers["Content-Range"].split("/")[1])
n = resolve(); end = n if A.end is None else min(A.end, n); side = A.out + ".parts"; lock = threading.Lock(); done = set(int(x) for x in open(side).read().split()) if os.path.exists(side) else set()
if os.path.exists(A.out) and os.path.getsize(A.out) == end - A.start and not os.path.exists(side): print(A.out, "already complete", flush=True); raise SystemExit
open(side, "a").close()
if not os.path.exists(A.out): open(A.out, "wb").close()
if os.path.getsize(A.out) < end - A.start: os.truncate(A.out, end - A.start)
starts = [s for s in range(A.start, end, A.chunk) if s not in done]; print(A.out, "size", n, "range", A.start, end, "chunks left", len(starts), flush=True); fd = os.open(A.out, os.O_RDWR); t0 = time.time(); got = [0]
def one(s):
    e = min(s + A.chunk, end) - 1
    for k in range(200):
        try:
            with op.open(urllib.request.Request(url[0], headers=dict(HD, Range=f"bytes={s}-{e}")), timeout=60) as r:
                b = r.read()
            if len(b) != e - s + 1: raise IOError(f"short read {len(b)}")
            os.pwrite(fd, b, s - A.start)
            with lock:
                done.add(s); open(side, "a").write(f"{s}\n"); got[0] += len(b)
                if len(done) % 16 == 0: print(A.out, len(done), "chunks", round(got[0] / 2 ** 20), "MiB", round(got[0] / 2 ** 20 / (time.time() - t0), 2), "MiB/s", flush=True)
            return
        except Exception as x:
            print("retry", s, repr(x)[:120], flush=True); time.sleep(min(30, 2 + 2 * k))
            if k % 3 == 2:
                try: resolve()
                except Exception: pass
    raise SystemExit(f"gave up on {s}")
with cf.ThreadPoolExecutor(A.threads) as ex: list(ex.map(one, starts))
os.close(fd); os.remove(side) if os.path.exists(side) else None; print(A.out, "complete", flush=True)
