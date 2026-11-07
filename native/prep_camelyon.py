#!/usr/bin/env python3
"""A slim subset of WILDS Camelyon17 (Koh et al. 2021) from the published parquet files, as one uint8 array. Row groups are drawn at random over all files of the split and only they are fetched, with HTTP range requests: images from the training hospitals, split at random into a training part and an in-distribution hold-out, and images from the official test hospital. Row groups already fetched are kept on disk, so the script resumes."""
import argparse, concurrent.futures as cf, io, json, os, time, urllib.request
import numpy as np, pyarrow.parquet as pq
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--native", default="."); ap.add_argument("--train", type=int, default=80000); ap.add_argument("--hold", type=int, default=10000); ap.add_argument("--test", type=int, default=10000); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--probe", action="store_true"); A = ap.parse_args()
P = os.environ.get("https_proxy", ""); op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": P, "https": P})); B = "https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/main/"; TMP = f"{A.native}/wilds/cam_parts"; os.makedirs(TMP, exist_ok=True)
class Remote:
    """a read-only file over HTTP range requests, enough for pyarrow"""
    def __init__(s, url): s.url, s.pos, s.closed = url, 0, False; s.size_ = s._size()
    def _size(s):
        with op.open(urllib.request.Request(s.url, headers={"User-Agent": "curl/8", "Range": "bytes=0-0"}), timeout=120) as r: s.cdn = r.geturl(); return int(r.headers["Content-Range"].split("/")[1])     # the URL behind the redirect, which saves requests against the rate limit of the hub
    def read(s, n=-1):
        if n < 0: n = s.size_ - s.pos
        n = min(n, s.size_ - s.pos)
        if n == 0: return b""
        for k in range(100):
            try:
                with op.open(urllib.request.Request(s.cdn, headers={"User-Agent": "curl/8", "Range": f"bytes={s.pos}-{s.pos + n - 1}"}), timeout=60) as r: b = r.read()
                if len(b) == n: s.pos += n; return b
            except Exception as x:
                if k % 3 == 2: print("retry", repr(x)[:100], flush=True); s._size()
            time.sleep(min(30, 1 + k))
        raise IOError("gave up")
    def seek(s, p, w=0): s.pos = p if w == 0 else (s.pos + p if w == 1 else s.size_ + p); return s.pos
    def tell(s): return s.pos
    def seekable(s): return True
    def readable(s): return True
    def writable(s): return False
    def close(s): s.closed = True
    def size(s): return s.size_
def listing():
    with op.open(urllib.request.Request("https://huggingface.co/api/datasets/wltjr1007/Camelyon17-WILDS/tree/main/data?limit=100", headers={"User-Agent": "curl/8"}), timeout=120) as r: return [x["path"] for x in json.load(r)]
files = listing(); meta = {}; gc = lambda m, name: [m.schema.to_arrow_schema().names.index(name)][0]
for f in files:
    k = f"{TMP}/{os.path.basename(f)}.footer.json"; fh = Remote(B + f); pf = pq.ParquetFile(fh); m = pf.metadata; col = next(i for i in range(m.num_columns) if m.schema.column(i).path == "center"); rows = [(i, m.row_group(i).num_rows, m.row_group(i).total_byte_size, m.row_group(i).column(col).statistics.min if m.row_group(i).column(col).statistics else None, m.row_group(i).column(col).statistics.max if m.row_group(i).column(col).statistics else None) for i in range(m.num_row_groups)]; meta[f] = (pf, fh, rows); print(f, "rows", m.num_rows, "groups", m.num_row_groups, "size of a group", rows[0][1], rows[0][2], "center range", sorted({(r[3], r[4]) for r in rows})[:8], flush=True)
if A.probe: raise SystemExit
def fetch(item):
    f, i = item; out = f"{TMP}/{os.path.basename(f)}.{i}.npz"
    if os.path.exists(out): return out
    fh = Remote.__new__(Remote); fh.url, fh.pos, fh.closed, fh.size_, fh.cdn = meta[f][1].url, 0, False, meta[f][1].size_, meta[f][1].cdn; pf = pq.ParquetFile(fh, metadata=meta[f][0].metadata); t = pf.read_row_group(i, columns=["image", "label", "center", "slide", "patient"]); ims = [np.array(Image.open(io.BytesIO(x["bytes"])).convert("RGB"), np.uint8) for x in t.column("image").to_pylist()]; np.savez(out + ".tmp.npz", x=np.stack(ims), y=np.array(t.column("label").to_pylist(), np.int8), center=np.array(t.column("center").to_pylist(), np.int8), slide=np.array(t.column("slide").to_pylist(), np.int16), patient=np.array(t.column("patient").to_pylist(), np.int16)); os.replace(out + ".tmp.npz", out); return out
rng = np.random.default_rng(17)
def draw(prefix, target):
    g = [(f, r[0], r[1]) for f in files if os.path.basename(f).startswith(prefix) for r in meta[f][2]]; o = rng.permutation(len(g)); s, n = [], 0
    for j in o:
        s.append(g[j][:2]); n += g[j][2]
        if n >= target: break
    return s
tr = draw("train", int(1.03 * (A.train + A.hold))); te = draw("test", int(1.03 * A.test)); print("row groups", len(tr), len(te), flush=True); done = 0; t0 = time.time()
with cf.ThreadPoolExecutor(A.threads) as ex:
    for _ in ex.map(fetch, tr + te):
        done += 1
        if done % 50 == 0: print(done, len(tr) + len(te), round(time.time() - t0), flush=True)
def stack(items):
    z = [np.load(f"{TMP}/{os.path.basename(f)}.{i}.npz") for f, i in items]; return {k: np.concatenate([q[k] for q in z]) for k in ("x", "y", "center", "slide", "patient")}
a, b = stack(tr), stack(te); print("training hospitals", np.unique(a["center"], return_counts=True), "test hospital", np.unique(b["center"], return_counts=True), flush=True); assert set(np.unique(b["center"])) == {2} and 2 not in set(np.unique(a["center"])) and 1 not in set(np.unique(a["center"]))
p = rng.permutation(len(a["y"]))[:A.train + A.hold]; q = rng.permutation(len(b["y"]))[:A.test]; x = np.lib.format.open_memmap(f"{A.native}/camelyon_x.npy", "w+", np.uint8, (len(p) + len(q),) + a["x"].shape[1:]); x[:len(p)] = a["x"][p]; x[len(p):] = b["x"][q]; x.flush()
part = np.r_[np.zeros(A.train, np.int8), np.ones(A.hold, np.int8), np.full(len(q), 2, np.int8)]; np.savez(f"{A.native}/camelyon_meta.npz", y=np.r_[a["y"][p], b["y"][q]], part=part, center=np.r_[a["center"][p], b["center"][q]], slide=np.r_[a["slide"][p], b["slide"][q]], patient=np.r_[a["patient"][p], b["patient"][q]]); print("saved", x.shape, "tumor share", [float(np.r_[a["y"][p], b["y"][q]][part == v].mean()) for v in (0, 1, 2)], flush=True)
