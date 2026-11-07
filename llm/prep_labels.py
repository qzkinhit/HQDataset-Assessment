#!/usr/bin/env python3
"""Texts of the test sets of Northcutt et al. (2021) for the language model as annotator, IMDB, 20news and a random sample of 50000 Amazon reviews plus the 1000 reviewed ones. Each set is aligned with its label array and the alignment is verified, on the labels of every record and on the text shown to the reviewers for a sample of flagged records, a set that fails is skipped.
The corrected label follows native/label_errors.py, a flagged label is wrong when at least three of the five reviewers chose the suggested label and fewer than three kept the given one, it is rejected when fewer than three kept it. Examples for the prompts come from the public training parts, for Amazon from reviews outside the sample."""
import argparse, ast, io, json, os, subprocess, tarfile, urllib.request
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--native"); ap.add_argument("--dl"); ap.add_argument("--out"); ap.add_argument("--check", type=int, default=40); ap.add_argument("--only", default="imdb,20news,amazon"); A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); AMZ = dict(Negative=0, Neutral=1, Positive=2); HDR = {"User-Agent": "curl/8.5"}; PX = os.environ.get("https_proxy", ""); OP = urllib.request.build_opener(urllib.request.ProxyHandler({"http": PX, "https": PX}) if PX else urllib.request.BaseHandler())
def votes(r): return ast.literal_eval(r["mturk"]) if isinstance(r["mturk"], str) else r["mturk"]
def review(ds, pos, code):
    """Arrays over all records: reviewed, wrong, rejected and the corrected label."""
    y = np.load(f"{A.native}/{ds}_orig.npy", mmap_mode="r"); m = json.load(open(f"{A.native}/{ds}_mturk.json")); c = {}; rev, wrong, rej = set(), set(), set()
    for r in m:
        v = votes(r); i = pos(r); rev.add(i)
        if "guessed" in v: fix = r["our_guessed_label"] if v["guessed"] >= 3 and v["given"] < 3 else None; kept = v["given"]
        else: t = {k: v.get(k, 0) for k in ("positive", "negative", "neutral")}; top = max(t, key=t.get); g = r["given_original_label"].lower(); kept = t[g]; fix = top.capitalize() if top != g and t[top] >= 3 else None
        if kept < 3: rej.add(i)
        if fix is not None: wrong.add(i); c[i] = code(fix)
    return y, m, rev, wrong, rej, c
def same_text(a, b):
    n = lambda s: "".join(s.split()); return n(a)[:2000] == n(b)[:2000]
def fetch(url):
    for _ in range(3):
        try: return OP.open(urllib.request.Request(url, headers=HDR), timeout=30).read().decode("utf-8", "replace")
        except Exception as e: err = e
    print("fetch failed", url, err); return None
def text_check(m, pos, text_of):
    k = np.random.default_rng(0).permutation(len(m))[:A.check]; ok = n = 0
    for j in k:
        t = fetch(m[j]["url"])
        if t is not None: n += 1; ok += same_text(t, text_of(pos(m[j])))
    return dict(text_checked=n, text_match=ok)
def write(ds, items, y, rev, wrong, rej, c, info, question, instruction, options, codes, shots):
    with open(f"{A.out}/{ds}.jsonl", "w") as f:
        for i, t in items: f.write(json.dumps(dict(i=int(i), y=int(y[i]), c=int(c.get(i, y[i])), rev=int(i in rev), wrong=int(i in wrong), rej=int(i in rej), text=t), ensure_ascii=False) + "\n")
    json.dump(dict(ds=ds, n=len(items), question=question, instruction=instruction, options=options, codes=codes, shots=shots, align=info), open(f"{A.out}/{ds}.meta.json", "w"), ensure_ascii=False, indent=0); print(ds, len(items), info, flush=True)
Q_S = "Is the sentiment of this {} {}?"; only = A.only.split(",")
if "imdb" in only:
    names = json.load(open(f"{A.native}/imdb_index.json")); idx = {x.replace(".txt", ""): i for i, x in enumerate(names)}; pos = lambda r: idx[r["id"].replace("test/", "", 1)]; y, m, rev, wrong, rej, c = review("imdb", pos, dict(Negative=0, Positive=1).__getitem__); tx = {}; shots_tr = {"neg": [], "pos": []}
    with tarfile.open(f"{A.dl}/imdb.tgz") as T:
        for e in T:
            p = (e.name[2:] if e.name.startswith("./") else e.name).split("/")
            if len(p) == 4 and p[1] in ("test", "train") and p[2] in ("neg", "pos") and p[3].endswith(".txt"):
                b = T.extractfile(e).read().decode("utf-8")
                if p[1] == "test": tx[f"{p[2]}/{p[3]}"] = b
                else: shots_tr[p[2]].append(b)
    lab = np.array([0 if x.startswith("neg/") else 1 for x in names]); agree = float((lab == np.asarray(y)).mean()); info = dict(label_agreement=agree, **text_check(m, pos, lambda i: tx[names[i]])); r = np.random.default_rng(0); sh = [dict(text=shots_tr[g][int(r.integers(len(shots_tr[g])))], answer=a) for g, a in (("neg", "Negative"), ("pos", "Positive"), ("neg", "Negative"), ("pos", "Positive"), ("neg", "Negative"))]
    if agree == 1.0 and info["text_match"] == info["text_checked"] > 0: write("imdb", [(i, tx[x]) for i, x in enumerate(names)], y, rev, wrong, rej, c, info, "Is the sentiment of this movie review positive or negative?", "Answer with one word, Positive or Negative.", ["Positive", "Negative"], [1, 0], sh)
    else: print("imdb skipped", info, flush=True)
if "20news" in only:
    import shutil, sklearn.datasets._twenty_newsgroups as N     # the archive is read from disk, sklearn checks nothing else
    from sklearn.datasets import fetch_20newsgroups
    N._fetch_remote = lambda remote, dirname=None: shutil.copy(f"{A.dl}/20news-bydate.tar.gz", f"{dirname}/20news-bydate.tar.gz")
    home = f"{A.dl}/skdata"; os.makedirs(home, exist_ok=True); te = fetch_20newsgroups(subset="test", data_home=home); trn = fetch_20newsgroups(subset="train", data_home=home); pos = lambda r: int(r["id"]); y, m, rev, wrong, rej, c = review("20news", pos, int); agree = float((te.target == np.asarray(y)).mean()); info = dict(label_agreement=agree, **text_check(m, pos, lambda i: te.data[i]))
    r = np.random.default_rng(0); cl = r.permutation(20)[:5]; letters = [chr(65 + i) for i in range(20)]; sh = [dict(text=trn.data[int(r.choice(np.where(trn.target == k)[0]))], answer=letters[k]) for k in cl]; q = "In which newsgroup was this post published?\n" + "\n".join(f"{a}. {b}" for a, b in zip(letters, te.target_names))
    if agree == 1.0 and info["text_match"] == info["text_checked"] > 0: write("20news", list(enumerate(te.data)), y, rev, wrong, rej, c, info, q, "Answer with the letter of the newsgroup.", letters, list(range(20)), sh)
    else: print("20news skipped", info, flush=True)
if "amazon" in only:
    pos = lambda r: int(r["id"]); y, m, rev, wrong, rej, c = review("amazon", pos, AMZ.__getitem__); yy = np.asarray(y); n = len(yy); r = np.random.default_rng(0); sample = set(int(i) for i in r.choice(n, 50000, replace=False)); keep = sample | rev; star = {"__label__1": 0, "__label__3": 1, "__label__5": 2}; tx = {}; agree = 0; k = 0; cand = [int(i) for i in r.choice(n, 2000, replace=False) if int(i) not in keep]; pool = {}; cs = set(cand)
    p = subprocess.Popen(f"cat {A.dl}/amazon5core.tar.gz-parta* | gzip -dc | tar -xO", shell=True, stdout=subprocess.PIPE)
    for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline="\n"):
        lab, _, t = raw.rstrip("\n").partition(" ")
        if k < n: agree += star.get(lab, -1) == yy[k]
        if k in keep: tx[k] = t
        elif k in cs: pool[k] = (star.get(lab), t)
        k += 1
    p.wait(); info = dict(lines=k, records=n, label_agreement=agree / n, **text_check(m, pos, lambda i: tx[i])); by = {g: [t for s, t in pool.values() if s == g] for g in (0, 1, 2)}; sh = [dict(text=by[g][j], answer=a) for j, (g, a) in enumerate(((0, "Negative"), (2, "Positive"), (1, "Neutral"), (2, "Positive"), (0, "Negative")))]; Q = "Is the sentiment of this product review positive, neutral or negative?"; I = "Answer with one word, Positive, Neutral or Negative."
    if k == n and agree == n and info["text_match"] == info["text_checked"] > 0:
        write("amazon", [(i, tx[i]) for i in sorted(sample)], y, rev, wrong, rej, c, info, Q, I, ["Positive", "Neutral", "Negative"], [2, 1, 0], sh); write("amazon_reviewed", [(i, tx[i]) for i in sorted(rev)], y, rev, wrong, rej, c, info, Q, I, ["Positive", "Neutral", "Negative"], [2, 1, 0], sh)
    else: print("amazon skipped", info, flush=True)
