#!/usr/bin/env python3
"""Error rates read from detectors on the 11 tables that come with a dirty and a clean version, nothing is injected here.
The detectors read the dirty feature columns only. D1 flags missing tokens and values with a rare character pattern, D2 applies the interquartile rule and the three standard deviation rule to the numeric columns, D3 flags minority values under approximate functional dependencies with one attribute on the left, D4 is an Isolation Forest on records, D5 is the union of D1, the interquartile rule and D3, D6 is Raha with 20 tuples labeled from the clean version.
The truth comes from load in run_real_tables.py. A record is dirty when one of its feature cells differs from the clean version, the unit of the static score. Label errors enter only the extra unit record_label. A cell sample draws n cells, and Isolation Forest has no cell unit.
For each sample size the script repeats four estimates. The method sample is a simple random sample with a Wilson interval. The method corrected applies the correction of Rogan and Gladen with sensitivity and specificity from that same sample and a percentile bootstrap, with the positive rate fixed because a full scan gives it. A sample or a bootstrap draw with an undefined correction uses the sample proportion. The method stratified draws half of n from the flagged items and half from the rest and weights the strata by their population shares, with a Wilson interval on the effective sample size. The method pooled is the proportion of that same sample without weights. The method poststratified is an addition beyond these, it weights the flagged and the unflagged part of the simple random sample by the population shares, so it spends the same labels as corrected. No interval applies a finite population correction, except that a stratum read in full has no sampling variance.
Nothing is injected by this script, yet only beers and flights hold errors that arose in collection. The errors of the other nine tables come from the benchmark authors, so the summary also reports these two tables on their own.
Stages run in the order detect, raha, measure, check. The stage raha needs the raha package and takes one table per process, so that a time limit can stop it from outside."""
import argparse, json, os, random, re, sys, time, zlib, warnings
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import IsolationForest
import run_real_tables as R
warnings.filterwarnings("ignore")
SIZES = (50, 100, 200, 400, 800)
Z = 1.959963984540054
RARE, SUPPORT, KEY, CONST = 0.01, 0.9, 0.9, 0.9
NATIVE = ("beers", "flights")
def seeded(*parts): return np.random.default_rng(zlib.crc32(json.dumps(parts, sort_keys=True).encode()) % (2 ** 32))
def pattern(v):
    v = re.sub("[0-9]+", "9", v); v = re.sub("[A-Z]+", "A", v); return re.sub("[a-z]+", "a", v)
def is_missing(X): return pd.DataFrame(X).isin(list(R.NA)).values
def numbers(col): return pd.to_numeric(pd.Series(col), errors="coerce").values.astype(float)
def d1(X):
    M = is_missing(X); F = M.copy(); F[:, M.all(axis=0)] = False      # a column that holds only the missing token says nothing about its records
    for j in range(X.shape[1]):
        ok = ~M[:, j]
        if not ok.any(): continue
        p = pd.Series(X[ok, j]); u = pd.Series(p.unique()); p = p.map(dict(zip(u, u.map(pattern)))); share = p.map(p.value_counts(normalize=True)).values; F[np.where(ok)[0][share < RARE], j] = True
    return F
def d2(X, num):
    Fi, Fs = np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for j in num:
        x = numbers(X[:, j]); ok = np.isfinite(x)
        if not ok.any(): continue
        v = x[ok]; q1, q3 = np.percentile(v, [25, 75]); w = q3 - q1; Fi[ok, j] = (v < q1 - 1.5 * w) | (v > q3 + 1.5 * w); s = v.std(); Fs[ok, j] = np.abs(v - v.mean()) > 3 * s
    return Fi, Fs
def d3(X, names):
    F = np.zeros(X.shape, bool); C, lhs, rhs, fds = [], [], [], []; miss = is_missing(X)
    for j in range(X.shape[1]):
        c = pd.factorize(X[:, j])[0]; c[miss[:, j]] = -1; C.append(c)
        if miss[:, j].all(): continue
        cnt = np.bincount(c[c >= 0]); cnt = cnt[cnt > 0]
        if cnt.max() / cnt.sum() >= CONST: continue         # such a column reaches the support with any left side
        rhs.append(j)
        if len(cnt) / cnt.sum() < KEY: lhs.append(j)
    for a in lhs:
        for b in rhs:
            mask = (C[a] >= 0) & (C[b] >= 0)
            if a == b or not mask.any(): continue
            x, y = C[a][mask], C[b][mask]; K = int(y.max()) + 1; size = np.bincount(x)
            pair, inv, cnt = np.unique(x * K + y, return_inverse=True, return_counts=True); top = np.zeros(len(size), int); np.maximum.at(top, pair // K, cnt); big = size >= 2     # singleton groups cannot violate
            if size[big].sum() == 0 or top[big].sum() / size[big].sum() < SUPPORT: continue
            F[np.where(mask)[0][cnt[inv] < top[x]], b] = True; fds.append([names[a], names[b], round(float(top[big].sum() / size[big].sum()), 4)])
    return F, fds
def d4(X, num, seed):
    cols, bad = [], np.zeros(len(X)); miss = is_missing(X)
    for j in range(X.shape[1]):
        if j in num:
            x = numbers(X[:, j]); ok = np.isfinite(x); x[~ok] = np.median(x[ok]) if ok.any() else 0.0; bad += ~ok; cols.append(x)
        else:
            s = pd.Series(X[:, j]); cols.append(s.map(s.value_counts(normalize=True)).values.astype(float)); bad += miss[:, j]
    Z_ = np.column_stack(cols + [bad]); return IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(Z_).predict(Z_) == -1
def detect(X, num, names, seed=0):
    t = {}; c = time.time(); A = d1(X); t["D1"] = time.time() - c; c = time.time(); Bi, Bs = d2(X, num); t["D2"] = time.time() - c; c = time.time(); G, fds = d3(X, names); t["D3"] = time.time() - c; c = time.time(); I = d4(X, num, seed); t["D4"] = time.time() - c
    return dict(D1=A, D2_iqr=Bi, D2_3sd=Bs, D3=G, D4=I, D5=A | Bi | G), fds, {k: round(v, 2) for k, v in t.items()}
def table(root, meta, name):
    D = R.load(root, name, meta[name]); X = D["a"][D["feat"]].apply(lambda s: s.str.strip()).values; return D, X, [j for j, c in enumerate(D["feat"]) if c in D["num"]]
def detect_table(args):
    root, meta, out, name, seed = args; c = time.time(); D, X, num = table(root, meta, name); F, fds, t = detect(X, num, D["feat"], seed)
    np.savez_compressed(f"{out}/flags/{name}.npz", E=D["E"], yE=D["yE"], **F); json.dump(dict(fds=fds, seconds=t, N=len(X), m=X.shape[1], feat=D["feat"], num=len(num)), open(f"{out}/flags/{name}.json", "w")); return name, round(time.time() - c, 1), t, len(fds)
def run_raha(root, meta, out, name, seed, procs):
    import multiprocessing, raha
    D, X, _ = table(root, meta, name); work = f"{out}/raha_work/{name}"; os.makedirs(work, exist_ok=True); c = time.time()
    dirty = D["a"][D["feat"]]; clean = pd.DataFrame(np.where(D["E"], D["b"][D["feat"]].values, dirty.values), columns=D["feat"])    # the labels of Raha then agree with the truth of this script
    dirty.to_csv(f"{work}/dirty.csv", index=False); clean.to_csv(f"{work}/clean.csv", index=False)
    pool = multiprocessing.Pool; multiprocessing.Pool = lambda *x, **k: pool(procs); random.seed(seed); np.random.seed(seed)
    app = raha.detection.Detection(); app.SAVE_RESULTS = False; assert app.LABELING_BUDGET == 20
    cells = app.run(dict(name=name, path=f"{work}/dirty.csv", clean_path=f"{work}/clean.csv")); F = np.zeros(D["E"].shape, bool)
    for i, j in cells: F[i, j] = True
    np.savez_compressed(f"{out}/flags/{name}.raha.npz", D6=F); E = D["E"]; tp = int((F & E).sum())
    json.dump(dict(seconds=round(time.time() - c, 1), detected=int(F.sum()), precision=tp / max(int(F.sum()), 1), recall=tp / max(int(E.sum()), 1), seed=seed, budget=20), open(f"{out}/flags/{name}.raha.json", "w")); print(name, "raha finished", round(time.time() - c, 1), int(F.sum()), flush=True)
def wilson(p, n):
    z2 = Z * Z; d = 1 + z2 / n; c = (p + z2 / (2 * n)) / d; h = Z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return np.where(p <= 0, 0.0, np.maximum(0.0, c - h)), np.where(p >= 1, 1.0, np.minimum(1.0, c + h))
def rogan_gladen(p, tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        se = tp / (tp + fn); sp = tn / (tn + fp); ok = se + sp > 1 + 1e-12; return np.where(ok, np.clip((p + sp - 1) / (se + sp - 1), 0, 1), np.nan), ok
def corrected(p, tp, fn, fp, tn, n, boot, rng):
    est, ok = rogan_gladen(p, tp, fn, fp, tn); s = (tp + fn) / n; est = np.where(ok, est, s)
    d = rng.multinomial(n, np.stack([tp, fn, fp, tn], 1) / n, size=(boot, len(tp))); e, o = rogan_gladen(p, d[..., 0], d[..., 1], d[..., 2], d[..., 3]); e = np.where(o, e, (d[..., 0] + d[..., 1]) / n)
    lo, hi = np.quantile(e, [.025, .975], axis=0); wl, wh = wilson(s, n); return est, np.where(ok, lo, wl), np.where(ok, hi, wh), ok
def weighted(k1, n1, k0, n0, M1, M0):
    a = Z * Z / 2; est = wide = V = 0.0
    for k, nh, Mh in ((k1, n1, M1), (k0, n0, M0)):
        W = Mh / (M1 + M0)
        with np.errstate(divide="ignore", invalid="ignore"): r = np.where(nh > 0, k / nh, 0.0); q = np.where(nh < Mh, (k + a) / (nh + 2 * a), r); v = np.where((nh > 0) & (nh < Mh), q * (1 - q) / nh, 0.0)
        est = est + W * r; wide = wide + W * q; V = V + W * W * v
    with np.errstate(divide="ignore", invalid="ignore"): neff = np.where(V > 0, wide * (1 - wide) / V, np.inf)
    lo, hi = wilson(est, neff); return est, lo, hi
def stratified(f, t, n, rep, rng):
    M1 = int(f.sum()); M0 = len(f) - M1; K1, K0 = int((f & t).sum()), int((~f & t).sum()); n1 = min(n // 2, M1); n0 = min(n - n1, M0); n1 = min(n - n0, M1)
    draw = lambda K, Mh, nh: rng.hypergeometric(K, Mh - K, nh, rep) if nh else np.zeros(rep, int)
    k1, k0 = draw(K1, M1, n1), draw(K0, M0, n0); return weighted(k1, n1, k0, n0, M1, M0), (k1 + k0) / n
def poststratified(f, tp, fn, fp, tn, n):
    M1 = int(f.sum()); M0 = len(f) - M1; e, lo, hi = weighted(tp, tp + fp, fn, fn + tn, M1, M0); s = (tp + fn) / n; wl, wh = wilson(s, n); ok = ((tp + fp > 0) | (M1 == 0)) & ((fn + tn > 0) | (M0 == 0))
    return np.where(ok, e, s), np.where(ok, lo, wl), np.where(ok, hi, wh), ok
def stat(est, lo, hi, truth): return dict(mean_est=float(est.mean()), mean_abs_error=float(np.abs(est - truth).mean()), coverage=float(((lo <= truth + 1e-12) & (hi >= truth - 1e-12)).mean()), mean_half_width=float(((hi - lo) / 2).mean()))
def rates(f, t): return (float(f[t].mean()) if t.any() else None), (float((~f[~t]).mean()) if (~t).any() else None)
def evaluate(ds, unit, n, t, flags, rep, boot, seed):
    pi, M = float(t.mean()), len(t); rng = seeded(ds, unit, n, seed, "srs"); idx = np.stack([rng.choice(M, n, replace=False) for _ in range(rep)]); ts = t[idx]; k = ts.sum(1); s = k / n; wl, wh = wilson(s, n); samp = stat(s, wl, wh, pi); rows = []
    for det, f in flags.items():
        fs = f[idx]; tp = (fs & ts).sum(1); fp = (fs & ~ts).sum(1); e, lo, hi, ok = corrected(float(f.mean()), tp, k - tp, fp, n - k - fp, n, boot, seeded(ds, unit, n, seed, det, "boot")); pe, pl_, ph_, pk = poststratified(f, tp, k - tp, fp, n - k - fp, n)
        st, pl = stratified(f, t, n, rep, seeded(ds, unit, n, seed, det, "strat")); pwl, pwh = wilson(pl, n); se, sp = rates(f, t); base = dict(ds=ds, detector=det, unit=unit, n=n, true_rate=pi, positive_rate=float(f.mean()), se=se, sp=sp, N=M, repeats=rep)
        rows += [dict(base, method="sample", share_undefined=None, **samp), dict(base, method="corrected", share_undefined=float(1 - ok.mean()), **stat(e, lo, hi, pi)), dict(base, method="stratified", share_undefined=None, **stat(*st, pi)), dict(base, method="pooled", share_undefined=None, **stat(pl, pwl, pwh, pi)), dict(base, method="poststratified", share_undefined=float(1 - pk.mean()), **stat(pe, pl_, ph_, pi))]
    return rows
STORE = {}
def load_store(out, names):
    for name in names:
        z = np.load(f"{out}/flags/{name}.npz"); F = {k: z[k] for k in z.files if k not in ("E", "yE")}; r = f"{out}/flags/{name}.raha.npz"
        if os.path.exists(r): F["D6"] = np.load(r)["D6"]
        STORE[name] = dict(E=z["E"], yE=z["yE"], F=F)
def items(T, unit):
    if unit == "cell": return T["E"].ravel(), {d: f.ravel() for d, f in T["F"].items() if f.ndim == 2}
    t = T["E"].any(1)
    return (t | T["yE"]) if unit == "record_label" else t, {d: (f.any(1) if f.ndim == 2 else f) for d, f in T["F"].items()}
def units(T): return ["record", "cell"] + (["record_label"] if T["yE"].any() else [])
def raw_rows(ds, unit, t, flags):
    pi = float(t.mean()); out = []
    for det, f in flags.items():
        p = float(f.mean()); se, sp = rates(f, t); out.append(dict(ds=ds, detector=det, unit=unit, n=0, method="raw", true_rate=pi, positive_rate=p, se=se, sp=sp, N=len(t), repeats=0, share_undefined=None, mean_est=p, mean_abs_error=abs(p - pi), coverage=None, mean_half_width=None))
    return out
def job(key):
    t, flags = items(STORE[key["ds"]], key["unit"]); return evaluate(key["ds"], key["unit"], key["n"], t, flags, key["rep"], key["boot"], key["seed"])
def clean_json(x):
    if isinstance(x, dict): return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, (float, np.floating)): return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer): return int(x)
    return x
def summarize(rows, out, names, seconds):
    D = pd.DataFrame(rows)
    for c in ("coverage", "mean_half_width", "share_undefined", "se", "sp"): D[c] = pd.to_numeric(D[c], errors="coerce")
    D["bias"] = D.mean_est - D.true_rate; D["abs_bias"] = D.bias.abs(); D["informative"] = D.true_rate.between(.01, .99); G = []
    for scope, S in (("all_tables", D), ("informative", D[D.informative]), ("informative_flagging", D[D.informative & (D.positive_rate > 0)]), ("native_tables", D[D.informative & D.ds.isin(NATIVE)])):
        for by in (["unit", "n", "method"], ["unit", "n", "method", "detector"]):
            for keys, g in S.groupby(by): G.append(dict(zip(by, keys), scope=scope, pairs=len(g), tables=int(g.ds.nunique()), bias=g.bias.mean(), abs_bias=g.abs_bias.mean(), mean_abs_error=g.mean_abs_error.mean(), coverage=g.coverage.mean(), mean_half_width=g.mean_half_width.mean(), share_undefined=g.share_undefined.mean()))
    P = {}
    for name in names:
        m = json.load(open(f"{out}/flags/{name}.json")); T = STORE[name]; rp = f"{out}/flags/{name}.raha.json"; P[name] = dict(N=m["N"], m=m["m"], true_record=T["E"].any(1).mean(), true_cell=T["E"].mean(), label_errors=T["yE"].mean(), true_record_label=(T["E"].any(1) | T["yE"]).mean(), fds=len(m["fds"]), fd_examples=m["fds"][:8], detector_seconds=m["seconds"], raha=json.load(open(rp)) if os.path.exists(rp) else None)
        for unit in D[D.ds == name].unit.unique():
            A = D[(D.ds == name) & (D.unit == unit)]; raw = A[A.method == "raw"].sort_values("abs_bias"); P[name][unit] = dict(best_raw=raw.iloc[0][["detector", "bias"]].to_dict(), worst_raw=raw.iloc[-1][["detector", "bias"]].to_dict(), raw_by_detector=raw.set_index("detector")[["positive_rate", "bias", "se", "sp"]].to_dict("index"), at_200=A[A.n == 200].groupby("method")[["mean_abs_error", "coverage", "mean_half_width", "share_undefined", "bias"]].mean().to_dict("index"), raw_mean_abs=raw.abs_bias.mean())
    W = []; piv = D[D.informative & (D.positive_rate > 0) & (D.n > 0)].pivot_table(index=["unit", "n", "ds", "detector"], columns="method", values="mean_abs_error")
    for (unit, n), g in piv.groupby(level=[0, 1]): W.append(dict(unit=unit, n=int(n), pairs=len(g), **{f"{m}_beats_sample": int((g[m] < g["sample"]).sum()) for m in ("corrected", "stratified", "poststratified")}, corrected_beats_stratified=int((g["corrected"] < g["stratified"]).sum()), corrected_mae_over_sample=float((g["corrected"] / g["sample"]).median()), stratified_mae_over_sample=float((g["stratified"] / g["sample"]).median()), poststratified_mae_over_sample=float((g["poststratified"] / g["sample"]).median())))
    K = []; J = D[D.informative & (D.positive_rate > 0) & D.se.notna() & D.sp.notna()].copy(); J["power"] = pd.cut(J.se + J.sp - 1, [-2, 0, .1, .3, .6, .9, 1.01], labels=["at most 0", "0 to 0.1", "0.1 to 0.3", "0.3 to 0.6", "0.6 to 0.9", "above 0.9"]).astype(str)     # the power is se + sp - 1 on the full table
    for (unit, n, power, method), g in J.groupby(["unit", "n", "power", "method"]): K.append(dict(unit=unit, n=int(n), power=power, method=method, pairs=len(g), mean_abs_error=g.mean_abs_error.mean(), coverage=g.coverage.mean(), lowest_coverage=g.coverage.min(), mean_half_width=g.mean_half_width.mean(), share_undefined=g.share_undefined.mean()))
    import sklearn
    return clean_json(dict(seconds=seconds, versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, raha="1.26"), sizes=SIZES, thresholds=dict(rare=RARE, support=SUPPORT, key=KEY, constant=CONST), scopes=dict(all_tables="every table and detector", informative="true rate between 0.01 and 0.99", informative_flagging="the same without detectors that flag nothing on the table", native_tables="informative rows of beers and flights"), tables=P, wins_on_informative_flagging=W, by_power=K, overall=G))
def measure(a, meta):
    t0 = time.time(); names = [n for n in meta if os.path.exists(f"{a.out}/flags/{n}.npz")]; load_store(a.out, names); rows = []; jobs = []
    for name in names:
        T = STORE[name]
        for unit in units(T):
            t, flags = items(T, unit); rows += raw_rows(name, unit, t, flags); jobs += [dict(ds=name, unit=unit, n=n, rep=a.repeats, boot=a.boot, seed=a.seed) for n in SIZES if n <= len(t)]
    print("jobs", len(jobs), flush=True)
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1): rows += r
    rows = sorted(clean_json(rows), key=lambda r: (r["ds"], r["unit"], r["detector"], r["n"], r["method"]))
    with open(f"{a.out}/results_detectors.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(summarize(rows, a.out, names, round(time.time() - t0, 1)), open(f"{a.out}/detectors_summary.json", "w"), indent=0); print("rows", len(rows), "seconds", round(time.time() - t0, 1), flush=True)
def check(a, meta):
    from hqd_assess import sampling as H
    load_store(a.out, [n for n in meta if os.path.exists(f"{a.out}/flags/{n}.npz")]); res = []
    def note(name, ok, **kw): res.append(dict(check=name, passed=bool(ok), **kw)); print("PASS" if ok else "FAIL", name, kw, flush=True)
    one = lambda x: np.array([x]); e, o = rogan_gladen(.25, one(30), one(10), one(20), one(140)); note("closed form 30 10 20 140 at p 0.25 gives 0.2", abs(e[0] - .2) < 1e-12 and H.rogan_gladen(.25, .75, .875) == e[0], value=float(e[0]))
    gap = max(np.abs(np.array(wilson(np.array(k / n), n)).ravel() - np.array(H.wilson_interval(k, n))).max() for n in (20, 50, 200, 800) for k in range(0, n + 1, max(1, n // 20))); note("Wilson equals the package", gap < 1e-9, gap=float(gap))
    # the same audit through this script and through the package, whose bootstrap also redraws the positive rate
    if "hospitals" in STORE:
        t, fl = items(STORE["hospitals"], "record"); f = fl["D5"]; rng = seeded("check"); pts, los, his = [], [], []
        for r in range(40):
            i = rng.choice(len(t), 200, replace=False); C = H.calibrate_from_audit(f, i, t[i], n_boot=1000, seed=r); tp = int((f[i] & t[i]).sum()); fp = int((f[i] & ~t[i]).sum()); k = int(t[i].sum()); E_, L_, U_, _ = corrected(float(f.mean()), one(tp), one(k - tp), one(fp), one(200 - k - fp), 200, 1000, seeded("check", r)); pts.append(abs(C.estimate - E_[0])); los.append(abs(C.low - L_[0])); his.append(abs(C.high - U_[0]))
        note("corrected against calibrate_from_audit on hospitals D5", max(pts) < 1e-12, point_gap=float(max(pts)), mean_low_gap=float(np.mean(los)), mean_high_gap=float(np.mean(his)))
    # the flags have to follow from the dirty file and the schema, the clean file is not opened here
    for name in [n for n in ("hospitals", "flights", "breast_cancer") if n in STORE]:
        A_ = pd.read_csv(f"{a.root}/{name}/dirty_index.csv", dtype=str, keep_default_na=False).drop(columns=["index"], errors="ignore"); feat = [c for c in A_.columns if c != meta[name]["label_col"]]; X = A_[feat].apply(lambda s: s.str.strip()).values; F_, _, _ = detect(X, [j for j, c in enumerate(feat) if c not in meta[name]["categorical_cols"]], feat, a.seed); note(f"{name} flags from the dirty file alone equal the stored flags", all((F_[k] == STORE[name]["F"][k]).all() for k in F_))
    # synthetic populations with known truth, the intervals have to hold before they are read on real tables
    grid = []
    for pi in (.02, .1, .3, .5, .8):
        for sen, spe in ((.95, .95), (.7, .9), (.4, .98)):
            M = 20000; r0 = seeded("syn", pi, sen, spe); t = r0.random(M) < pi; f = np.where(t, r0.random(M) < sen, r0.random(M) > spe)
            for n in (50, 200, 800):
                rows = evaluate("syn", "record", n, t, {"d": f}, 1000, 1000, 1); grid += [dict(pi=pi, se=sen, sp=spe, n=n, method=x["method"], coverage=x["coverage"], mae=x["mean_abs_error"], half=x["mean_half_width"], bias=x["mean_est"] - x["true_rate"], undefined=x["share_undefined"]) for x in rows]
    G = pd.DataFrame(grid); g = G.groupby("method").agg(coverage=("coverage", "mean"), low=("coverage", "min"), mae=("mae", "mean"), half=("half", "mean"), bias=("bias", lambda s: s.abs().mean())); note("synthetic populations", g.loc["sample", "coverage"] > .93 and g.loc["stratified", "coverage"] > .93 and g.loc["pooled", "bias"] > 5 * g.loc["stratified", "bias"], table=clean_json(g.reset_index().to_dict("records")), weakest=clean_json(G[G.method == "stratified"].sort_values("coverage").head(5).to_dict("records")))
    # detectors with a known outcome on a real table
    if "hospitals" in STORE:
        t = items(STORE["hospitals"], "record")[0]; r0 = seeded("coin"); det = dict(none=np.zeros(len(t), bool), every=np.ones(len(t), bool), oracle=t.copy(), coin=r0.random(len(t)) < .3); R_ = {(x["detector"], x["method"]): x for x in evaluate("hospitals", "record", 200, t, det, 500, 1000, 7)}
        c, s = R_[("none", "corrected")], R_[("none", "sample")]; note("no flags falls back to the sample in all repeats", c["share_undefined"] == 1 and c["mean_abs_error"] == s["mean_abs_error"] and c["coverage"] == s["coverage"], raw_error=float(t.mean()), corrected_mae=c["mean_abs_error"], stratified_mae=R_[("none", "stratified")]["mean_abs_error"], sample_half=s["mean_half_width"], stratified_half=R_[("none", "stratified")]["mean_half_width"], stratified_coverage=R_[("none", "stratified")]["coverage"])
        c = R_[("none", "poststratified")]; note("post-stratification without flags is the sample", c["mean_abs_error"] == s["mean_abs_error"] and c["mean_half_width"] == s["mean_half_width"], post_mae=c["mean_abs_error"]); c = R_[("oracle", "poststratified")]; note("post-stratification with an oracle has no error", c["mean_abs_error"] < 1e-12, post_mae=c["mean_abs_error"], post_coverage=c["coverage"], post_half=c["mean_half_width"])
        c = R_[("every", "corrected")]; note("flags on everything falls back in all repeats", c["share_undefined"] == 1 and c["mean_abs_error"] == R_[("every", "sample")]["mean_abs_error"], corrected_mae=c["mean_abs_error"])
        c = R_[("oracle", "corrected")]; d = R_[("oracle", "stratified")]; note("an oracle gives the truth with no error", c["mean_abs_error"] < 1e-12 and d["mean_abs_error"] < 1e-12 and c["mean_half_width"] < 1e-12 and d["coverage"] == 1, stratified_half=d["mean_half_width"], pooled_bias=R_[("oracle", "pooled")]["mean_est"] - float(t.mean()))
        c = R_[("coin", "corrected")]; note("a coin has no power, the correction is undefined about half the time and worse than the sample", .25 < c["share_undefined"] < .75 and c["mean_abs_error"] > R_[("coin", "sample")]["mean_abs_error"], undefined=c["share_undefined"], corrected_mae=c["mean_abs_error"], sample_mae=R_[("coin", "sample")]["mean_abs_error"], se=c["se"], sp=c["sp"])
        a1 = evaluate("hospitals", "record", 100, t, {"coin": det["coin"]}, 200, 200, 3); a2 = evaluate("hospitals", "record", 100, t, {"coin": det["coin"]}, 200, 200, 3); note("same seed, same rows", a1 == a2)
        # the counts of the strata come from a hypergeometric draw, a physical draw of the records has to agree
        f = items(STORE["hospitals"], "record")[1]["D5"]; (e1, _, _), pl1 = stratified(f, t, 200, 4000, seeded("hyper")); r1 = seeded("phys"); i1, i0 = np.where(f)[0], np.where(~f)[0]; e2, pl2 = [], []
        for _ in range(4000):
            n1 = min(100, len(i1)); n0 = min(200 - n1, len(i0)); n1 = min(200 - n0, len(i1)); a_, b_ = t[r1.choice(i1, n1, replace=False)], t[r1.choice(i0, n0, replace=False)]; e2.append(f.mean() * a_.mean() + (1 - f.mean()) * b_.mean()); pl2.append((a_.sum() + b_.sum()) / 200)
        note("hypergeometric counts agree with a physical draw", abs(e1.mean() - np.mean(e2)) < .003 and abs(pl1.mean() - np.mean(pl2)) < .003, hyper=float(e1.mean()), physical=float(np.mean(e2)), hyper_pooled=float(pl1.mean()), physical_pooled=float(np.mean(pl2)), truth=float(t.mean()))
    # the tables at the ends of the scale, from the stored rows when measure has run
    path = f"{a.out}/results_detectors.jsonl"; rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    for ds, unit in (("adult", "record"), ("adult", "cell"), ("soilmoisture", "record"), ("mercedes", "record"), ("beers", "record")):
        A = [r for r in rows if r["ds"] == ds and r["unit"] == unit and r["n"] == 200]
        if not A: continue
        C = [r for r in A if r["method"] == "corrected"]; S = [r for r in A if r["method"] == "stratified"]; B = [r for r in A if r["method"] == "sample"]; degenerate = C[0]["true_rate"] in (0.0, 1.0)
        note(f"{ds} {unit} at the end of the scale", (not degenerate) or (all(r["share_undefined"] == 1 for r in C) and all(r["coverage"] == 1 for r in B + S)), true_rate=C[0]["true_rate"], undefined=float(np.mean([r["share_undefined"] for r in C])), corrected_mae=float(np.mean([r["mean_abs_error"] for r in C])), sample_mae=B[0]["mean_abs_error"], sample_half=B[0]["mean_half_width"], stratified_half=float(np.mean([r["mean_half_width"] for r in S])), positive_rates={r["detector"]: r["positive_rate"] for r in C})
    ratio, agree = [], []
    for r in [x for x in rows if x["method"] == "corrected" and x["share_undefined"] == 0 and x["se"] is not None and x["sp"] is not None and x["unit"] in ("record", "cell") and .01 < x["true_rate"] < .99 and x["n"] >= 200]:
        J = r["se"] + r["sp"] - 1; pi = r["true_rate"]; fpc = 1 - r["n"] / r["N"]; v = ((1 - pi) * r["sp"] * (1 - r["sp"]) + pi * r["se"] * (1 - r["se"])) / J ** 2; ratio.append(r["mean_abs_error"] / max(np.sqrt(2 / np.pi * fpc * v / r["n"]), 1e-9)); b = [x for x in rows if x["ds"] == r["ds"] and x["unit"] == r["unit"] and x["detector"] == r["detector"] and x["n"] == r["n"] and x["method"] == "sample"][0]; agree.append((v < pi * (1 - pi)) == (r["mean_abs_error"] < b["mean_abs_error"]))
    if ratio: note("the error of the correction follows its first order variance, and that variance predicts when it beats the sample", .8 < np.median(ratio) < 1.25, pairs=len(ratio), median_ratio=float(np.median(ratio)), q1=float(np.percentile(ratio, 25)), q3=float(np.percentile(ratio, 75)), break_even_agreement=float(np.mean(agree)))
    if "adult" in STORE:
        T = STORE["adult"]; note("adult has label errors and no wrong feature cell", not T["E"].any() and T["yE"].any(), label_rate=float(T["yE"].mean()), positive_rates={k: float((v.any(1) if v.ndim == 2 else v).mean()) for k, v in T["F"].items()})
    json.dump(clean_json(res), open(f"{a.out}/sanity_detectors.json", "w"), indent=0); print("checks", len(res), "failed", sum(not r["passed"] for r in res), flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--meta"); ap.add_argument("--out"); ap.add_argument("--stage", choices=("detect", "raha", "measure", "check")); ap.add_argument("--only", default=""); ap.add_argument("--procs", type=int, default=40); ap.add_argument("--repeats", type=int, default=500); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    meta = json.load(open(a.meta)); os.makedirs(f"{a.out}/flags", exist_ok=True); only = [x for x in a.only.split(",") if x]
    if only: meta = {k: v for k, v in meta.items() if k in only}
    if a.stage == "detect":
        with Pool(min(a.procs, len(meta))) as pool:
            for r in pool.imap_unordered(detect_table, [(a.root, meta, a.out, n, a.seed) for n in meta]): print(*r, flush=True)
    elif a.stage == "raha":
        for n in meta: run_raha(a.root, meta, a.out, n, a.seed, a.procs)
    elif a.stage == "measure": measure(a, meta)
    else: check(a, meta)
