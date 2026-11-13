#!/usr/bin/env python3
"""The two result figures of the survey, Figure 5 and Figure 6. All values are read from the result files, and the plotted values are written next to the figures as figure_data_a.json and figure_data_b.json.
Usage: plot_results.py [RESULTS_DIR] [OUT_DIR]"""
import json, os, sys, statistics as st, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from paper_style import *
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import big_tables
from matplotlib.colors import SymLogNorm
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results"); OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "plots"); os.makedirs(OUT, exist_ok=True)
def load(fn):
    p = os.path.join(R, fn); return json.load(open(p)) if os.path.exists(p) else None
def jsonl(*fns):
    out = {}
    for fn in fns:
        p = os.path.join(R, fn)
        if os.path.exists(p):
            for i, l in enumerate(open(p)):
                try: r = json.loads(l)
                except Exception: continue
                if "error" not in r: out[r.get("id", (fn, i))] = r
    return list(out.values())
def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]; return st.mean(x) if x else float("nan")
setup()
BROWN, PINK, OLIVE = "#8C564B", "#E377C2", "#A3A300"
DS = [("adult", "Ad"), ("beers", "Be"), ("breast_cancer", "BC"), ("flights", "Fl"), ("hospitals", "Ho"), ("smartfactory", "SF"), ("soccer_full", "So"), ("bike", "Bi"), ("mercedes", "Me"), ("nasa", "Na"), ("soilmoisture", "SM")]
DSC = dict(adult=RED, beers=BLUE, breast_cancer=ORANGE, flights=GREEN, hospitals=PURPLE, smartfactory=TEAL, soccer_full=BROWN, bike=PINK, mercedes=OLIVE, nasa=GRAY, soilmoisture=DARK)
MOD = [("linear", "Lin"), ("rf", "RF"), ("gb", "GB"), ("knn", "KNN"), ("mlp", "MLP")]; MODS = [("gb", "GB", RED, "*"), ("rf", "RF", BLUE, "o"), ("mlp", "MLP", ORANGE, "D"), ("linear", "Lin", GREEN, "^"), ("knn", "KNN", PURPLE, "s")]; MC = {m: (c, k) for m, _, c, k in MODS}
def line(ax, x, y, col, mk=None, lw=1.5, ms=None, ls="-", z=3):
    ms = ms if ms else (6.0 if mk == "*" else 3.4)
    return ax.plot(x, y, color=col, marker=mk, ms=ms, lw=lw, ls=ls, mec="white", mew=.25, zorder=z)
def hl(col, mk=None, n="", lw=1.3, ms=3.4, ls="-"): return Line2D([], [], color=col, marker=mk, ms=ms, lw=lw, ls=ls, label=n)
DASH = (0, (3, 2)); XS = [.39, 2.10, 3.83, 5.58]; PWS = [1.23, 1.28, 1.28, 1.28]; PW, PH = 1.28, 1.02; LD = .25
def cell(fig, i, y): return box(fig, XS[i], y, PWS[i], PH)
def ylab(ax, text, dist=LD, **kw):
    """The y label at one fixed distance from the axis in all panels, whatever the width of the tick labels."""
    ax.set_ylabel(text, **kw); ax.yaxis.set_label_coords(-dist / (ax.get_position().width * ax.figure.get_figwidth()), .5)
def label_room(fig):
    """Reports a y label that touches its tick labels."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.get_ylabel(): continue
        lo_, hi_ = sorted(ax.get_ylim()); tk = [t.get_window_extent(r).x0 for t, q in zip(ax.get_yticklabels(), ax.get_yticks()) if t.get_text() and lo_ <= q <= hi_]; lb = ax.yaxis.label.get_window_extent(r)
        if tk and lb.x1 > min(tk) - 1: print("label touches ticks:", ax.get_ylabel(), round((min(tk) - lb.x1) / fig.dpi, 3))
from matplotlib.ticker import FuncFormatter, MaxNLocator
def short_y(ax, n=4):
    """Compact tick labels, without the leading zero, so that the panels keep the same width."""
    ax.yaxis.set_major_locator(MaxNLocator(n)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:.3g}".replace("0.", ".", 1) if abs(v) < 1 and v != 0 else f"{v:.3g}").replace("-", "\u2212")))
def zoom(ax, ys, pad=.08):
    v = [y for y in np.ravel(np.array(ys, dtype=float)) if np.isfinite(y)]
    if v: a, b = min(v), max(v); d = (b - a) or 1; ax.set_ylim(a - pad * d, b + pad * d)
# ======== figure A: quality and utility on tables, protocols and label noise
T, C, Q = load("tables_summary.json"), load("case_tep.json"), load("results_rieth.json"); V = jsonl("results_vision.jsonl", "results_vision_fine.jsonl"); science = {}
PH_B = PH; PH = PH_B; H = 5.02; fig = plt.figure(figsize=(W, H), facecolor="white"); ty, th = 3.72, 0.94; y2, y3 = 2.08, .34
EX = .10; a = box(fig, .34, ty - EX, 1.56, th + EX); b = box(fig, 1.96, ty - EX, 1.56, th + EX); c = box(fig, 3.58, ty - EX, 1.46, th + EX); d = box(fig, 5.10, ty - EX, 1.78, th + EX)
NORM = SymLogNorm(linthresh=2, linscale=1, vmin=-40, vmax=40, base=10); HMAP = plt.get_cmap("RdBu_r").copy(); HMAP.set_bad("#B9BFC5")
def ink(v):
    """The colour of a number on a cell, from the luminance of the cell."""
    r_, g_, b_, _ = HMAP(NORM(v)); return "white" if .299 * r_ + .587 * g_ + .114 * b_ < .46 else DARK
def cells(ax, M, xlabels, xticks, ylabels, size=6.6):
    im = ax.imshow(np.ma.masked_invalid(M), cmap=HMAP, norm=NORM, aspect="auto", interpolation="nearest")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7.2); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(ylabels if ylabels else [], fontsize=size)
    ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True); ax.grid(which="minor", color="white", lw=.45); ax.tick_params(which="both", length=0, pad=1.7)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.8)
    return im
def scale(fig, im, x, y, w, tk=(-40, -10, -2, 0, 2, 10, 40)):
    """The thin colour bar above a pair of heat maps, on the scale of the cells."""
    cax = box(fig, x, y, w, .055); tk = list(tk); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks(tk, labels=[str(t).replace("-", "−") for t in tk]); cb.minorticks_off(); cb.ax.tick_params(labelsize=7.0, length=1.2, pad=.6); cb.ax.xaxis.set_ticks_position("top"); cb.outline.set_linewidth(.6); fig.text((x + w + .03) / W, (y + .03) / fig.get_figheight(), "pp", fontsize=7.2, va="center")
def heat(ax, action, show):
    M = np.full((len(DS), len(MOD)), np.nan)
    for i, (ds, _) in enumerate(DS):
        for j, (m, _) in enumerate(MOD):
            row = T["table"].get(f"{ds}|{m}", {})
            if action in row and "keep" in row: M[i, j] = 100 * (row[action] - row["keep"])
    science[f"gain_{action}"] = M.tolist(); im = cells(ax, M, [n for _, n in MOD], range(len(MOD)), [n for _, n in DS] if show else None)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v) and abs(v) >= .5: ax.text(j, i, (f"{v:.0f}" if abs(v) >= 9.95 else f"{v:.1f}").replace("-", "−"), ha="center", va="center", fontsize=5.8, color=ink(v))
    return im
im = heat(a, "repair_records", True); heat(b, "delete_records", False)
scale(fig, im, .40, ty + th + .05, 2.90)
# the table versions of (c) and (f), seed means of the task score and of the static score per table, model, operation and share
RT = jsonl("results_real_tables.jsonl", "results_tables_fine.jsonl"); G = collections.defaultdict(dict); PS = collections.defaultdict(dict)
seen = set()
for r in RT:
    if "score" not in r: continue
    k = (r["ds"], r["model"], r["seed"], r["action"], r["f"])
    if k in seen: continue
    seen.add(k); G[(r["ds"], r["model"], r["action"], round(r["f"], 3))][r["seed"]] = r["score"]
    if r.get("rate_missing") is not None: PS[(r["ds"], r["model"], r["action"], round(r["f"], 3))][r["seed"]] = 100 * (8 - r["rate_missing"] - r["rate_domain"] - r["rate_value"]) / 8
def path(ds, m, action):
    """The raw table and every share of one operation, as (share, static score, task score), seed means."""
    return sorted((k[3], mean(PS[k].values()), mean(G[k].values())) for k in G if k[:2] == (ds, m) and k[2] in ("keep", action) and PS.get(k))
# (c) rank correlation of the static score with the task score along the repair path and along the delete path, one point per table and model. A delete path enters when deleting raises the static score by more than one point
from scipy.stats import spearmanr
PATHS = (("repair_records", "Repair path", BLUE, "o", -.17), ("delete_records", "Delete path", ORANGE, "D", .17)); science["path_spearman"] = {}
for a_, n_, col, mk, off in PATHS:
    allr = []
    for y_, (ds, _) in enumerate(DS):
        for m, _ in MOD:
            pt = path(ds, m, a_)
            if len(pt) < 5 or len({round(q[1], 9) for q in pt}) < 3 or pt[-1][1] - pt[0][1] <= 1: continue
            rho = float(spearmanr([q[1] for q in pt], [q[2] for q in pt]).statistic); allr.append(rho); science["path_spearman"].setdefault(n_, {}).setdefault(ds, {})[m] = rho
            c.plot(rho, y_ + off, ls="none", marker=mk, ms=3.6 if mk == "o" else 3.2, color=col, mec="white", mew=.25, zorder=4)
    md = float(np.median(allr)); science["path_spearman"][n_ + "/median"] = md; science["path_spearman"][n_ + "/pairs"] = len(allr); science["path_spearman"][n_ + "/negative"] = sum(q < 0 for q in allr)
    c.axvline(md, color=col, ls=DASH, lw=1.1, zorder=3)
pos = np.arange(len(DS)); c.axvline(0, color=DARK, lw=.7, zorder=2); c.set_ylim(len(DS) - .5, -.5); c.set_yticks(pos); c.set_xlim(-1.08, 1.08); c.set_xticks([-1, -.5, 0, .5, 1]); c.set_xticklabels(["−1", "−.5", "0", ".5", "1"]); frame(c, "x"); c.tick_params(labelleft=False)
key_over(fig, c, [Line2D([], [], ls="none", marker=mk, ms=3.4, color=col, label=n_) for _, n_, col, mk, _ in PATHS] + [hl(DARK, None, "Median", lw=1.1, ls=DASH)], cols=2, size=5.8)
# (d) joint gain minus the sum of the single gains
OFF = dict(linear=-.28, rf=-.14, gb=0, knn=.14, mlp=.28)
I = T["interaction_list"]; rows_ = {n: i for i, (n, _) in enumerate(DS)}; science["interaction"] = I; d.axvline(0, color=DARK, lw=.7, zorder=1)
for r in I:
    y_ = rows_[r["ds"]] + OFF[r["model"]]; v = 100 * r["inter"]; sig = r["q"] < .05
    d.plot(v, y_, marker="o", ms=4.6 if sig else 3.4, ls="none", color=RED if sig else GRAY, mfc=RED if sig else "white", mec="white" if sig else GRAY, mew=.5 if sig else .8, zorder=5 if sig else 4, clip_on=False)
d.set_xscale("symlog", linthresh=2, linscale=.6); d.set_xlim(-60, 60); d.set_xticks([-30, -2, 0, 2, 30]); d.set_xticklabels(["−30", "−2", "0", "2", "30"]); d.minorticks_off(); d.set_ylim(len(DS) - .5, -.5); d.set_yticks(pos); frame(d, "x"); d.tick_params(labelleft=False)
key_over(fig, d, [Line2D([], [], ls="none", marker="o", ms=4.6, color=RED, mec="white", mew=.5, label="Significant"), Line2D([], [], ls="none", marker="o", ms=3.4, color=GRAY, mfc="white", mec=GRAY, mew=.8, label="Not significant")], cols=2)
titles(fig, [a, b, c, d], ["(a) Repair all errors", "(b) Delete dirty rows", "(c) Repair vs. delete", "(d) Joint minus single gains"])
# second row
e = cell(fig, 0, y2); f = cell(fig, 1, y2); j_ = box(fig, 3.98, y2, 1.08, PH); h = cell(fig, 3, y2)
# (e) cells the language model repaired and damaged, per 100 rows it read, the same rows as (a)
LC = load("results_llm_cells.json") or {}; pos_e = np.arange(len(DS)); science["qwen_cells"] = {}
for y_, (ds, _) in zip(pos_e, DS):
    v_ = LC.get(ds)
    if not v_: continue
    c_ = v_.get("cells_features", v_["cells"]); rp = 100 * c_["repaired"] / v_["rows_read"]; dm = 100 * c_["damaged"] / v_["rows_read"]; science["qwen_cells"][ds] = dict(repaired=rp, damaged=dm)
    e.plot([rp, dm], [y_, y_], color=GRAY, lw=1.0, zorder=2); e.plot(rp, y_, ls="none", marker="o", ms=4.0, color=BLUE, mec="white", mew=.3, zorder=4); e.plot(dm, y_, ls="none", marker="X", ms=4.6, color=RED, mec="white", mew=.3, zorder=3)
e.set_xscale("symlog", linthresh=1, linscale=.5); e.set_xlim(-.2, 9000); e.set_xticks([0, 1, 10, 100, 1000]); e.set_xticklabels(["0", "1", "10", "100", "1k"]); e.minorticks_off(); e.set_ylim(len(DS) - .5, -.5); e.set_yticks(pos_e); e.set_yticklabels([n for _, n in DS], fontsize=6.6); e.set_xlabel("Cells per 100 rows", labelpad=1.5); frame(e, "x")
key_over(fig, e, [Line2D([], [], ls="none", marker="o", ms=4.0, color=BLUE, label="Repaired"), Line2D([], [], ls="none", marker="X", ms=4.4, color=RED, label="Damaged")], cols=2, size=6.4)
# (f) partial repair. For each table and model whose full repair gains at least 0.02, the gain at each share of repaired records as a fraction of the full gain, the median per model family, against the same fraction of the static score
SH = [0.0] + [round(.1 * q, 1) for q in range(1, 11)]; science["share"] = dict(shares=SH); FR = collections.defaultdict(list); FS_ = []
for ds, _ in DS:
    for m, _ in MOD:
        k0, k1 = (ds, m, "keep", 0.0), (ds, m, "repair_records", 1.0)
        if not G.get(k0) or not G.get(k1): continue
        s0, g1 = mean(G[k0].values()), mean(G[k1].values()) - mean(G[k0].values())
        if g1 < .02: continue
        p0, p1 = mean(PS[k0].values()), mean(PS[k1].values()); FR[m].append([0.0] + [(mean(G[(ds, m, "repair_records", x)].values()) - s0) / g1 for x in SH[1:]]); FS_.append([0.0] + [(mean(PS[(ds, m, "repair_records", x)].values()) - p0) / (p1 - p0) for x in SH[1:]])
fsm = list(np.median(FS_, axis=0)); science["share"]["static"] = fsm; f.plot(SH, fsm, color=GRAY, lw=1.3, ls=DASH, zorder=2)
for m, n, col, mk in MODS:
    if FR[m]: yy = list(np.median(FR[m], axis=0)); science["share"][n] = dict(gain_fraction=yy, pairs=len(FR[m])); line(f, SH, yy, col, mk, lw=1.5, z=4 if m == "gb" else 3)
f.set_xlim(-.04, 1.04); f.set_ylim(-.06, 1.08); f.set_xticks([0, .5, 1]); f.set_xticklabels(["0", "50%", "100%"]); f.set_yticks([0, .5, 1]); f.set_yticklabels(["0", ".5", "1"]); f.set_xlabel("Dirty records repaired", labelpad=1.5); ylab(f, "Share of full gain"); frame(f)
key_over(fig, f, [hl(col, mk, n, ms=3.0) for _, n, col, mk in MODS] + [hl(GRAY, None, "Static score", lw=1.2, ls=DASH)], cols=3, size=5.8)
# (j) CleanML tables with their real errors. For each table and error type the gain of each cleaning method over the dirty version, averaged over model families and seeds, and the choice of the static rule and of the task test within each error type, for every model and seed
CM = [r for r in jsonl("results_cleanml.jsonl") if "test" in r]; GC = collections.defaultdict(dict)
for r in CM: GC[(r["ds"], r["error_type"], r["model"], r["seed"])][r["method"]] = r
VAL_N = 400
def at_n(r, n=VAL_N):
    av = sorted(int(x) for x in r["val"]); return r["val"][str(max([x for x in av if x <= n] or av[:1]))]
EG = collections.defaultdict(list); CS = collections.defaultdict(lambda: collections.defaultdict(list)); CT = collections.defaultdict(lambda: collections.defaultdict(list))
for (ds, et, _, _), arms in GC.items():
    if "dirty" not in arms: continue
    d0 = arms["dirty"]["test"]
    for me, r in arms.items():
        if me != "dirty": EG[(ds, et, me)].append(100 * (r["test"] - d0))
    CS[ds][et].append(100 * (mean(r["test"] for r in big_tables._static_cleanml(arms, et)) - d0) if len(arms) > 1 else 0.0); CT[ds][et].append(100 * (max(arms.values(), key=at_n)["test"] - d0))
science["cleanml"] = {}
for ds in sorted(CS):
    gm = {k: mean(v) for k, v in EG.items() if k[0] == ds}; lo_, hi_ = min(gm, key=gm.get), max(gm, key=gm.get)
    science["cleanml"][ds] = dict(worst=gm[lo_], worst_method=lo_[1:], best=gm[hi_], best_method=hi_[1:], static=mean(mean(v) for v in CS[ds].values()), task=mean(mean(v) for v in CT[ds].values()), versions=len(gm), error_types=len(CS[ds]))
ROWS = sorted(science["cleanml"], key=lambda q: science["cleanml"][q]["best"] - science["cleanml"][q]["worst"], reverse=True); ys = np.arange(len(ROWS)); RC = [science["cleanml"][q] for q in ROWS]
j_.barh(ys, [q["best"] - q["worst"] for q in RC], left=[q["worst"] for q in RC], height=.72, color=BLUE, alpha=.35, edgecolor=DARK, lw=.3, hatch="///", zorder=2)
j_.plot([q["static"] for q in RC], ys, ls="none", marker="D", ms=2.8, color=ORANGE, mec="white", mew=.25, zorder=4); j_.plot([q["task"] for q in RC], ys, ls="none", marker="*", ms=5.0, color=RED, mec="white", mew=.25, zorder=5)
j_.axvline(0, color=DARK, lw=.7, zorder=3); j_.set_xscale("symlog", linthresh=1, linscale=1); j_.set_xlim(-20, 9); j_.set_xticks([-10, -1, 0, 1, 5]); j_.set_xticklabels(["−10", "−1", "0", "1", "5"]); j_.minorticks_off(); j_.set_ylim(len(ROWS) - .5, -.5); j_.set_yticks(ys); j_.set_yticklabels(ROWS); j_.set_xlabel("Gain over dirty (pp)", labelpad=1.5); frame(j_, "x"); j_.tick_params(axis="y", labelsize=5.4, pad=1.2, length=1.2)
key_over(fig, j_, [Patch(facecolor=BLUE, alpha=.35, hatch="///", edgecolor=DARK, label="Range"), Line2D([], [], ls="none", marker="D", ms=2.8, color=ORANGE, label="Static"), Line2D([], [], ls="none", marker="*", ms=4.6, color=RED, label="Task test")], cols=3, size=5.8, dx=-.15)
# (h) score changes on the tables, sorted, against the gaps caused by the protocol on TEP
def mf(m, arm, fld="macro_f1"): return mean(r[fld] for r in C["runs"] if r["model"] == m and r["arm"] == arm)
TB = []
for k, v in T["table"].items():
    if T["by_task"][k.split("|")[0]] != "cls": continue
    for a_ in ("repair_records", "delete_records"):
        if a_ in v and "keep" in v: TB.append((100 * abs(v[a_] - v["keep"]), a_))
TB.sort(reverse=True); xs_ = np.arange(1, len(TB) + 1); h.bar(xs_, [max(t[0], .02) for t in TB], width=1.0, color=[BLUE if t[1] == "repair_records" else ORANGE for t in TB], lw=0, zorder=2, bottom=0)
gaps = [("Rows", 100 * (mf("rf", "row_random") - mf("rf", "official")), RED, "-"), ("Windows", 100 * (mf("rf", "window_random") - mf("rf", "window_run")), PURPLE, "-"), ("Labels", 100 * abs(mf("rf", "official", "eval_onset_labels") - mf("rf", "official", "eval_file_name_labels")), DARK, DASH)]
for n, v, col, ls in gaps: h.axhline(v, color=col, lw=1.2, ls=ls, zorder=4)
science["protocol_vs_data"] = dict(changes=[t[0] for t in TB], gaps={n: v for n, v, _, _ in gaps})
h.set_yscale("log"); h.set_ylim(.02, 110); h.set_yticks([.1, 1, 10]); h.set_yticklabels([".1", "1", "10"]); h.minorticks_off(); h.set_xlim(0, len(TB) + 1); h.set_xticks([1, 20, 40, len(TB)]); h.set_xticklabels(["1", "20", "40", str(len(TB))]); h.set_xlabel("Sorted pairs", labelpad=1.5); ylab(h, "Macro-F1 gap (pp)"); frame(h)
key_over(fig, h, [Patch(facecolor=BLUE, label="Repair"), Patch(facecolor=ORANGE, label="Delete")] + [hl(col, None, n, lw=1.2, ls=ls) for n, _, col, ls in gaps], cols=3, size=5.8)
titles(fig, [e, f, j_, h], ["(e) Qwen repair", "(f) Partial repair", "(g) Real errors, CleanML", "(h) Protocol vs. data"])
# third row
j_, i_, k_, l_ = [cell(fig, q, y3) for q in range(4)]
# (i) error rate of a table estimated from a detector and from labeled records, record level, tables with a true rate between 1 and 99 percent
DE = [r for r in jsonl("results_detectors.jsonl") if r.get("unit") == "record" and 0.01 < r.get("true_rate", 0) < 0.99]; science["detectors"] = {}
for m_, n_, col, mk, ls_ in (("corrected", "Corrected", ORANGE, "D", "-"), ("pooled", "Pooled", GRAY, "v", "-"), ("sample", "Sample", BLUE, "o", "-"), ("poststratified", "Post-strat.", RED, "*", "-")):
    ns = sorted({r["n"] for r in DE if r["method"] == m_}); yy = [100 * mean(r["mean_abs_error"] for r in DE if r["method"] == m_ and r["n"] == x) for x in ns]; science["detectors"][n_] = dict(n=ns, mae=yy); line(i_, ns, yy, col, mk, lw=1.5, ms=5.6 if mk == "*" else 3.4)
rawv = 100 * mean(r["mean_abs_error"] for r in DE if r["method"] == "raw"); science["detectors"]["Raw"] = rawv; i_.axhline(rawv, color=DARK, ls=DASH, lw=1.2, zorder=2)
i_.set_xscale("log"); i_.set_yscale("log"); i_.set_xticks([50, 100, 200, 400, 800]); i_.set_xticklabels(["50", "100", "200", "400", "800"], fontsize=7.0); i_.set_yticks([1, 3, 10, 30]); i_.set_yticklabels(["1", "3", "10", "30"]); i_.minorticks_off(); i_.set_ylim(.6, 50); i_.set_xlabel("Labeled records", labelpad=1.5); ylab(i_, "Absolute error (pp)"); frame(i_)
key_over(fig, i_, [hl(DARK, None, "Raw", lw=1.2, ls=DASH), hl(ORANGE, "D", "Corrected", ms=3.0), hl(GRAY, "v", "Pooled", ms=3.0), hl(BLUE, "o", "Sample", ms=3.0), hl(RED, "*", "Post-strat.", ms=4.0)], cols=3, size=5.6)
# (j) large TEP, the same number of records from k independent runs
ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]; RM = jsonl("results_rieth_models.jsonl"); rows = [r for r in Q["runs"] if r.get("arm") == "runs_fixed_records"]; science["runs"] = {}
FAM = [("GB", rows, RED, "*"), ("RF", [r for r in RM if r["model"] == "rf" and r["n"] == 480], BLUE, "o"), ("MLP", [r for r in RM if r["model"] == "mlp" and r["n"] == 480], ORANGE, "D"), ("Lin", [r for r in RM if r["model"] == "linear" and r["n"] == 480], GREEN, "^"), ("KNN", [r for r in RM if r["model"] == "knn" and r["n"] == 480], PURPLE, "s")]; hf = []
for n_, rr, col, mk in FAM:
    kk = [k for k in ks if any(r["k"] == k for r in rr)]
    if kk: base1 = mean(r["macro_f1"] for r in rr if r["k"] == 1); yy = [100 * (mean(r["macro_f1"] for r in rr if r["k"] == k) - base1) for k in kk]; science["runs"][n_] = dict(k=kk, gain=yy, f1_one_run=base1); line(j_, kk, yy, col, mk, lw=1.6, z=4 if n_ == "GB" else 3); hf.append(hl(col, mk, n_, ms=3.0))
j_.set_xscale("log"); j_.set_xticks([1, 10, 100]); j_.set_xticklabels(["1", "10", "100"]); j_.minorticks_off(); j_.set_xlabel("Runs per class", labelpad=1.5); ylab(j_, "Macro-F1 gain (pp)"); frame(j_); short_y(j_); key_over(fig, j_, hf, cols=5, size=5.6)
# (k) human label noise of CIFAR-N, two architectures
NZ = [("cifar10", "aggre_label", "Agg.", BLUE, "o"), ("cifar10", "random_label1", "Rand.", GREEN, "^"), ("cifar10", "worse_label", "Worst", ORANGE, "D"), ("cifar100", "noisy_label", "C100", RED, "*")]
def acc(ds, nz, act, fr, arch="resnet18"): return mean(r["score"] for r in V if r["ds"] == ds and r.get("noise") == nz and r.get("action") == act and abs(r.get("f", -1) - fr) < 1e-9 and r["arch"] == arch)
def clean(ds, arch="resnet18"): return mean(r["score"] for r in V if r["ds"] == ds and "noise" not in r and "coverage" not in r and "scale" not in r and r["arch"] == arch)
hs = []; fr = [0, .25, .5, .75, 1]; science["noise"] = {}
for ds, nz, n, col, mk in NZ:
    k0 = acc(ds, nz, "keep", 0); yr = [0.0] + [100 * (acc(ds, nz, "repair", x) - k0) for x in fr[1:4]] + [100 * (clean(ds) - k0)]; yd = [0.0, 100 * (acc(ds, nz, "delete", .5) - k0), 100 * (acc(ds, nz, "delete", 1.0) - k0)]; science["noise"][n] = dict(keep=k0, repair=yr, delete=yd)
    line(k_, fr, yr, col, mk, lw=1.8); k_.plot([0, .5, 1], yd, color=col, marker=mk, ms=3.6, lw=1.1, ls=DASH, mfc="white", mec=col, mew=.8, zorder=2); hs.append(hl(col, mk, n, ms=3.2))
    v0 = acc(ds, nz, "keep", 0, "vgg11"); yv = [0.0] + [100 * (acc(ds, nz, "repair", x, "vgg11") - v0) for x in fr[1:4]] + [100 * (clean(ds, "vgg11") - v0)]; science["noise"][n + "/vgg"] = dict(keep=v0, repair=yv)
k_.set_xticks([0, .5, 1]); k_.set_xticklabels(["0", "50%", "100%"]); k_.set_xlabel("Label errors handled", labelpad=1.5); ylab(k_, "Accuracy gain (pp)"); frame(k_); short_y(k_); key_over(fig, k_, hs + [hl(DARK, None, "Repair", lw=1.8), hl(DARK, None, "Delete", lw=1.1, ls=DASH)], cols=3, size=6.0)
# (l) coverage against scale on CIFAR-100 superclasses, two architectures
def cv(k, fld, arch): return mean(r.get(fld) for r in V if r.get("coverage") == k and r["arch"] == arch)
K = [4, 3, 2, 1]; sh = [.8, .6, .4, .2]; science["coverage"] = {}
SC = sorted({r["scale"] for r in V if r.get("scale") is not None}, reverse=True)
for arch, ls, hollow in (("resnet18", "-", False), ("vgg11", DASH, True)):
    seen_ = [cv(k, "score_seen_sub", arch) for k in K]; un = [cv(k, "score_unseen_sub", arch) for k in K]; rnd = [mean(r["score"] for r in V if r.get("scale") == x and r["arch"] == arch) for x in SC]; science["coverage"][arch] = dict(share=sh, seen=seen_, unseen=un, scale=SC, random=rnd)
    for xx, yy, col, mk, ms, lw in ((sh, seen_, BLUE, "o", 3.8, 1.6), (SC, rnd, GREEN, "s", 3.4, 1.6), (sh, un, RED, "*", 6.4, 1.9)):
        l_.plot(xx, yy, color=col, marker=mk, ms=ms if not hollow else 3.0, lw=lw if not hollow else 1.0, ls=ls, mec=col if hollow else "white", mfc="white" if hollow else col, mew=.7 if hollow else .25, zorder=4 if not hollow else 3)
l_.invert_xaxis(); l_.set_xticks([.8, .6, .4, .2]); l_.set_xticklabels([".8", ".6", ".4", ".2"]); l_.set_xlabel("Kept share", labelpad=1.5); ylab(l_, "Accuracy"); l_.set_ylim(.2, 1); frame(l_)
key_over(fig, l_, [hl(BLUE, "o", "Seen"), hl(RED, "*", "Unseen", ms=5.0), hl(GREEN, "s", "Random")], cols=3, size=6.4, dx=-.02)
titles(fig, [j_, i_, k_, l_], ["(i) Independent runs", "(j) Error-rate estimates", "(k) Real label noise", "(l) Coverage, scale"]); label_room(fig)
print("A", save(fig, f"{OUT}/results_a")); json.dump(science, open(f"{OUT}/figure_data_a.json", "w"), indent=0, default=float)
# ======== figure B: choosing a version and large real data
fine = jsonl("results_decision_fine.jsonl"); D = jsonl("results_decision.jsonl"); D = fine if len(fine) >= len(D) else D
X, X2, X3, CO = load("results_taxi.json"), load("results_taxi_100k.json"), load("results_taxi_sizes.json"), jsonl("results_corpus.jsonl"); sci2 = {}
PH = PH_B; H = 3.47; fig = plt.figure(figsize=(W, H), facecolor="white"); y1, y2 = 2.12, .34
a, d, e, f = [cell(fig, q, y1) for q in range(4)]
D = [r for r in D if "test" in r]; SZ_ = sorted({int(x) for r in D for x in r["val"] if int(x) <= 1600}); GD = collections.defaultdict(dict)
for r in D: GD[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r
def replay(group, model=None):
    """A table with fewer validation records than n uses all it has, so that every point covers the same tables."""
    out = {"static": [], "oracle": [], **{n: [] for n in SZ_}}
    for k, arms in GD.items():
        if k[0] not in group or (model and k[1] != model) or ("keep", 0.0) not in arms or len(arms) < 14: continue
        base_ = arms[("keep", 0.0)]["test"]; P_ = lambda r: r["rate_missing"] + r["rate_domain"] + r["rate_value"]; st_ = min(arms.values(), key=lambda r: (P_(r), r["rate_label"], -r["n_train"])); out["static"].append(st_["test"] - base_); out["oracle"].append(max(r["test"] for r in arms.values()) - base_)
        for n in SZ_:
            def v(r):
                av = sorted(int(x) for x in r["val"]); use = max([x for x in av if x <= n] or av[:1]); return r["val"][str(use)]
            out[n].append(max(arms.values(), key=v)["test"] - base_)
    return out
CLS = {r["ds"] for r in D if r.get("task") == "cls"}; REG = {r["ds"] for r in D if r.get("task") == "reg"}
for ax, grp, lab in ((a, CLS, "cls"),):
    o = replay(grp); s0 = 100 * mean(o["static"]); b0 = 100 * mean(o["oracle"]) - s0; yy = [100 * mean(o[n]) - s0 for n in SZ_]; sci2["decision_" + lab] = dict(n=SZ_, gain_over_static=yy, static=s0, oracle_over_static=b0, cases=len(o["static"]), models={})
    ax.axhline(0, color=DARK, ls=DASH, lw=1.1, zorder=2); ax.axhline(b0, color=DARK, ls=":", lw=1.0, zorder=2)
    for m, n, col, mk in MODS:
        om = replay(grp, m); sm = 100 * mean(om["static"]); ym = [100 * mean(om[n_]) - sm for n_ in SZ_]; sci2["decision_" + lab]["models"][n] = dict(gain_over_static=ym, static=sm, oracle_over_static=100 * mean(om["oracle"]) - sm); line(ax, SZ_, ym, col, mk, lw=1.1, ms=4.6 if mk == "*" else 2.6)
    line(ax, SZ_, yy, DARK, None, lw=2.0, z=5)
    ax.set_xscale("log"); ax.set_xticks([10, 100, 1000] if SZ_[0] < 25 else [25, 100, 400, 1600]); ax.set_xticklabels(["10", "100", "1000"] if SZ_[0] < 25 else ["25", "100", "400", "1600"]); ax.minorticks_off(); ax.set_xlabel("Validation records", labelpad=1.5); ylab(ax, "Gain over static (pp)"); frame(ax)
    ax.set_yscale("symlog", linthresh=1, linscale=1.3); ax.set_ylim(-3.2, 16); ax.set_yticks([-2, 0, 1, 10]); ax.set_yticklabels(["−2", "0", "1", "10"]); ax.minorticks_off()
    # regression tables on the same axis, the mean and the family that carries it
    o = replay(REG); sr = 100 * mean(o["static"]); yr = [100 * mean(o[n_]) - sr for n_ in SZ_]; om = replay(REG, "mlp"); sm = 100 * mean(om["static"]); ym = [100 * mean(om[n_]) - sm for n_ in SZ_]; sci2["decision_reg"] = dict(n=SZ_, gain_over_static=yr, mlp=ym, static=sr, cases=len(o["static"]))
    ax.plot(SZ_, yr, color=DARK, lw=1.3, ls=DASH, zorder=4); ax.plot(SZ_, ym, color=ORANGE, lw=1.0, ls=DASH, zorder=3)
    key_over(fig, ax, [hl(col, mk, n, ms=3.0) for _, n, col, mk in MODS] + [hl(DARK, None, "Mean", lw=1.8), hl(DARK, None, "Reg.", lw=1.2, ls=DASH), hl(DARK, None, "Best", lw=1.0, ls=":")], cols=4, size=5.4)
# (c) taxi, rule violations per month, every rule
keys = sorted(X["months"]); idx = {k: i for i, k in enumerate(keys)}; xm = [2019 + i / 12 for i in range(len(keys))]; RU = [("r_location", "Location", BLUE), ("r_passengers", "Passengers", ORANGE), ("r_ratecode", "Rate code", GREEN), ("r_distance", "Distance", PURPLE), ("r_duration", "Duration", TEAL), ("r_fare", "Fare", BROWN), ("r_speed", "Speed", PINK), ("any_rule", "Any rule", RED)]; sci2["rules"] = {}
# (b) taxi, mean R2 on two fixed later test periods against the training month, raw records, mean over seeds. A training month enters a period only when it precedes every test month of the period
TMS = sorted({r["train_month"] for r in X["runs"] if r["arm"] == "raw"}); sci2["taxi_by_train_month"] = {}; ym_ = lambda k: int(k[:4]) + (int(k[5:7]) - 1) / 12; ALLB = []
for n_, test_, col, mk, ls in (("Test 2023", [f"2023-{q:02d}" for q in range(2, 13)], RED, "*", "-"), ("Test 2024", [f"2024-{q:02d}" for q in range(8, 13)], BLUE, "o", DASH)):
    xx, yy = [], []
    for tm in TMS:
        v = [mean(r["r2"][k] for k in test_) for r in X["runs"] if r["arm"] == "raw" and r["train_month"] == tm and all(k in r["r2"] and k > tm for k in test_)]
        if v: xx.append(ym_(tm)); yy.append(mean(v))
    sci2["taxi_by_train_month"][n_] = dict(train=[t for t in TMS if ym_(t) in xx], test=test_, r2=yy); ALLB += yy; line(d, xx, yy, col, mk, lw=1.5, ls=ls)
d.axvline(ym_("2020-03"), color=GRAY, ls=":", lw=1.0, zorder=1); d.text(ym_("2020-03") - .16, .04, "2020-03", transform=d.get_xaxis_transform(), fontsize=6.4, color=DARK, ha="right", va="bottom")
d.set_xlim(2018.2, 2024.9); d.set_xticks([2019, 2021, 2023]); d.set_xticklabels(["2019", "2021", "2023"]); d.set_xlabel("Train month", labelpad=1.5); ylab(d, "Mean R\u00b2"); frame(d); zoom(d, ALLB); short_y(d)
key_over(fig, d, [hl(RED, "*", "Test 2023", ms=4.4), hl(BLUE, "o", "Test 2024", ls=DASH)], cols=2, size=6.4)
# second row
g, h, k_, l_ = [cell(fig, q, y2) for q in range(4)]
# (e) monthly change of the violation rate against monthly change of R2, the models trained before the first large change
fx = [r for r in X["runs"] if r["arm"] == "raw" and r["train_month"] <= "2020-01"]; last = max(r["train_month"] for r in fx); ks_ = [k for k in keys if k > last and all(k in r["r2"] for r in fx)]; mr = {k: mean(r["r2"][k] for r in fx) for k in ks_}; ar = {k: 100 * X["months"][k]["any_rule"] for k in keys}
tx_ = [abs(ar[q] - ar[p]) for p, q in zip(ks_, ks_[1:])]; ty_ = [abs(mr[q] - mr[p]) for p, q in zip(ks_, ks_[1:])]; sci2["trigger"] = dict(month=ks_[1:], dviol=tx_, dr2=ty_)
for x_, y_ in zip(tx_, ty_):
    above = x_ > 2; e.plot(x_, y_, ls="none", marker="o", ms=4.2 if above else 3.0, color=RED if above else BLUE, mec="white", mew=.3, zorder=4 if above else 3)
for m_, x_, y_ in zip(ks_[1:], tx_, ty_):
    if m_ in ("2020-04", "2024-03"): e.annotate(m_, (x_, y_), xytext=(2, -5) if y_ > .2 else (9, 28), textcoords="offset points", ha="right", va="top" if y_ > .2 else "bottom", fontsize=6.0, color=DARK, arrowprops=None if y_ > .2 else dict(arrowstyle="-", lw=.5, color=DARK, shrinkA=1, shrinkB=2.5))
e.axvline(2, color=DARK, ls=DASH, lw=.9, zorder=2); e.axhline(.05, color=DARK, ls=DASH, lw=.9, zorder=2); e.set_xscale("symlog", linthresh=1, linscale=.6); e.set_xticks([0, 1, 2, 5, 10]); e.set_xticklabels(["0", "1", "2", "5", "10"]); e.set_xlim(-.3, 16); e.minorticks_off(); e.set_xlabel("Violation change (pp)", labelpad=1.5); ylab(e, "R\u00b2 change"); frame(e); short_y(e)
key_over(fig, e, [Line2D([], [], ls="none", marker="o", ms=4.0, color=RED, label="Above 2 pp"), Line2D([], [], ls="none", marker="o", ms=3.0, color=BLUE, label="Other months")], cols=2, size=6.4)
# (f) taxi, R2 against the months after training for three training sizes, raw and rule-filtered, the same six training months at every horizon
SZT = [(10000, "10k", RED, "*"), (100000, "100k", BLUE, "o"), (3000000, "3M", GREEN, "^")]; RUNS = []; sci2["taxi_size"] = {}; ALLY = []
for src in (X3, X2, X):
    if src: RUNS += src["runs"]
HZ = [1, 3, 6, 9, 12, 18, 24, 30, 36]; TM = ("2019-01", "2019-07", "2020-01", "2020-07", "2021-01", "2021-07")
for s_, n, col, mk in SZT:
    for arm, ls, lw_, mk_ in (("raw", "-", 1.5, mk), ("rule_filtered", DASH, 1.0, None)):
        xx, yy = [], []
        for hz_ in HZ:
            v = [r["r2"].get(keys[idx[r["train_month"]] + hz_]) for r in RUNS if r.get("requested_size", r.get("size")) == s_ and r["arm"] == arm and r["train_month"] in TM and idx[r["train_month"]] + hz_ < len(keys)]; v = [q for q in v if q is not None]
            if v: xx.append(hz_); yy.append(st.mean(v))
        sci2["taxi_size"][f"{n}/{arm}"] = dict(months=xx, r2=yy); ALLY += yy; line(f, xx, yy, col, mk_, lw=lw_, ls=ls)
f.set_xticks([1, 12, 24, 36]); f.set_xticklabels(["1", "12", "24", "36"]); f.set_xlabel("Months after training", labelpad=1.5); ylab(f, "R\u00b2"); frame(f); zoom(f, ALLY); short_y(f); key_over(fig, f, [hl(col, mk, n, ms=3.0) for _, n, col, mk in SZT] + [hl(DARK, None, "Raw", lw=1.3), hl(DARK, None, "Rules", lw=1.0, ls=DASH)], cols=3, size=5.8)
# (g) C4 corpus, filters against random selection on three references, in hundredths of the per-token loss, each arm against the random arm of the same seed
BU = 30_000_000; REFS = [("nll_c4", "C4", BLUE, "///"), ("nll_c4_unique", "Unique", GREEN, "---"), ("nll_wiki", "Wiki", ORANGE, "\\\\\\")]
def delta(arm, size, budget, fld):
    runs = {r["seed"]: r[fld] for r in CO if r.get("arm") == "random" and r.get("size") == size and r.get("budget") == budget and fld in r}; v = [100 * (r[fld] - runs[r["seed"]]) for r in CO if r.get("arm") == arm and r.get("size") == size and r.get("budget") == budget and r.get("seed") in runs and fld in r]; return mean(v)
ARM = [(a_, n) for a_, n in [("ppl_low", "Low PPL"), ("ppl_mid", "Mid PPL"), ("ppl_high", "High PPL"), ("rule", "Rules"), ("dedup", "Dedup"), ("rule_dedup", "Both"), ("llm_high", "Qwen high"), ("llm_mid", "Qwen mid")] if np.isfinite(delta(a_, "M", BU, "nll_c4"))]
g.remove(); g = box(fig, XS[0] + .30, y2, .93, PH); pos = np.arange(len(ARM)); sci2["corpus"] = {}
for off, (fld, n, col, hat) in zip((-.26, 0, .26), REFS):
    vals = [delta(a_, "M", BU, fld) for a_, _ in ARM]; sci2["corpus"][n] = vals; g.barh(pos + off, vals, height=.25, color=col, edgecolor=DARK, lw=.25, hatch=hat, zorder=3)
g.axvline(0, color=DARK, lw=.7); g.set_ylim(len(ARM) - .5, -.5); g.set_yticks(pos); g.set_yticklabels([n for _, n in ARM], fontsize=6.4); GAP = "Loss gap (\u00d710\u207b\u00b2)"; g.set_xlabel(GAP, labelpad=1.5); frame(g, "x"); key_over(fig, g, [Patch(facecolor=col, hatch=hat, edgecolor=DARK, label=n) for _, n, col, hat in REFS], cols=3, size=5.8, dx=-.15)
# (h, i) C4 corpus, filter effect by model size and by token budget, web text solid and Wikipedia dashed with hollow markers, so that a budget with one point already shows
LN = [("ppl_low", "Low PPL", RED, "*"), ("ppl_high", "High PPL", GRAY, "v"), ("rule_dedup", "Both", PURPLE, "s")]
SIZES_ = ["XS", "T", "S", "SM", "MS", "M", "ML", "L"]
def versus(ax, axis, name):
    sci2[name] = {}; ally = [0.0]
    if axis == "size":
        PAR = {s_: next((r["params"] for r in CO if r.get("size") == s_ and r.get("params")), None) for s_ in SIZES_}; pts = [(PAR[s_], s_, BU) for s_ in SIZES_ if PAR[s_]]
    else:
        BG = sorted({r["budget"] for r in CO if r.get("size") == "M" and r.get("budget") and r["budget"] <= BU}); pts = [(b_, "M", b_) for b_ in BG]
    for a_, n, col, mk in LN:
        for fld, ls, lw_, hollow in (("nll_c4", "-", 1.5, False), ("nll_wiki", DASH, 1.1, True)):
            xx, yy = [], []
            for x_, s_, b_ in pts:
                v = delta(a_, s_, b_, fld)
                if np.isfinite(v): xx.append(x_ / 1e6); yy.append(v)
            sci2[name][f"{n}/{fld}"] = dict(x=xx, delta=yy); ally += yy; ax.plot(xx, yy, color=col, marker=mk, ms=(5.6 if mk == "*" else 3.4) if not hollow else (4.6 if mk == "*" else 3.0), lw=lw_, ls=ls, mec=col if hollow else "white", mfc="white" if hollow else col, mew=.7 if hollow else .25, zorder=4 if not hollow else 3)
    ax.axhline(0, color=DARK, lw=.7, zorder=1); ax.set_xscale("log"); ax.minorticks_off(); ylab(ax, GAP); frame(ax); zoom(ax, ally); short_y(ax)
NK = [hl(col, mk, n, ms=3.0) for _, n, col, mk in LN] + [hl(DARK, None, "C4", lw=1.3), hl(DARK, None, "Wiki", lw=1.0, ls=DASH), hl(DARK, None, "Random", lw=.7)]
versus(h, "size", "corpus_size"); h.set_xticks([3, 10, 30, 100]); h.set_xticklabels(["3M", "10M", "30M", "100M"]); h.set_xlabel("Model parameters", labelpad=1.5); key_over(fig, h, NK, cols=3, size=5.6)
titles(fig, [a, d, e, f], ["(a) Choice of version", "(b) R\u00b2 by train month", "(c) Violations and R\u00b2", "(d) Training size, time"])
# third row
# (k) published synthetic images, share of synthetic images in the training set, replacing real images or added to them, each run against the real-only run of the same seed
SY = jsonl("results_synth.jsonl"); GENS = [("edm", "EDM", RED, "*"), ("ddpm", "DDPM", BLUE, "o"), ("cifake", "CIFAKE", ORANGE, "D")]; sci2["synthetic"] = {}
REAL = {r["seed"]: r["score"] for r in SY if r.get("gen") == "real" and r["arch"] == "resnet18"}; sci2["synthetic"]["real"] = mean(REAL.values())
def over_real(rr): return mean(100 * (r["score"] - REAL[r["seed"]]) for r in rr if r["seed"] in REAL)
for g_, n, col, mk in GENS:
    rr = [r for r in SY if r.get("gen") == g_ and r["arch"] == "resnet18"]
    sh_ = sorted({r["share"] for r in rr if "share" in r}); ad_ = sorted({r["add"] for r in rr if "add" in r})
    if sh_: yy = [over_real([r for r in rr if r.get("share") == x]) for x in sh_]; sci2["synthetic"][n + "/replace"] = dict(share=sh_, gain=yy); line(k_, sh_, yy, col, mk, lw=1.6)
    if ad_:
        byshare = collections.defaultdict(list)
        for r in rr:
            if "add" in r: byshare[round(r["synthetic_share"], 3)].append(r)
        xx = sorted(byshare); yy = [over_real(byshare[x]) for x in xx]; sci2["synthetic"][n + "/add"] = dict(share=xx, gain=yy); line(k_, xx, yy, col, None, lw=1.1, ls=DASH)
k_.axhline(0, color=DARK, lw=.7, ls=":", zorder=1); k_.set_yscale("symlog", linthresh=1, linscale=1); k_.set_xticks([0, .5, 1]); k_.set_xticklabels(["0", ".5", "1"]); k_.set_xlabel("Synthetic share", labelpad=1.5); ylab(k_, "Accuracy gain (pp)"); frame(k_); k_.set_ylim(-60, 3); k_.set_yticks([-30, -5, -1, 0, 1]); k_.set_yticklabels(["−30", "−5", "−1", "0", "1"]); k_.tick_params(axis="y", labelsize=7.2); k_.minorticks_off(); key_over(fig, k_, [hl(col, mk, n, ms=3.0) for _, n, col, mk in GENS] + [hl(DARK, None, "Real", lw=.9, ls=":")], cols=4, size=5.8)
# (l) native image-text pools, kept fraction by each filter, retrieval on clean COCO pairs, against the random arm at the same fraction and seed
DC = jsonl("results_datacomp.jsonl"); MF = jsonl("results_mirflickr.jsonl"); FILT = [("sim_top", "Top CLIP", RED, "*"), ("sim_mid", "Mid CLIP", BLUE, "o"), ("sim_bottom", "Low CLIP", GRAY, "v"), ("text_rule", "Text rule", ORANGE, "D"), ("dedup", "Dedup", GREEN, "^")]; sci2["image_text"] = {}
for src, tag, ls, lw_, mkon in ((DC, "DataComp", "-", 1.5, True), (MF, "MIRFLICKR", DASH, 1.0, False)):
    rnd = {(r["frac"], r["seed"]): r["test_r5"] for r in src if r.get("arm") == "random"}
    for a_, n, col, mk in FILT:
        rr = [r for r in src if r.get("arm") == a_ and (r.get("frac"), r.get("seed")) in rnd]; fs = sorted({r["frac"] for r in rr})
        if fs: yy = [mean(100 * (r["test_r5"] - rnd[(x, r["seed"])]) for r in rr if r["frac"] == x) for x in fs]; sci2["image_text"][f"{tag}/{n}"] = dict(frac=fs, gain=yy); line(l_, fs, yy, col, mk if mkon else None, lw=lw_, ls=ls)
l_.axhline(0, color=DARK, lw=.7, zorder=1); l_.set_yscale("symlog", linthresh=5, linscale=1); l_.set_ylim(-70, 30); l_.set_yticks([-30, -5, 0, 5, 20]); l_.set_yticklabels(["−30", "−5", "0", "5", "20"]); l_.set_xticks([.1, .3, .5, .7, .9]); l_.set_xticklabels([".1", ".3", ".5", ".7", ".9"]); l_.set_xlabel("Kept share of the pool", labelpad=1.5); ylab(l_, "Recall@5 gain (pp)"); frame(l_); l_.tick_params(axis="y", labelsize=7.2); l_.minorticks_off(); key_over(fig, l_, [hl(col, mk, n, ms=3.0) for _, n, col, mk in FILT] + [hl(DARK, None, "Random", lw=.7)], cols=3, size=5.4, dx=-.04)
titles(fig, [g, h, k_, l_], ["(e) Filters on C4", "(f) Filters, model size", "(g) Synthetic images", "(h) Image-text filters"]); label_room(fig)
print("B", save(fig, f"{OUT}/results_b")); json.dump(sci2, open(f"{OUT}/figure_data_b.json", "w"), indent=0, default=float)
