#!/usr/bin/env python3
"""Statistics behind the cross-experiment findings of the survey. The script reads only the result files and the table data, trains nothing and
draws no random numbers, so every run gives the same output. It writes results/insights_summary.json, and macros(results_dir) returns the values
that the text may cite as {macro name: formatted string}, for make_numbers.py.
Usage: insights.py [RESULTS_DIR]

Sections, in the order of the findings:
  1  static score against task score along the repair path and along the deletion path, the two fully clean versions, and the population
     represented by the records that survive deletion; with results_random_delete.jsonl also deletion of the records with errors against
     deletion of the same number of random records
  2  amount of error against the gain of repair, size of the errors, share of the gain reached at half repair by model family
  3  concentration of the loss of the static rule over the table and model pairs
  4  independent runs against records, split gap at 500 runs, window length under the two splits
  5  dataset-level defects against record-level errors, seen subclasses under low coverage, deletion on image labels
  6  training period of the taxi model and the two rises of the rule violation rate
  7  corpus filters against random sampling, recall of the synthetic images
  8  detector corrections, language-model repair and the language-model judge
  main  the rows of the main table as big_tables.py prints them: versions against the raw data, the task test against the static rule, the
        gaps of the evaluation-practice rows by kind of defect
"""
import collections, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "study")]
TREES, MODELS = ("rf", "gb"), ("linear", "rf", "gb", "knn", "mlp")
SIG = 0.05
SUMMARY_FILE = "insights_summary.json"
SUMMARY_VERSION = 2     # raised whenever compute() adds fields, so that macros() recomputes an older stored summary


# ---------------------------------------------------------------- reading
def _jsonl(R, fn):
    p = os.path.join(R, fn); out = []
    if os.path.exists(p):
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return out


def _load(R, fn):
    p = os.path.join(R, fn); return json.load(open(p)) if os.path.exists(p) else None


def _data_dir(R):
    """The table data sit next to the results folder in the repository."""
    return os.path.join(os.path.dirname(os.path.abspath(R)), "data", "tables")


def static_score(df):
    """Score of the standard for a state of the training set, as in tables_summary.py: three record-level rates and five full items, out of 100."""
    return 100 * ((1 - df.rate_missing) + (1 - df.rate_domain) + (1 - df.rate_value) + 5) / 8


def table_runs(R):
    """Runs of the tabular track, the coarse grid first, the finer fractions added where the coarse grid has no run with the same key."""
    keep = ["ds", "model", "seed", "action", "f", "task", "score", "n_train", "n0", "rate_missing", "rate_domain", "rate_value", "rate_label", "rate_clean_records", "cell_error"]
    parts = [pd.DataFrame([r for r in _jsonl(R, fn) if "score" in r]) for fn in ("results_real_tables.jsonl", "results_tables_fine.jsonl")]
    D = pd.concat([p[keep] for p in parts if len(p)]).drop_duplicates(["ds", "model", "seed", "action", "f"]).reset_index(drop=True)
    D["P"] = static_score(D); return D


def fdr_by(p):
    """Benjamini-Yekutieli adjusted p values, the correction of tables_summary.py."""
    p = np.asarray(p, float); m = len(p)
    if m == 0: return p
    o = np.argsort(p); c = np.sum(1 / np.arange(1, m + 1)); q = np.minimum(1, p[o] * m * c / np.arange(1, m + 1)); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty(m); r[o] = q; return r


def paired_flags(D, base_action="keep"):
    """Each version against the raw table of the same seed, paired t test and Benjamini-Yekutieli over all versions, as in tables_summary.py.
    flag N: significantly lower, P: significantly higher, S: no significant difference."""
    K = D[D.action == base_action].set_index(["ds", "model", "seed"]).score; Pk = D[D.action == base_action].groupby(["ds", "model"]).P.mean(); rows = []
    for (d, m, a, f), g in D[D.action != base_action].groupby(["ds", "model", "action", "f"]):
        x = g.set_index(["ds", "model", "seed"]).score; y = K.reindex(x.index); ok = y.notna(); dif = (x - y)[ok]
        if len(dif) < 3: continue
        p = 1.0 if dif.abs().sum() == 0 else float(stats.ttest_rel(x[ok], y[ok]).pvalue)
        rows.append(dict(ds=d, model=m, action=a, f=f, delta=float(dif.mean()), p=p if np.isfinite(p) else 1.0, dP=float(g.P.mean() - Pk.get((d, m), np.nan))))
    C = pd.DataFrame(rows); C["q"] = fdr_by(C.p); C["flag"] = np.where(C.q >= SIG, "S", np.where(C.delta > 0, "P", "N")); return C


def _sp(x, y):
    r = stats.spearmanr(x, y); return float(r.statistic), float(r.pvalue)


def _tables(R, names=None):
    """Table data as run_real_tables.py reads them, so that errors are defined exactly as in the experiments."""
    import run_real_tables as rt
    root = _data_dir(R); meta = json.load(open(os.path.join(root, "table_meta.json")))
    return {k: rt.load(root, k, m) for k, m in meta.items() if names is None or k in names}


def population_shift(D, keep):
    """Shift between the records in keep and all records, on the clean values. Label: total variation distance of the class shares, or the
    absolute standardized mean difference for regression. Features: the largest standardized mean difference over numeric columns with more than
    10 values, or total variation distance otherwise, over the model features. Returns the label shift, the feature shift and their maximum."""
    b = D["b"]; y = b[D["y"]].astype(str).str.strip()
    if D["task"] == "cls":
        pa = y.value_counts(normalize=True); pk = y[keep].value_counts(normalize=True).reindex(pa.index).fillna(0); ls = float(0.5 * (pa - pk).abs().sum())
    else:
        v = pd.to_numeric(y, errors="coerce"); ls = float(abs(v[keep].mean() - v.mean()) / v.std())
    fs = 0.0
    for c in D.get("model_feat", D["feat"]):
        s = b[c].astype(str).str.strip(); x = pd.to_numeric(s, errors="coerce")
        if x.notna().mean() > 0.95 and s.nunique() > 10: d = float(abs(x[keep].mean() - x.mean()) / x.std()) if x.std() > 0 else 0.0
        else:
            pa = s.value_counts(normalize=True); pk = s[keep].value_counts(normalize=True).reindex(pa.index).fillna(0); d = float(0.5 * (pa - pk).abs().sum())
        if np.isfinite(d): fs = max(fs, d)
    return ls, fs, max(ls, fs)


# ---------------------------------------------------------------- 1. repair and deletion
def sec1(R, D, TD):
    out = {}
    G = D.groupby(["ds", "model", "action", "f"]).agg(score=("score", "mean"), P=("P", "mean"), n=("n_train", "mean"), n0=("n0", "mean")).reset_index()
    rows = []
    for (d, m), g in G.groupby(["ds", "model"]):
        k = g[g.action == "keep"]
        for a in ("repair_records", "delete_records"):
            p = pd.concat([k, g[g.action == a]]).sort_values("f")
            if len(p) >= 5 and p.P.nunique() > 2:
                rows.append(dict(ds=d, model=m, path=a, rho=_sp(p.P, p.score)[0], dP=float(p.P.iloc[-1] - p.P.iloc[0])))
    S = pd.DataFrame(rows); rep = S[S.path == "repair_records"]; dele = S[(S.path == "delete_records") & (S.dP > 1)]
    out["path"] = dict(repair_n=len(rep), repair_med=float(rep.rho.median()), repair_tables=int(rep.ds.nunique()), delete_n=len(dele), delete_med=float(dele.rho.median()),
                       delete_tables=int(dele.ds.nunique()), delete_neg=int((dele.rho < 0).sum()), repair_neg=int((rep.rho < 0).sum()))
    # versions that raise the static score, and whether the task score drops significantly, on the coarse grid that tables_summary.py tests
    A = pd.DataFrame([r for r in _jsonl(R, "results_real_tables.jsonl") if "score" in r]); A["P"] = static_score(A)
    C = paired_flags(A); up = C[C.dP > 1e-6]; ur = up[up.action.str.startswith("repair")]; ud = up[up.action == "delete_records"]
    out["score_up"] = dict(n=len(up), repair_n=len(ur), repair_down=int((ur.flag == "N").sum()), repair_down_pct=float(100 * (ur.flag == "N").mean()),
                           delete_n=len(ud), delete_down=int((ud.flag == "N").sum()), delete_down_pct=float(100 * (ud.flag == "N").mean()))
    # the two fully clean versions
    def get(a, f): return G[(G.action == a) & (G.f == f)].set_index(["ds", "model"])
    k, r, d = get("keep", 0.0), get("repair_records", 1.0), get("delete_records", 1.0)
    T = pd.DataFrame(dict(keep=k.score, rep=r.score, dele=d.score, Pr=r.P, Pd=d.P, nd=d.n, n0=k.n0)).dropna(subset=["dele"])
    gap = T.rep - T.dele
    out["full"] = dict(n=len(T), tables=int(T.index.get_level_values(0).nunique()), P_min=float(min(T.Pr.min(), T.Pd.min())), med_abs_gap=float(gap.abs().median()),
                       over05=int((gap.abs() > .05).sum()), delete_better=int((gap < 0).sum()), mean_gap=float(gap.mean()))
    # the population represented by the records that survive deletion, per table, on the full table and the clean values
    tab = []
    for ds in sorted(T.index.get_level_values(0).unique()):
        X = TD[ds]; rec = X["E"].any(axis=1) | X["yE"]; keep = ~rec; ls, fs, sh = population_shift(X, keep); g = T.loc[ds]
        tab.append(dict(ds=ds, shift=sh, shift_label=ls, shift_feature=fs, gap=float((g.rep - g.dele).mean()), deleted=float(1 - (g.nd / g.n0).mean()), kept=int(keep.sum())))
    S1 = pd.DataFrame(tab); rs, ps = _sp(S1["shift"], S1.gap); rd, pd_ = _sp(S1.deleted, S1.gap)
    out["shift"] = dict(n=len(S1), rho=rs, p=ps, rho_deleted=rd, p_deleted=pd_, tables=S1.round(4).to_dict("records"))
    # Bike: records without errors on the full table
    if "bike" in TD:
        X = TD["bike"]; keep = ~(X["E"].any(axis=1) | X["yE"]); b = X["b"]; wd = pd.to_numeric(b["workingday"], errors="coerce"); mn = pd.to_numeric(b["mnth"], errors="coerce")
        out["bike"] = dict(kept=int(keep.sum()), n=int(len(keep)), workday_kept=float(100 * (wd[keep] == 1).mean()), workday_all=float(100 * (wd == 1).mean()),
                           months_8_12_kept=float(100 * mn[keep].between(8, 12).mean()), months_kept=sorted(int(x) for x in mn[keep].dropna().unique()))
    if "soccer_full" in TD:
        X = TD["soccer_full"]; keep = ~(X["E"].any(axis=1) | X["yE"]); y = X["b"][X["y"]].astype(str).str.strip(); share = (y[keep].value_counts() / y.value_counts()).reindex(y.value_counts().index).fillna(0).sort_values()
        out["soccer"] = dict(classes=int(len(share)), lowest=[float(100 * v) for v in share.iloc[:3]])
    # CleanML Titanic, deleting the records with missing values against the dirty table
    Cm = [r for r in _jsonl(R, "results_cleanml.jsonl") if r.get("ds") == "Titanic" and r.get("error_type") == "missing_values" and "test" in r]
    if Cm:
        base = {(r["model"], r["seed"]): r["test"] for r in Cm if r["method"] == "dirty"}; dl = [r for r in Cm if r["method"] == "delete" and (r["model"], r["seed"]) in base]
        out["titanic"] = dict(drop_pp=float(100 * np.mean([base[(r["model"], r["seed"])] - r["test"] for r in dl])), records=int(dl[0]["records"] + dl[0]["records_removed"]), removed=int(dl[0]["records_removed"]))
    return out, G, C


def sec1_random(R, D, C):
    """Deletion of the records with errors against deletion of the same number of random training records, from results_random_delete.jsonl."""
    rows = [r for r in _jsonl(R, "results_random_delete.jsonl") if "score" in r]
    if not rows: return None
    X = pd.DataFrame(rows); X["P"] = static_score(X); out = dict(runs=len(X), errors=len([r for r in _jsonl(R, "results_random_delete.jsonl") if "error" in r]))
    # the rerun of delete_records repeats run_real_tables.py, so its scores must equal the stored ones
    ref = D.set_index(["ds", "model", "seed", "action", "f"]).score; x = X[X.action == "delete_records"].set_index(["ds", "model", "seed", "action", "f"]).score; common = x.index.intersection(ref.index)
    out["rerun_pairs"] = int(len(common)); out["rerun_max_abs_diff"] = float((x[common] - ref[common]).abs().max()) if len(common) else None
    # significance against the raw table of the same seed, with the keep runs of the main grid, corrected together with all other versions as in tables_summary.py
    K = D[D.action == "keep"]; seeds = set(X.seed); both = pd.concat([K[K.seed.isin(seeds)], X.assign(action=X.action.where(X.action == "delete_records", "random_delete"))[K.columns.intersection(X.columns)]])
    F = paired_flags(both); F = F[F.action.isin(["delete_records", "random_delete"])]
    res = {}
    for frac in sorted(X.fraction.unique()):
        Ff = F[F.f == frac]; d_ = Ff[Ff.action == "delete_records"].set_index(["ds", "model"]); r_ = Ff[Ff.action == "random_delete"].set_index(["ds", "model"])
        G = X[X.fraction == frac].pivot_table(index=["ds", "model", "seed"], columns="action", values="score").dropna()
        pairs = []
        for (ds, m), g in G.groupby(level=[0, 1]):
            dif = g["delete_records"] - g["random_delete"]; p = 1.0 if dif.abs().sum() == 0 else float(stats.ttest_rel(g["delete_records"], g["random_delete"]).pvalue)
            pairs.append(dict(ds=ds, model=m, diff=float(dif.mean()), p=p if np.isfinite(p) else 1.0))
        Pp = pd.DataFrame(pairs); Pp["q"] = fdr_by(Pp.p)
        sh = X[X.fraction == frac].groupby(["ds", "action"])["shift"].mean().unstack()
        res[str(frac)] = dict(pairs=len(Pp), tables=int(Pp.ds.nunique()), selective_down=int((d_.flag == "N").sum()), selective_down_pct=float(100 * (d_.flag == "N").mean()),
                              random_down=int((r_.flag == "N").sum()), random_down_pct=float(100 * (r_.flag == "N").mean()), selective_worse=int((Pp["diff"] < 0).sum()),
                              selective_worse_pct=float(100 * (Pp["diff"] < 0).mean()), selective_worse_sig=int(((Pp.q < SIG) & (Pp["diff"] < 0)).sum()),
                              selective_better_sig=int(((Pp.q < SIG) & (Pp["diff"] > 0)).sum()), med_diff=float(Pp["diff"].median()), mean_diff=float(Pp["diff"].mean()),
                              shift_selective=float(sh["delete_records"].median()), shift_random=float(sh["random_delete"].median()),
                              shift_by_table=sh.round(4).to_dict("index"), diff_by_table=Pp.groupby("ds")["diff"].mean().round(4).to_dict(),
                              n_train_deleted=X[(X.fraction == frac) & (X.action == "delete_records")].groupby("ds").n_train.mean().round(1).to_dict(),
                              n_train_all=X[X.fraction == frac].groupby("ds").n0.mean().round(1).to_dict())
        # does the gap between the two deletions follow the gap of their shifts, per table
        t = pd.DataFrame(dict(dd=Pp.groupby("ds")["diff"].mean(), ds_=sh["delete_records"] - sh["random_delete"])).dropna()
        if len(t) >= 4: res[str(frac)]["rho_shift_gap"], res[str(frac)]["p_shift_gap"] = _sp(-t.dd, t.ds_)
    out["by_fraction"] = res; return out


# ---------------------------------------------------------------- 2. amount and effect of errors
def sec2(R, D, G, TD):
    out = {}; T = load_tables_summary(R)
    def get(a, f): return G[(G.action == a) & (G.f == f)].set_index(["ds", "model"])
    k, r = get("keep", 0.0), get("repair_records", 1.0)
    g = pd.DataFrame(dict(gain=r.score - k.score, P=k.P)).groupby(level=0).mean()
    g10 = g.drop(index=[d for d in g.index if g.loc[d, "P"] >= 100 - 1e-9])   # tables whose static score already is 100 carry only label errors
    rho, p = _sp(100 - g10.P, g10.gain); out["amount_gain"] = dict(n=len(g10), rho=rho, p=p, excluded=sorted(set(g.index) - set(g10.index)))
    for ds in ("nasa", "mercedes"):
        if ds in g.index: out[ds] = dict(P=float(g.loc[ds, "P"]), gain=float(g.loc[ds, "gain"]))
    # size of the errors in two NASA columns, in standard deviations of the clean column
    if "nasa" in TD:
        X = TD["nasa"]; mags = {}
        for c in ("chord_length", "thickness"):
            j = X["feat"].index(c); a_ = pd.to_numeric(X["a"][c], errors="coerce").values; b_ = pd.to_numeric(X["b"][c], errors="coerce").values; e = X["E"][:, j] & np.isfinite(a_) & np.isfinite(b_)
            mags[c] = float(np.median(np.abs(a_[e] - b_[e])) / np.nanstd(b_, ddof=1))
        out["nasa_magnitude"] = mags
    # NASA, linear model, repair of 90 percent of the records with errors against all
    x = G[(G.ds == "nasa") & (G.model == "linear")].set_index(["action", "f"])
    if ("repair_records", 0.9) in x.index:
        b0 = x.loc[("keep", 0.0), "score"]; out["nasa_linear"] = dict(gain90=float(x.loc[("repair_records", 0.9), "score"] - b0), gain100=float(x.loc[("repair_records", 1.0), "score"] - b0), P90=float(x.loc[("repair_records", 0.9), "P"]))
    # share of the full gain reached by repairing half of the records with errors, pairs with a full gain of at least 0.02
    h = get("repair_records", 0.5); H = pd.DataFrame(dict(full=r.score - k.score, half=h.score - k.score)).dropna(); H = H[H.full >= .02]; H["share"] = H.half / H.full
    fam = H.index.get_level_values(1).map(lambda m: "tree" if m in TREES else m)
    out["half"] = dict(n=len(H), **{f"{f}_share": float(100 * H.share[fam == f].median()) for f in ("tree", "linear", "mlp", "knn") if (fam == f).any()}, **{f"{f}_n": int((fam == f).sum()) for f in ("tree", "linear", "mlp", "knn")})
    # interactions of the error classes, by model family
    if T and T.get("interaction_list"):
        I = pd.DataFrame(T["interaction_list"]); s = I[I.q < SIG]
        out["interaction"] = dict(sig=len(s), tree_sig=int(s.model.isin(TREES).sum()), tree_ant=int((s.model.isin(TREES) & (s.inter < 0)).sum()), syn=int((s.inter > 0).sum()),
                                  syn_models=sorted(s[s.inter > 0].model.unique().tolist()), syn_on_trees=int((s.model.isin(TREES) & (s.inter > 0)).sum()))
    return out


def load_tables_summary(R): return _load(R, "tables_summary.json")


# ---------------------------------------------------------------- 3. loss of the static rule
def decision_gain(R, n_top=1600):
    """Gain of the task test over the static rule on the classification tables, in points, by validation size and by model, replayed as
    summaries.py and Figure 5(a) do: a table with fewer validation records than n uses all it has."""
    rows = [r for r in _jsonl(R, "results_decision_fine.jsonl") if "test" in r] or [r for r in _jsonl(R, "results_decision.jsonl") if "test" in r]
    if not rows: return None
    G = collections.defaultdict(dict)
    for r in rows: G[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r
    sizes = sorted({int(x) for r in rows for x in r["val"] if int(x) <= n_top})
    def val(r, n):
        av = sorted(int(x) for x in r["val"]); return r["val"][str(max([x for x in av if x <= n] or av[:1]))]
    P = lambda r: r["rate_missing"] + r["rate_domain"] + r["rate_value"]
    per = collections.defaultdict(lambda: {"static": [], **{n: [] for n in sizes}})
    for k, arms in G.items():
        if ("keep", 0.0) not in arms or len(arms) < 14 or arms[("keep", 0.0)]["task"] != "cls": continue
        base = arms[("keep", 0.0)]["test"]; s0 = min(arms.values(), key=lambda r: (P(r), r["rate_label"], -r["n_train"]))
        for grp in ("all", k[1]):
            per[grp]["static"].append(s0["test"] - base)
            for n in sizes: per[grp][n].append(max(arms.values(), key=lambda r: val(r, n))["test"] - base)
    gain = {g: {str(n): float(100 * (np.mean(v[n]) - np.mean(v["static"]))) for n in sizes} for g, v in per.items()}
    return dict(sizes=sizes, gain=gain)


def sec3(R):
    rows = [r for r in _jsonl(R, "results_decision_fine.jsonl") if "test" in r]
    if not rows: return None
    G = collections.defaultdict(dict)
    for r in rows: G[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r
    def val(r, n):
        av = sorted(int(x) for x in r["val"]); return r["val"][str(max([x for x in av if x <= n] or av[:1]))]
    out = []
    for k, arms in G.items():
        if ("keep", 0.0) not in arms or len(arms) < 14: continue
        s0 = min(arms.items(), key=lambda kv: (kv[1]["rate_missing"] + kv[1]["rate_domain"] + kv[1]["rate_value"], kv[1]["rate_label"], -kv[1]["n_train"]))
        best = max(r["test"] for r in arms.values()); t100 = max(arms.values(), key=lambda r: val(r, 100))["test"]
        out.append(dict(ds=k[0], model=k[1], static=s0[1]["test"], static_arm=s0[0], best=best, t100=t100))
    X = pd.DataFrame(out); X["regret"] = X.best - X.static; X["reg_t"] = X.best - X.t100
    C = X.groupby(["ds", "model"])[["regret", "reg_t"]].mean().reset_index().sort_values("regret", ascending=False); tot = C.regret.sum(); d = C.regret - C.reg_t
    T = load_tables_summary(R); top = C.head(3)
    return dict(pairs=len(C), small=int((C.regret < .01).sum()), top3_share=float(100 * top.regret.sum() / tot), top3=[dict(ds=a, model=b, regret=float(c), P=float(T["base"][a]["P_data"]) if T else None) for a, b, c in top[["ds", "model", "regret"]].values],
                cases=len(X), static_is_full_repair=int((X.static_arm == ("repair_records", 1.0)).sum()), task_better=int((d > .002).sum()), task_worse=int((d < -.002).sum()), task_tie=int((d.abs() <= .002).sum()),
                same_static={k: T["same_static"][k] for k in ("n", "ge05")} if T and "same_static" in T else None, decision=decision_gain(R))


# ---------------------------------------------------------------- 4. runs and records
def sec4(R):
    out = {}; Q = _load(R, "results_rieth.json")
    if Q:
        rr = Q["runs"]
        def a(arm, k, n): return float(np.mean([r["macro_f1"] for r in rr if r["arm"] == arm and r.get("k") == k and r.get("n") == n]))
        n0 = min(r["n"] for r in rr if r["arm"] == "runs_fixed_records")
        out["rieth"] = dict(per_class=n0 // 21, run1=a("runs_fixed_records", 1, n0), run5=a("runs_fixed_records", 5, n0), gain_runs=a("runs_fixed_records", 5, n0) - a("runs_fixed_records", 1, n0),
                            gain_rec5x4=a("records_fixed_runs", 5, 4 * n0) - a("records_fixed_runs", 5, n0), gain_rec500x16=a("records_fixed_runs", 500, 16 * n0) - a("records_fixed_runs", 500, n0))
        sp = [r for r in rr if r["arm"] == "split_protocol"]
        if sp: out["rieth_split"] = dict(record=float(np.mean([r["record_split"] for r in sp])), run=float(np.mean([r["run_split"] for r in sp])))
    W = _jsonl(R, "results_tep_windows.jsonl")
    if W:
        M = pd.DataFrame(W).groupby(["split", "model", "length"]).macro_f1.mean(); res = {}
        for sp_ in ("random", "run"):
            m = M[sp_]; L = sorted(m.index.get_level_values(1).unique()); lo, hi = L[0], L[-1]
            ch = {md: float(m[(md, hi)] - m[(md, lo)]) for md in m.index.get_level_values(0).unique()}; best = {md: int(m[md].idxmax()) for md in ch}
            res[sp_] = dict(shortest=int(lo), longest=int(hi), change=ch, best=best)
        out["windows"] = res
    return out


# ---------------------------------------------------------------- 5. dataset-level defects
def sec5(R):
    out = {}; V = list({r["id"]: r for r in _jsonl(R, "results_vision.jsonl")}.values())
    c1 = [r for r in V if r.get("coverage") == 1 and r["arch"] == "resnet18"]; c5 = [r for r in V if r.get("coverage") == 5 and r["arch"] == "resnet18"]
    if c1 and c5: out["coverage"] = dict(seen_low=float(np.mean([r["score_seen_sub"] for r in c1])), all_low=float(np.mean([r["score"] for r in c1])), full=float(np.mean([r["score"] for r in c5])))
    def vm(f): x = [r["score"] for r in V if f(r)]; return float(np.mean(x)) if x else None
    out["cifar10n_worse"] = dict(keep=vm(lambda r: r["ds"] == "cifar10" and r.get("noise") == "worse_label" and r.get("action") == "keep" and r["arch"] == "resnet18"),
                                 delete=vm(lambda r: r["ds"] == "cifar10" and r.get("noise") == "worse_label" and r.get("action") == "delete" and r.get("f") == 1.0 and r["arch"] == "resnet18"),
                                 clean=vm(lambda r: r["ds"] == "cifar10" and "noise" not in r and r["arch"] == "resnet18"),
                                 deleted_pct=float(100 * (1 - np.mean([r["retained"] for r in V if r["ds"] == "cifar10" and r.get("noise") == "worse_label" and r.get("action") == "delete" and r.get("f") == 1.0]))))
    # CIFAR-N, every label set and network: gain of deleting all images with a wrong label and of training on the clean labels, in points
    lab = {}
    for ds, nz in sorted({(r["ds"], r["noise"]) for r in V if r.get("noise") and r.get("action") == "keep"}):
        for arch in sorted({r["arch"] for r in V if r["ds"] == ds and r.get("noise") == nz}):
            k0 = vm(lambda r: r["ds"] == ds and r.get("noise") == nz and r.get("action") == "keep" and r["arch"] == arch)
            dl = vm(lambda r: r["ds"] == ds and r.get("noise") == nz and r.get("action") == "delete" and r.get("f") == 1.0 and r["arch"] == arch)
            cl = vm(lambda r: r["ds"] == ds and "noise" not in r and "coverage" not in r and "scale" not in r and r["arch"] == arch)
            if None not in (k0, dl, cl): lab[f"{ds}/{nz}/{arch}"] = dict(delete=float(100 * (dl - k0)), repair=float(100 * (cl - k0)))
    if lab: out["label_sets"] = lab
    # right half of the main table: gaps of the evaluation-protocol rows against the leakage and test-label rows
    import big_tables, datasets
    P = big_tables.protocol_rows(R); grp = {g: [d["key"] for d in ds] for g, ds in datasets.GROUPS}
    def gaps(prefix): return [abs(P[k][0] - P[k][1]) for g, ks in grp.items() if g.startswith(prefix) for k in ks if k in P and None not in P[k]]
    pr, lk = gaps("Evaluation protocol"), gaps("Data leakage") + gaps("Test-set labels")
    out["main_right"] = dict(protocol_n=len(pr), protocol_min=float(min(pr)), protocol_max=float(max(pr)), protocol_med=float(np.median(pr)), record_n=len(lk), record_max=float(max(lk)))
    return out


# ---------------------------------------------------------------- 6. taxi
def sec6(R):
    X = _load(R, "results_taxi.json")
    if not X: return None
    months = sorted(X["months"]); test = [m for m in months if "2023-02" <= m <= "2023-12"]; out = {}
    by = collections.defaultdict(list)
    for r in X["runs"]:
        if r["arm"] == "raw" and r["train_month"] <= "2023-01": by[r["train_month"]].append(np.mean([r["r2"][m] for m in test if m in r["r2"]]))
    out["train_2023"] = {tm: float(np.mean(v)) for tm, v in sorted(by.items())}
    # month to month change of the rule violation rate against the change of the mean score of the models trained up to 2020-01, as in summaries.py
    g = collections.defaultdict(list)
    for r in X["runs"]:
        if r["arm"] == "raw" and r["train_month"] <= "2020-01": g[r["train_month"]].append(r["r2"])
    cv = {tm: {k: np.mean([c[k] for c in cs if k in c]) for k in set().union(*cs)} for tm, cs in g.items()}
    last = max(cv); ks = [k for k in months if k > last and all(k in c for c in cv.values())]; mr = {k: np.mean([c[k] for c in cv.values()]) for k in ks}
    rise = {}
    for a, b in zip(ks, ks[1:]):
        rise[b] = dict(rule_pp=float(100 * (X["months"][b]["any_rule"] - X["months"][a]["any_rule"])), r2_change=float(abs(mr[b] - mr[a])), miss_pp=float(100 * (X["months"][b]["missing_passengers"] - X["months"][a]["missing_passengers"])))
    # the largest rise in the first pandemic year and the largest rise afterwards
    early = max((k for k in rise if k <= "2020-12"), key=lambda k: rise[k]["rule_pp"]); late = max((k for k in rise if k >= "2021-01"), key=lambda k: rise[k]["rule_pp"])
    out["rises"] = {early: rise[early], late: rise[late]}
    # training size: mean R2 over the same test months by training month, arm and number of training records, from every taxi result file
    runs = {}
    for fn in ("results_taxi.json", "results_taxi_100k.json", "results_taxi_sizes.json", "results_taxi_monthly.json"):
        Y = _load(R, fn)
        for r in (Y or {}).get("runs", []):
            n = r.get("requested_size", r.get("size")); runs[(r["train_month"], r["arm"], n, r.get("seed", 0))] = r
    sz = collections.defaultdict(list)
    for (tm, arm, n, _), r in runs.items():
        v = [r["r2"][m] for m in test if m in r["r2"]]
        if n and len(v) == len(test): sz[(tm, arm, int(n))].append(float(np.mean(v)))
    size = {}
    for (tm, arm, n), v in sorted(sz.items()): size.setdefault(tm, {}).setdefault(arm, {})[str(n)] = float(np.mean(v))
    out["size"] = size
    return out


# ---------------------------------------------------------------- 7. corpus filters and synthetic images
def sec7(R):
    out = {}; Co = [r for r in _jsonl(R, "results_corpus.jsonl") if "error" not in r]
    if Co:
        sel = [r for r in Co if r["size"] == "M" and r.get("budget") == 30000000]; arms = sorted({r["arm"] for r in sel} - {"random", "all", "dclm_pool", "dclm_baseline"}); res = {}
        for ref in ("nll_c4", "nll_wiki"):
            b = {r["seed"]: r[ref] for r in sel if r["arm"] == "random"}; rr = {}
            for a in arms:
                pr = [(r[ref], b[r["seed"]]) for r in sel if r["arm"] == a and r["seed"] in b]; x, y = np.array(pr).T
                t = stats.ttest_rel(x, y); one = stats.ttest_rel(x, y, alternative="less")
                rr[a] = dict(diff=float(100 * np.mean(x - y)), p=float(t.pvalue), p_better=float(one.pvalue), n=len(pr))
            res[ref] = rr
        c4 = res["nll_c4"]; better = [a for a in arms if c4[a]["diff"] < 0]
        out["corpus"] = dict(filters=arms, c4_min=min(v["diff"] for v in c4.values()), c4_max=max(v["diff"] for v in c4.values()), c4_min_p_better=min(c4[a]["p"] for a in arms if c4[a]["diff"] < 0), c4_min_p_better_one_sided=min(v["p_better"] for v in c4.values()),
                             c4_better=better, c4_sig_better=int(sum(v["p_better"] < SIG for v in c4.values())), c4_sig_worse=sorted(a for a, v in c4.items() if v["diff"] > 0 and v["p"] < SIG),
                             wiki_min=min(v["diff"] for v in res["nll_wiki"].values()), detail=res)
    F = _load(R, "results_fidelity.json"); Sy = [r for r in _jsonl(R, "results_synth.jsonl") if "error" not in r]
    if F and Sy:
        acc = {"real_test": float(np.mean([r["score"] for r in Sy if r["gen"] == "real" and r["arch"] == "resnet18"]))}
        for gname in ("edm", "ddpm", "cifake"): acc[gname] = float(np.mean([r["score"] for r in Sy if r["gen"] == gname and r["arch"] == "resnet18" and r.get("share") == 1.0]))
        keys = list(acc); rec = [F[k]["recall"] for k in keys]; ac = [acc[k] for k in keys]; den = [F[k]["density"] for k in keys]; pre = [F[k]["precision"] for k in keys]
        out["synth"] = dict(recall={k: float(F[k]["recall"]) for k in keys}, acc=acc, density={k: float(F[k]["density"]) for k in keys}, recall_same_order=bool(np.argsort(rec).tolist() == np.argsort(ac).tolist()),
                            rho_recall=_sp(rec, ac)[0], rho_density=_sp(den, ac)[0], rho_precision=_sp(pre, ac)[0])
    return out


# ---------------------------------------------------------------- 8. measuring tools
def sec8(R):
    out = {}; Dt = [r for r in _jsonl(R, "results_detectors.jsonl") if r.get("unit") == "record" and 0.01 < r.get("true_rate", 0) < 0.99]
    if Dt:
        X = pd.DataFrame(Dt); raw = X[X.method == "raw"].set_index(["ds", "detector"]); J = (raw.se + raw.sp); out_ = dict(pairs=len(raw), tables=int(raw.index.get_level_values(0).nunique()))
        piv = X[X.n > 0].pivot_table(index=["n", "ds", "detector"], columns="method", values="mean_abs_error")
        out_["corrected_below_sample"] = {int(n): int((g.corrected < g["sample"]).sum()) for n, g in piv.groupby(level=0)}
        out_["post_below_sample"] = {int(n): int((g.poststratified < g["sample"]).sum()) for n, g in piv.groupby(level=0)}
        out_["post_above_sample_max_pp"] = float(100 * (piv.poststratified - piv["sample"]).max())
        cov = X[(X.method == "corrected") & (X.n == 50)].set_index(["ds", "detector"]).coverage; z = pd.DataFrame(dict(J=J, cov=cov)).dropna()
        out_["rho_cov_sesp"], out_["p_cov_sesp"] = _sp(z.J, z["cov"]); band = z[(z.J >= 1.7) & (z.J <= 1.95)]; out_["band_n"] = len(band); out_["band_cov"] = float(100 * band["cov"].mean())
        c200 = X[(X.method == "corrected") & (X.n == 200)].set_index(["ds", "detector"]).coverage; top = z.J.idxmax()
        out_["best"] = dict(ds=top[0], detector=top[1], sesp=float(z.J.max()), cov50=float(100 * z["cov"][top]), cov200=float(100 * c200[top]))
        out_["d1"] = {ds: float(J[(ds, "D1")]) for ds in ("smartfactory", "bike") if (ds, "D1") in J.index}
        out["detectors"] = out_
    Lc = _load(R, "results_llm_cells.json"); Lt = [r for r in _jsonl(R, "results_llm_tables.jsonl") if "error" not in r and "test" in r]; Df = [r for r in _jsonl(R, "results_decision_fine.jsonl") if "test" in r]
    if Lc:
        per = {k: 100 * v["cells"]["changed"] / v["rows_read"] for k, v in Lc.items()}
        out["llm"] = dict(min_changed_per100=float(min(per.values())), min_table=min(per, key=per.get), adult_errors_per100=float(100 * Lc["adult"]["cells"]["wrong_before"] / Lc["adult"]["rows_read"]), adult_changed_per100=float(per["adult"]))
        if Lt and Df:
            kp = {(r["ds"], r["model"], r["seed"]): r["test"] for r in Df if r["action"] == "keep"}; ch = collections.defaultdict(list); bym = collections.defaultdict(list)
            for r in Lt:
                k = (r["ds"], r["model"], r["seed"])
                if k in kp: ch[r["ds"]].append(r["test"] - kp[k]); bym[(r["ds"], r["model"])].append(r["test"] - kp[k])
            f1 = {k: (v["cells"]["f1"] or 0.0) for k, v in Lc.items() if k in ch}; dt = {k: float(np.mean(v)) for k, v in ch.items()}; ks = sorted(f1)
            rho, p = _sp([f1[k] for k in ks], [dt[k] for k in ks]); bestf = max(f1, key=f1.get)
            out["llm"].update(rho_f1_gain=rho, p_f1_gain=p, n=len(ks), best_f1_table=bestf, best_f1=float(f1[bestf]), best_f1_models_down=int(sum(np.mean(v) < 0 for (d, m), v in bym.items() if d == bestf)),
                              soil_f1=float(f1.get("soilmoisture", np.nan)), soil_gain=dt.get("soilmoisture"))
    Lj = _load(R, "results_llm_judge.json")
    if Lj: out["judge"] = dict(rho_words=float(Lj["spearman"]["words"]), rho_ppl=float(Lj["spearman"]["ppl"]), median=float(Lj["quantiles"]["0.5"]), q75=float(Lj["quantiles"]["0.75"]), q90=float(Lj["quantiles"]["0.9"]), q99=float(Lj["quantiles"]["0.99"]) if "0.99" in Lj["quantiles"] else None)
    return out


# ---------------------------------------------------------------- main table
SHIFT_SKIP = ("tep", "tepl")     # the two TEP rows of the evaluation-protocol group are a protocol of independent runs, the other rows a shift of coverage or distribution


def sec_main(R):
    """Rows of the main table with the values it prints, from big_tables.main_rows. Versions: the rows of the first seven groups with all
    three scores. Practices: the absolute printed gap by kind of defect, and the static score of the rows outside the detector group."""
    import big_tables, datasets
    rows = big_tables.main_rows(R); gname = [g for g, _ in datasets.GROUPS]; out = {}
    V = [r for r in rows if r["versions"] and None not in r["shown"]]
    def better(a, b, low): return a < b if low else a > b
    below = [r["d"]["name"] for r in V if better(r["shown"][0], r["shown"][1], r["low"])]
    tt = [(round((r["shown"][2] - r["shown"][1]) * (-1 if r["low"] else 1), 3), r["d"]["name"]) for r in V]
    loss = [x for x in tt if x[0] < 0]; gain = [x for x in tt if x[0] > 0]
    out["versions"] = dict(n=len(V), static_below_raw=len(below), static_below_raw_rows=below, task_loss_n=len(loss), task_gain_n=len(gain), task_tie_n=len(tt) - len(loss) - len(gain),
                           task_loss_max=-min(loss)[0] if loss else 0.0, task_loss_max_ds=min(loss)[1] if loss else None, task_gain_max=max(gain)[0] if gain else 0.0, task_gain_max_ds=max(gain)[1] if gain else None)
    P = [r for r in rows if not r["versions"] and r["gap"] is not None]
    def gap(r): return round(abs(r["gap"]), 3)
    shift = [gap(r) for r in P if gname[r["group"]].startswith("Evaluation protocol") and r["d"]["key"] not in SHIFT_SKIP]
    record = [gap(r) for r in P if gname[r["group"]].startswith(("Data leakage", "Test-set labels"))]
    stat = [r["static"] for r in rows if not r["versions"] and not gname[r["group"]].startswith("Detector") and r["static"] is not None]
    out["practices"] = dict(shift_n=len(shift), shift_min=min(shift) if shift else None, shift_max=max(shift) if shift else None, shift_med=float(np.median(shift)) if shift else None,
                            record_n=len(record), record_med=float(np.median(record)) if record else None, static_min=min(stat) if stat else None)
    return out


# ---------------------------------------------------------------- driver
def compute(R):
    D = table_runs(R); TD = _tables(R)
    s1, G, C = sec1(R, D, TD)
    out = {"_version": SUMMARY_VERSION, "1": s1, "1_random": sec1_random(R, D, C), "2": sec2(R, D, G, TD), "3": sec3(R), "4": sec4(R), "5": sec5(R), "6": sec6(R), "7": sec7(R), "8": sec8(R), "main": sec_main(R)}
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _neg(s): return s.replace("-", "$-$") if s.startswith("-") else s
def f1(x): return _neg(f"{x:.1f}")
def f2(x): return _neg(f"{x:.2f}")
def f3(x): return _neg(f"{x:.3f}")
def f0(x): return f"{x:.0f}"
def fp(p): return "0.001" if p < 0.001 else f"{p:.3f}"


def macros_from(S):
    """Formatting follows the manuscript: score differences with three decimals, points and percentages with one, correlations with two."""
    M = {}; a = S["1"]
    M.update(InsPathRepRho=f2(a["path"]["repair_med"]), InsPathRepN=str(a["path"]["repair_n"]), InsPathDelRho=f2(a["path"]["delete_med"]), InsPathDelN=str(a["path"]["delete_n"]),
             InsPathDelTables=str(a["path"]["delete_tables"]), InsPathDelNeg=str(a["path"]["delete_neg"]),
             InsUpN=str(a["score_up"]["n"]), InsUpRepN=str(a["score_up"]["repair_n"]), InsUpRepDown=str(a["score_up"]["repair_down"]), InsUpRepDownPct=f1(a["score_up"]["repair_down_pct"]),
             InsUpDelN=str(a["score_up"]["delete_n"]), InsUpDelDown=str(a["score_up"]["delete_down"]), InsUpDelDownPct=f1(a["score_up"]["delete_down_pct"]),
             InsFullN=str(a["full"]["n"]), InsFullGapMed=f3(a["full"]["med_abs_gap"]), InsFullGapOver=str(a["full"]["over05"]),
             InsShiftRho=f2(a["shift"]["rho"]), InsShiftP=fp(a["shift"]["p"]), InsShiftN=str(a["shift"]["n"]), InsShiftDelRho=f2(a["shift"]["rho_deleted"]), InsShiftDelP=fp(a["shift"]["p_deleted"]))
    if "bike" in a: M.update(InsBikeKept=str(a["bike"]["kept"]), InsBikeWorkKept=f1(a["bike"]["workday_kept"]), InsBikeWorkAll=f1(a["bike"]["workday_all"]))
    if "soccer" in a: M.update(InsSoccerClasses=str(a["soccer"]["classes"]), InsSoccerLowA=f1(a["soccer"]["lowest"][0]), InsSoccerLowB=f1(a["soccer"]["lowest"][1]))
    if "titanic" in a: M.update(InsTitanicDrop=f1(a["titanic"]["drop_pp"]), InsTitanicRecords=str(a["titanic"]["records"]), InsTitanicRemoved=str(a["titanic"]["removed"]))
    r = S.get("1_random")
    if r:
        for frac, tag in (("0.5", "Half"), ("0.9", "Most"), ("1.0", "All")):
            x = r["by_fraction"].get(frac)
            if not x: continue
            M.update({f"InsRnd{tag}SelDownPct": f1(x["selective_down_pct"]), f"InsRnd{tag}RndDownPct": f1(x["random_down_pct"]), f"InsRnd{tag}SelDown": str(x["selective_down"]), f"InsRnd{tag}RndDown": str(x["random_down"]),
                      f"InsRnd{tag}Pairs": str(x["pairs"]), f"InsRnd{tag}WorsePct": f1(x["selective_worse_pct"]), f"InsRnd{tag}WorseSig": str(x["selective_worse_sig"]), f"InsRnd{tag}BetterSig": str(x["selective_better_sig"]),
                      f"InsRnd{tag}DiffMed": f3(x["med_diff"]), f"InsRnd{tag}ShiftSel": f2(x["shift_selective"]), f"InsRnd{tag}ShiftRnd": f2(x["shift_random"])})
        x = r["by_fraction"].get("1.0") or {}; dt = x.get("diff_by_table", {}); nk = x.get("n_train_deleted", {}); na = x.get("n_train_all", {})
        if "nasa" in dt: M["InsRndNasaDiff"] = f3(dt["nasa"])     # mean over the model families of deletion of the records with errors minus random deletion
        if "bike" in dt: M["InsRndBikeDiff"] = f3(dt["bike"])
        if "bike" in nk and "bike" in na: M.update(InsBikeTrainKept=f0(nk["bike"]), InsBikeTrainAll=f0(na["bike"]))
    b = S["2"]
    M.update(InsAmountRho=f2(b["amount_gain"]["rho"]), InsAmountP=f2(b["amount_gain"]["p"]), InsAmountN=str(b["amount_gain"]["n"]))
    if "nasa" in b: M.update(InsNasaP=f1(b["nasa"]["P"]), InsNasaGain=f3(b["nasa"]["gain"]), InsMercP=f1(b["mercedes"]["P"]), InsMercGain=f3(b["mercedes"]["gain"]))
    if "nasa_linear" in b: M.update(InsNasaLinNinety=f3(b["nasa_linear"]["gain90"]), InsNasaLinAll=f3(b["nasa_linear"]["gain100"]), InsNasaLinP=f1(b["nasa_linear"]["P90"]))
    h = b["half"]; M.update(InsHalfN=str(h["n"]), InsHalfTree=f0(h["tree_share"]), InsHalfTreeN=str(h["tree_n"]), InsHalfLinear=f0(h["linear_share"]), InsHalfMLP=f0(h["mlp_share"]))
    if "interaction" in b: M.update(InsIntTreeAnt=str(b["interaction"]["tree_ant"]), InsIntTreeSig=str(b["interaction"]["tree_sig"]), InsIntSyn=str(b["interaction"]["syn"]))
    c = S["3"]
    if c: M.update(InsLossPairs=str(c["pairs"]), InsLossSmall=str(c["small"]), InsLossTopShare=f0(c["top3_share"]), InsLossTaskBetter=str(c["task_better"]), InsLossTaskWorse=str(c["task_worse"]),
                   InsLossTaskTie=str(c["task_tie"]), InsLossStaticFull=str(c["static_is_full_repair"]), InsLossCases=str(c["cases"]),
                   InsLossHospKNN=f3(c["top3"][0]["regret"]), InsLossFlightsKNN=f3(c["top3"][1]["regret"]), InsLossMercMLP=f3(c["top3"][2]["regret"]),
                   InsLossStaticFullPct=f0(100 * c["static_is_full_repair"] / c["cases"]))
    if c and c.get("same_static"): M.update(InsSameN=str(c["same_static"]["n"]), InsSameGeFive=str(c["same_static"]["ge05"]))
    dg = (c or {}).get("decision")
    if dg and "100" in dg["gain"].get("all", {}):
        oth = [v["100"] for m, v in dg["gain"].items() if m not in ("all", "knn")]; top = str(dg["sizes"][-1])
        if "knn" in dg["gain"]: M["InsDecKnnGain"] = f1(dg["gain"]["knn"]["100"])
        if oth: M.update(InsDecOtherLo=f2(min(oth)), InsDecOtherHi=f2(max(oth)))
        if dg["gain"]["all"][top]: M["InsDecHundredShare"] = f0(100 * dg["gain"]["all"]["100"] / dg["gain"]["all"][top])
    d = S["4"]
    if "rieth" in d: M.update(InsRunOne=f3(d["rieth"]["run1"]), InsRunFive=f3(d["rieth"]["run5"]), InsRunGain=f3(d["rieth"]["gain_runs"]), InsRecFourGain=f3(d["rieth"]["gain_rec5x4"]), InsRecSixteenGain=f3(d["rieth"]["gain_rec500x16"]))
    if "rieth_split" in d: M.update(InsSplitRecord=f"{d['rieth_split']['record']:.4f}", InsSplitRun=f"{d['rieth_split']['run']:.4f}")
    if "windows" in d:
        w = d["windows"]; rnd = w["random"]["change"]; run = w["run"]["change"]; tri = [rnd[m] for m in ("gb", "rf", "knn") if m in rnd]
        M.update(InsWinRandLo=f3(min(tri)), InsWinRandHi=f3(max(tri)), InsWinRunLo=f3(min(run.values())), InsWinRunHi=f3(max(run.values())),
                 InsWinRunBestLo=str(min(w["run"]["best"].values())), InsWinRunBestHi=str(max(w["run"]["best"].values())))
    e = S["5"]
    if "coverage" in e: M.update(InsCovSeen=f3(e["coverage"]["seen_low"]), InsCovFull=f3(e["coverage"]["full"]), InsCovAll=f3(e["coverage"]["all_low"]))
    if e.get("cifar10n_worse", {}).get("delete") is not None: x = e["cifar10n_worse"]; M.update(InsTenNKeep=f3(x["keep"]), InsTenNDel=f3(x["delete"]), InsTenNClean=f3(x["clean"]), InsTenNDelPct=f0(x["deleted_pct"]))
    ls = e.get("label_sets")
    if ls:
        de = [v["delete"] for v in ls.values()]; rp = [v["repair"] for v in ls.values()]
        M.update(InsLabN=str(len(ls)), InsLabDelLo=f1(min(de)), InsLabDelHi=f1(max(de)), InsLabRepLo=f1(min(rp)), InsLabRepHi=f1(max(rp)))
    x = e["main_right"]; M.update(InsProtoGapLo=f3(x["protocol_min"]), InsProtoGapHi=f3(x["protocol_max"]), InsProtoGapMed=f3(x["protocol_med"]), InsProtoRows=str(x["protocol_n"]), InsRecGapMax=f3(x["record_max"]), InsRecRows=str(x["record_n"]))
    t = S["6"]
    if t:
        tr = t["train_2023"]; M.update(InsTaxiNineteen=f3(tr["2019-07"]), InsTaxiTwentyThree=f3(tr["2023-01"]), InsTaxiTwenty=f3(tr["2020-07"]), InsTaxiTwentyOne=f3(tr["2021-01"]))
        ri = t["rises"]; (ka, va), (kb, vb) = sorted(ri.items())
        M.update(InsTaxiRiseA=f1(va["rule_pp"]), InsTaxiRiseAChange=f3(va["r2_change"]), InsTaxiRiseB=f1(vb["rule_pp"]), InsTaxiRiseBChange=f3(vb["r2_change"]))
        sz = t.get("size", {}); big = sz.get("2020-07", {}).get("raw", {}); small = sz.get("2019-07", {}).get("raw", {})
        if big: M["InsTaxiTwentyBig"] = f3(big[max(big, key=int)])     # the largest number of training records
        if small: M["InsTaxiNineteenSmall"] = f3(small[min(small, key=int)])     # the smallest number of training records
    s = S["7"]
    if "corpus" in s: M.update(InsCorpusMinP=f3(s["corpus"]["c4_min_p_better"]), InsCorpusLo=f2(s["corpus"]["c4_min"]), InsCorpusHi=f1(s["corpus"]["c4_max"]), InsCorpusN=str(len(s["corpus"]["filters"])))
    if "synth" in s:
        y = s["synth"]; M.update(InsRecallReal=f3(y["recall"]["real_test"]), InsRecallEDM=f3(y["recall"]["edm"]), InsRecallDDPM=f3(y["recall"]["ddpm"]), InsRecallCIFAKE=f3(y["recall"]["cifake"]),
                                 InsAccReal=f3(y["acc"]["real_test"]), InsAccEDM=f3(y["acc"]["edm"]), InsAccDDPM=f3(y["acc"]["ddpm"]), InsAccCIFAKE=f3(y["acc"]["cifake"]),
                                 InsDensDDPM=f3(y["density"]["ddpm"]), InsDensReal=f3(y["density"]["real_test"]))
    g = S["8"]
    if "detectors" in g:
        z = g["detectors"]; pb = list(z["post_below_sample"].values())
        M.update(InsDetPairs=str(z["pairs"]), InsDetCorBelow=str(z["corrected_below_sample"].get("200")), InsDetPostBelow=str(z["post_below_sample"].get("200")), InsDetPostBelowLo=str(min(pb)), InsDetPostBelowHi=str(max(pb)),
                 InsDetCovRho=f2(z["rho_cov_sesp"]), InsDetBandN=str(z["band_n"]), InsDetBandCov=f1(z["band_cov"]), InsDetBestSeSp=f"{z['best']['sesp']:.3f}",
                 InsDetBestCovFifty=f1(z["best"]["cov50"]), InsDetBestCovTwoHundred=f1(z["best"]["cov200"]), InsDetSmartSeSp=f"{z['d1']['smartfactory']:.3f}", InsDetBikeSeSp=f"{z['d1']['bike']:.3f}")
    if "llm" in g:
        l = g["llm"]; M.update(InsLLMMinChanged=f0(l["min_changed_per100"]), InsLLMAdultErr=f1(l["adult_errors_per100"]), InsLLMAdultChanged=f1(l["adult_changed_per100"]))
        if "rho_f1_gain" in l: M.update(InsLLMRho=f2(l["rho_f1_gain"]), InsLLMRhoN=str(l["n"]), InsLLMBestF=f2(l["best_f1"]), InsLLMSoilF=f2(l["soil_f1"]), InsLLMSoilGain=f3(l["soil_gain"]))
    if "judge" in g:
        M.update(InsJudgeWords=f2(g["judge"]["rho_words"]), InsJudgePPL=f2(g["judge"]["rho_ppl"]), InsJudgeQc=f2(g["judge"]["q75"]), InsJudgeMed=f2(g["judge"]["median"]))
        if g["judge"].get("q99") is not None: M["InsJudgeQnn"] = f2(g["judge"]["q99"])
    mt = S.get("main")
    if mt:
        v = mt["versions"]; p = mt["practices"]
        M.update(InsStaticBelowRaw=str(v["static_below_raw"]), InsTaskLossMax=f3(v["task_loss_max"]), InsTaskGainMax=f3(v["task_gain_max"]))
        if v["task_loss_max_ds"]: M["InsTaskLossDs"] = v["task_loss_max_ds"]
        if v["task_gain_max_ds"]: M["InsTaskGainDs"] = v["task_gain_max_ds"]
        if p["shift_n"]: M.update(InsShiftRowsLo=f3(p["shift_min"]), InsShiftRowsHi=f3(p["shift_max"]), InsShiftRowsMed=f3(p["shift_med"]), InsShiftRowsN=str(p["shift_n"]))
        if p["record_n"]: M["InsRecGapMed"] = f3(p["record_med"])
        if p["static_min"] is not None: M["InsLowStaticMin"] = f1(p["static_min"])
    bad = [k for k in M if not k.isalpha()]
    if bad: raise ValueError(f"macro names must be letters only: {bad}")
    return M


def macros(results_dir):
    """{macro name: formatted string}. Reads insights_summary.json when it is newer than every result file it depends on, otherwise recomputes."""
    p = os.path.join(results_dir, SUMMARY_FILE); dep = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith("results_") or f == "tables_summary.json"]
    S = json.load(open(p)) if os.path.exists(p) and os.path.getmtime(p) >= max(os.path.getmtime(f) for f in dep) else None
    if not S or S.get("_version") != SUMMARY_VERSION: S = compute(results_dir)
    return macros_from(S)


if __name__ == "__main__":
    R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results")
    S = compute(R); json.dump(S, open(os.path.join(R, SUMMARY_FILE), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(macros_from(S), indent=0, ensure_ascii=False))
