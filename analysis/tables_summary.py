#!/usr/bin/env python3
"""Summary of the tabular track. Reads results_real_tables.jsonl, computes the score of the standard for every state of the training set,
and reports how far that score determines the task score. Prints JSON."""
import json, sys, collections, math
import numpy as np, pandas as pd
from scipy import stats

def max_task_gap_within_score_band(static_scores, task_scores, tolerance=0.5):
    """Largest task-score gap among pairs whose static-score difference is within tolerance."""
    points = sorted(zip(static_scores, task_scores))
    minima, maxima = collections.deque(), collections.deque()
    left, best = 0, 0.0
    for right, (static_score, task_score) in enumerate(points):
        while static_score - points[left][0] > tolerance:
            if minima and minima[0] == left:
                minima.popleft()
            if maxima and maxima[0] == left:
                maxima.popleft()
            left += 1
        while minima and points[minima[-1]][1] >= task_score:
            minima.pop()
        while maxima and points[maxima[-1]][1] <= task_score:
            maxima.pop()
        minima.append(right)
        maxima.append(right)
        best = max(best, points[maxima[0]][1] - points[minima[0]][1])
    return float(best)

path = sys.argv[1]; rows = []
for l in open(path):
    try:
        r = json.loads(l)
        if "score" in r: rows.append(r)
    except Exception: pass
D = pd.DataFrame(rows)
D["s_struct"] = 1 - D.rate_missing; D["s_format"] = 1 - D.rate_domain; D["s_auth"] = 1 - D.rate_value; D["s_label"] = 1 - D.rate_label
D["P_data"] = 100 * (D.s_struct + D.s_format + D.s_auth + 5) / 8
D["scale"] = D.n_train / D.n0
out = dict(n_runs=len(D), n_ds=D.ds.nunique(), by_task=D.groupby("ds").task.first().to_dict())
base = D[D.action == "keep"].groupby("ds")[["cell_error", "rate_clean_records", "rate_missing", "rate_domain", "rate_value", "rate_label", "P_data", "n0"]].mean()
out["base"] = base.round(4).to_dict("index")
# mean task score per dataset, model, action
G = D.groupby(["ds", "model", "action", "f"]).agg(score=("score", "mean"), sd=("score", "std"), n=("score", "size"), P=("P_data", "mean"), scale=("scale", "mean"), lab=("s_label", "mean")).reset_index()
# Spearman of the static score against the task score across the states of the training set, per dataset and model
sp = []
for (d, m), g in G.groupby(["ds", "model"]):
    if g.P.nunique() > 2 and len(g) >= 6:
        sp.append(dict(ds=d, model=m, rho=float(stats.spearmanr(g.P, g.score).statistic), span=float(g.score.max() - g.score.min())))
SP = pd.DataFrame(sp); out["spearman"] = dict(n=len(SP), median=float(SP.rho.median()), q1=float(SP.rho.quantile(.25)), q3=float(SP.rho.quantile(.75)), neg=int((SP.rho < 0).sum()), lt03=int((SP.rho < .3).sum()), gt07=int((SP.rho > .7).sum()), min=float(SP.rho.min()), max=float(SP.rho.max()))
out["spearman_list"] = SP.round(4).to_dict("records"); out["spearman_by_ds"] = SP.groupby("ds").rho.median().round(3).to_dict(); out["spearman_by_model"] = SP.groupby("model").rho.median().round(3).to_dict()
# paired comparison of every action against keep, same seed, as in CleanML: P, S, N with Benjamini-Yekutieli
K = D[D.action == "keep"].set_index(["ds", "model", "seed"]).score
cmp = []
for (d, m, a, f), g in D[D.action != "keep"].groupby(["ds", "model", "action", "f"]):
    x = g.set_index(["ds", "model", "seed"]).score; y = K.reindex(x.index); ok = y.notna(); dif = (x - y)[ok]
    if len(dif) < 3: continue
    p = 1.0 if dif.abs().sum() == 0 else float(stats.ttest_rel(x[ok], y[ok]).pvalue)
    if not np.isfinite(p): p = 1.0
    cmp.append(dict(ds=d, model=m, action=a, f=f, delta=float(dif.mean()), p=p, dP=float(g.P_data.mean() - D[(D.ds == d) & (D.model == m) & (D.action == "keep")].P_data.mean())))
def fdr_by(p):
    """Benjamini-Yekutieli adjusted p values, the correction CleanML uses."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); c = np.sum(1 / np.arange(1, m + 1)); q = np.minimum(1, p[o] * m * c / np.arange(1, m + 1)); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty(m); r[o] = q; return r
C = pd.DataFrame(cmp); C["q"] = fdr_by(C.p); C["flag"] = np.where(C.q >= .05, "S", np.where(C.delta > 0, "P", "N"))
def psn(x): return {k: int((x.flag == k).sum()) for k in "PSN"}
out["psn_all"] = psn(C); out["psn_by_action"] = {a: psn(g) for a, g in C[C.f == 1.0].groupby("action")}
up = C[C.dP > 1e-6]; out["static_up"] = dict(n=len(up), task_down=int((up.flag == "N").sum()), task_up=int((up.flag == "P").sum()), task_same=int((up.flag == "S").sum()))
# sign flips of one action between two models on one dataset
F = C[C.f == 1.0]; flips = []
for (d, a), g in F.groupby(["ds", "action"]):
    pos = g[g.flag == "P"]; neg = g[g.flag == "N"]
    if len(pos) and len(neg): flips.append(dict(ds=d, action=a, pos=pos.sort_values("delta").iloc[-1][["model", "delta"]].to_dict(), neg=neg.sort_values("delta").iloc[0][["model", "delta"]].to_dict()))
out["flips"] = flips; out["flips_of"] = int(F.groupby(["ds", "action"]).ngroups)
# non-additivity on real errors: repair of one error class at a time against the joint repair
inter = []
M = D.pivot_table(index=["ds", "model", "seed"], columns=["action", "f"], values="score")
for (d, m), g in M.groupby(level=[0, 1]):
    need = [("keep", 0.0), ("repair_records", 1.0)]
    parts = [("repair_type_missing", 1.0), ("repair_type_domain", 1.0), ("repair_type_value", 1.0), ("repair_labels", 1.0)]
    if not all(c in g.columns for c in need + parts): continue
    h = g[need + parts].dropna()
    if len(h) < 3: continue
    single = sum(h[p] - h[("keep", 0.0)] for p in parts); joint = h[("repair_records", 1.0)] - h[("keep", 0.0)]; i = joint - single
    active = [p[0] for p in parts if float((h[p] - h[("keep", 0.0)]).abs().mean()) > 1e-9]
    if len(active) < 2: continue
    t = stats.ttest_1samp(i, 0); inter.append(dict(ds=d, model=m, joint=float(joint.mean()), single=float(single.mean()), inter=float(i.mean()), p=float(t.pvalue) if np.isfinite(t.pvalue) else 1.0, n=len(h), active=len(active)))
I = pd.DataFrame(inter)
if len(I):
    I["q"] = fdr_by(I.p); out["interaction_list"] = I.round(5).to_dict("records"); out["interaction"] = dict(n=len(I), sig=int((I.q < .05).sum()), syn=int(((I.q < .05) & (I.inter > 0)).sum()), ant=int(((I.q < .05) & (I.inter < 0)).sum()), med_abs=float(I.inter.abs().median()), max_abs=float(I.inter.abs().max()), top=I.reindex(I.inter.abs().sort_values(ascending=False).index).head(6).round(4).to_dict("records"))
# same static score, different task score: for each dataset and model, states whose static score is within half a point
same = []
for (d, m), g in G.groupby(["ds", "model"]):
    best = max_task_gap_within_score_band(g.P.values, g.score.values)
    same.append(dict(ds=d, model=m, gap=float(best)))
S = pd.DataFrame(same); out["same_static"] = dict(median=float(S.gap.median()), q3=float(S.gap.quantile(.75)), max=float(S.gap.max()), ge05=int((S.gap >= .05).sum()), ge01=int((S.gap >= .01).sum()), n=len(S))
# score of the standard against pass line on the raw table
out["pass90"] = dict(keep_pass=int((base.P_data >= 90).sum()), of=len(base), min_P=float(base.P_data.min()))
# best model changes with the state of the training set
W = []
for (d, a, f), g in G.groupby(["ds", "action", "f"]):
    if g.model.nunique() == 5: W.append(dict(ds=d, action=a, f=f, best=g.sort_values("score").iloc[-1].model))
W = pd.DataFrame(W); out["best_model_changes"] = {d: sorted(g.best.unique().tolist()) for d, g in W.groupby("ds")}
# overall view of keep/repair/delete per dataset and model
T = G[((G.action == "keep") | ((G.action.isin(["repair_records", "delete_records"])) & (G.f == 1.0)))].pivot_table(index=["ds", "model"], columns="action", values="score")
out["table"] = {f"{d}|{m}": {k: round(float(v), 3) for k, v in row.items() if np.isfinite(v)} for (d, m), row in T.iterrows()}
json.dump(out, sys.stdout, indent=0, default=float)
