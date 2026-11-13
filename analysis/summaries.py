#!/usr/bin/env python3
"""Numbers of the tracks other than the tables, read from the result files and written as JSON for make_numbers.py. No value is typed by hand."""
import collections, json, math, os, statistics as st, sys
import big_tables
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
def load(fn):
    p = os.path.join(R, fn); return json.load(open(p)) if os.path.exists(p) else None
def jsonl(fn):
    p = os.path.join(R, fn); out = []
    if os.path.exists(p):
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return out
def dump(fn, d): json.dump(d, open(os.path.join(R, fn), "w"), indent=0, ensure_ascii=False)
def m(x):
    x = list(x); return st.mean(x) if x else float("nan")
def sci(n):
    if n >= 1e8: return f"${n / 1e8:.2f}\\times 10^8$"
    if n >= 1e6: return f"${n / 1e6:.2f}\\times 10^6$"
    if n >= 1e4: return f"${n / 1e4:.2f}\\times 10^4$"
    return f"{int(n)}"
# ---- vision
V = {}; rows = {}
for r in jsonl("results_vision.jsonl"): rows[r["id"]] = r
rows = list(rows.values())
if rows:
    def sel(f): return [r for r in rows if f(r)]
    base = sel(lambda r: r["ds"] == "cifar100" and "noise" not in r and "coverage" not in r and "scale" not in r and r["arch"] == "resnet18")
    noisy = sel(lambda r: r["ds"] == "cifar100" and r.get("noise") == "noisy_label" and r.get("action") == "keep" and r["arch"] == "resnet18"); dele = sel(lambda r: r["ds"] == "cifar100" and r.get("noise") == "noisy_label" and r.get("action") == "delete" and r.get("f") == 1.0 and r["arch"] == "resnet18")
    cov1 = sel(lambda r: r.get("coverage") == 1 and r["arch"] == "resnet18"); sc2 = sel(lambda r: r.get("scale") == .2 and r["arch"] == "resnet18"); cov5 = sel(lambda r: r.get("coverage") == 5 and r["arch"] == "resnet18")
    if noisy and base: V["VisNoisyAcc"] = f"{m(r['score'] for r in noisy):.3f}"; V["VisCleanAcc"] = f"{m(r['score'] for r in base):.3f}"; V["VisGain"] = f"{m(r['score'] for r in base) - m(r['score'] for r in noisy):.3f}"
    if dele and base: V["VisDelGap"] = f"{100 * (m(r['score'] for r in base) - m(r['score'] for r in dele)):.1f}"
    if cov1:
        gap = 100 * (m(r["score_seen_sub"] for r in cov1) - m(r["score_unseen_sub"] for r in cov1)); V["VisCovGap"] = f"{gap:.0f}"; V["VisCovGapB"] = f"{gap:.0f}"; V["VisCovK"] = "1"
    if sc2 and cov5: V["VisScaleDrop"] = f"{100 * (m(r['score'] for r in cov5) - m(r['score'] for r in sc2)):.0f}"
    if cov1 and cov5: V["VisUnseenDrop"] = f"{100 * (m(r['score'] for r in cov5) - m(r['score_unseen_sub'] for r in cov1)):.0f}"; V["VisCovDrop"] = f"{100 * (m(r['score'] for r in cov5) - m(r['score'] for r in cov1)):.0f}"; V["VisCovOverScale"] = f"{(m(r['score'] for r in cov5) - m(r['score'] for r in cov1)) / (m(r['score'] for r in cov5) - m(r['score'] for r in sc2)):.1f}" if sc2 else ""
    V["RunsVisionN"] = f"{len(sel(lambda r: 'coverage' not in r and 'scale' not in r))}"; V["RunsVisionC"] = f"{len(sel(lambda r: 'coverage' in r or 'scale' in r))}"
F = load("results_fair.json")
if F and F["cifar100"]["runs"]:
    c = F["cifar100"]; V["FairShare"] = f"{100 * c['share']:.1f}"; V["FairDrop"] = f"{100 * (m(r['dup'] for r in c['runs']) - m(r['rest'] for r in c['runs'])):.1f}"
dump("numbers_vision.json", V)
# ---- taxi
X = load("results_taxi.json"); X2 = load("results_taxi_100k.json"); Tx = {}
if X:
    months = sorted(X["months"]); idx = {k: i for i, k in enumerate(months)}; anyr = [100 * X["months"][k]["any_rule"] for k in months]
    Tx.update(TaxiMonths=f"{len(months)}", TaxiRecords=sci(X.get("records_total", sum(v["records"] for v in X["months"].values()))), TaxiRuleLo=f"{min(anyr):.1f}", TaxiRuleHi=f"{max(anyr):.1f}")
    H = collections.defaultdict(lambda: collections.defaultdict(list)); byk = collections.defaultdict(list)
    for r in X["runs"]:
        for k, v in r["r2"].items():
            H[r["arm"]][idx[k] - idx[r["train_month"]]].append(v)
            if r["arm"] == "raw" and k > r["train_month"]: byk[k].append(v)
    hz = 48 if H["raw"].get(48) else max(H["raw"]); Tx.update(TaxiHorizon=f"{hz}", TaxiRzero=f"{m(H['raw'][0]):.3f}", TaxiRlate=f"{m(H['raw'][hz]):.3f}")
    def by_month(arm):
        g = collections.defaultdict(list)
        for r in X["runs"]:
            if r["arm"] == arm: g[r["train_month"]].append(r["r2"])
        return {tm: {k: m(c[k] for c in cs if k in c) for k in set().union(*cs)} for tm, cs in g.items()}
    raw, flt = by_month("raw"), by_month("rule_filtered")
    def at(curve, tm, h):
        i = idx[tm] + h; return curve.get(months[i]) if i < len(months) else None
    drop = {tm: at(cv, tm, 1) - at(cv, tm, hz) for tm, cv in raw.items() if at(cv, tm, 1) is not None and at(cv, tm, hz) is not None}
    pre = [v for k, v in drop.items() if k <= "2020-01"]; cov = [v for k, v in drop.items() if "2020-01" < k <= "2020-12"]
    if pre: Tx.update(TaxiDropPre=f"{m(pre):.3f}", TaxiNPre=f"{len(pre)}")
    if cov: Tx.update(TaxiDropCovid=f"{m(cov):.3f}", TaxiNCovid=f"{len(cov)}")
    # filtering against the raw model of the same training month, at the same horizon
    pair = [abs(at(flt[k], k, hz) - at(raw[k], k, hz)) for k in flt if k in raw and at(flt[k], k, hz) is not None and at(raw[k], k, hz) is not None]
    if pair: Tx["TaxiFilterGain"] = f"{m(pair):.3f}"
    # rule violations as a signal: models trained before the first large change, month to month after the last training month
    fixed = {k: cv for k, cv in raw.items() if k <= "2020-01"}
    if fixed:
        anyr_ = {k: 100 * X["months"][k]["any_rule"] for k in months}; cs = []
        def corr(a, b):
            ma, mb = m(a), m(b); d = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)); return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / d if d else float("nan")
        for tm, cv in fixed.items():
            ks = [k for k in months if k > tm and k in cv]
            if len(ks) > 3: cs.append(corr([anyr_[k] for k in ks], [cv[k] for k in ks]))
        if cs: Tx["TaxiCorrMed"] = f"{st.median(cs):.2f}".replace("-", "$-$")
        last = max(fixed); ks = [k for k in months if k > last and all(k in cv for cv in fixed.values())]; mean_r = {k: m(cv[k] for cv in fixed.values()) for k in ks}
        trig = [(abs(anyr_[b] - anyr_[a]) > 2, abs(mean_r[b] - mean_r[a]) > .05) for a, b in zip(ks, ks[1:])]
        Tx.update(TaxiTrigN=f"{sum(1 for t, _ in trig if t)}", TaxiTrigHit=f"{sum(1 for t, h in trig if t and h)}", TaxiRestN=f"{sum(1 for t, _ in trig if not t)}", TaxiRestHit=f"{sum(1 for t, h in trig if not t and h)}")
    X3 = load("results_taxi_sizes.json"); Tx["RunsTaxi"] = f"{len(X['runs']) + (len(X2['runs']) if X2 else 0) + (len(X3['runs']) if X3 else 0)}"
dump("numbers_taxi.json", Tx)
# ---- corpus and leakage
Cc = {}; CO = jsonl("results_corpus.jsonl")
if CO: Cc["RunsCorpus"] = f"{len(CO)}"
Z = load("corpus_stats.json")
if Z: Cc["CorpusDocs"] = sci(Z["docs"])
TX = list({r.get("id", i): r for i, r in enumerate(jsonl("results_text.jsonl") + jsonl("results_text_more.jsonl"))}.values())
if TX:
    def tm(fld, rate): return m(r[fld] for r in TX if r.get("defects") == {"contamination": rate} and fld in r)
    Cc["LeakLeaked"] = f"{tm('nll_leaked', .2):.3f}"; Cc["LeakUnseen"] = f"{tm('nll_unseen', .2):.3f}"; Cc["LeakDrop"] = f"{tm('nll', .05) - tm('nll', .4):.3f}"; Cc["LeakGap"] = f"{tm('nll_unseen', .2) - tm('nll_leaked', .2):.3f}"; Cc["RunsText"] = f"{len(TX)}"
dump("numbers_corpus.json", Cc)
# ---- multimodal
MM = {}; M = list({r.get("id", i): r for i, r in enumerate(jsonl("results_multimodal.jsonl") + jsonl("results_multimodal_fine.jsonl"))}.values())
if M:
    def mm(d, rate, fld="score"): return m(r[fld] for r in M if r.get("defects") == {d: rate} and r.get(fld) is not None)
    MM["MMrecords"] = f"{max(r['n_train'] for r in M) + 1000}"; MM["MMauc"] = f"{mm('mismatch', .2, 'detect_auc'):.3f}"; MM["MMdrop"] = f"{100 * (mm('mismatch', .1) - mm('mismatch', .4)):.1f}"; MM["MMother"] = f"{100 * max(mm(d, .1) - mm(d, .4) for d in ('duplicate', 'scale')):.1f}"; MM["RunsMM"] = f"{len(M)}"
    MM["MMraw"] = f"{mm('mismatch', .2, 'consistency_est'):.3f}"; MM["MMtrue"] = f"{mm('mismatch', .2, 'consistency_true'):.3f}"; MM["MMcorr"] = f"{mm('mismatch', .2, 'consistency_corrected'):.3f}"; MM["MMgap"] = f"{100 * abs(mm('mismatch', .2, 'consistency_est') - mm('mismatch', .2, 'consistency_true')):.1f}"
dump("numbers_mm.json", MM)
# ---- decision value and totals
D = {}; rows = [r for r in jsonl("results_decision.jsonl") if "test" in r]; fine = [r for r in jsonl("results_decision_fine.jsonl") if "test" in r]
if len(fine) >= len(rows): rows = fine   # same runs and scores with more validation sizes
if rows:
    G = collections.defaultdict(dict)
    for r in rows: G[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r
    def replay(task, ns, only=None):
        out = {"static": [], "oracle": [], **{n: [] for n in ns}}
        for k, arms in G.items():
            if ("keep", 0.0) not in arms or len(arms) < 14 or arms[("keep", 0.0)]["task"] != task or (only and k[0] != only): continue
            base = arms[("keep", 0.0)]["test"]; P = lambda r: r["rate_missing"] + r["rate_domain"] + r["rate_value"]; s0 = min(arms.values(), key=lambda r: (P(r), r["rate_label"], -r["n_train"])); out["static"].append(s0["test"] - base); out["oracle"].append(max(r["test"] for r in arms.values()) - base)
            for n in ns:
                def v(r):
                    av = sorted(int(x) for x in r["val"]); use = max([x for x in av if x <= n] or av[:1]); return r["val"][str(use)]
                out[n].append(max(arms.values(), key=v)["test"] - base)
        return out
    cl = replay("cls", (25, 100)); re_ = replay("reg", (100,))
    D.update(DecSmallN="25", DecMidN="100", DecStaticCls=f"{100 * m(cl['static']):.1f}", DecSmallCls=f"{100 * m(cl[25]):.1f}", DecMidCls=f"{100 * m(cl[100]):.1f}", DecOracleCls=f"{100 * m(cl['oracle']):.1f}", DecStaticReg=f"{100 * m(re_['static']):.1f}", DecMidReg=f"{100 * m(re_[100]):.1f}", RunsDecision=f"{len(rows)}")
    if m(cl["static"]): D.update(DecRatioSmall=f"{m(cl[25]) / m(cl['static']):.1f}", DecRatioMid=f"{m(cl[100]) / m(cl['static']):.1f}")
    if m(cl["oracle"]): D["DecOracleShare"] = f"{100 * m(cl[100]) / m(cl['oracle']):.0f}"
    for ds, tag in (("flights", "Flights"), ("hospitals", "Hospitals")):
        o = replay("cls", (100,), ds)
        if o["static"]: D[f"Dec{tag}"] = f"{100 * m(o[100]):.1f}"; D[f"Dec{tag}Static"] = f"{100 * m(o['static']):.1f}"; D[f"Dec{tag}Ratio"] = f"{m(o[100]) / m(o['static']):.1f}"
    # the size at which the mean gain on the classification tables passes the static rule, and the tables where a task test is not below it
    SZ = sorted({int(x) for r in rows for x in r["val"]}); allc = replay("cls", SZ); cross = [n for n in SZ if m(allc[n]) >= m(allc["static"])]
    if cross: D["DecCrossN"] = f"{cross[0]}"
    WIN = 400; D["DecWinSize"] = f"{WIN}"; win = 0; tabs = sorted({k[0] for k in G})
    for ds in tabs:
        task = next(a[("keep", 0.0)]["task"] for k, a in G.items() if k[0] == ds and ("keep", 0.0) in a); o = replay(task, (WIN,), ds); win += round(m(o[WIN]), 3) >= round(m(o["static"]), 3)
    D["DecWinN"] = f"{win}"
dump("numbers_decision.json", D)
# the four table grids share fits with the same configuration, each fit counts once
fits = {r["fit_id"] for fn in ("results_real_tables.jsonl", "results_tables_fine.jsonl", "results_decision_fine.jsonl", "results_cleanml.jsonl") for r in jsonl(fn) if "error" not in r and "fit_id" in r}
tot = len(fits); T = load("tables_summary.json"); C = load("case_tep.json"); tot += sum(3 if r["arm"] == "repair" else 1 for r in C["runs"]) if C else 0
base_k = {(r["ds"], r["model"], r["seed"], r["action"], r["f"]) for r in jsonl("results_real_tables.jsonl") if "score" in r}; fine_k = {(r["ds"], r["model"], r["seed"], r["action"], r["f"]) for r in jsonl("results_tables_fine.jsonl") if "score" in r} - base_k; W_ = [r for r in jsonl("results_tep_windows.jsonl") if "macro_f1" in r]; tot += len(W_)
D["RunsTablesAll"] = f"{(T['n_runs'] if T else 0) + len(fine_k) + int(D.get('RunsDecision', 0))}"; D["RunsCaseAll"] = f"{(sum(3 if r['arm'] == 'repair' else 1 for r in C['runs']) if C else 0) + len(W_)}"; dump("numbers_decision.json", D)
Q = load("results_rieth.json"); tot += sum(2 if r["arm"] == "split_protocol" else 1 for r in Q["runs"]) if Q else 0; tot += len(jsonl("results_rieth_models.jsonl")); tot += int(V.get("RunsVisionN", 0)) + int(V.get("RunsVisionC", 0)) + int(Tx.get("RunsTaxi", 0)) + int(Cc.get("RunsCorpus", 0)) + len(jsonl("results_multimodal.jsonl")) + sum(len(jsonl(fn)) for fn in ("results_wilds.jsonl", "results_datacomp.jsonl", "results_mirflickr.jsonl", "results_synth.jsonl", "results_llm_tables.jsonl"))
rec = 0; rec += X.get("records_total", 0) if X else 0; rec += Q.get("records_total", 0) if Q else 0
tot += sum(1 for r in jsonl("results_vision_fine.jsonl") if "error" not in r); rec = sum(big_tables.records(R).values())
tot += sum(1 for r in jsonl("results_random_delete.jsonl") if r.get("action") == "random_delete" and "error" not in r)     # the reruns of delete_records repeat stored fits and are not counted
dump("numbers_totals.json", dict(StudyRuns=f"{tot}", StudyRecords=sci(rec)))
print(dict(V=V, Tx=Tx, Cc=Cc, MM=MM, D=D, tot=tot, rec=rec))
