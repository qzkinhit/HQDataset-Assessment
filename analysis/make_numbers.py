#!/usr/bin/env python3
"""Every number the survey reports from its own experiments passes through this script. It reads the result files and prints the numbers as JSON.
With a LaTeX file as the second argument, it rewrites the macro block between the markers NUMBERS BEGIN and NUMBERS END of that file and the main table between its markers, and a macro without a value becomes a marked placeholder.
Usage: make_numbers.py RESULTS_DIR [MAIN_TEX]"""
import json, math, os, re, statistics, sys
R = sys.argv[1]; TEX = sys.argv[2] if len(sys.argv) > 2 else None; N = {}
def load(fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else None
def jsonl(fn):
    p = os.path.join(R, fn); out = []
    if os.path.exists(p):
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return out
def mean(x):
    x = list(x); return sum(x) / len(x) if x else float("nan")
def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def f1(x): return f"{x:.1f}"
def sci(n):
    if n >= 1e8: return f"${n / 1e8:.2f}\\times 10^8$"
    if n >= 1e6: return f"${n / 1e6:.2f}\\times 10^6$"
    return f"{int(n)}"
# ---- running example
C = load("case_tep.json")
if C:
    runs = C["runs"]
    def arm(model, a, field="macro_f1"): return mean(r[field] for r in runs if r["model"] == model and r["arm"] == a)
    N["TEPrecords"] = f"{C['scan']['records']}"; N["TEPdupshare"] = f1(100 * C["scan"]["duplicate_records"] / C["scan"]["records"]); N["TEPpreonset"] = f1(100 * C["pre_onset"]["share"])
    N["TEPofficialRF"] = f3(arm("rf", "official")); N["TEPofficialGB"] = f3(arm("gb", "official")); N["TEProwRF"] = f3(arm("rf", "row_random")); N["TEPwinrunRF"] = f3(arm("rf", "window_run")); N["TEPwinrandRF"] = f3(arm("rf", "window_random"))
    N["TEPnaiveRF"] = f3(arm("rf", "official", "eval_file_name_labels")); N["TEPonsetRF"] = f3(arm("rf", "official", "eval_onset_labels")); N["TEPhardRF"] = f3(arm("rf", "official", "hard")); N["TEPotherRF"] = f3(arm("rf", "official", "other_faults"))
    N["TEPkeepRF"] = f3(arm("rf", "repair", "keep_defect")); N["TEPrelabelRF"] = f3(arm("rf", "repair", "relabel_normal")); N["TEPdropRF"] = f3(arm("rf", "repair", "drop_pre_onset")); N["TEPkeepMLP"] = f3(arm("mlp", "repair", "keep_defect")); N["TEPrelabelMLP"] = f3(arm("mlp", "repair", "relabel_normal"))
    b = [(r["boot_hi"] - r["boot_lo"]) / 2 for r in runs if r["model"] == "rf" and r["arm"] == "official"]; N["TEPbootHalf"] = f3(mean(b))
    sm = arm("rf", "window_run") / .8; fm = math.exp(sm - 1) if sm < 1 else 1.0; N["TEPsmRun"] = f3(sm); N["TEPfmRun"] = f3(fm); N["TEPscoreRun"] = f1(100 * fm); N["RunsCase"] = f"{sum(3 if r['arm'] == 'repair' else 1 for r in runs)}"
    N["TEPgapWin"] = f3(arm("rf", "window_random") - arm("rf", "window_run")); N["TEPgapRow"] = f3(arm("rf", "row_random") - arm("rf", "official")); N["TEPgapLabel"] = f3(arm("rf", "official", "eval_onset_labels") - arm("rf", "official", "eval_file_name_labels")); N["TEPdRelabelRF"] = f3(arm("rf", "repair", "keep_defect") - arm("rf", "repair", "relabel_normal")); N["TEPdRelabelMLP"] = f3(arm("mlp", "repair", "relabel_normal") - arm("mlp", "repair", "keep_defect"))
    N["TEPoverlap"] = f1(100 * mean(r["windows_with_any_overlap"] for r in runs if r["arm"] == "window_random"))
# ---- large TEP
Q = load("results_rieth.json")
if Q:
    rr = Q["runs"]; a = lambda k: mean(r["macro_f1"] for r in rr if r.get("k") == k and r.get("arm") == "runs_fixed_records"); N["RiethOne"] = f3(a(1)); N["RiethAll"] = f3(a(500)); N["RiethRecords"] = sci(Q["records_total"]); QM = jsonl("results_rieth_models.jsonl"); N["RunsRieth"] = f"{sum(2 if r['arm'] == 'split_protocol' else 1 for r in rr) + len(QM)}"; N["RiethGain"] = f3(a(500) - a(1))
    qm = lambda md, k: mean(r["macro_f1"] for r in QM if r["model"] == md and r["k"] == k and r.get("n") == 480)
    if QM: N["RiethGainMLP"] = f3(qm("mlp", 500) - qm("mlp", 1))
    sp = [r for r in rr if r["arm"] == "split_protocol"]
    if sp: N["RiethSplitGap"] = f1(100 * mean(r["record_split"] - r["run_split"] for r in sp))
# ---- tables, from tables_summary.py
T = load("tables_summary.json")
if T:
    B = T["base"]; pc = lambda d, k: 100 * B[d][k]
    for d, tag in (("hospitals", "Hospitals"), ("flights", "Flights"), ("beers", "Beers")):
        N[f"TCell{tag}"] = f1(pc(d, "cell_error")); N[f"TRec{tag}"] = f1(100 - pc(d, "rate_clean_records"))
    N["TPHospitals"] = f1(B["hospitals"]["P_data"]); N["TDsN"] = f"{T['n_ds']}"; N["TPassN"] = f"{T['pass90']['keep_pass']}"; N["RunsTables"] = f"{T['n_runs']}"; S_ = load("table_sizes.json"); N["TRecordsTotal"] = sci(sum(S_.values())) if S_ else sci(sum(v["n0"] / .7 for v in B.values()))
    S = T["spearman"]; N["TSpMed"] = f2(S["median"]); N["TSpQa"] = f2(S["q1"]); N["TSpQc"] = f2(S["q3"]); N["TSpN"] = f"{S['n']}"; N["TSpNeg"] = f"{S['neg']}"; N["TSameMed"] = f3(T["same_static"]["median"])
    def share(a):
        x = T["psn_by_action"][a]; t = sum(x.values()); return {k: f1(100 * x[k] / t) for k in "PSN"}
    r = share("repair_records"); N["TRepP"], N["TRepS"], N["TRepN"] = r["P"], r["S"], r["N"]; r = share("delete_records"); N["TDelP"], N["TDelS"], N["TDelN"] = r["P"], r["S"], r["N"]
    N["TRepPairs"] = f"{sum(T['psn_by_action']['repair_records'].values())}"; N["TDelPairs"] = f"{sum(T['psn_by_action']['delete_records'].values())}"
    N["TUpN"] = f"{T['static_up']['n']}"; N["TUpDown"] = f"{T['static_up']['task_down']}"; N["TUpUp"] = f"{T['static_up']['task_up']}"; N["TUpSame"] = f"{T['static_up']['task_same']}"; N["TFlipN"] = f"{len(T['flips'])}"; N["TFlipOf"] = f"{T['flips_of']}"; N["TBestChange"] = f"{sum(1 for v in T['best_model_changes'].values() if len(v) > 1)}"
    N["TRecBreast"] = f1(100 - pc("breast_cancer", "rate_clean_records")); N["TPBreast"] = f1(B["breast_cancer"]["P_data"])
    if T.get("interaction", {}).get("top"): x_ = T["interaction"]["top"][0]; N["TIntExSingle"] = f"{x_['single']:.3f}".replace("-", "$-$"); N["TIntExJoint"] = f"{x_['joint']:.3f}"
    if C:
        lim = arm("rf", "window_random") - arm("rf", "window_run"); dl = [abs(v[a_] - v["keep"]) for k_, v in T["table"].items() if T["by_task"][k_.split("|")[0]] == "cls" for a_ in ("repair_records", "delete_records") if a_ in v and "keep" in v]; N["TProtoN"] = f"{len(dl)}"; N["TProtoBeat"] = f"{sum(1 for d_ in dl if d_ < lim)}"; lab_ = arm("rf", "official", "eval_onset_labels") - arm("rf", "official", "eval_file_name_labels"); N["TProtoBeatLabel"] = f"{sum(1 for d_ in dl if d_ < lab_)}"
    I = T.get("interaction")
    if I: N["TIntN"] = f"{I['n']}"; N["TIntSig"] = f"{I['sig']}"; N["TIntSyn"] = f"{I['syn']}"; N["TIntAnt"] = f"{I['ant']}"
    fl = sorted(T["flips"], key=lambda x: -min(x["pos"]["delta"], -x["neg"]["delta"]))
    if fl:
        x = fl[0]; ACT = dict(delete_records="删除全部错误记录", repair_records="修复全部错误", repair_labels="修复标签", repair_type_value="修复语义错误", repair_type_domain="修复语法错误", repair_type_missing="修复缺失值"); MOD = dict(linear="线性模型", rf="随机森林", gb="梯度提升", knn=" K 近邻 (k-nearest neighbors, \\textbf{KNN}) ", mlp="多层感知机"); DSN = dict(soccer_full="Soccer", breast_cancer="Breast Cancer", soilmoisture="Soil Moisture", smartfactory="SmartFactory", nasa="NASA")
        ex = f"{DSN.get(x['ds'], x['ds'].capitalize())} 上{ACT.get(x['action'], x['action'])}使{MOD[x['pos']['model']]}提高 {x['pos']['delta']:.3f}, 使{MOD[x['neg']['model']]}降低 {-x['neg']['delta']:.3f}"; N["TFlipExample"] = ex; N["TFlipExampleA"] = ex; p_, n_ = x["pos"]["delta"], -x["neg"]["delta"]; N["TFlipRegret"] = f"{p_ * n_ / (p_ + n_):.3f}"; N["TFlipDetRegret"] = f"{min(p_, n_):.3f}"     # theorem 1 (2) on the two uses of the flip
    n = next((v["n0"] for k, v in B.items() if k == "flights"), None)
    if n:
        te = round(n / .7 * .3); N["TFlightsTest"] = f"{te}"; L = math.log(2 * 20 / .05); lo, hi = 1e-4, 1
        for _ in range(60):
            g = (lo + hi) / 2; lo, hi = (lo, g) if (8 * .1 / g ** 2 + 8 / (3 * g)) * L <= te else (g, hi)
        N["TFlightsRes"] = f"{100 * g:.0f}"
    d_, v_, K_ = .02, .1, 20; N["TSelN"] = f"{round((2 * v_ / d_ ** 2 + 4 / (3 * d_)) * math.log((K_ - 1) / .05), -2):.0f}"     # theorem 3 (1): candidates more than 2 points below the best are excluded
# ---- taxi, vision, corpus, multimodal and decision numbers come from the files that summaries.py writes
for extra in ("numbers_taxi.json", "numbers_vision.json", "numbers_corpus.json", "numbers_mm.json", "numbers_decision.json", "numbers_search.json", "numbers_totals.json"):
    e = load(extra)
    if e: N.update(e)
# ---- the main table and the share at which repair is best, from the runs of the tables
def main_table():
    rows = [r for r in jsonl("results_decision.jsonl") if "test" in r]; fine = [r for r in jsonl("results_decision_fine.jsonl") if "test" in r]
    if len(fine) >= len(rows): rows = fine
    if not rows: return None
    G = {}
    for r in rows: G.setdefault((r["ds"], r["model"], r["seed"]), {})[(r["action"], r["f"])] = r
    CLS = [("adult", "Adult"), ("beers", "Beers"), ("breast_cancer", "Breast"), ("flights", "Flights"), ("hospitals", "Hosp."), ("smartfactory", "SmartF."), ("soccer_full", "Soccer")]; REG = [("bike", "Bike"), ("mercedes", "Merc."), ("nasa", "NASA"), ("soilmoisture", "Soil")]
    def arm(a, f): return lambda arms: arms[(a, f)]["test"] if (a, f) in arms else None
    def task(n):
        def pick(arms):
            def v(r):
                av = sorted(int(x) for x in r["val"]); use = max([x for x in av if x <= n] or av[:1]); return r["val"][str(use)]
            return max(arms.values(), key=v)["test"]
        return pick
    NS = (25, 50, 100, 200, 400, 800)
    GROUPS = [("Fixed action", [("Keep the raw table", arm("keep", 0.0)), (r"Repair 25\%", arm("repair_records", .25)), (r"Repair 50\%", arm("repair_records", .5)), (r"Repair 75\%", arm("repair_records", .75)), ("Repair all (static rule)", arm("repair_records", 1.0)), (r"Delete 25\%", arm("delete_records", .25)), (r"Delete 50\%", arm("delete_records", .5)), (r"Delete 75\%", arm("delete_records", .75))]),
              ("Repair one error type", [("Missing values", arm("repair_type_missing", 1.0)), ("Out-of-domain values", arm("repair_type_domain", 1.0)), ("In-domain wrong values", arm("repair_type_value", 1.0)), ("Labels", arm("repair_labels", 1.0))]),
              (r"Repair half of the errors by column", [("Important columns first", arm("repair_cols_high", .5)), ("Unimportant columns first", arm("repair_cols_low", .5))]),
              (r"Choice by task test on $n$ clean validation records", [(f"$n={n}$", task(n)) for n in NS])]
    ORACLE = ("Best in hindsight", lambda arms: max(r["test"] for r in arms.values()))
    cols = [d for d, _ in CLS] + [d for d, _ in REG]
    def score(fn, ds):
        v = [fn(arms) for k, arms in G.items() if k[0] == ds]; v = [x for x in v if x is not None]; return round(mean(v), 3)
    vals = {}
    for _, items in GROUPS:
        for name, fn in items: vals[name] = {d: score(fn, d) for d in cols}
    orc = {d: score(ORACLE[1], d) for d in cols}
    for name, v in list(vals.items()) + [(ORACLE[0], orc)]:
        v["mc"] = round(mean(v[d] for d, _ in CLS), 3); v["mr"] = round(mean(v[d] for d, _ in REG), 3)
    names = list(vals)
    def rank_col(c):
        xs = sorted(((vals[n][c], n) for n in names), reverse=True); out = {}; i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]: j += 1
            for k in range(i, j + 1): out[xs[k][1]] = (i + j) / 2 + 1
            i = j + 1
        return out
    per = {c: rank_col(c) for c in cols}; avg = {n: mean(per[c][n] for c in cols) for n in names}
    order = sorted(names, key=lambda n: (avg[n], -(vals[n]["mc"] + vals[n]["mr"]))); rank = {n: i + 1 for i, n in enumerate(order)}
    def cell(n, c):
        v = vals[n][c]; s_ = f"{v:.3f}"; top = sorted({vals[m_][c] for m_ in names}, reverse=True)
        if v == top[0]: return f"\\best{{{s_}}}"
        if len(top) > 1 and v == top[1]: return f"\\secondbest{{{s_}}}"
        return s_
    ALL = [d for d, _ in CLS] + ["mc"] + [d for d, _ in REG] + ["mr"]; ncol = 2 + len(ALL)
    best_task = min((n for n in names if n.startswith("$n=")), key=lambda n: rank[n])
    L = ["\\begin{table}[!t]", "\\cnentablecaption{11 张表上各种数据版本的测试分数，每个数值为 5 类模型的平均，分数越大越好。行按得到版本的方式分组，每列的第一名和第二名分别用深色和浅色标出，Rank 为各列名次平均后的排名，事后最优只作为上界，不参与排名。静态规则总是选择错误最少的版本，在这些表上就是修复全部错误的版本，依据任务测试的选择只用少量干净验证样本。}{Test scores of the data versions on the 11 tables, each value is the mean of 5 model families, higher is better. Rows are grouped by how the version is obtained, the first and second place of each column are shaded dark and light, Rank is the rank of the mean place over the columns, and the best version in hindsight is an upper bound outside the ranking. The static rule always picks the version with the fewest errors, which on these tables is the fully repaired one, and the choice by a task test uses only a small clean validation sample.}", "\\label{tab:main}", "\\footnotesize", "\\tabcolsep 1.2pt", "\\renewcommand{\\arraystretch}{0.92}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rr|" + "c" * (len(CLS) + 1) + "|" + "c" * (len(REG) + 1) + "@{}}", "\\toprule", f"\\multicolumn{{2}}{{r|}}{{Task / Dataset}} & \\multicolumn{{{len(CLS) + 1}}}{{c|}}{{Classification (Macro-F1)}} & \\multicolumn{{{len(REG) + 1}}}{{c}}{{Regression ($R^2$)}}\\\\", "Rank & Data version & " + " & ".join(n for _, n in CLS) + " & Mean & " + " & ".join(n for _, n in REG) + " & Mean\\\\", "\\midrule"]
    for g, items in GROUPS:
        L.append(f"\\grp{{{ncol}}}{{{g}}}\\\\")
        for name, _ in items:
            pre = "\\oursrow " if name == best_task else ""
            L.append(pre + f"{rank[name]} & {name} & " + " & ".join(cell(name, c) for c in ALL) + "\\\\")
    L += ["\\midrule", f" & {ORACLE[0]} & " + " & ".join(f"{orc[c]:.3f}" for c in ALL) + "\\\\", "\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    print("main table ranks", {n: (rank[n], round(avg[n], 2)) for n in order})
    return "\n".join(L)
RT = {}
for fn in ("results_real_tables.jsonl", "results_tables_fine.jsonl"):
    for r in jsonl(fn):
        if "score" in r: RT[(r["ds"], r["model"], r["seed"], r["action"], r["f"])] = r["score"]
if RT:
    A_ = {}
    for (d_, m_, s_, a_, f_), v in RT.items():
        if a_ == "repair_records": A_.setdefault((d_, f_), []).append(v)
    tabs = {d_ for d_, _ in A_}; non = 0
    for d_ in tabs:
        cur = {f_: round(mean(v), 3) for (x, f_), v in A_.items() if x == d_}
        if 1.0 in cur and max(cur.values()) > cur[1.0]: non += 1
    N["TShareNonMono"] = f"{non}"
import big_tables as BT
N.update(BT.counts()); N.update(BT.main_table(R)[1])
# differences and ratios are computed from unrounded means where the values are read above and in summaries.py
def num(k):
    try: return float(str(N[k]).replace("$-$", "-"))
    except (KeyError, ValueError): return None
# ---- split gap over all window lengths and families
Tw = jsonl("results_tep_windows.jsonl")
if Tw:
    g_ = {}
    for r in Tw: g_.setdefault((r["model"], r["length"]), {}).setdefault(r["split"], []).append(r["macro_f1"])
    N["TEPgapWinMin"] = f"{min(100 * (mean(v['random']) - mean(v['run'])) for v in g_.values()):.1f}"
# ---- detectors on the tables with benchmark errors, 200 labeled records, record level, tables with a true rate between 1 and 99 percent
Dt = [r for r in jsonl("results_detectors.jsonl") if r.get("unit") == "record" and 0.01 < r.get("true_rate", 0) < 0.99]
if Dt:
    def dt(method, field="mean_abs_error", n=200): return mean(r[field] for r in Dt if r["method"] == method and (r["n"] == n or method == "raw") and r.get(field) is not None)
    for k_, m_ in (("DetRawMAE", "raw"), ("DetCorMAE", "corrected"), ("DetSamMAE", "sample"), ("DetPostMAE", "poststratified"), ("DetStrMAE", "stratified"), ("DetPoolMAE", "pooled")): N[k_] = f1(100 * dt(m_))
    N["DetCorOverSam"] = f1(dt("corrected") / dt("sample"))
    N["DetPoolCov"] = f"{100 * dt('pooled', 'coverage'):.0f}"; N["DetSamCov"] = f"{100 * dt('sample', 'coverage'):.0f}"; N["DetPostCov"] = f"{100 * dt('poststratified', 'coverage'):.0f}"; N["DetTables"] = f"{len({r['ds'] for r in Dt})}"
# ---- documentation scores of the public documents, judgment items are those whose quote needs a reading of the data
Do = load("dataset_docs.json")
if Do:
    N["DocN"] = f"{len(Do)}"; N["DocMean"] = f1(mean(r["score"] for r in Do)); N["DocPass"] = f"{sum(1 for r in Do if r['score'] >= 90)}"
    def strict(r):  # each item is worth 5 points, items that needed a judgment of truth are listed in the note and count zero here
        m = re.search(r"判断项[:：]\s*([A-Z]\d(?:[、,]\s*[A-Z]\d)*)", r.get("note") or ""); return r["score"] - 5 * sum(r["items"].get(x.strip(), 0) for x in re.split("[、,]", m.group(1))) if m else r["score"]
    N["DocMeanJudge"] = f1(mean(strict(r) for r in Do)); N["DocPassJudge"] = f"{sum(1 for r in Do if strict(r) >= 90)}"
# ---- documentation scores against the other columns of the main table
if Do:
    try:
        from scipy import stats as _st2
        import big_tables as _bt
        from datasets import all_datasets as _ad, DOC_NAME as _dn
        Dm, Sm, Bm = _bt._docs(R), _bt._static(R), _bt.protocol_rows(R); Am = {}
        for f_ in (_bt.table_versions, _bt.cleanml_versions, _bt.corpus_versions, _bt.taxi_versions, _bt.native_versions): Am.update(f_(R))
        x1, x2, x3 = [], [], []
        for d_ in _ad():
            v_ = Dm.get(_dn.get(d_["name"], d_["name"]))
            if v_ is None: continue
            if Sm.get(d_["key"]) is not None: x1.append((v_, Sm[d_["key"]]))
            a_ = Am.get(d_["key"]) if d_["part"] == "versions" else None
            if a_ and None not in a_: x2.append((v_, a_[0] - a_[3] if d_["metric"] == "Loss" else a_[3] - a_[0]))
            b_ = Bm.get(d_["key"]) if d_["part"] != "versions" else None
            if b_ and None not in b_: x3.append((v_, abs(b_[0] - b_[1])))
        for k_, q_ in (("DocRhoStatic", x1), ("DocRhoGain", x2), ("DocRhoGap", x3)): N[k_] = f2(float(_st2.spearmanr(*zip(*q_)).statistic))
    except Exception as e_: print("doc rho", repr(e_), file=sys.stderr)
# ---- CleanML, gain of each cleaning method over the dirty version
Cm = [r for r in jsonl("results_cleanml.jsonl") if "error" not in r and "test" in r]
if Cm:
    from scipy import stats as _st
    G_ = {}
    for r in Cm: G_.setdefault((r["ds"], r["error_type"], r["model"], r["method"]), {})[r["seed"]] = r["test"]
    P_ = []
    for k_, v in G_.items():
        if k_[3] == "dirty": continue
        b_ = G_.get((k_[0], k_[1], k_[2], "dirty"), {}); ss = sorted(set(v) & set(b_))
        if len(ss) >= 3:
            d_ = [v[x] - b_[x] for x in ss]
            P_.append((1.0 if all(abs(x) < 1e-12 for x in d_) else float(_st.ttest_1samp(d_, 0).pvalue), mean(d_)))
    m_ = len(P_); cm_ = sum(1 / (i + 1) for i in range(m_)); order = sorted(range(m_), key=lambda i: P_[i][0]); q_ = [1.0] * m_; low = 1.0
    for rank, i in reversed(list(enumerate(order))):
        low = min(low, P_[i][0] * m_ * cm_ / (rank + 1)); q_[i] = low
    up = sum(1 for i in range(m_) if q_[i] < 0.05 and P_[i][1] > 0); dn = sum(1 for i in range(m_) if q_[i] < 0.05 and P_[i][1] < 0)
    N["CMUp"] = f"{100 * up / m_:.1f}"; N["CMDown"] = f"{100 * dn / m_:.1f}"; N["CMSame"] = f"{100 * (m_ - up - dn) / m_:.1f}"; N["CMSettings"] = f"{m_}"
    T_ = {}
    for r in Cm: T_.setdefault(r["ds"], {}).setdefault((r["error_type"], r["method"]), []).append(r["test"])
    cross = 0
    for ds_, a_ in T_.items():
        g_ = [100 * (mean(v) - mean(a_[(e_, "dirty")])) for (e_, m2), v in a_.items() if m2 != "dirty" and (e_, "dirty") in a_]
        if g_ and min(g_) < 0 < max(g_): cross += 1
    N["CMTables"] = f"{len(T_)}"; N["CMCross"] = f"{cross}"
# ---- WILDS, CIFAR-10.1, synthetic images, image-text pools
W_ = jsonl("results_wilds.jsonl")
x_ = [r for r in W_ if str(r.get("ds", "")).startswith("camelyon")]
if x_: N["CamID"] = f3(mean(r["score_id"] for r in x_)); N["CamOOD"] = f3(mean(r["score_ood"] for r in x_))
x_ = [r for r in W_ if str(r.get("ds", "")).startswith("civil")]
if x_: N["CivAvg"] = f3(mean(r["avg_acc"] for r in x_)); N["CivWorst"] = f3(mean(r["worst_group_acc"] for r in x_)); N["CivGap"] = f3(mean(r["avg_acc"] for r in x_) - mean(r["worst_group_acc"] for r in x_))
Sy = [r for r in jsonl("results_synth.jsonl") if "error" not in r]
x_ = [r for r in Sy if r.get("gen") == "real" and "score_c101" in r]
if x_: N["COneGap"] = f1(100 * mean(r["score"] - r["score_c101"] for r in x_))
base_ = {}
for r in Sy:
    if r.get("gen") == "real" and r.get("arch") == "resnet18": base_[r["seed"]] = r["score"]
def syn(gen, field, value):
    v = [100 * (r["score"] - base_[r["seed"]]) for r in Sy if r.get("gen") == gen and r.get("arch") == "resnet18" and r.get(field) == value and r["seed"] in base_]
    return mean(v) if v else None
for g_, k_ in (("edm", "SynRepEDM"), ("ddpm", "SynRepDDPM"), ("cifake", "SynRepCIFAKE")):
    if syn(g_, "share", 1.0) is not None: N[k_] = f1(-syn(g_, "share", 1.0))
if syn("edm", "add", 100000) is not None: N["SynAddEDM"] = f1(syn("edm", "add", 100000))
def itp(fn, arm_, fr): return mean(r["test_r5"] for r in jsonl(fn) if r.get("arm") == arm_ and abs(r.get("frac", -1) - fr) < 1e-9)
if jsonl("results_datacomp.jsonl"): N["TopGainDC"] = f1(100 * (itp("results_datacomp.jsonl", "sim_top", .05) - itp("results_datacomp.jsonl", "random", .05))); N["LowLossDC"] = f1(100 * (itp("results_datacomp.jsonl", "random", .2) - itp("results_datacomp.jsonl", "sim_bottom", .2)))
if jsonl("results_mirflickr.jsonl"): N["TopGainMF"] = f1(100 * (itp("results_mirflickr.jsonl", "sim_top", .05) - itp("results_mirflickr.jsonl", "random", .05)))
# ---- choice among versions, where the gain of the task test comes from, the same replay as the figure
Df = [r for r in jsonl("results_decision_fine.jsonl") if "test" in r]
if Df:
    G_ = {}
    for r in Df: G_.setdefault((r["ds"], r["model"], r["seed"], r["task"]), {})[(r["action"], r["f"])] = r
    def pick_(a, n):
        def v(r):
            av = sorted(int(x) for x in r["val"]); return r["val"][str(max([x for x in av if x <= n] or av[:1]))]
        return max(a.values(), key=v)
    Rw = []
    for k_, a in G_.items():
        if ("keep", 0.0) not in a or len(a) < 14: continue
        b_ = a[("keep", 0.0)]["test"]; st_ = min(a.values(), key=lambda r: (r["rate_missing"] + r["rate_domain"] + r["rate_value"], r["rate_label"], -r["n_train"]))["test"]; or_ = max(r["test"] for r in a.values())
        Rw.append(dict(ds=k_[0], model=k_[1], task=k_[3], s=100 * (st_ - b_), t=100 * (pick_(a, 100)["test"] - b_), o=100 * (or_ - b_), low=st_ < b_ - 1e-9, best=abs(st_ - or_) < 1e-9))
    N["DecCases"] = f"{len(Rw)}"; N["DecStaticBest"] = f"{sum(r['best'] for r in Rw)}"; N["DecStaticLow"] = f"{sum(r['low'] for r in Rw)}"
    def rw(f, fld): return f1(mean(r[fld] for r in Rw if f(r)))
    for k_, f_ in (("Flights", lambda r: r["ds"] == "flights"), ("Hospitals", lambda r: r["ds"] == "hospitals"), ("Mercedes", lambda r: r["ds"] == "mercedes"), ("OtherCls", lambda r: r["task"] == "cls" and r["ds"] not in ("flights", "hospitals")), ("KnnCls", lambda r: r["task"] == "cls" and r["model"] == "knn"), ("RestCls", lambda r: r["task"] == "cls" and r["model"] != "knn"), ("MlpReg", lambda r: r["task"] == "reg" and r["model"] == "mlp"), ("RestReg", lambda r: r["task"] == "reg" and r["model"] != "mlp")):
        N["Dc" + k_ + "S"] = rw(f_, "s").replace("-", "$-$"); N["Dc" + k_ + "T"] = rw(f_, "t").replace("-", "$-$")
# ---- corpus, loss gaps in hundredths, each arm against random of the same seed, model M at 30M tokens unless stated
Co = [r for r in jsonl("results_corpus.jsonl") if "error" not in r]
if Co:
    def gap(arm_, fld, size="M", bud=30000000):
        b_ = {r["seed"]: r[fld] for r in Co if r["arm"] == "random" and r["size"] == size and r.get("budget") == bud and r.get(fld) is not None}
        v = [r[fld] - b_[r["seed"]] for r in Co if r["arm"] == arm_ and r["size"] == size and r.get("budget") == bud and r["seed"] in b_ and r.get(fld) is not None]; return 100 * mean(v) if v else float("nan")
    for k_, a_ in (("Low", "ppl_low"), ("High", "ppl_high"), ("Qwen", "llm_high"), ("Both", "rule_dedup")):
        N["Corpus" + k_ + "Web"] = f1(gap(a_, "nll_c4")); N["Corpus" + k_ + "Wiki"] = f1(gap(a_, "nll_wiki"))
    N["CorpusLowWikiXS"] = f1(gap("ppl_low", "nll_wiki", "XS")); N["CorpusLowWikiL"] = f1(gap("ppl_low", "nll_wiki", "L")); N["CorpusLowWikiRatio"] = f1(gap("ppl_low", "nll_wiki", "L") / gap("ppl_low", "nll_wiki", "XS")); N["CorpusHighWebXS"] = f1(gap("ppl_high", "nll_c4", "XS")); N["CorpusHighWebL"] = f1(gap("ppl_high", "nll_c4", "L"))
    N["CorpusLowWebThree"] = f1(gap("ppl_low", "nll_c4", "M", 3000000))
    for k_ in list(N):
        if k_.startswith("Corpus") and N[k_].startswith("-"): N[k_] = "$-$" + N[k_][1:]
    def nll(arm_, fld, size="M", bud=30000000): return mean(r[fld] for r in Co if r["arm"] == arm_ and r["size"] == size and r.get("budget") == bud and r.get(fld) is not None)
    N["CorpusBudgetDrop"] = f"{100 * (nll('random', 'nll_c4') - nll('all', 'nll_c4', 'M', None)):.0f}"; N["CorpusDCLMWeb"] = f1(100 * (nll("dclm_pool", "nll_c4") - nll("dclm_baseline", "nll_c4"))); N["CorpusDCLMWiki"] = f1(100 * (nll("dclm_pool", "nll_wiki") - nll("dclm_baseline", "nll_wiki")))
# ---- language model baselines
Lc = load("results_llm_cells.json"); Lt = [r for r in jsonl("results_llm_tables.jsonl") if "error" not in r and "test" in r]
if Lc:
    cf = {k_: v.get("cells_features", v["cells"]) for k_, v in Lc.items()}
    N["LLMTables"] = f"{len(cf)}"; N["LLMDamage"] = f"{sum(1 for v in cf.values() if v['damaged'] > v['repaired'])}"; N["LLMBestF"] = f2(max(v["f1"] or 0 for v in cf.values())); fs_ = [v["f1"] for v in cf.values() if v["f1"] is not None]; N["LLMMedF"] = f2(statistics.median(fs_))
if Lt and Df:
    kp, ra = {}, {}
    for r in Df:
        if r["action"] == "keep": kp[(r["ds"], r["model"], r["seed"])] = r["test"]
        if r["action"] == "repair_records" and r["f"] == 1.0: ra[(r["ds"], r["model"], r["seed"])] = r["test"]
    by = {}
    for r in Lt:
        k_ = (r["ds"], r["model"], r["seed"])
        if k_ in kp and k_ in ra: by.setdefault(r["ds"], []).append((r["test"] - kp[k_], r["test"] - ra[k_]))
    N["LLMBelowKeep"] = f"{sum(1 for v in by.values() if mean(x[0] for x in v) < 0)}"; N["LLMBelowRepair"] = f"{sum(1 for v in by.values() if mean(x[1] for x in v) < 0)}"; N["LLMOnlyMerc"] = ", ".join(sorted(k_ for k_, v in by.items() if mean(x[1] for x in v) >= 0))
Ll = jsonl("results_llm_labels.jsonl")
if Ll:
    def ll(ds, sh): return next(r for r in Ll if r["ds"] == ds and r["shots"] == sh)
    N["LLMImdbAcc"] = f3(ll("imdb", 5)["acc_original"]); N["LLMNewsAcc"] = f3(ll("20news", 5)["acc_original"]); x_ = ll("imdb", 5); N["LLMImdbRecall"] = f"{100 * x_['confirmed_recall']:.0f}"; N["LLMImdbPrec"] = f"{100 * x_['confirmed_flagged'] / x_['flagged_reviewed']:.0f}"
Lj = load("results_llm_judge.json")
if Lj: N["LLMJudgeN"] = f"{Lj['scored']}"; N["LLMJudgeMed"] = f2(Lj["quantiles"]["0.5"]); N["LLMJudgeQ"] = f2(Lj["quantiles"]["0.9"])
# ---- intervals of the finite-population method against the bootstrap on the corrected rate, record level
Fp = [r for r in jsonl("results_finite_population.jsonl") if r.get("unit") == "record" and r.get("n") == 200 and 0.01 < r.get("true_rate", 0) < 0.99 and r.get("method") == "finite_poststratified"]
if Fp:
    N["FpCov"] = f"{100 * mean(r['coverage'] for r in Fp):.0f}"; N["FpHalf"] = f1(100 * mean(r["mean_half_width"] for r in Fp)); N["DetPostHalf"] = f1(100 * mean(r["mean_half_width"] for r in Dt if r["method"] == "poststratified" and r["n"] == 200))
    N["DetCorHalf"] = f1(100 * mean(r["mean_half_width"] for r in Dt if r["method"] == "corrected" and r["n"] == 200)); N["DetCorCovMin"] = f"{100 * min(r['coverage'] for r in jsonl('results_detectors.jsonl') if r.get('method') == 'corrected' and r.get('unit') == 'record' and r.get('n') == 50 and r.get('ds') == 'smartfactory' and r.get('detector') == 'D1'):.1f}"
if num("MMcorr") is not None and num("MMtrue") is not None: N["MMcorrGap"] = f"{100 * abs(num('MMcorr') - num('MMtrue')):.1f}"
# ---- cross-experiment findings, computed in insights.py; a name defined twice stops the script
import insights
INS = insights.macros(R); clash = sorted(set(INS) & set(N))
if clash: raise SystemExit(f"macro names defined both here and in insights.py: {' '.join(clash)}")
N.update(INS)
if TEX is None:
    print(json.dumps(N, indent=1, ensure_ascii=False, sort_keys=True)); raise SystemExit
s = open(TEX, encoding="utf-8").read(); used = sorted(set(re.findall(r"\\(Fp[A-Z][a-z]+|Dc[A-Z][A-Za-z]*|Det[A-Z][A-Za-z]*|Doc[A-Z][A-Za-z]*|CM[A-Z][a-z]+|Cam[A-Z]+|Civ[A-Z][a-z]+|COneGap|Syn[A-Z][A-Za-z]*|TopGain[A-Z]+|LowLoss[A-Z]+|LLM[A-Z][A-Za-z]*|TEP[A-Za-z]*|Srch[A-Za-z]*|Study[A-Za-z]*|Cards[A-Za-z]*|Refs[A-Za-z]*|T[A-Z][a-zA-Z]*|Vis[A-Z][A-Za-z]*|Rieth[A-Za-z]*|MM[A-Za-z]+|Corpus[A-Za-z]*|Taxi[A-Za-z]*|Runs[A-Za-z]*|Fair[A-Za-z]*|Dec[A-Z][A-Za-z]*|Leak[A-Z][a-z]*|Ins[A-Z][A-Za-z]*)\b", s[s.index("\\begin{document}"):])))
used = [u for u in used if u not in ("TeX",)]
lines = []; missing = []
for k in used:
    if k in N: lines.append(f"\\newcommand{{\\{k}}}{{{N[k]}}}")
    else: lines.append(f"\\newcommand{{\\{k}}}{{\\tbd{{{k}}}}}"); missing.append(k)
b = "% ===== NUMBERS END ====="
i, j = re.search(r"% ===== NUMBERS BEGIN[^\n]*\n", s).end(), s.index(b); s = s[:i] + "\n".join(lines) + "\n" + s[j:]
tab = BT.main_table(R)[0]; ma, mb = "% ===== MAIN TABLE BEGIN =====\n", "% ===== MAIN TABLE END ====="
if tab and ma in s: i, j = s.index(ma) + len(ma), s.index(mb); s = s[:i] + tab + "\n" + s[j:]
da, db = "% ===== DATASET TABLE BEGIN =====\n", "% ===== DATASET TABLE END ====="
if da in s: i, j = s.index(da) + len(da), s.index(db); s = s[:i] + BT.dataset_table(R) + "\n" + s[j:]
open(TEX, "w", encoding="utf-8").write(s); print("defined", len(used) - len(missing), "missing", len(missing)); print(" ".join(missing))
