#!/usr/bin/env python3
"""The dataset table and the main table of the survey, generated line by line from the result files.
Run as a script with the results directory as argument to print the red cells and the win counts per group."""
import json, os, statistics as st
from datasets import DOC_NAME, GROUPS, STRATEGIES, all_datasets
TBD = r"\tbd{.}"
BARL = "l@{\\hspace{2pt}}|@{\\hspace{2pt}}"     # the end of a left panel
def mean(x):
    x = [v for v in x if v is not None]; return st.mean(x) if x else None
def _f(v, d=3): return TBD if v is None else (f"{v:.{d}f}" if d else f"{v:.0f}")
def _signed(v):
    """a difference, with a real minus sign and no negative zero"""
    if v is None: return TBD
    s = f"{abs(v):.3f}"; return s if v >= 0 or float(s) == 0 else f"$-${s}"
def _load(R, fn): return json.load(open(f"{R}/{fn}")) if os.path.exists(f"{R}/{fn}") else None
def _jsonl(R, fn):
    p = f"{R}/{fn}"; out = []
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
                if "error" not in r: out.append(r)
            except Exception: pass
    return out
def _wilds(R):
    """the WILDS runs, by id, whichever file of the servers they arrived in"""
    return list({r["id"]: r for fn in ("results_camelyon.jsonl", "results_wilds.jsonl") for r in _jsonl(R, fn)}.values())
def _label_errors(R):
    L = _load(R, "label_errors.json") or {}
    for k, v in (_load(R, "results_audioset_labels.json") or {}).items(): L.setdefault(k, {}).update(v)     # AudioSet is multi-label and has its own script
    return L
def _multimodal(R): return list({r.get("id", i): r for i, r in enumerate(_jsonl(R, "results_multimodal.jsonl") + _jsonl(R, "results_multimodal_fine.jsonl"))}.values())
MISMATCH = .2     # the mismatch share at which the text of Section 6 reports the detector
def _coco(R): return [r for r in _multimodal(R) if r.get("defects") == {"mismatch": MISMATCH} and r.get("consistency_est") is not None]
# ---- the dataset table
def records(R):
    """Records used in the assessment per dataset, a static scan over a whole table counts as use. Read from the result files where they are stored, the sizes in the first line are those of the published files."""
    out = dict(cifar10n=60000, cifar100n=60000, edm=200000, ddpm=100000, cifake=50000, cifar101=2000, coverage=60000, cifair10=10000, cifair100=10000, mirflickr=25000, dclm=495393, c4dup=4000)     # dclm: the two tokenized corpora, dclm_prep.log. c4dup: the documents scored in run_corpus.py
    out.update(_load(R, "table_sizes.json") or {}); C = _load(R, "case_tep.json"); Q = _load(R, "results_rieth.json"); X = _load(R, "results_taxi.json"); Z = _load(R, "corpus_stats.json")
    if C: out["tep"] = C["scan"]["records"]
    if Q: out["tepl"] = Q["records_total"]
    if X: out["taxi"] = X["records_total"]
    if Z: out["c4"] = Z["docs"]
    for k, v in _label_errors(R).items(): out[k] = v["n"]
    for r in _jsonl(R, "results_cleanml.jsonl"):
        if r.get("method") == "dirty": out[r["ds"]] = max(out.get(r["ds"], 0), r["records_dirty"] + r["records_dirty_test"])
    for r in _jsonl(R, "results_datacomp.jsonl")[:1]: out["datacomp"] = r["n_pool"]
    for r in _wilds(R):
        if r["id"].startswith("civil"): out["civil"] = r["n_train"] + r["n_val"] + r["n"]
        elif r["id"].startswith("cam"): out["camelyon"] = r["n_train"] + r["n_id"] + r["n_ood"]
    M = _coco(R)
    if M: out["coco"] = max(r["n_train"] for r in M) + M[0]["n_test"]
    out.update(_load(R, "results_sizes_native.json") or {}); return out
def rec(n):
    if n is None: return TBD
    if n < 1e6: return f"{n}"
    return f"{n / 1e6:.2f}M" if n / 1e6 < 9.95 else f"{n / 1e6:.1f}M"
S3, S4, S5 = "\\S\\ref{sec:dimension}", "\\S\\ref{sec:method}", "\\S\\ref{sec:newdata}"
HEAD = [("Tables with benchmark errors", S3, "", "Macro-F1 or $R^2$"), ("Tables with real errors", S3, "", "Macro-F1"), ("Image labels", S3, "", "accuracy"), ("Language-model corpora", S5, "", "loss"), ("Multimodal data", S5, "", "Recall@5"), ("Synthetic data", S5, "", "accuracy"), ("Dynamic data", S5, "", "$R^2$"), ("Evaluation protocol", S3, "", "Macro-F1 or accuracy"), ("Data leakage", S5, "", "accuracy or loss"), ("Test-set labels", S3, "", "accuracy"), ("Detector estimates", S4, "", "share of matched pairs")]
# every dataset row carries its citation. Where the dataset list gives none or only the benchmark, the original source is added when it is in the bibliography.
# NYC taxi: Ferreira et al. study the same TLC trip records.
ROW_CITE = {"taxi": "ferreira2013taxi", "cifar10": "krizhevsky2009cifar,northcutt2021pervasive", "cifar100": "krizhevsky2009cifar,northcutt2021pervasive", "beers": "mahdavi2019raha,abdelaal2023rein", "flights": "mahdavi2019raha,ding2025uniclean", "hospitals": "mahdavi2019raha,ding2025uniclean"}
ABBR = {"Classification": "C", "Regression": "R", "Language modeling": "LM", "Retrieval": "IR", "Fault diagnosis": "C"}
INDENT = "\\hspace{0.8ex}"
PANEL_STUDY = 5     # the groups before this index fill the left panel of the dataset table
def _ds_lines(d, S):
    """the printed lines of one dataset, the other columns stand on the first line"""
    name = d["display"].split("|"); errs = d["errors"].split("|"); cite = ROW_CITE.get(d["key"], d.get("cite"))
    if cite: name[-1] += f"~\\cite{{{cite}}}"
    L = [[(INDENT if i else "") + (name[i] if i < len(name) else ""), "" if i else f"{d['type']}, {ABBR[d['task']]}", errs[i] if i < len(errs) else "", "" if i else rec(S.get(d["key"]))] for i in range(max(len(name), len(errs)))]
    return L
def dataset_table(R):
    S = records(R); left, right = [], []
    for gi, (g, ds) in enumerate(GROUPS):
        t, sec, cite, metric = HEAD[gi]; tgt = left if gi < PANEL_STUDY else right
        if tgt: tgt.append(("rule",))
        tgt.append(("g", f"{t} ({sec}){cite}, {metric}"))
        for d in ds: tgt += [("r", x) for x in _ds_lines(d, S)]
    L = ["\\begin{table}[!tp]", "\\cnentablecaption{按数据类别汇总的公开数据集, 第~\\ref{sec:empirical}~节在这些数据集上测评. 分组行给出对应章节和评价指标, 数据集列写全称, 缩写加粗. Type, task 列为数据类型和任务, 其中 Mixed, Num., Cat. 和 Pairs 分别为数值和类别混合、数值型、类别型和图文对, C, R, LM 和 IR 分别为分类、回归、语言建模和信息检索. Errors 列为数据问题的来源, 其中 Syn. 和 Sem. 分别为语法错误 (syntactic error) 和语义错误 (semantic error), Dup. 为重复记录 (duplicate). Records 为测评使用的记录数, M 表示百万.}{Public datasets by data category, assessed in Section~\\ref{sec:empirical}. Each group row gives the section and the metric, the dataset column gives the full name with the abbreviation in bold. Type, task gives the data type and the task, Mixed, Num., Cat. and Pairs mean numerical and categorical columns together, numerical, categorical and image-text pairs, C, R, LM and IR mean classification, regression, language modeling and information retrieval. Errors gives the origin of the data problem, Syn. and Sem. mean syntactic and semantic errors, Dup. means duplicates. Records is the number of records used in the assessment, M means million.}", "\\label{tab:study}", "\\footnotesize", "\\tabcolsep 1pt", "\\renewcommand{\\arraystretch}{0.91}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllr@{\\hspace{2pt}}|@{\\hspace{2pt}}lllr@{}}", "\\toprule", "Dataset & Type, task & Errors & Records & Dataset & Type, task & Errors & Records\\\\", "\\midrule"]
    def cells(x, bar):
        if x is None: return "&&&"
        if x[0] == "g": return f"\\multicolumn{{4}}{{{BARL if bar else 'l'}}}{{\\textbf{{\\textit{{{x[1]}}}}}}}"
        return " & ".join(x[1])
    def seq(side):
        """the lines of a panel, a rule between two groups stands before the group row, as in the main table"""
        out = []; rule = False
        for x in side:
            if x[0] == "rule": rule = True; continue
            out.append((x, rule)); rule = False
        return out
    a_, b_ = seq(left), seq(right)
    for i in range(max(len(a_), len(b_))):
        x, rx = a_[i] if i < len(a_) else (None, False); y, ry = b_[i] if i < len(b_) else (None, False)
        if rx or ry: L.append(("\\cmidrule(r){1-4}" if rx else "") + ("\\cmidrule(l){5-8}" if ry else ""))
        L.append(cells(x, True) + " & " + cells(y, False) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(L)
def counts():
    return dict(StudyDatasets=f"{len(all_datasets())}", StudyCategories=f"{len(GROUPS)}", StudyStrategies=f"{sum(len(v) for v in STRATEGIES.values())}", StudyCleanmlN=f"{len(GROUPS[1][1])}")
# ---- data versions: raw, static rule, task test, best in hindsight
WIN = 400
def _val(r, n=WIN):
    av = sorted(int(x) for x in r["val"]); use = max([x for x in av if x <= n] or av[:1]); return r["val"][str(use)]
def _pick_val(arms, n=WIN):
    """the test score of the version that a validation sample of n records chooses, versions tied on it count equally"""
    top = max(_val(r, n) for r in arms.values()); return mean(r["test"] for r in arms.values() if _val(r, n) == top)
def _complete(G):
    """the groups with every version of their model, a run still in progress must not choose among fewer versions"""
    most = {}
    for k, a in G.items(): most[k[0]] = max(most.get(k[0], 0), len(a))
    return {k: a for k, a in G.items() if len(a) == most[k[0]]}
def _versions(rows, version, static):
    """Raw, static rule, task test and best in hindsight from the runs of one dataset, the mean over models and seeds."""
    G = {}
    for r in rows: G.setdefault((r["model"], r["seed"]), {})[version(r)] = r
    raw, sta, tas, bes = [], [], [], []
    for arms in _complete(G).values():
        v0 = static["raw"](arms)
        if v0 is None: continue
        raw.append(v0["test"]); s_ = static["pick"](arms); sta.append(s_["test"] if s_ else None); tas.append(_pick_val(arms)); bes.append(max(r["test"] for r in arms.values()))
    return mean(raw), mean(sta), mean(tas), mean(bes)
def _static_tables():
    P = lambda r: r["rate_missing"] + r["rate_domain"] + r["rate_value"]
    return dict(raw=lambda a: a.get(("keep", 0.0)), pick=lambda a: min(a.values(), key=lambda r: (P(r), r["rate_label"], -r["n_train"])))
SYNTH_MODEL = "resnet18"
def table_versions(R):
    """Dataset key to (raw, static rule, task test, best), for the tables with benchmark errors, the image labels and the synthetic images."""
    out = {}; D = _jsonl(R, "results_decision.jsonl"); F = _jsonl(R, "results_decision_fine.jsonl"); D = F if len(F) >= len(D) else D; ver = lambda r: (r["action"], r["f"])
    for ds in {r["ds"] for r in D}: out[ds] = _versions([r for r in D if r["ds"] == ds and "test" in r], ver, _static_tables())
    V = _jsonl(R, "results_versions.jsonl"); labels = dict(raw=lambda a: a.get(("keep", 0.0)), pick=lambda a: a.get(("repair", 1.0))); real = dict(raw=lambda a: a.get(("share", 1.0)), pick=lambda a: a.get(("real", 0.0)))     # the raw data of a synthetic row are the generated images alone, which the Static column also scores. The rate of a version is its synthetic share times one minus precision, so the real images alone have the highest static score
    for ds, st_ in (("cifar10n", labels), ("cifar100n", labels), ("edm", real), ("ddpm", real), ("cifake", real)):
        rows = [r for r in V if r["ds"] == ds and (st_ is labels or r["model"] == SYNTH_MODEL)]     # the synthetic rows use ResNet-18 alone, the network with the full set of synthetic shares and additions, as Figure 5(g)
        if rows: out[ds] = _versions(rows, ver, st_)
    return out
def _static_cleanml(a, err):
    """the versions of CleanML with the highest static score. After any repair of missing values no value is missing, so these versions tie and the ones that keep the most records stay, otherwise the score rises with the cells changed and the records removed."""
    v = [r for k, r in a.items() if k != "dirty"]; size = (lambda r: r["records"]) if err == "missing_values" else (lambda r: r["cells_changed"] + r["records_removed"]); top = max(size(r) for r in v); return [r for r in v if size(r) == top]
def cleanml_versions(R):
    G = {}; out = {}
    for r in _jsonl(R, "results_cleanml.jsonl"):
        if "test" in r: G.setdefault(r["ds"], {}).setdefault((r["model"], r["seed"], r["error_type"]), {})[r["method"]] = r
    for ds, groups in G.items():
        raw, sta, tas, bes = [], [], [], []
        for (_, _, err), a in groups.items():
            if "dirty" not in a or len(a) < 2: continue
            raw.append(a["dirty"]["test"]); sta.append(mean(r["test"] for r in _static_cleanml(a, err))); tas.append(_pick_val(a)); bes.append(max(v["test"] for v in a.values()))
        out[ds] = (mean(raw), mean(sta), mean(tas), mean(bes))
    return out
def corpus_versions(R):
    CO = _jsonl(R, "results_corpus.jsonl"); out = {}; BU = 30000000
    def cross(arms, raw, sta, fld="nll_c4", size="M"):
        """raw, static rule, task test and best on the seeds that have every version. These corpora have no validation part, the task test takes the version with the lowest loss on the other seeds."""
        G = {}
        for r in CO:
            if r.get("arm") in arms and r.get("size") == size and r.get("budget") == BU and fld in r: G.setdefault(r["seed"], {})[r["arm"]] = r[fld]
        G = {s: v for s, v in G.items() if len(v) == len(arms)}
        if not G: return None, None, None, None
        pick = []
        for s, v in G.items():
            others = [u for t, u in G.items() if t != s] or [v]; a = min(arms, key=lambda x: mean(u[x] for u in others)); pick.append(v[a])
        return mean(v[raw] for v in G.values()), mean(v[sta] for v in G.values()), mean(pick), mean(min(v.values()) for v in G.values())
    out["c4"] = cross(["random", "ppl_low", "ppl_mid", "ppl_high", "rule", "dedup", "rule_dedup"], "random", "rule_dedup")     # the static score counts rule failures and near duplicates, rule_dedup has neither
    out["dclm"] = cross(["dclm_pool", "dclm_baseline"], "dclm_pool", "dclm_baseline")
    return out
def taxi_versions(R, H=48):
    X = _load(R, "results_taxi.json")
    if not X: return {}
    keys = sorted(X["months"]); idx = {k: i for i, k in enumerate(keys)}; G = {}
    for r in X["runs"]: G.setdefault((r["train_month"], r.get("seed", 0)), {})[r["arm"]] = r["r2"]
    raw, sta, tas, bes = [], [], [], []
    for (tm, _), a in G.items():
        i = idx[tm] + H
        if i >= len(keys) or len(a) < 2: continue
        k, nx = keys[i], keys[idx[tm] + 1]; raw.append(a["raw"].get(k)); sta.append(a["rule_filtered"].get(k)); ch = max(a, key=lambda z: a[z].get(nx, -9)); tas.append(a[ch].get(k)); bes.append(max(v.get(k, -9) for v in a.values()))
    return dict(taxi=(mean(raw), mean(sta), mean(tas), mean(bes)))
def native_versions(R):
    """Image-text pools. Raw is the whole pool, or the largest random share while the whole pool is missing. The static rule keeps the version with the highest consistency, among equal consistency the one with the most pairs. The task test takes the highest Recall@5 on the COCO validation part, Best the highest on the test part."""
    out = {}
    for key, fn in (("datacomp", "results_datacomp.jsonl"), ("mirflickr", "results_mirflickr.jsonl")):
        G = {}
        for r in _jsonl(R, fn): G.setdefault(r["seed"], {})[(r["arm"], r["frac"])] = r
        raw, sta, tas, bes = [], [], [], []
        for a in G.values():
            rnd = [v for k, v in a.items() if k[0] == "random"]; big = a.get(("all", 1.0)) or (max(rnd, key=lambda v: v["n_kept"]) if rnd else None); top = max(v["consistency"] for v in a.values()); best_val = max(v["val_r5"] for v in a.values())
            raw.append(big["test_r5"] if big else None); sta.append(max((v for v in a.values() if v["consistency"] == top), key=lambda v: v["n_kept"])["test_r5"]); tas.append(mean(v["test_r5"] for v in a.values() if v["val_r5"] == best_val)); bes.append(max(v["test_r5"] for v in a.values()))
        if G: out[key] = (mean(raw), mean(sta), mean(tas), mean(bes))
    return out
# ---- evaluation practices: reported and stricter
def protocol_rows(R):
    """Dataset key to (reported score, stricter score)."""
    out = {}; C = _load(R, "case_tep.json"); Q = _load(R, "results_rieth.json"); F = _load(R, "results_fair.json"); V = _jsonl(R, "results_vision.jsonl"); SY = _jsonl(R, "results_synth.jsonl"); W = _wilds(R); CO = _jsonl(R, "results_corpus.jsonl")
    if C: out["tep"] = (mean(r["macro_f1"] for r in C["runs"] if r["arm"] == "window_random"), mean(r["macro_f1"] for r in C["runs"] if r["arm"] == "window_run"))
    if Q:
        a = lambda k: mean(r["macro_f1"] for r in Q["runs"] if r.get("k") == k and r.get("arm") == "runs_fixed_records")
        out["tepl"] = (a(500), a(1))
    x = [r for r in W if r["id"].startswith("cam")]
    if x: out["camelyon"] = (mean(r["score_id"] for r in x), mean(r["score_ood"] for r in x))
    x = [r for r in W if r["id"].startswith("civil")]
    if x: out["civil"] = (mean(r["avg_acc"] for r in x), mean(r["worst_group_acc"] for r in x))
    x = [r for r in SY if r.get("gen") == "real" and "score_c101" in r]
    if x: out["cifar101"] = (mean(r["score"] for r in x), mean(r["score_c101"] for r in x))
    x = [r for r in V if r.get("coverage") == 1]
    if x: out["coverage"] = (mean(r.get("score_seen_sub") for r in x), mean(r.get("score_unseen_sub") for r in x))
    if F:
        for k, d in (("cifair10", "cifar10"), ("cifair100", "cifar100")):
            if F[d]["runs"]: out[k] = (mean(r["all"] for r in F[d]["runs"]), mean(r["rest"] for r in F[d]["runs"]))
    x = [r for r in CO if r.get("arm") == "random" and r.get("size") == "M" and r.get("budget") == 30000000]
    if x: out["c4dup"] = (mean(r["nll_c4"] for r in x), mean(r["nll_c4_unique"] for r in x))
    for k, v in _label_errors(R).items():
        if "acc_original" in v: out[k] = (v["acc_original"], v["acc_corrected"])
    x = _coco(R)
    if x: out["coco"] = (mean(r["consistency_est"] for r in x), mean(r["consistency_true"] for r in x))
    return out
# ---- the two score columns of every row
def _docs(R):
    return {r["name"]: r.get("score") for r in _load(R, "dataset_docs.json") or []}
def _static(R):
    """Static score of the raw data by the formula of GB/T 48562-2026 on the data quality dimension, the rates that can be measured on a dataset enter its sub-indicators."""
    P = lambda *rates: 100 * (8 - sum(rates)) / 8; out = {}; T = _load(R, "tables_summary.json")
    if T:
        for k, v in T["base"].items(): out[k] = v["P_data"]
    V = _jsonl(R, "results_vision.jsonl")
    for key, ds, nz in (("cifar10n", "cifar10", "worse_label"), ("cifar100n", "cifar100", "noisy_label")):
        x = [r["noise_rate"] for r in V if r["ds"] == ds and r.get("noise") == nz and "noise_rate" in r]
        if x: out[key] = P(mean(x))
    X = _load(R, "results_taxi.json")
    if X: out["taxi"] = P(mean(v["any_rule"] for v in X["months"].values()))
    C = _load(R, "case_tep.json")
    if C: out["tep"] = P(C["scan"]["duplicate_records"] / C["scan"]["records"])
    F = _load(R, "results_fair.json")
    if F: out["cifair10"] = P(F["cifar10"]["share"]); out["cifair100"] = P(F["cifar100"]["share"])
    for k, v in _label_errors(R).items():
        if "rate" in v: out[k] = P(v["rate"])
    S = _load(R, "corpus_static.json")
    if S: out["c4"] = P(S["rule_fail"], S["near_duplicate"]); out["c4dup"] = P(S["near_duplicate"])
    out.update(_static_cleanml_rates(R))
    for key, fn in (("datacomp", "results_datacomp.jsonl"), ("mirflickr", "results_mirflickr.jsonl")):
        for r in _jsonl(R, fn)[:1]: out[key] = P(1 - r["consistency_pool"])
    Fi = _load(R, "results_fidelity.json") or {}
    for k in ("edm", "ddpm", "cifake"):
        if k in Fi: out[k] = P(1 - Fi[k]["precision"])
    x = _coco(R)
    if x: out["coco"] = P(1 - mean(r["consistency_est"] for r in x))     # the estimate of the detector, as for the image-text pools
    out.update(_load(R, "static_native.json") or {})     # measured on the data as published, native/static_native.py
    for k in ("coverage", "cifar101", "camelyon", "civil", "tepl"): out.setdefault(k, 100.0)
    return out
def _static_cleanml_rates(R):
    """The static score of the CleanML tables from the runs, used until static_native.json is there. Missing: the records removed by deletion. Duplicates: the records removed by the cleaned version. Inconsistency: the cells changed by the cleaned version per record. Outliers: the median over the three detectors of the cells changed per record."""
    G = {}; out = {}
    for r in _jsonl(R, "results_cleanml.jsonl"): G.setdefault(r["ds"], {}).setdefault(r["error_type"], {}).setdefault(r["method"], r)
    for ds, errs in G.items():
        rates = []
        for err, a in errs.items():
            cap = lambda v: min(1, v)
            if err == "missing_values" and "delete" in a: rates.append(a["delete"]["records_removed"] / a["delete"]["records_dirty"])
            elif err == "duplicates" and "clean" in a: rates.append(a["clean"]["records_removed"] / a["clean"]["records_dirty"])
            elif err == "inconsistency" and "clean" in a: rates.append(cap(a["clean"]["cells_changed"] / a["clean"]["records"]))
            elif err == "outliers":
                det = [cap(st.median(r["cells_changed"] for k, r in a.items() if k.startswith(f"clean_{d}_")) / a["dirty"]["records"]) for d in ("IQR", "SD", "IF") if any(k.startswith(f"clean_{d}_") for k in a)]
                if det: rates.append(st.median(det))
        if rates: out[ds] = 100 * (8 - sum(rates)) / 8
    return out
def _lower(metric): return metric == "Loss"
# ---- the main table
PANEL_MAIN = 4     # the groups before this index fill the left panel of the main table
WIN_TAGS = ("Bench", "Real", "Image", "Corpus", "Multimodal", "Synthetic", "Dynamic")
def main_rows(R):
    """Every row of the main table as the table prints it. A version row has the three scores rounded as shown (Raw, Static rule, Task test),
    the unrounded Best and the lower-is-better flag. A practice row has the usual score, the alternative score and their difference. Both carry
    the group index, the dataset entry, the static score and the documentation score. insights.py reads the same rows."""
    A = {}; A.update(table_versions(R)); A.update(cleanml_versions(R)); A.update(corpus_versions(R)); A.update(taxi_versions(R)); A.update(native_versions(R)); B = protocol_rows(R); ST = _static(R); DOC = _docs(R); out = []
    for gi, (g, ds) in enumerate(GROUPS):
        versions = ds[0]["part"] == "versions"
        for d in ds:
            k = d["key"]; row = dict(group=gi, versions=versions, d=d, low=_lower(d["metric"]), static=ST.get(k), doc=DOC.get(DOC_NAME.get(d["name"], d["name"])))
            if versions:
                vals = list(A.get(k, (None, None, None, None))); row.update(shown=[None if x is None else round(x, 3) for x in vals[:3]], best=vals[3])
            else:
                v = B.get(k, (None, None)); row.update(usual=v[0], alt=v[1], gap=None if None in v else v[0] - v[1])
            out.append(row)
    return out
def place(shown, low):
    """Competition place of each of the three scores, one plus the number of strictly better scores, so tied scores share the better place."""
    return [1 + sum(1 for j in range(3) if (shown[j] < shown[i] if low else shown[j] > shown[i])) for i in range(3)]
def main_table(R):
    left, top, low_ = [], [], []; ranks = {}; wins = {}; n_row = {}; rows = main_rows(R)
    for gi, (g, ds) in enumerate(GROUPS):
        versions = ds[0]["part"] == "versions"; tgt = left if gi < PANEL_MAIN else (top if versions else low_); tgt.append(("g", HEAD[gi][0]))
        for row in [r for r in rows if r["group"] == gi]:
            d = row["d"]; low = row["low"]; s_ = row["static"]; nm = d["name"] + (r"$\downarrow$" if low else ""); dc = _f(row["doc"], 0)
            if versions:
                shown = row["shown"]; pl = place([x if x is not None else (float("inf") if low else -float("inf")) for x in shown], low)
                def c(i):
                    x = shown[i]
                    if x is None: return TBD
                    t = f"{x:.3f}"; return f"\\best{{{t}}}" if pl[i] == 1 else (f"\\secondbest{{{t}}}" if pl[i] == 2 else t)
                if all(x is not None for x in shown):
                    ranks.setdefault(gi, []).append([1 + sum(1 for j in range(3) if (shown[j] < shown[i] if low else shown[j] > shown[i])) + (sum(1 for j in range(3) if j != i and shown[j] == shown[i]) / 2) for i in range(3)])
                    n_row[gi] = n_row.get(gi, 0) + 1; wins[gi] = wins.get(gi, 0) + bool((shown[2] <= shown[1]) if low else (shown[2] >= shown[1]))
                tgt.append(("r", [nm, dc, _f(s_, 1), c(0), c(1), c(2), _f(row["best"])]))
            else:
                tgt.append(("p", [nm, dc, _f(s_, 1), _f(row["usual"]), _f(row["alt"]), _signed(row["gap"])]))
    def mp(label, groups):
        x = [r for g_ in groups for r in ranks.get(g_, [])]
        return ("m", label, [f"{mean(r[i] for r in x):.2f}" if x else TBD for i in range(3)])
    left += [("rule",), mp("Mean place, benchmark errors", [0]), mp("Mean place, real errors", [1]), mp("Mean place, image labels, corpora", [2, 3])]
    top.append(mp("Mean place, all seven groups", range(len(WIN_TAGS))))
    right = top + [("rule",), ("h", ["Dataset", "Doc", "Static", "Usual", "Alternative", "Gap"]), ("rule",)] + low_
    H = ["\\begin{table}[!t]", "\\cnentablecaption{全部数据集的测评结果, 分组和指标与表~\\ref{tab:study} 一致, $\\downarrow$ 表示数值越小越好. Doc 和 Static 为文档得分和静态质量得分. 前七组比较数据版本, Raw 为原始数据, Static rule, Task test 和 Best 为静态规则、任务测试和事后最优三种选择规则选出的版本, 每行 Raw, Static rule 和 Task test 中的第一和第二名分别用深色和浅色标出, Mean place 为这三列在所列分组上的平均名次. 后四组比较常见做法与替代做法, Usual 和 Alternative 为两种做法下的分数, Gap 为两者之差. 各组做法见第~\\ref{sec:setup}~节.}{Assessment results on all datasets, groups and metrics as in Table~\\ref{tab:study}, $\\downarrow$ marks lower is better. Doc and Static: documentation score and static quality score. The first seven groups compare data versions. Raw: the raw data. Static rule, Task test and Best: versions chosen by the static rule, the task test and the best choice in hindsight. The first and second among Raw, Static rule and Task test in each row are shaded dark and light. Mean place: mean place of these three columns over the named groups. The last four groups compare a common and an alternative practice. Usual and Alternative: scores under the two practices. Gap: their difference. Section~\\ref{sec:setup} describes each group.}", "\\label{tab:main}", "\\footnotesize", "\\tabcolsep 1pt", "\\renewcommand{\\arraystretch}{0.92}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{\\hspace{2pt}}|@{\\hspace{2pt}}lcccccc@{}}", "\\toprule", "Dataset & Doc & Static & Raw & Static rule & Task test & Best & Dataset & Doc & Static & Raw & Static rule & Task test & Best\\\\", "\\midrule"]
    def cl(x, bar):
        """seven cells of one panel"""
        if x is None: return "&&&&&&"
        if x[0] == "g": return f"\\multicolumn{{7}}{{{BARL if bar else 'l'}}}{{\\textbf{{\\textit{{{x[1]}}}}}}}"
        if x[0] == "m": return f"\\multicolumn{{3}}{{l}}{{\\textit{{{x[1]}}}}} & " + " & ".join(x[2]) + " & "
        if x[0] == "r": return " & ".join(x[1])
        c_ = x[1]; return " & ".join(c_[:4]) + f" & \\multicolumn{{2}}{{c}}{{{c_[4]}}} & {c_[5]}"     # three scores spread over the four columns of the versions
    def seq(side):
        """the lines of a panel, a rule stands before the following line"""
        out = []; rule = False
        for x in side:
            if x[0] == "rule": rule = True; continue
            out.append((x, rule)); rule = False
        return out
    a_, b_ = seq(left), seq(right)
    for i in range(max(len(a_), len(b_))):
        x, rx = a_[i] if i < len(a_) else (None, False); y, ry = b_[i] if i < len(b_) else (None, False)
        if rx or ry: H.append(("\\cmidrule(r){1-7}" if rx else "") + ("\\cmidrule(l){8-14}" if ry else ""))
        H.append(cl(x, True) + " & " + cl(y, False) + "\\\\")
    H += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    macros = dict(StudyTaskWins=f"{sum(wins.values())}", StudyVersionN=f"{sum(n_row.values())}")
    for g_, tag in enumerate(WIN_TAGS): macros[f"StudyWins{tag}"] = f"{wins.get(g_, 0)}"; macros[f"StudyN{tag}"] = f"{n_row.get(g_, 0)}"
    return "\n".join(H), macros
if __name__ == "__main__":
    import sys
    R = sys.argv[1]; tab, m = main_table(R); print(m)
    for name, t in (("dataset table", dataset_table(R)), ("main table", tab)):
        print(name, "red cells", t.count(TBD))
        for l in t.split("\n"):
            if TBD in l: print("  ", l)
