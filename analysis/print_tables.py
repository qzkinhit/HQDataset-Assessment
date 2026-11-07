#!/usr/bin/env python3
"""Print the main results per dataset as plain tables. A section whose result file is absent is skipped.
Usage: print_tables.py [--results DIR] [--only SECTION,...] [--latex]
The sections are tables, decision, tep, taxi, vision, corpus and multimodal. With --latex the script prints the dataset table and the main table of the survey in LaTeX, from big_tables.py."""
import argparse, collections, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from hqd_assess import paired_flags, replay_rules
R = None

def load(fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else None

def jsonl(*fns):
    """Rows of JSONL files without failed runs. A later row with the same id replaces an earlier one."""
    out = {}
    for fn in fns:
        p = os.path.join(R, fn)
        if os.path.exists(p):
            for i, line in enumerate(open(p)):
                try: r = json.loads(line)
                except Exception: continue
                if "error" not in r: out[r.get("id", (fn, i))] = r
    return list(out.values())

def mean(x):
    x = [v for v in x if v is not None and v == v]
    return st.mean(x) if x else None

def f(x, digits=3):
    return "" if x is None else f"{x:.{digits}f}"

def table(title, headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    if not rows: return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    print(f"\n{title}\n{line(headers)}\n{'  '.join('-' * w for w in widths)}")
    for r in rows: print(line(r))

MODELS = ("linear", "rf", "gb", "knn", "mlp")

def tables():
    T = load("tables_summary.json"); sizes = load("table_sizes.json") or {}
    if not T: return
    B = T["base"]
    table("Tables, the raw table of each dataset (percent of records, static score of the standard)",
          ["dataset", "task", "records", "wrong cells", "wrong records", "missing", "out of domain", "in-domain value", "label", "static score"],
          [[d, T["by_task"][d], sizes.get(d, ""), f(100 * b["cell_error"], 1), f(100 - 100 * b["rate_clean_records"], 1), f(100 * b["rate_missing"], 1), f(100 * b["rate_domain"], 1), f(100 * b["rate_value"], 1), f(100 * b["rate_label"], 1), f(b["P_data"], 1)] for d, b in sorted(B.items())])
    scores = collections.defaultdict(dict)
    for r in jsonl("results_real_tables.jsonl"):
        if "score" in r: scores[(r["ds"], r["model"], r["action"], r["f"])][r["seed"]] = r["score"]
    comparisons = {}
    for (d, m, a, share), by_seed in scores.items():
        if a != "keep" and (d, m, "keep", 0.0) in scores:
            raw = scores[(d, m, "keep", 0.0)]; seeds = sorted(set(by_seed) & set(raw)); comparisons[(d, m, a, share)] = ([by_seed[s] for s in seeds], [raw[s] for s in seeds])
    flags = paired_flags(comparisons); rho = {(x["ds"], x["model"]): x["rho"] for x in T["spearman_list"]}; rows = []
    for key in sorted(T["table"]):
        d, m = key.split("|"); v = T["table"][key]; flag = lambda a: flags[(d, m, a, 1.0)].flag if (d, m, a, 1.0) in flags else ""
        rows.append([d, m, f(v.get("keep")), f(v.get("repair_records")), flag("repair_records"), f(v.get("delete_records")), flag("delete_records"), f(rho.get((d, m)), 2)])
    table("Tables, task score per dataset and model (P improves, S no significant change, N worsens, corrected by Benjamini and Yekutieli)", ["dataset", "model", "raw", "repair all", "flag", "delete all", "flag", "Spearman"], rows)
    verdict = lambda x: "none" if x["q"] >= .05 else ("synergy" if x["inter"] > 0 else "antagonism")
    table("Tables, joint repair against the sum of single repairs", ["dataset", "model", "sum of single gains", "joint gain", "interaction", "corrected p", "verdict"],
          [[x["ds"], x["model"], f(x["single"]), f(x["joint"]), f(x["inter"]), f(x["q"]), verdict(x)] for x in sorted(T.get("interaction_list", []), key=lambda x: (x["ds"], x["model"]))])

def decision():
    rows = [r for r in jsonl("results_decision.jsonl") if "test" in r]; fine = [r for r in jsonl("results_decision_fine.jsonl") if "test" in r]
    if len(fine) >= len(rows): rows = fine
    if not rows: return
    grouped = collections.defaultdict(dict); task = {}
    for r in rows: grouped[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r; task[r["ds"]] = r["task"]
    sizes = (25, 100, 400, 800); out = []
    for d in sorted(task, key=lambda x: (task[x], x)):
        o = replay_rules([arms for k, arms in grouped.items() if k[0] == d], sizes)
        if o["static"]: out.append([d, task[d], f(100 * mean(o["static"]), 1)] + [f(100 * mean(o[n]), 1) for n in sizes] + [f(100 * mean(o["hindsight"]), 1)])
    for t in ("cls", "reg"):
        o = replay_rules([arms for arms in grouped.values() if arms.get(("keep", 0.0), {}).get("task") == t], sizes)
        if o["static"]: out.append([f"all {t}", t, f(100 * mean(o["static"]), 1)] + [f(100 * mean(o[n]), 1) for n in sizes] + [f(100 * mean(o["hindsight"]), 1)])
    table("Decision value, gain over the raw table in points, by the rule that picks the training version", ["dataset", "task", "static"] + [f"n={n}" for n in sizes] + ["hindsight"], out)

def tep():
    C = load("case_tep.json")
    if C:
        runs = C["runs"]; arm = lambda m, a, fld="macro_f1": mean(r.get(fld) for r in runs if r["model"] == m and r["arm"] == a)
        table("Classic TEP, Macro-F1 by evaluation protocol", ["model", "rows by run", "rows random", "gap", "windows by run", "windows random", "gap", "hard faults", "other faults"],
              [[m, f(arm(m, "official")), f(arm(m, "row_random")), f(arm(m, "row_random") - arm(m, "official")), f(arm(m, "window_run")), f(arm(m, "window_random")), f(arm(m, "window_random") - arm(m, "window_run")), f(arm(m, "official", "hard")), f(arm(m, "official", "other_faults"))] for m in ("gb", "rf", "mlp", "linear", "knn")])
        table("Classic TEP, labels", ["model", "file-name labels", "onset labels", "gap", "train on test files, raw labels", "relabelled", "dropped"],
              [[m, f(arm(m, "official", "eval_file_name_labels")), f(arm(m, "official", "eval_onset_labels")), f(arm(m, "official", "eval_onset_labels") - arm(m, "official", "eval_file_name_labels")), f(arm(m, "repair", "keep_defect")), f(arm(m, "repair", "relabel_normal")), f(arm(m, "repair", "drop_pre_onset"))] for m in ("gb", "rf", "mlp", "linear", "knn")])
    W = jsonl("results_tep_windows.jsonl")
    if W:
        lengths = sorted({r["length"] for r in W}); g = lambda m, l, s: mean(r["macro_f1"] for r in W if r["model"] == m and r["length"] == l and r["split"] == s)
        table("Classic TEP, Macro-F1 by window length", ["model", "split"] + [str(l) for l in lengths], [[m, s] + [f(g(m, l, s)) for l in lengths] for m in ("gb", "rf", "mlp", "linear", "knn") for s in ("run", "random")])
    Q = load("results_rieth.json"); QM = jsonl("results_rieth_models.jsonl")
    if Q or QM:
        ks = (1, 2, 5, 10, 20, 50, 100, 200, 500); rows = []
        if Q: rows.append(["gb"] + [f(mean(r["macro_f1"] for r in Q["runs"] if r.get("arm") == "runs_fixed_records" and r["k"] == k)) for k in ks])
        for m in ("rf", "mlp", "linear", "knn"): rows.append([m] + [f(mean(r["macro_f1"] for r in QM if r["model"] == m and r["k"] == k and r["n"] == 480)) for k in ks])
        table("Large TEP, Macro-F1 at 480 records per class, by the number of independent runs", ["model"] + [str(k) for k in ks], rows)
        if Q:
            sp = [r for r in Q["runs"] if r.get("arm") == "split_protocol"]
            table("Large TEP, split protocol", ["record split", "run split", "gap"], [[f(mean(r["record_split"] for r in sp)), f(mean(r["run_split"] for r in sp)), f(mean(r["record_split"] - r["run_split"] for r in sp))]] if sp else [])

def taxi():
    X = load("results_taxi.json")
    if not X: return
    months = sorted(X["months"]); idx = {k: i for i, k in enumerate(months)}; horizons = (1, 12, 24, 36, 48)
    def at(arm, tm, h, runs):
        v = [r["r2"].get(months[idx[tm] + h]) for r in runs if r["arm"] == arm and r["train_month"] == tm and idx[tm] + h < len(months)]
        return mean(v)
    grid = sorted({r["train_month"] for r in X["runs"]}); rows = []
    for tm in grid:
        m = X["months"][tm]; rows.append([tm, f"{m['records']}", f(100 * m["any_rule"], 1)] + [f(at("raw", tm, h, X["runs"])) for h in horizons] + [f(at("rule_filtered", tm, h, X["runs"])) for h in horizons])
    table("Taxi records by training month, 1M training records, R2 after h months, raw and rule-filtered", ["train month", "records", "violations %"] + [f"raw +{h}" for h in horizons] + [f"filtered +{h}" for h in horizons], rows)
    runs = []
    for fn in ("results_taxi_sizes.json", "results_taxi_100k.json", "results_taxi.json"):
        src = load(fn)
        if src: runs += src["runs"]
    sizes = sorted({r.get("requested_size", r.get("size")) for r in runs}); early = ("2019-01", "2019-07", "2020-01", "2020-07", "2021-01", "2021-07"); rows = []
    for s in sizes:
        for arm in ("raw", "rule_filtered"):
            sel = [r for r in runs if r.get("requested_size", r.get("size")) == s and r["arm"] == arm and r["train_month"] in early]
            if sel: rows.append([s, arm] + [f(mean(r["r2"].get(months[idx[r["train_month"]] + h]) for r in sel if idx[r["train_month"]] + h < len(months))) for h in horizons[:4]])
    table("Taxi records by training size, training months up to 2021-07, R2 after h months", ["size", "arm"] + [f"+{h}" for h in horizons[:4]], rows)

def vision():
    V = jsonl("results_vision.jsonl", "results_vision_fine.jsonl")
    if V:
        def acc(ds, nz, act, share, arch): return mean(r["score"] for r in V if r["ds"] == ds and r.get("noise") == nz and r.get("action") == act and abs(r.get("f", -1) - share) < 1e-9 and r["arch"] == arch)
        def clean(ds, arch): return mean(r["score"] for r in V if r["ds"] == ds and "noise" not in r and "coverage" not in r and "scale" not in r and r["arch"] == arch)
        rows = []
        for ds, nz in (("cifar10", "aggre_label"), ("cifar10", "random_label1"), ("cifar10", "worse_label"), ("cifar100", "noisy_label")):
            for arch in ("resnet18", "vgg11"):
                rows.append([ds, nz, arch, f(acc(ds, nz, "keep", 0.0, arch))] + [f(acc(ds, nz, "repair", s, arch)) for s in (.25, .5, .75)] + [f(clean(ds, arch))] + [f(acc(ds, nz, "delete", s, arch)) for s in (.5, 1.0)])
        table("CIFAR-N, test accuracy under human label noise", ["dataset", "labels", "model", "keep", "repair 25%", "repair 50%", "repair 75%", "clean labels", "delete 50%", "delete all"], rows)
        rows = []
        for arch in ("resnet18", "vgg11"):
            for k in (5, 4, 3, 2, 1):
                sel = [r for r in V if r.get("coverage") == k and r["arch"] == arch]
                if sel: rows.append([arch, f"{k} fine classes", f(mean(r["score"] for r in sel)), f(mean(r.get("score_seen_sub") for r in sel)), f(mean(r.get("score_unseen_sub") for r in sel))])
            for share in sorted({r["scale"] for r in V if r.get("scale") is not None}, reverse=True):
                sel = [r for r in V if r.get("scale") == share and r["arch"] == arch]
                if sel: rows.append([arch, f"random {share:.0%}", f(mean(r["score"] for r in sel)), "", ""])
        table("CIFAR-100 superclasses, coverage against scale", ["model", "training set", "accuracy", "seen fine classes", "unseen fine classes"], rows)
    F = load("results_fair.json")
    if F:
        table("CIFAR test images with a near duplicate in the training set", ["dataset", "share of test images", "accuracy on them", "accuracy on the rest"],
              [[d, f(c["share"]), f(mean(r["dup"] for r in c["runs"])), f(mean(r["rest"] for r in c["runs"]))] for d, c in F.items() if c.get("runs")])

def corpus():
    C = jsonl("results_corpus.jsonl")
    if C:
        rows = []
        for size, budget in sorted({(r["size"], r.get("budget") or 0) for r in C}, key=lambda x: (x[1], x[0])):
            sel = [r for r in C if r["size"] == size and (r.get("budget") or 0) == budget]; base = [r for r in sel if r["arm"] == "random"]
            for arm in ("all", "random", "ppl_low", "ppl_mid", "ppl_high", "rule", "dedup", "rule_dedup"):
                a = [r for r in sel if r["arm"] == arm]
                if a: rows.append([size, f"{budget // 1000000}M" if budget else "all", arm, len(a), f(mean(r["nll_c4"] for r in a)), f(mean(r["nll_c4_unique"] for r in a)), f(mean(r["nll_wiki"] for r in a)), f(mean(r["corpus_ppl"] for r in a), 2)])
        table("C4 web corpus, held-out loss by filter, model size and token budget", ["model", "tokens", "filter", "runs", "C4", "C4 unique", "Wikipedia", "corpus perplexity"], rows)
    T = jsonl("results_text.jsonl", "results_text_more.jsonl")
    if T:
        rates = sorted({r["defects"]["contamination"] for r in T if list(r.get("defects", {})) == ["contamination"]})
        table("Text pool, held-out documents leaked into training", ["leaked share", "all held-out", "leaked", "unseen"],
              [[x, f(mean(r["nll"] for r in T if r.get("defects") == {"contamination": x})), f(mean(r.get("nll_leaked") for r in T if r.get("defects") == {"contamination": x})), f(mean(r.get("nll_unseen") for r in T if r.get("defects") == {"contamination": x}))] for x in rates])
        rows = []
        for d in ("truncation", "template", "crossdomain", "lowtier", "duplicate", "coverage", "scale"):
            for x in sorted({r["defects"][d] for r in T if list(r.get("defects", {})) == [d]}): rows.append([d, x, f(mean(r["nll"] for r in T if r.get("defects") == {d: x}))])
        table("Text pool, held-out loss under one defect", ["defect", "rate", "loss"], rows)

def multimodal():
    M = jsonl("results_multimodal.jsonl", "results_multimodal_fine.jsonl")
    if not M: return
    rows = []; clean = [r for r in M if r.get("defects") == {}]
    if clean: rows.append(["clean", 0, f(mean(r["score"] for r in clean)), f(mean(r["r1"] for r in clean)), "", "", "", ""])
    for d in ("mismatch", "missing", "duplicate", "coverage", "scale", "contamination"):
        for x in sorted({r["defects"][d] for r in M if list(r.get("defects", {})) == [d]}):
            sel = [r for r in M if r.get("defects") == {d: x}]
            rows.append([d, x, f(mean(r["score"] for r in sel)), f(mean(r["r1"] for r in sel)), f(mean(r.get("detect_auc") for r in sel)), f(mean(r.get("consistency_est") for r in sel)), f(mean(r.get("consistency_true") for r in sel)), f(mean(r.get("consistency_corrected") for r in sel))])
    table("COCO image-text pairs, retrieval under one defect and the consistency estimate", ["defect", "rate", "Recall@5", "Recall@1", "AUC of CLIP similarity", "consistency read", "true", "corrected"], rows)

SECTIONS = dict(tables=tables, decision=decision, tep=tep, taxi=taxi, vision=vision, corpus=corpus, multimodal=multimodal)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results", default=os.path.join(HERE, "..", "results")); ap.add_argument("--only", default=",".join(SECTIONS)); ap.add_argument("--latex", action="store_true"); a = ap.parse_args(); R = a.results
    if a.latex:
        import big_tables
        print(big_tables.dataset_table(R)); print(big_tables.main_table(R)[0]); raise SystemExit
    for name in a.only.split(","): SECTIONS[name]()
