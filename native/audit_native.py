#!/usr/bin/env python3
"""Audit of the result files of the native tracks against their job lists: unique ids, rows with an error field, missing and unexpected ids, values that are not finite or out of range, runs that used a fallback, and a summary per dataset, method, model and seed. Needs only the standard library. Usage: audit_native.py RESULTS_DIR."""
import collections, json, math, os, statistics as st, sys
R = sys.argv[1]; out = {}
def rows(fn):
    p = f"{R}/{fn}"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def bad_numbers(x, path=""):
    if isinstance(x, float) and not math.isfinite(x): yield path
    elif isinstance(x, dict):
        for k, v in x.items(): yield from bad_numbers(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from bad_numbers(v, f"{path}[{i}]")
def base(name, rs, want, metrics, lo=0.0, hi=1.0):
    ids = [r.get("id") for r in rs]; c = collections.Counter(ids); good = [r for r in rs if "error" not in r]; have = {r["id"] for r in good}; rng = [r["id"] for r in good for m in metrics if not (isinstance(r.get(m), (int, float)) and lo <= r[m] <= hi)]; nf = [(r["id"], p) for r in rs for p in bad_numbers(r)]
    d = dict(rows=len(rs), unique_ids=len(c), repeated_ids=[k for k, v in c.items() if v > 1][:20], error_rows=[r["id"] for r in rs if "error" in r][:20], expected=len(want), missing=sorted(set(want) - have)[:40], missing_count=len(set(want) - have), unexpected=sorted(have - set(want))[:20], not_finite=nf[:20], metric_out_of_range=rng[:20]); out[name] = d
    print(f"== {name}: rows {d['rows']}, unique {d['unique_ids']}, repeated {len(d['repeated_ids'])}, errors {len(d['error_rows'])}, expected {d['expected']}, missing {d['missing_count']}, unexpected {len(d['unexpected'])}, not finite {len(nf)}, out of range {len(rng)}", flush=True)
    return good
def by(rs, *keys): return dict(sorted(collections.Counter("/".join(str(r.get(k)) for k in keys) for r in rs).items()))
# CleanML
MV = ["dirty", "delete", "impute_holoclean", "impute_mean_mode", "impute_mean_dummy", "impute_median_mode", "impute_median_dummy", "impute_mode_mode", "impute_mode_dummy"]; OUT = ["dirty", "clean_HC_impute_holoclean"] + [f"clean_{d}_impute_{i}_dummy" for d in ("SD", "IQR", "IF") for i in ("mean", "mode", "median")]; DUP = ["dirty", "clean", "AutoER"]; INC = ["dirty", "clean"]; KIND = dict(missing_values=MV, outliers=OUT, duplicates=DUP, inconsistency=INC)
SCHEMA = dict(Airbnb=["duplicates", "outliers", "missing_values"], Citation=["duplicates"], Company=["inconsistency"], Credit=["outliers", "missing_values"], EEG=["outliers"], KDD=["missing_values", "outliers"], Marketing=["missing_values"], Movie=["duplicates", "inconsistency"], Restaurant=["duplicates", "inconsistency"], Sensor=["outliers"], Titanic=["missing_values"], University=["inconsistency"], USCensus=["missing_values"])
want = [f"cm_{d}_{e}_{m}_{mo}_s{s}" for d, es in SCHEMA.items() for e in es for m in KIND[e] if not (d == "KDD" and "holoclean" in m) for mo in ("linear", "rf", "gb", "knn", "mlp") for s in range(5)]; C = rows("results_cleanml.jsonl"); G = base("cleanml", C, want, ("score", "test"))
print(" per dataset", by(G, "ds")); print(" per model", by(G, "model")); print(" per seed", by(G, "seed")); arms = by(G, "ds", "error_type", "method"); odd = {k: v for k, v in arms.items() if v != 25}; print(" arms", len(arms), "arms without 25 runs", odd)
fb = [r["id"] for r in G if r.get("histgb_singleton_fallback")]; print(" fallback runs (gradient boosting without internal early stopping, a class with one training record)", len(fb), fb[:12]); out["cleanml"].update(fallback=fb, per_dataset=by(G, "ds"), per_model=by(G, "model"), per_seed=by(G, "seed"), arms_not_25=odd)
noop = sorted({f"{r['ds']}/{r['error_type']}/{r['method']}" for r in G if r["method"] != "dirty" and r["cells_changed"] == 0 and r["records_removed"] == 0 and r["cells_changed_test"] == 0 and r["records_removed_test"] == 0}); print(" arms identical to dirty", noop); out["cleanml"]["arms_identical_to_dirty"] = noop
small = sorted({(r["ds"], r["method"], r["n_val"], r["n_test"]) for r in G if r["n_test"] < 100}); print(" test parts under 100 rows", small); out["cleanml"]["small_test_parts"] = small
low = sorted(G, key=lambda r: r["score"])[:6]; print(" lowest Macro-F1", [(r["id"], round(r["score"], 3)) for r in low]); empty = [r["id"] for r in G if not r.get("val")]; print(" runs without a validation score", len(empty)); dist = collections.defaultdict(list)
for r in G: dist[r["ds"]].append(r["score"])
print(" Macro-F1 range per dataset", {k: (round(min(v), 3), round(st.mean(v), 3), round(max(v), 3)) for k, v in sorted(dist.items())})
# image-text pools
FR = (.05, .1, .2, .3, .5, .7, .9); AR = ("random", "sim_top", "sim_mid", "sim_bottom", "text_rule", "dedup")
for name, fn, pre in (("datacomp", "results_datacomp.jsonl", "dc"), ("mirflickr", "results_mirflickr.jsonl", "mf")):
    want = [f"{pre}_{a}_f{f}_s{s}" for a in AR for f in FR for s in range(5)] + [f"{pre}_all_f1.0_s{s}" for s in range(5)]; rs = rows(fn); G = base(name, rs, want, ("val_r1", "val_r5", "test_r1", "test_r5", "consistency")); print(" per arm", by(G, "arm")); print(" per seed", by(G, "seed"))
    inc = [r["id"] for r in G if r["test_r5"] < r["test_r1"] or r["val_r5"] < r["val_r1"]]; short = sorted({(r["arm"], r["frac"], round(r["kept_share"], 3)) for r in G if r["arm"] != "all" and r["kept_share"] < r["frac"] - .001}); print(" recall at 5 below recall at 1", inc, " arms that kept less than asked (too few candidates)", short); print(" tau", {round(r["tau"], 6) for r in G}, "consistency of the pool", {round(r["consistency_pool"], 6) for r in G}, "n_train max", max(r["n_train"] for r in G)); out[name].update(kept_less_than_asked=short, per_arm=by(G, "arm"), per_seed=by(G, "seed"))
    if name == "mirflickr": print(" concept precision range", round(min(r["concept_p5"] for r in G), 3), round(max(r["concept_p5"] for r in G), 3), "chance", {round(r["concept_chance"], 4) for r in G})
# WILDS
W = rows("results_wilds.jsonl"); want = [f"civil_{m}_s{s}" for m in ("tfidf_lr", "tfidf_xgb") for s in range(5)] + [f"cam_resnet18_s{s}" for s in range(5)]; G = base("wilds", W, want, ()); print(" per dataset and model", by(G, "ds", "model"))
for key, fs in (("tfidf_lr", ("avg_acc", "worst_group_acc", "macro_f1")), ("tfidf_xgb", ("avg_acc", "worst_group_acc", "macro_f1")), ("resnet18", ("score_id", "score_ood", "f1_id", "f1_ood"))):
    x = [r for r in G if r.get("model") == key]
    if x: print(" ", key, {f: (round(min(r[f] for r in x), 4), round(st.mean(r[f] for r in x), 4), round(max(r[f] for r in x), 4)) for f in fs}, "runs", len(x))
# DCLM
CO = rows("results_corpus.jsonl"); want = [f"c_s{s}_{z}_{a}_30M" for s in range(3) for z in ("M", "S", "L", "XS") for a in ("dclm_pool", "dclm_baseline")]; D = [r for r in CO if str(r.get("arm", "")).startswith("dclm_")]; G = base("dclm", D, want, ()); print(" per size and arm", by(G, "size", "arm"))
for z in ("XS", "S", "M", "L"):
    for a in ("dclm_pool", "dclm_baseline", "random", "ppl_low", "rule_dedup"):
        x = [r for r in CO if r.get("arm") == a and r.get("size") == z and r.get("budget") == 30000000]
        if x: print(f"  {z} {a}: runs {len(x)}, C4 {st.mean(r['nll_c4'] for r in x):.3f}, C4 unique {st.mean(r['nll_c4_unique'] for r in x):.3f}, Wikipedia {st.mean(r['nll_wiki'] for r in x):.3f}, tokens {st.mean(r['n_tokens'] for r in x):.0f}")
nofield = [r["id"] for r in G if r.get("corpus_ppl") is not None]; print(" DCLM rows that report C4 quality fields", nofield)
# synthetic images
SY = rows("results_synth.jsonl"); want = [f"sy_real_{a}_s{s}" for s in range(5) for a in ("resnet18", "vgg11")]
for s in range(3):
    for g in ("edm", "ddpm", "cifake"): want += [f"sy_{g}_resnet18_s{s}_share{x}" for x in (1.0, .5, .9, .75, .25, .1)] + [f"sy_{g}_resnet18_s{s}_add{x}" for x in (50000, 10000, 25000, 100000)] + [f"sy_{g}_vgg11_s{s}_share{x}" for x in (1.0, .5)] + [f"sy_{g}_vgg11_s{s}_add50000"]
    want.append(f"sy_edm_resnet18_s{s}_add200000")
G = base("synth", SY, want, ("score", "score_c101")); print(" per generator and architecture", by(G, "gen", "arch"))
for g in ("real", "edm", "ddpm", "cifake"):
    x = [r for r in G if r.get("gen") == g and r["arch"] == "resnet18" and (g == "real" or r.get("share") == 1.0)]
    if x: print(f"  {g} resnet18 {'all real' if g == 'real' else 'all synthetic'}: runs {len(x)}, CIFAR-10 {st.mean(r['score'] for r in x):.4f}, CIFAR-10.1 {st.mean(r['score_c101'] for r in x):.4f}")
# single files
for fn in ("results_fidelity.json", "label_errors.json", "results_fair.json", "static_native.json", "results_sizes_native.json"):
    p = f"{R}/{fn}"
    if not os.path.exists(p): print("==", fn, "absent"); continue
    x = json.load(open(p)); print("==", fn, "keys", list(x), "not finite", list(bad_numbers(x))[:10])
L = json.load(open(f"{R}/label_errors.json")) if os.path.exists(f"{R}/label_errors.json") else {}; PUB = dict(mnist=15, cifar10=54, cifar100=585, caltech256=458, imagenet=2916, quickdraw=1870, imdb=725, amazon=732, audioset=275, **{"20news": 82}); print(" validated counts against the published table", {k: (L[k].get("given_rejected"), v) for k, v in PUB.items() if k in L}, "all equal" if all(L[k].get("given_rejected") == v for k, v in PUB.items() if k in L) else "DIFFERENT")
F = json.load(open(f"{R}/results_fair.json")) if os.path.exists(f"{R}/results_fair.json") else {}; print(" ciFAIR runs", {k: len(v["runs"]) for k, v in F.items()})
json.dump(out, open(f"{R}/audit_native.json", "w"), indent=0, default=str); print("written", f"{R}/audit_native.json")
