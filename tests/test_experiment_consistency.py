"""The package computes what the experiments compute, checked on the stored results when they are present."""
import collections
import json
import math
import os

import numpy as np
import pytest

from hqd_assess import interaction_test, model_application_score, model_fit_value, paired_flags, replay_rules, trigger_counts

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is absent")
    return json.load(open(path))


def load_jsonl(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is absent")
    return [json.loads(line) for line in open(path) if line.strip()]


@pytest.fixture(scope="module")
def table_scores():
    scores = collections.defaultdict(dict)
    for r in load_jsonl("results_real_tables.jsonl"):
        if "score" in r:
            scores[(r["ds"], r["model"], r["action"], r["f"])][r["seed"]] = r["score"]
    return scores


def test_paired_flags_reproduce_the_summary(table_scores):
    summary = load_json("tables_summary.json")
    comparisons = {}
    for (ds, model, action, f), by_seed in table_scores.items():
        if action != "keep":
            raw = table_scores[(ds, model, "keep", 0.0)]
            seeds = sorted(set(by_seed) & set(raw))
            comparisons[(ds, model, action, f)] = ([by_seed[s] for s in seeds], [raw[s] for s in seeds])
    flags = paired_flags(comparisons)
    counts = collections.Counter(v.flag for v in flags.values())
    assert {k: counts.get(k, 0) for k in "PSN"} == summary["psn_all"]
    for action in ("repair_records", "delete_records"):
        c = collections.Counter(v.flag for k, v in flags.items() if k[2] == action and k[3] == 1.0)
        assert {k: c.get(k, 0) for k in "PSN"} == summary["psn_by_action"][action]


def test_interaction_test_reproduces_the_summary(table_scores):
    summary = load_json("tables_summary.json")["interaction"]
    parts = ("repair_type_missing", "repair_type_domain", "repair_type_value", "repair_labels")
    experiments = {}
    for ds, model in {(k[0], k[1]) for k in table_scores}:
        columns = [(ds, model, a, 1.0) for a in parts] + [(ds, model, "keep", 0.0), (ds, model, "repair_records", 1.0)]
        if not all(c in table_scores for c in columns):
            continue
        seeds = sorted(set.intersection(*(set(table_scores[c]) for c in columns)))
        experiments[(ds, model)] = dict(raw=[table_scores[columns[4]][s] for s in seeds], joint=[table_scores[columns[5]][s] for s in seeds],
                                        single={a: [table_scores[(ds, model, a, 1.0)][s] for s in seeds] for a in parts})
    results = interaction_test(experiments)
    assert len(results) == summary["n"]
    verdicts = collections.Counter(r.verdict for r in results.values())
    assert verdicts["synergy"] == summary["syn"]
    assert verdicts["antagonism"] == summary["ant"]


def test_replay_reproduces_the_decision_numbers():
    numbers = load_json("numbers_decision.json")
    grouped = collections.defaultdict(dict)
    for r in load_jsonl("results_decision.jsonl"):
        if "test" in r:
            grouped[(r["ds"], r["model"], r["seed"])][(r["action"], r["f"])] = r
    runs = [arms for arms in grouped.values() if arms.get(("keep", 0.0), {}).get("task") == "cls"]
    out = replay_rules(runs, [25, 100])
    assert f"{100 * np.mean(out['static']):.1f}" == numbers["DecStaticCls"]
    assert f"{100 * np.mean(out[100]):.1f}" == numbers["DecMidCls"]
    assert f"{100 * np.mean(out['hindsight']):.1f}" == numbers["DecOracleCls"]


def test_model_fit_of_the_running_example():
    runs = load_json("case_tep.json")["runs"]
    by_run = np.mean([r["macro_f1"] for r in runs if r["model"] == "rf" and r["arm"] == "window_run"])
    s_m = model_fit_value(by_run, 0.8)
    assert model_application_score(s_m) == pytest.approx(100 * math.exp(s_m - 1))


def test_trigger_counts_reproduce_the_taxi_numbers():
    taxi, numbers = load_json("results_taxi.json"), load_json("numbers_taxi.json")
    months = sorted(taxi["months"])
    by_month = collections.defaultdict(list)
    for r in taxi["runs"]:
        if r["arm"] == "raw" and r["train_month"] <= "2020-01":
            by_month[r["train_month"]].append(r["r2"])
    fixed = {month: {k: float(np.mean([c[k] for c in curves if k in c])) for k in set().union(*curves)} for month, curves in by_month.items()}
    assert all(len(curves) == 2 for curves in by_month.values())
    last = max(fixed)
    keys = [k for k in months if k > last and all(k in curve for curve in fixed.values())]
    rates = [100 * taxi["months"][k]["any_rule"] for k in keys]
    scores = [float(np.mean([curve[k] for curve in fixed.values()])) for k in keys]
    counts = trigger_counts(rates, scores)
    assert counts["triggered"] == int(numbers["TaxiTrigN"])
    assert counts["triggered_hits"] == int(numbers["TaxiTrigHit"])
    assert counts["rest"] == int(numbers["TaxiRestN"])
    assert counts["rest_hits"] == int(numbers["TaxiRestHit"])
