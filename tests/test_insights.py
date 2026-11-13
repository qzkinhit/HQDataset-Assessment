"""Smoke test of analysis/insights.py: it runs on the stored results, the macro names are valid TeX names that do not clash with the
manuscript macros of make_numbers.py, and the key numbers lie in plausible ranges."""
import json
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
RESULTS = os.path.join(ROOT, "results")
need = pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, "results_real_tables.jsonl")), reason="stored results are not present")


@pytest.fixture(scope="module")
def summary():
    import insights
    return insights.compute(RESULTS)


@pytest.fixture(scope="module")
def macros(summary):
    import insights
    return insights.macros_from(summary)


@need
def test_summary_is_json(summary):
    json.dumps(summary)
    assert {"1", "2", "3", "4", "5", "6", "7", "8"} <= set(summary)


@need
def test_macro_names_are_letters_and_new(macros):
    assert macros
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in macros)
    assert all(k.startswith("Ins") for k in macros)
    src = open(os.path.join(ROOT, "analysis", "make_numbers.py"), encoding="utf-8").read() + open(os.path.join(ROOT, "analysis", "summaries.py"), encoding="utf-8").read()
    assert not [k for k in macros if re.search(r"\b" + k + r"\b", src)]
    assert all(isinstance(v, str) and v and "nan" not in v for v in macros.values())


@need
def test_key_numbers_in_range(summary):
    a = summary["1"]
    assert -1 <= a["path"]["delete_med"] < 0 < a["path"]["repair_med"] <= 1
    assert a["path"]["repair_n"] > a["path"]["delete_n"] > 0
    up = a["score_up"]
    assert up["repair_n"] + up["delete_n"] <= up["n"] and up["delete_down_pct"] > up["repair_down_pct"]
    assert a["full"]["P_min"] == pytest.approx(100) and 0 < a["full"]["med_abs_gap"] < 1
    assert -1 <= a["shift"]["rho"] <= 1 and a["shift"]["n"] >= 5
    b = summary["2"]
    assert -1 <= b["amount_gain"]["rho"] <= 1 and 0 < b["half"]["tree_share"] <= 150
    c = summary["3"]
    assert c["small"] <= c["pairs"] and 0 < c["top3_share"] <= 100 and c["task_better"] + c["task_worse"] + c["task_tie"] == c["pairs"]
    d = summary["4"]
    assert 0 < d["rieth"]["run1"] < d["rieth"]["run5"] < 1 and abs(d["rieth_split"]["record"] - d["rieth_split"]["run"]) < 0.05
    e = summary["5"]
    assert e["main_right"]["protocol_med"] > e["main_right"]["record_max"] >= 0
    assert e["coverage"]["all_low"] < e["coverage"]["full"]
    t = summary["6"]
    assert all(-1 < v <= 1 for v in t["train_2023"].values()) and len(t["rises"]) == 2
    s = summary["7"]
    assert len(s["corpus"]["filters"]) >= 4 and 0 < s["corpus"]["c4_min_p_better"] <= 1
    assert all(0 <= v <= 1 for v in s["synth"]["recall"].values())
    g = summary["8"]
    assert 0 <= g["detectors"]["corrected_below_sample"]["200"] <= g["detectors"]["pairs"]
    assert -1 <= g["llm"]["rho_f1_gain"] <= 1 and 0 <= g["judge"]["median"] <= 5


@need
def test_random_deletion_summary_on_a_constructed_file(tmp_path):
    """The comparison of the two deletions runs on a file with the layout of results_random_delete.jsonl. The selective arm reuses stored
    deletion runs, the random arm is the same runs shifted down, so the selective deletion must come out ahead in every pair."""
    import insights
    for fn in ("results_real_tables.jsonl", "results_tables_fine.jsonl"):
        os.symlink(os.path.join(RESULTS, fn), tmp_path / fn)
    D = insights.table_runs(str(tmp_path))
    sel = D[(D.action == "delete_records") & (D.f.isin([0.5, 1.0])) & (D.seed < 5)].copy()
    rnd = sel.copy(); rnd["action"] = "random_delete"; rnd["score"] = rnd["score"] - 0.05 - 0.001 * rnd["seed"]
    rows = pd.concat([sel, rnd]).drop(columns=["P"]); rows["fraction"] = rows["f"]
    rows["shift"] = [len(d) / 10 if a == "delete_records" else 0.1 for d, a in zip(rows["ds"], rows["action"])]
    with open(tmp_path / "results_random_delete.jsonl", "w") as f:
        for r in rows.to_dict("records"): f.write(json.dumps(r) + "\n")
    out = insights.sec1_random(str(tmp_path), D, None)
    assert out["rerun_max_abs_diff"] == pytest.approx(0)
    for frac in ("0.5", "1.0"):
        x = out["by_fraction"][frac]
        assert x["selective_worse"] == 0 and x["pairs"] > 0 and x["med_diff"] > 0
        assert 0 <= x["selective_down_pct"] <= 100 and 0 <= x["random_down_pct"] <= 100
