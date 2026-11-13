"""Table 4 and the numbers of Section 3.2 of the survey, computed by analysis/requirement_levels.py from data/requirements/items.jsonl."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
SCRIPT = os.path.join(ROOT, "analysis", "requirement_levels.py")
ITEMS = os.path.join(ROOT, "data", "requirements", "items.jsonl")
CODES = ("given", "needed", "form")
LEVELS = ("L1", "L2", "L3", "L4", "L5")
FIELDS = ["id", "file", "document", "clause", "name", "requirement", "formula", "inputs", "sampling", "threshold", "aggregation", "coverage",
          "given", "needed", "form", "round1", "round2"]
INPUTS = {"documents", "metadata", "records_by_rule", "records_by_human_or_detector", "external_source_or_log", "labeled_sample", "trained_model", "controlled_variant"}
FORMS = {"doc_presence", "record_ratio", "scale_ratio", "model_ratio", "sampled_rate", "other"}
# given levels on which both rounds agreed and the rules decide otherwise: a composite score takes the lowest given level of its components (G48562-S2),
# an item that lists deliverables reads the delivery list (G48560-0013, G48560-0015)
RULE_CORRECTED = {"G48562-S2": "L1", "G48560-0013": "L1", "G48560-0015": "L1"}
# Table 4 of the survey: Items, L1 to L5, Sampling, Threshold, Scored, N>G
TABLE4 = {
    "G48562 (T04)": [44, 8, 18, 17, 1, 0, 0, 4, 20, 30],
    "G48560": [225, 20, 187, 18, 0, 0, 3, 1, 0, 178],
    "G48561 (T03)": [8, 3, 5, 0, 0, 0, 0, 0, 0, 3],
    "G48563 (T02)": [17, 0, 0, 17, 0, 0, 0, 16, 0, 0],
    "DPQ": [74, 11, 43, 20, 0, 0, 0, 0, 74, 35],
    "T01": [6, 1, 3, 0, 2, 0, 0, 0, 0, 4],
    "ISO/IEC 25012, 25024": [4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    "NIST AI 100-1": [6, 2, 3, 0, 1, 0, 0, 0, 0, 1],
    "NIST AI 600-1": [27, 13, 11, 3, 0, 0, 0, 0, 0, 17],
    "Total": [411, 62, 270, 75, 4, 0, 3, 21, 94, 268],
    "Share (%)": [100, 15.1, 65.7, 18.2, 1.0, 0, 0.7, 5.1, 22.9, 65.2],
}


@pytest.fixture(scope="module")
def summary():
    import requirement_levels
    return requirement_levels.compute()


def test_items_file_has_the_public_fields():
    items = [json.loads(line) for line in open(ITEMS, encoding="utf-8")]
    assert len(items) == len({r["id"] for r in items}) == 479
    for r in items:
        assert list(r) == FIELDS, r["id"]
        assert all(isinstance(r[k], bool) for k in ("sampling", "threshold", "aggregation")), r["id"]
        assert r["coverage"] in ("full", "partial") and r["inputs"] and set(r["inputs"]) <= INPUTS, r["id"]
        for codes in (r, r["round1"], r["round2"]):
            assert codes["given"] in LEVELS + ("none",) and codes["needed"] in LEVELS + ("none",) and codes["form"] in FORMS, r["id"]
        assert list(r["round1"]) == list(r["round2"]) == list(CODES)
        agreed = [c for c in CODES if r["round1"][c] == r["round2"][c] and not (c == "given" and r["id"] in RULE_CORRECTED)]
        assert all(r[c] == r["round1"][c] for c in agreed), r["id"]
        if r["id"] in RULE_CORRECTED: assert r["given"] == RULE_CORRECTED[r["id"]] != r["round1"]["given"] == r["round2"]["given"], r["id"]
        assert r["given"] in LEVELS and r["needed"] in LEVELS, r["id"]


def test_documents_and_items(summary):
    x = summary["section_3_2"]
    assert (x["documents"], x["domestic"], x["international"], x["items"]) == (13, 9, 4, 479)
    pairs = summary["pairs"]
    assert {k: v["items"] for k, v in pairs.items()} == {"G48562, T04": [44, 44], "G48561, T03": [8, 8], "G48563, T02": [17, 16]}
    assert [p["only_in_draft"] for p in pairs.values()] == [[], [], ["G48563-0016"]] and not any(p["only_in_technical_document"] for p in pairs.values())
    assert [p["coded_differently"] for p in pairs.values()] == [["G48562-0205", "G48562-S2"], [], []]
    assert (x["documents_counted"], x["items_counted"]) == (10, 411)


def test_table_4(summary):
    rows = summary["table4"]["rows"]
    assert list(rows) == list(TABLE4)
    assert rows == TABLE4


def test_matched_is_the_most_frequent_given_level_and_most_items_need_a_higher_level(summary):
    total = dict(zip(summary["table4"]["columns"], summary["table4"]["rows"]["Total"]))
    assert max(LEVELS, key=total.get) == "L2"
    assert 2 * total["N>G"] > total["Items"]
    assert summary["section_3_2"]["threshold_by_file"] == {"G48562": 4, "G48560": 1, "G48563": 16}


@pytest.mark.parametrize("group, items, agreement, kappa", [
    ("all", 479, (91.6, 86.4, 95.2), (0.85, 0.70, 0.88)),
    ("domestic", 442, None, (0.85, 0.69, 0.89)),
    ("international", 37, None, (0.79, 0.66, 0.74)),
])
def test_agreement_of_the_two_rounds(summary, group, items, agreement, kappa):
    a = summary["agreement"][group]
    assert a["items"] == items
    assert [round(a[c]["kappa"], 2) for c in CODES] == list(kappa)
    if agreement:
        assert [round(100 * a[c]["agreement"], 1) for c in CODES] == list(agreement)


def test_items_with_a_disagreement(summary):
    assert summary["agreement"]["items_with_disagreement"] == 89


def test_items_judged_by_people_or_detectors_lack_sampling(summary):
    x = summary["section_3_2"]
    assert len(x["files_with_given_L2"]) == 7
    assert x["files_with_given_L2_without_sampling"] == ["G48562", "G48561", "DPQ", "T01", "NAI100", "NAI600"]
    assert x["items_with_sampling"] == ["G48560-0002", "G48560-0003", "G48560-0009"]
    assert x["given_L2_without_sampling"] == 268
    assert x["scored_with_sampling"] == 0 and x["files_with_scored"] == ["G48562", "DPQ"]


def test_items_that_read_only_documents(summary):
    x = summary["section_3_2"]
    assert len(x["files_with_given_L1"]) == 9
    assert x["given_L1_needed_higher"] == 40
    assert len(x["files_where_most_L1_need_higher"]) == 5
    n = x["given_L1_by_needed"]
    assert (n["L2"], n["L3"] + n["L4"] + n["L5"], n["L1"]) == (34, 6, 22)


def test_no_item_reaches_the_interventional_level(summary):
    x = summary["section_3_2"]
    assert x["given_L5"] == 0 and x["reading_a_version_with_one_change"] == 0
    assert x["given_L4"] == ["G48562-0305", "T01-0005", "T01-0006", "NAI100-01"]
    assert x["needed_L4_or_L5"] == 13 and len(x["files_with_needed_L4_or_L5"]) == 6


def test_command_line():
    out = subprocess.run([sys.executable, SCRIPT, "--json"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["table4"]["rows"]["Total"] == TABLE4["Total"]
    out = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "Table 4" in out.stdout and "Agreement of the two coding rounds" in out.stdout
