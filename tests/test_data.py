"""Numbers of the survey that come from the data in this repository."""
import hashlib
import json
import os

import numpy as np
import pytest

from hqd_assess import data_quality_score, sliding_window_starts, window_row_overlap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEP = os.path.join(ROOT, "data", "tep")
TABLES = os.path.join(ROOT, "data", "tables")


def read(name):
    a = np.loadtxt(os.path.join(TEP, name))
    return a.T if a.shape[1] != 52 and a.shape[0] == 52 else a


@pytest.fixture(scope="module")
def tep():
    return {name: np.loadtxt(os.path.join(TEP, name)) for name in sorted(os.listdir(TEP)) if name.endswith(".dat")}


def test_checksums_of_the_tep_files():
    for line in open(os.path.join(TEP, "SHA256SUMS.txt")):
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            with open(os.path.join(TEP, name), "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest, name


def test_static_scan_of_classic_tep(tep):
    assert len(tep) == 44
    swapped = [name for name, a in tep.items() if a.shape[1] != 52]
    assert swapped == ["d00.dat"]
    assert tep["d00.dat"].shape == (52, 500)
    records = np.vstack([a.T if a.shape[1] != 52 else a for a in tep.values()])
    assert records.shape == (31700, 52)
    assert np.isnan(records).sum() == 0
    assert records.shape[0] - np.unique(records.round(8), axis=0).shape[0] == 160


def test_label_numbers_of_classic_tep(tep):
    test_files = [name for name in tep if name.endswith("_te.dat")]
    assert all(tep[name].shape == (960, 52) for name in test_files)
    assert round(100 * 160 / 960, 1) == 16.7
    wrong = 21 * 160
    total = sum(len(tep[name]) for name in test_files)
    assert round(1 - wrong / total, 3) == 0.841


def test_every_test_window_overlaps_under_a_random_split(tep):
    runs, starts = [], []
    for run, name in enumerate(sorted(n for n in tep if n != "d00.dat")):
        s = sliding_window_starts(len(tep[name]), 20)
        runs.append(np.full(len(s), run))
        starts.append(s)
    runs, starts = np.concatenate(runs), np.concatenate(starts)
    order = np.random.default_rng(7).permutation(len(runs))
    cut = int(0.7 * len(order))
    result = window_row_overlap(runs[order[:cut]], starts[order[:cut]], runs[order[cut:]], starts[order[cut:]], 20)
    assert result["windows_with_any_overlap"] == 1.0


@pytest.fixture(scope="module")
def tables():
    pytest.importorskip("sklearn")
    from run_real_tables import load
    meta = json.load(open(os.path.join(TABLES, "table_meta.json")))
    return {name: load(TABLES, name, meta[name]) for name in ("hospitals", "flights", "beers", "breast_cancer")}


def base_rates(data, seeds=20):
    """Means over the training records of the 20 seeds, the state of the raw table in the experiment."""
    rows = []
    n = len(data["a"])
    for seed in range(seeds):
        train = np.random.default_rng(seed + 500).permutation(n)[:int(0.7 * n)]
        kinds, labels = data["T"][train], data["yE"][train]
        missing, domain, value = [float((kinds == t).any(axis=1).mean()) for t in (1, 2, 3)]
        rows.append((float((kinds > 0).mean()), float(((kinds > 0).any(axis=1) | labels).mean()), data_quality_score(missing, domain, value)))
    return np.mean(rows, axis=0)


@pytest.mark.parametrize("name, cell, record, score", [
    ("hospitals", 2.6, 38.8, 95.2),
    ("flights", 34.6, 80.2, None),
    ("beers", 15.5, 100.0, None),
    ("breast_cancer", None, 57.0, 91.0),
])
def test_error_rates_and_scores_of_the_tables(tables, name, cell, record, score):
    measured = base_rates(tables[name])
    if cell is not None:
        assert round(100 * measured[0], 1) == cell
    assert round(100 * measured[1], 1) == record
    if score is not None:
        assert round(measured[2], 1) == score


def test_table_sizes_match_the_metadata():
    meta = json.load(open(os.path.join(TABLES, "table_meta.json")))
    assert len(meta) == 11
    sizes = {name: sum(1 for _ in open(os.path.join(TABLES, name, "dirty_index.csv"), encoding="utf-8")) - 1 for name in ("flights", "hospitals", "nasa")}
    assert sizes == dict(flights=2376, hospitals=1000, nasa=1503)
