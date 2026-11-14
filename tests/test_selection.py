"""Algorithm 3, its precision, the static rule and the hindsight best."""
import math

import numpy as np
import pytest

from hqd_assess.selection import (accuracy, gain_precision_bootstrap, hindsight_best, macro_f1, r2, replay_rules, resolution_for_sample_size,
                                  select_version, split_holdout, static_rule, validation_score_at)


def test_metrics_match_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 5, 200))
    assert macro_f1(y, p) == pytest.approx(metrics.f1_score(y, p, average="macro"))
    assert accuracy(y, p) == pytest.approx(metrics.accuracy_score(y, p))
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert r2(a, b) == pytest.approx(metrics.r2_score(a, b))
    assert r2(np.ones(3), np.ones(3)) == 1.0
    assert r2(np.ones(3), np.zeros(3)) == 0.0


def test_split_holdout_is_a_partition():
    validation, test = split_holdout(11, seed=4)
    assert len(validation) == 5 and len(test) == 6
    assert sorted(np.concatenate([validation, test]).tolist()) == list(range(11))


def make_candidates(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    flip = lambda rate: np.where(rng.random(n) < rate, 1 - y, y)
    return y, dict(raw=flip(0.30), better=flip(0.10), worse=flip(0.45))


def test_algorithm_3_selects_the_best_candidate_and_measures_on_the_test_part():
    y, preds = make_candidates()
    result = select_version(list(preds), preds.get, y, accuracy, "raw", n=500, per_record_metric=True)
    assert result.selected == "better"
    assert result.gain == pytest.approx(0.2, abs=0.03)
    assert result.gain == pytest.approx(result.test_scores["better"] - result.test_scores["raw"])
    assert result.gain_low < result.gain < result.gain_high
    assert result.precision_method == "Bernstein bound"
    assert set(result.validation_scores) == {"raw", "better", "worse"}


def test_algorithm_3_requires_the_reference():
    y, preds = make_candidates(n=100)
    with pytest.raises(ValueError):
        select_version(["better"], preds.get, y, accuracy, "raw", n=10)


def test_algorithm_3_uses_the_bootstrap_for_macro_f1():
    y, preds = make_candidates()
    result = select_version(list(preds), preds.get, y, macro_f1, "raw", n=500, n_boot=50)
    assert result.precision_method == "bootstrap"
    assert result.precision > 0


def test_algorithm_3_reads_only_the_validation_records_for_the_choice():
    y, preds = make_candidates(n=1000, seed=3)
    validation, test = split_holdout(len(y), seed=7)
    preds["trap"] = y.copy()
    preds["trap"][validation[:50]] = 1 - y[validation[:50]]
    result = select_version(list(preds), preds.get, y, accuracy, "raw", n=50, seed=7, per_record_metric=True)
    assert result.selected != "trap"
    assert result.test_scores["trap"] == 1.0


def test_tie_goes_to_the_earliest_candidate():
    y = np.array([0, 1] * 50)
    preds = dict(raw=y.copy(), same=y.copy())
    assert select_version(list(preds), preds.get, y, accuracy, "raw", n=20, per_record_metric=True).selected == "raw"


def test_q_defaults_to_the_share_of_differing_predictions():
    y = np.zeros(1000, int)
    raw = y.copy()
    other = y.copy()
    validation, _ = split_holdout(1000, seed=0)
    other[validation[:50]] = 1
    preds = dict(raw=raw, other=other)
    result = select_version(list(preds), preds.get, y, accuracy, "raw", n=500, per_record_metric=True)
    assert result.precision == pytest.approx(resolution_for_sample_size(500, 0.1, 1, 0.05))
    forced = select_version(list(preds), preds.get, y, accuracy, "raw", n=500, per_record_metric=True, q=0.5)
    assert forced.precision == pytest.approx(resolution_for_sample_size(500, 0.5, 1, 0.05))


@pytest.mark.parametrize("n", [500, 2500, 15000])
def test_resolution_solves_the_bernstein_size(n):
    gamma = resolution_for_sample_size(n, 0.1, 20, 0.05)
    assert (8 * 0.1 / gamma ** 2 + 8 / (3 * gamma)) * math.log(2 * 20 / 0.05) == pytest.approx(n)
    assert resolution_for_sample_size(4 * n, 0.1, 20, 0.05) < gamma


def test_resolution_holds_in_simulation():
    rng = np.random.default_rng(0)
    q, n_candidates, delta, n = 0.1, 20, 0.05, 2500
    gamma = resolution_for_sample_size(n, q, n_candidates, delta)
    failures = 0
    trials = 200
    for _ in range(trials):
        z = rng.choice([-1, 0, 1], size=(n, n_candidates), p=[q / 2, 1 - q, q / 2])
        failures += np.abs(z.mean(axis=0)).max() > gamma / 2
    assert failures / trials <= delta


def test_bootstrap_precision_covers_the_movement_of_the_gains():
    y, preds = make_candidates(n=800, seed=1)
    gamma = gain_precision_bootstrap(preds, y, accuracy, "raw", n_boot=300)
    assert 0.0 < gamma < 0.3
    assert gain_precision_bootstrap(dict(raw=preds["raw"]), y, accuracy, "raw") == 0.0
    larger = gain_precision_bootstrap({k: v[:200] for k, v in preds.items()}, y[:200], accuracy, "raw", n_boot=300)
    assert larger > gamma


def state(missing, domain, value, label=0.0, n_train=100):
    return dict(rate_missing=missing, rate_domain=domain, rate_value=value, rate_label=label, n_train=n_train)


def test_static_rule_picks_the_fewest_errors_then_labels_then_size():
    assert static_rule(dict(a=state(0.2, 0, 0), b=state(0.1, 0, 0))) == "b"
    assert static_rule(dict(a=state(0.1, 0, 0, 0.2), b=state(0.05, 0.05, 0, 0.1))) == "b"
    assert static_rule(dict(a=state(0, 0, 0, 0, 50), b=state(0, 0, 0, 0, 100))) == "b"
    assert static_rule(dict(a=state(0, 0, 0), b=state(0, 0, 0))) == "a"


def test_hindsight_best():
    assert hindsight_best(dict(a=0.5, b=0.7, c=0.6)) == "b"


def test_validation_score_uses_the_largest_sample_within_n():
    scores = {"25": 0.1, "100": 0.2, "400": 0.3}
    assert validation_score_at(scores, 100) == 0.2
    assert validation_score_at(scores, 300) == 0.2
    assert validation_score_at(scores, 3200) == 0.3
    assert validation_score_at(scores, 10) == 0.1


def run_of_three():
    keep = dict(test=0.60, val={"25": 0.50, "100": 0.60}, n_train=100, **{k: v for k, v in state(0.2, 0.1, 0).items() if k != "n_train"})
    clean = dict(test=0.65, val={"25": 0.40, "100": 0.70}, n_train=100, **{k: v for k, v in state(0, 0, 0).items() if k != "n_train"})
    other = dict(test=0.70, val={"25": 0.90, "100": 0.65}, n_train=80, **{k: v for k, v in state(0.1, 0, 0).items() if k != "n_train"})
    return {("keep", 0.0): keep, ("repair_records", 1.0): clean, ("delete_records", 0.5): other}


def test_replay_of_the_decision_rules():
    out = replay_rules([run_of_three()], [25, 100], min_candidates=3)
    assert out["static"] == pytest.approx([0.05])
    assert out["hindsight"] == pytest.approx([0.10])
    assert out[25] == pytest.approx([0.10])
    assert out[100] == pytest.approx([0.05])
    assert replay_rules([run_of_three()], [25], min_candidates=4)["static"] == []
