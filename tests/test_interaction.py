"""The interaction quantity, the correction and the paired flags."""
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from hqd_assess.interaction import benjamini_yekutieli, constant_predictor_risk, interaction_quantity, interaction_test, one_sample_t, paired_flags


def test_interaction_quantity_on_hand_made_scores():
    assert interaction_quantity(0.5, [0.6, 0.7], 0.8) == pytest.approx(0.3 - 0.3)
    assert interaction_quantity(0.5, [0.6, 0.7], 0.65) == pytest.approx(0.15 - 0.3)
    assert interaction_quantity(0.5, [0.6, 0.7], 0.9) == pytest.approx(0.4 - 0.3)
    values = interaction_quantity(np.array([0.5, 0.4]), [np.array([0.6, 0.5])], np.array([0.7, 0.7]))
    assert values.tolist() == pytest.approx([0.1, 0.2])
    assert interaction_quantity(Fraction(0), [Fraction(4, 121), Fraction(4, 121)], Fraction(0)) == -Fraction(8, 121)


def test_one_sample_t_matches_scipy():
    values = [0.1, 0.2, -0.05, 0.15, 0.3]
    assert one_sample_t(values) == pytest.approx(stats.ttest_1samp(values, 0).pvalue)
    assert one_sample_t([0.0, 0.0, 0.0]) == 1.0
    assert one_sample_t([0.1]) == 1.0


def test_benjamini_yekutieli_by_hand():
    c = 1 + 1 / 2 + 1 / 3
    assert benjamini_yekutieli([0.01, 0.02, 0.03]) == pytest.approx([0.03 * c] * 3)
    assert benjamini_yekutieli([0.5, 0.001, 0.04]) == pytest.approx([0.5 * c, 0.003 * c, 0.06 * c])
    two = benjamini_yekutieli([0.03, 0.031])
    assert two == pytest.approx([0.0465, 0.0465])
    assert benjamini_yekutieli([0.9, 0.95]).max() == 1.0
    assert len(benjamini_yekutieli([])) == 0


def experiment(raw, single, joint):
    return dict(raw=raw, single=single, joint=joint)


def test_algorithm_4_labels_synergy_antagonism_and_no_interaction():
    rng = np.random.default_rng(0)
    raw = 0.5 + 0.01 * rng.normal(size=10)
    a = raw + 0.1 + 0.005 * rng.normal(size=10)
    b = raw + 0.1 + 0.005 * rng.normal(size=10)
    results = interaction_test({
        "antagonism": experiment(raw, dict(a=a, b=b), raw + 0.05 + 0.002 * rng.normal(size=10)),
        "synergy": experiment(raw, dict(a=a, b=b), raw + 0.40 + 0.002 * rng.normal(size=10)),
        "additive": experiment(raw, dict(a=a, b=b), a + b - raw),
    })
    assert results["antagonism"].verdict == "antagonism"
    assert results["antagonism"].mean < 0
    assert results["synergy"].verdict == "synergy"
    assert results["synergy"].mean > 0
    assert results["additive"].verdict == "none"
    assert results["antagonism"].active_types == 2
    assert results["antagonism"].runs == 10
    assert results["synergy"].joint_gain == pytest.approx(0.40, abs=0.01)


def test_algorithm_4_corrects_all_comparisons_together():
    rng = np.random.default_rng(1)
    raw = np.full(5, 0.5)
    weak = raw + 0.1 + np.array([0.01, -0.01, 0.02, -0.02, 0.0])
    comparisons = {"weak": experiment(raw, dict(a=weak, b=raw + 0.1), raw + 0.25 + 0.03 * rng.normal(size=5))}
    alone = interaction_test(comparisons)["weak"]
    for i in range(30):
        comparisons[f"null{i}"] = experiment(raw, dict(a=raw + 0.1 + 0.01 * rng.normal(size=5), b=raw + 0.1), raw + 0.2 + 0.01 * rng.normal(size=5))
    together = interaction_test(comparisons)["weak"]
    assert together.p == alone.p
    assert together.p_by >= alone.p_by


def test_algorithm_4_leaves_out_tables_with_one_active_error_type_or_few_runs():
    raw = np.array([0.5, 0.6, 0.7, 0.5])
    results = interaction_test({
        "one type": experiment(raw, dict(a=raw + 0.1, b=raw.copy()), raw + 0.2),
        "few runs": experiment(raw[:2], dict(a=raw[:2] + 0.1, b=raw[:2] + 0.1), raw[:2] + 0.3),
    })
    assert results == {}


def test_paired_flags():
    rng = np.random.default_rng(2)
    raw = 0.5 + 0.01 * rng.normal(size=20)
    results = paired_flags({
        "up": (raw + 0.1 + 0.002 * rng.normal(size=20), raw),
        "down": (raw - 0.1 + 0.002 * rng.normal(size=20), raw),
        "flat": (raw + 0.02 * rng.normal(size=20), raw),
        "same": (raw.copy(), raw),
    })
    assert results["up"].flag == "P"
    assert results["down"].flag == "N"
    assert results["flat"].flag == "S"
    assert results["same"].flag == "S"
    assert results["same"].p == 1.0
    assert results["up"].delta == pytest.approx(0.1, abs=0.01)


def test_paired_flags_align_runs_and_need_three_pairs():
    assert paired_flags({"few": ([0.1, 0.2], [0.1, 0.1])}) == {}
    result = paired_flags({"gaps": ([0.2, np.nan, 0.35, 0.4], [0.1, 0.1, 0.2, 0.3])})["gaps"]
    assert result.delta == pytest.approx(0.35 / 3)


def test_constant_predictor_risk():
    assert constant_predictor_risk([1.0, 3.0], 2.0) == 0.0
    assert constant_predictor_risk([Fraction(1), Fraction(0)]) == Fraction(1, 4)
