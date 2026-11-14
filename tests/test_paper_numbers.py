"""Every number that the survey and docs/framework.md derive by calculation, reproduced by the package."""
import math
from fractions import Fraction

import pytest

from hqd_assess import scoring
from hqd_assess.interaction import additivity_counterexample
from hqd_assess.sampling import (jeffreys_interval, minimum_observed_rate, proportion_half_width, proportion_sample_size, wilson_interval,
                                 zero_error_upper_bound)
from hqd_assess.scoring import (SUB_INDICATORS, data_quality_score, dimension_score, dimension_size, expected_performance_band,
                                model_application_score, model_fit_threshold, model_fit_value, passes, single_item_floor,
                                weighted_item_floor, within_compensation_budget)


def test_sample_size_of_a_proportion_is_865():
    assert proportion_sample_size(0.9, 0.02, 0.95) == 865


def test_half_width_at_4000_records_and_96_percent_is_near_six_tenths_of_a_point():
    half = 100 * proportion_half_width(0.96, 4000)
    assert half == pytest.approx(0.6, abs=0.05)


@pytest.mark.parametrize("n", [100, 300, 1000, 4000, 20000])
def test_zero_error_upper_bound_is_near_three_over_n(n):
    assert zero_error_upper_bound(n) * n == pytest.approx(3.0, abs=0.06)


def test_observed_rate_of_96_6_percent_reaches_a_lower_bound_of_96_percent():
    rate = minimum_observed_rate(0.96, 4000)
    assert round(100 * rate, 1) == 96.6
    assert wilson_interval(round(rate * 4000), 4000)[0] >= 0.96
    assert wilson_interval(round(rate * 4000) - 1, 4000)[0] < 0.96
    assert wilson_interval(round(0.96 * 4000), 4000)[0] < 0.96


def test_single_item_floors_of_the_dimensions():
    assert dimension_size("data quality") == 8
    assert dimension_size("documentation") == 4
    assert single_item_floor(8) == pytest.approx(0.2)
    assert single_item_floor(4) == pytest.approx(0.6)


@pytest.mark.parametrize("k", [4, 8])
def test_a_floor_is_sharp_for_equal_weights(k):
    floor = single_item_floor(k)
    assert passes(dimension_score([floor + 1e-9] + [1.0] * (k - 1)))
    assert not passes(dimension_score([floor - 1e-6] + [1.0] * (k - 1)))


def test_equal_weight_equivalence_of_the_compensation_budget():
    for values in ([0.9] * 8, [0.2] + [1.0] * 7, [0.95] * 4, [0.61, 1, 1, 1], [0.59, 1, 1, 1], [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]):
        assert passes(dimension_score(values)) == within_compensation_budget(values)


def test_weighted_equivalence_of_the_compensation_budget():
    weights = [0.5, 0.3, 0.2]
    for values in ([0.9, 0.9, 0.9], [0.8, 1.0, 1.0], [0.79, 1.0, 1.0], [1.0, 0.7, 0.95], [1.0, 1.0, 0.5], [1.0, 1.0, 0.49]):
        assert passes(dimension_score(values, weights)) == within_compensation_budget(values, weights)


def test_floors_under_general_weights():
    assert weighted_item_floor(0.5) == pytest.approx(0.8)
    assert weighted_item_floor(1 / 8) == pytest.approx(0.2)
    assert weighted_item_floor(0.1) == pytest.approx(0.0, abs=1e-12)
    assert weighted_item_floor(0.05) == 0.0
    weights = [0.1] + [0.9 / 7] * 7
    assert passes(dimension_score([0.0] + [1.0] * 7, weights))
    weights = [0.2] + [0.8 / 7] * 7
    assert not passes(dimension_score([0.0] + [1.0] * 7, weights))
    assert passes(dimension_score([weighted_item_floor(0.2)] + [1.0] * 7, weights))


def test_model_fit_threshold_is_0_895():
    threshold = model_fit_threshold()
    assert round(threshold, 3) == 0.895
    assert threshold == pytest.approx(1 + math.log(0.9))
    assert not passes(model_application_score(threshold - 1e-6, [1.0, 1.0, 1.0, 1.0]))
    assert passes(model_application_score(threshold + 1e-6, [1.0, 1.0, 1.0, 1.0]))
    assert not passes(model_application_score(threshold - 1e-6, [2.0, 2.0]))


def test_two_kinds_of_items_compensate_differently():
    assert passes(dimension_score([0.2] + [1.0] * 7))
    assert not passes(model_application_score(0.2, [1.0, 1.0, 1.0, 1.0]))


def test_excess_ratio_compensates_only_without_truncation():
    values = [2.0, 0.6]
    assert dimension_score(values) == pytest.approx(80.0)
    assert dimension_score(values, truncate=False) == pytest.approx(130.0)


def test_tep_model_fit_example():
    performance_by_run, performance_random = 0.6966, 0.8742
    s_run = model_fit_value(performance_by_run, 0.80)
    assert round(s_run, 3) == 0.871
    assert round(model_application_score(s_run), 1) == 87.9
    assert not passes(model_application_score(s_run))
    s_random = model_fit_value(performance_random, 0.80)
    assert s_random > 1
    assert model_application_score(s_random) == 100.0
    low, high = expected_performance_band(performance_by_run, performance_random)
    assert low < 0.78 and 0.97 < high
    for expected in (0.78, 0.85, 0.90, 0.97):
        assert not passes(model_application_score(model_fit_value(performance_by_run, expected)))
        assert passes(model_application_score(model_fit_value(performance_random, expected)))


def test_risk_example():
    example = additivity_counterexample()
    assert example["share_wrong_raw"] == Fraction(3, 11)
    assert example["share_wrong_repair_a"] == Fraction(2, 11)
    assert example["raw"] == 0
    assert example["repair_a"] == Fraction(4, 121)
    assert example["repair_b_c"] == Fraction(4, 121)
    assert example["repair_all"] == 0
    assert example["single_sum"] == Fraction(8, 121)
    assert example["joint"] == 0
    assert example["interaction"] == -Fraction(8, 121)


def test_seventeen_sub_indicators():
    assert len(SUB_INDICATORS) == 17
    assert dimension_size("model application") == 5
    assert all(code[:2] in scoring.DIMENSIONS.values() for code in SUB_INDICATORS)


def test_sub_indicator_table_flags_the_three_weak_links():
    gaps = {code for code, s in SUB_INDICATORS.items() if s.given < s.lowest_needed}
    assert {"0101", "0102", "0103", "0104"} <= gaps
    assert {"0202", "0205", "0206", "0207", "0208", "0301", "0302", "0303", "0304", "0305"} <= gaps
    assert not {"0201", "0204"} & gaps


def test_score_of_a_table_follows_the_standard_formula():
    assert data_quality_score(0.0, 0.388, 0.0) == pytest.approx(100 * (8 - 0.388) / 8)
    assert data_quality_score(0.1, 0.1, 0.1) == pytest.approx(96.25)


def test_jeffreys_and_wilson_agree_with_known_values():
    assert wilson_interval(5, 10)[0] == pytest.approx(0.2366, abs=1e-3)
    assert wilson_interval(5, 10)[1] == pytest.approx(0.7634, abs=1e-3)
    assert jeffreys_interval(0, 10)[0] == 0.0
    assert jeffreys_interval(10, 10)[1] == 1.0
