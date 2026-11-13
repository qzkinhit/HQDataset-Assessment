"""Every number that the survey derives by calculation, reproduced by the package, and the theorems and propositions it proves."""
import json
import math
import os
from fractions import Fraction

import numpy as np
import pytest

from hqd_assess import scoring, selection
from hqd_assess.interaction import (additivity_counterexample, interaction_quantity, operation_biases, repair_interaction,
                                    run_mean_excess_risk)
from hqd_assess.sampling import (estimator_variances, jeffreys_interval, minimum_observed_rate, proportion_half_width, proportion_sample_size,
                                 wilson_interval, zero_error_upper_bound)
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


def test_estimation_records_of_proposition_3_are_about_14300():
    n = selection.task_test_sample_size(0.02, q=0.1, n_candidates=20, delta=0.05)
    assert round(n, -2) == 14300
    assert n == pytest.approx((8 * 0.1 / 0.02 ** 2 + 8 / (3 * 0.02)) * math.log(2 * 20 / 0.05), abs=1)


def test_selection_records_of_proposition_3_are_about_3400():
    n = selection.selection_sample_size(0.02, q=0.1, n_candidates=20, delta=0.05)
    assert round(n, -2) == 3400
    assert n == pytest.approx((2 * 0.1 / 0.02 ** 2 + 4 / (3 * 0.02)) * math.log(19 / 0.05), abs=1)
    assert n < selection.task_test_sample_size(0.02, q=0.1, n_candidates=20, delta=0.05)


@pytest.mark.parametrize("gamma", [0.01, 0.02, 0.05])
def test_selection_size_keeps_the_bound_of_proposition_3_below_delta(gamma):
    n = selection.selection_sample_size(gamma, q=0.1, n_candidates=20, delta=0.05)
    worst = selection.wrong_choice_bound(n, [gamma * (1 + 1e-9)] * 19, [0.1] * 19, gamma)
    assert worst <= 0.05
    assert selection.wrong_choice_bound(n - 2, [gamma * (1 + 1e-9)] * 19, [0.1] * 19, gamma) > 0.05
    assert selection.wrong_choice_bound(n, [gamma, gamma / 2], [0.1, 0.1], gamma) == 0.0


def test_flights_test_records_guarantee_about_ten_points():
    test_records = round(2376 * 0.3)
    assert test_records == 713
    assert round(100 * selection.resolution_for_sample_size(test_records, 0.1, 20, 0.05)) == 10


@pytest.mark.parametrize("gamma", [0.01, 0.02, 0.05, 0.1])
def test_resolution_inverts_the_sample_size(gamma):
    n = selection.task_test_sample_size(gamma)
    assert selection.resolution_for_sample_size(n) == pytest.approx(gamma, rel=1e-3)


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


def test_equal_weight_equivalence_of_proposition_1():
    for values in ([0.9] * 8, [0.2] + [1.0] * 7, [0.95] * 4, [0.61, 1, 1, 1], [0.59, 1, 1, 1], [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]):
        assert passes(dimension_score(values)) == within_compensation_budget(values)


def test_weighted_equivalence_of_proposition_1():
    weights = [0.5, 0.3, 0.2]
    for values in ([0.9, 0.9, 0.9], [0.8, 1.0, 1.0], [0.79, 1.0, 1.0], [1.0, 0.7, 0.95], [1.0, 1.0, 0.5], [1.0, 1.0, 0.49]):
        assert passes(dimension_score(values, weights)) == within_compensation_budget(values, weights)


def test_general_weights_of_proposition_1():
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


def test_static_rule_loss_is_the_minimax_loss_of_theorem_1_at_equal_gaps():
    assert selection.static_rule_expected_loss(0.3) == selection.minimax_loss(0.3, 0.3)
    assert selection.static_rule_expected_loss(0.3) == pytest.approx(0.15)


def test_flip_example_of_theorem_1():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables_summary.json")
    if not os.path.exists(path):
        pytest.skip("tables_summary.json is absent")
    flips = json.load(open(path))["flips"]
    flip = max(flips, key=lambda x: min(x["pos"]["delta"], -x["neg"]["delta"]))
    assert (flip["ds"], flip["action"], flip["pos"]["model"], flip["neg"]["model"]) == ("hospitals", "delete_records", "knn", "gb")
    a_plus, a_minus = selection.two_version_gaps([flip["pos"]["delta"], flip["neg"]["delta"]])
    assert (f"{a_plus:.3f}", f"{a_minus:.3f}") == ("0.487", "0.579")
    assert f"{selection.minimax_loss(a_plus, a_minus):.3f}" == "0.265"
    assert f"{selection.minimax_loss(a_plus, a_minus, randomized=False):.3f}" == "0.487"
    assert round(selection.minimax_loss(0.487, 0.579), 3) == 0.265


@pytest.mark.parametrize("a_plus, a_minus", [(0.487, 0.579), (0.1, 0.1), (0.02, 0.3), (1.0, 0.25)])
def test_minimax_loss_of_theorem_1(a_plus, a_minus):
    u = np.linspace(0, 1, 100001)
    worst = np.maximum((1 - u) * a_plus, u * a_minus)
    best = selection.minimax_loss(a_plus, a_minus)
    assert worst.min() == pytest.approx(best, abs=1e-5)
    u_star = selection.minimax_choice_probability(a_plus, a_minus)
    assert max((1 - u_star) * a_plus, u_star * a_minus) == pytest.approx(best)
    assert selection.minimax_loss(a_plus, a_minus, randomized=False) == min(worst[0], worst[-1])
    assert best <= min(a_plus, a_minus)


def test_zero_loss_rule_of_theorem_1():
    best = dict(w1={"repair"}, w2={"repair", "delete"}, w3={"delete"}, w4={"keep"})
    fine = dict(w1="e1", w2="e1", w3="e2", w4="e3")
    coarse = dict(w1="e1", w2="e1", w3="e1", w4="e3")
    assert selection.zero_loss_rule_exists(fine, best)
    assert not selection.zero_loss_rule_exists(coarse, best)
    assert selection.minimax_loss(0.0, 0.2) == 0.0 and selection.minimax_choice_probability(0.0, 0.2) == 0.0
    assert selection.minimax_choice_probability(0.2, 0.0) == 1.0


def test_coarser_result_does_not_lower_the_minimax_loss_of_theorem_1():
    differences = dict(w1=0.3, w2=-0.1, w3=0.2, w4=-0.4)
    fine = dict(w1="e1", w2="e1", w3="e2", w4="e2")
    coarse = {w: "e" for w in fine}
    def worst(result):
        groups = {}
        for w, e in result.items():
            groups.setdefault(e, []).append(differences[w])
        return max(selection.minimax_loss(*selection.two_version_gaps(d)) for d in groups.values())
    assert worst(fine) <= worst(coarse)
    assert worst(coarse) == pytest.approx(0.3 * 0.4 / 0.7)


def test_estimator_variances_of_section_4_1_order_the_labels():
    for pi in (0.02, 0.1, 0.3, 0.6):
        for phi1, phi0 in ((0.99, 0.01), (0.9, 0.2), (0.6, 0.4), (0.55, 0.5), (0.3, 0.1)):
            v = estimator_variances(pi, phi1, phi0)
            p = pi * phi1 + (1 - pi) * phi0
            assert v["rho2"] == pytest.approx((phi1 - phi0) ** 2 * pi * (1 - pi) / (p * (1 - p)))
            assert v["post_stratified"] <= v["direct"] + 1e-15
            assert v["post_stratified"] <= v["corrected"] + 1e-15
            assert (v["corrected"] < v["direct"]) == (v["rho2"] > 0.5)
            assert v["corrected"] == pytest.approx(p * (1 - p) / (phi1 - phi0) ** 2 - pi * (1 - pi))
    perfect = estimator_variances(0.1, 1.0, 0.0)
    assert perfect["rho2"] == pytest.approx(1.0) and perfect["post_stratified"] == pytest.approx(0.0, abs=1e-15)
    assert estimator_variances(0.1, 0.4, 0.4)["corrected"] == math.inf


@pytest.mark.parametrize("pi, phi1, phi0", [(0.1, 0.8, 0.05), (0.2, 0.6, 0.3), (0.3, 0.95, 0.02)])
def test_estimator_variances_of_section_4_1_in_simulation(pi, phi1, phi0):
    rng = np.random.default_rng(0)
    n, reps = 2000, 2000
    y = rng.random((reps, n)) < pi
    h = np.where(y, rng.random((reps, n)) < phi1, rng.random((reps, n)) < phi0)
    p = pi * phi1 + (1 - pi) * phi0
    direct = y.mean(axis=1)
    post = p * (y & h).sum(axis=1) / h.sum(axis=1) + (1 - p) * (y & ~h).sum(axis=1) / (~h).sum(axis=1)
    rate1, rate0 = (y & h).sum(axis=1) / y.sum(axis=1), (~y & h).sum(axis=1) / (~y).sum(axis=1)
    corrected = (p - rate0) / (rate1 - rate0)
    v = estimator_variances(pi, phi1, phi0)
    for draws, key in ((direct, "direct"), (post, "post_stratified"), (corrected, "corrected")):
        assert n * np.var(draws) == pytest.approx(v[key], rel=0.12), key


def test_biases_of_proposition_2_on_a_finite_population():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    errors = np.where(rng.random(1000) < 0.2, rng.normal(1.0, 1.0, 1000), 0.0)
    keep = np.where(errors != 0, 0.3, 0.9) * (1 + 0.1 * (x > 0))
    selection_bias, error_bias = operation_biases(x, errors, keep)
    kept_mean = np.sum(keep * (x + errors)) / np.sum(keep)
    assert kept_mean - x.mean() == pytest.approx(selection_bias + error_bias)
    assert operation_biases(x, np.zeros(1000), np.ones(1000)) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize("b1, b2", [(0.3, -0.2), (0.3, 0.2), (-0.5, -0.1), (0.4, 0.0)])
def test_interaction_of_proposition_2(b1, b2):
    utility = dict(raw=-(b1 + b2) ** 2, first=-b2 ** 2, second=-b1 ** 2, both=0.0)
    measured = interaction_quantity(utility["raw"], (utility["first"], utility["second"]), utility["both"])
    assert measured == pytest.approx(repair_interaction(b1, b2))
    assert (repair_interaction(b1, b2) > 0) == (b1 * b2 < 0)


def test_counterexample_is_a_case_of_proposition_2():
    example = additivity_counterexample()
    assert -example["interaction"] == repair_interaction(Fraction(2, 11), Fraction(-2, 11)) == Fraction(8, 121)


def test_runs_of_proposition_2():
    rng = np.random.default_rng(2)
    runs, per_run, tau2, sigma2, bias = 5, 8, 0.5, 2.0, 0.1
    draws = bias + rng.normal(0, math.sqrt(tau2), (20000, runs)).mean(axis=1) + rng.normal(0, math.sqrt(sigma2), (20000, runs * per_run)).mean(axis=1)
    assert np.mean(draws ** 2) == pytest.approx(run_mean_excess_risk(bias, tau2, sigma2, runs, per_run), rel=0.05)
    assert run_mean_excess_risk(0, tau2, sigma2, 40, 1) < run_mean_excess_risk(0, tau2, sigma2, 5, 8)
    assert run_mean_excess_risk(0, tau2, sigma2, 5, 8) - run_mean_excess_risk(0, tau2, sigma2, 5, 80) == pytest.approx(sigma2 / 40 - sigma2 / 400)


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5])
def test_coverage_loss_of_theorem_2(alpha):
    assert selection.coverage_minimax_loss(alpha) == pytest.approx(selection.minimax_loss(alpha, alpha)) == pytest.approx(alpha / 2)
    assert selection.coverage_minimax_loss(alpha, 10) == pytest.approx(alpha / 2 * (1 - alpha) ** 10)


@pytest.mark.parametrize("beta", [0.0, 0.05, 0.3])
def test_scene_shift_bound_of_theorem_2_is_sharp(beta):
    g = np.array([1.0, -1.0])
    target, test = np.array([1 - beta, beta]), np.array([1.0, 0.0])
    gain_target, gain_test = float(target @ g), float(test @ g)
    assert 0.5 * np.abs(target - test).sum() == pytest.approx(beta)
    assert gain_test - gain_target == pytest.approx(2 * beta)
    assert selection.shifted_gain_lower_bound(gain_test, 0.0, beta) == pytest.approx(gain_target)


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
