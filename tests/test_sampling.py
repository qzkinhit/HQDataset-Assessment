"""Intervals, the correction of Rogan and Gladen and Algorithm 1."""
import numpy as np
import pytest
from scipy import stats

from hqd_assess.sampling import (bootstrap_metric_interval, calibrate_error_rate, calibrate_from_audit, detector_rates, jeffreys_interval,
                                 proportion_half_width, rogan_gladen, stratified_audit, wald_interval, wilson_interval, z_value)
from hqd_assess.selection import accuracy


def coverage(interval, p, n):
    """Exact coverage probability of an interval for a binomial proportion."""
    total = 0.0
    for k in range(n + 1):
        low, high = interval(k, n)
        if low <= p <= high:
            total += stats.binom.pmf(k, n, p)
    return total


def test_z_value():
    assert z_value(0.95) == pytest.approx(1.95996, abs=1e-4)


def test_jeffreys_matches_the_beta_quantiles():
    low, high = jeffreys_interval(30, 100)
    assert low == pytest.approx(stats.beta.ppf(0.025, 30.5, 70.5))
    assert high == pytest.approx(stats.beta.ppf(0.975, 30.5, 70.5))


@pytest.mark.parametrize("p, n", [(0.01, 50), (0.02, 50), (0.02, 100)])
def test_wald_coverage_is_unstable_and_wilson_stays_closer(p, n):
    assert coverage(wald_interval, p, n) < 0.9
    assert coverage(wilson_interval, p, n) > 0.9
    assert coverage(jeffreys_interval, p, n) > 0.9


def test_half_width_shrinks_with_the_square_root_of_n():
    assert proportion_half_width(0.5, 400) == pytest.approx(2 * proportion_half_width(0.5, 1600))


def test_correction_recovers_the_true_rate():
    pi, se, sp = 0.2, 0.9, 0.95
    p_hat = pi * se + (1 - pi) * (1 - sp)
    assert rogan_gladen(p_hat, se, sp) == pytest.approx(pi)


def test_correction_needs_a_discriminating_detector():
    with pytest.raises(ValueError):
        rogan_gladen(0.3, 0.5, 0.5)


def test_detector_rates_on_hand_made_audit():
    flags = np.array([1, 1, 0, 0, 1, 0], bool)
    truth = np.array([1, 1, 1, 0, 0, 0], bool)
    assert detector_rates(flags, truth) == pytest.approx((2 / 3, 2 / 3))
    se, sp = detector_rates(flags[:2], truth[:2])
    assert se == 1.0 and np.isnan(sp)


def synthetic(n=20000, pi=0.2, se=0.9, sp=0.95, seed=1):
    rng = np.random.default_rng(seed)
    wrong = rng.random(n) < pi
    flags = np.where(wrong, rng.random(n) < se, rng.random(n) >= sp)
    return wrong, flags


def test_algorithm_1_reads_the_detector_error_from_the_audit():
    wrong, flags = synthetic()
    result = calibrate_error_rate(flags, lambda i: wrong[i], n=2000, n_boot=200, seed=3)
    assert result.method.startswith("Rogan")
    assert result.estimate == pytest.approx(wrong.mean(), abs=0.015)
    assert result.low <= wrong.mean() <= result.high
    assert abs(result.estimate - wrong.mean()) < abs(flags.mean() - wrong.mean())


def test_algorithm_1_matches_the_formula_of_the_image_text_experiment():
    wrong, flags = synthetic(seed=2)
    audit = np.random.default_rng(5).choice(len(flags), 200, replace=False)
    result = calibrate_from_audit(flags, audit, wrong[audit], n_boot=50)
    audited = wrong[audit]
    se = flags[audit][audited].mean()
    sp = (~flags[audit][~audited]).mean()
    assert result.estimate == pytest.approx((flags.mean() + sp - 1) / (se + sp - 1))


def test_algorithm_1_returns_the_audit_rate_for_a_detector_without_power():
    rng = np.random.default_rng(0)
    wrong = rng.random(5000) < 0.3
    flags = rng.random(5000) < 0.5
    audit = np.arange(0, 5000, 5)
    forced = flags.copy()
    forced[audit] = ~wrong[audit]
    result = calibrate_from_audit(forced, audit, wrong[audit], n_boot=20)
    assert result.method.startswith("audit error rate")
    assert result.estimate == pytest.approx(wrong[audit].mean())
    assert (result.low, result.high) == pytest.approx(wilson_interval(int(wrong[audit].sum()), len(audit)))


def test_algorithm_1_returns_the_audit_rate_when_the_audit_holds_no_error():
    flags = np.array([0] * 90 + [1] * 10, bool)
    audit = np.arange(50)
    result = calibrate_from_audit(flags, audit, np.zeros(50, bool), n_boot=20)
    assert result.estimate == 0.0
    assert result.low == 0.0
    assert result.high == pytest.approx(wilson_interval(0, 50)[1])


def test_corrected_rate_is_cut_to_the_unit_interval():
    flags = np.array([0] * 100, bool)
    audit = np.arange(40)
    truth = np.array([True] * 2 + [False] * 38)
    flags[:2] = True
    flags[2:40] = False
    result = calibrate_from_audit(flags, audit, truth, n_boot=20)
    assert 0.0 <= result.estimate <= 1.0


def test_stratified_audit_allocates_in_proportion():
    strata = np.array([0] * 800 + [1] * 150 + [2] * 50)
    picked = stratified_audit(1000, 100, strata, seed=1)
    assert len(picked) == len(set(picked.tolist())) == 100
    assert np.bincount(strata[picked]).tolist() == [80, 15, 5]
    uneven = stratified_audit(1000, 7, strata, seed=1)
    assert len(uneven) == 7
    assert len(stratified_audit(10, 4, seed=0)) == 4
    with pytest.raises(ValueError):
        stratified_audit(10, 11)


def test_bootstrap_interval_contains_the_metric():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.where(rng.random(500) < 0.8, y, 1 - y)
    low, high = bootstrap_metric_interval(y, p, accuracy, n_boot=200)
    assert low < accuracy(y, p) < high
