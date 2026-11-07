"""Sampling calibration of ratio sub-indicators, Algorithm 1 of docs/framework.md."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def z_value(confidence: float = 0.95) -> float:
    """Two-sided standard normal quantile of a confidence level."""
    return float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def proportion_sample_size(p: float, half_width: float, confidence: float = 0.95) -> int:
    """Records needed to estimate a proportion p with a given half-width.

    The size is n = z^2 p (1 - p) divided by the squared half-width, rounded up. With
    p = 0.9, half-width 0.02 and 95 percent confidence it gives 865 records.
    """
    z = z_value(confidence)
    return int(math.ceil(z * z * p * (1.0 - p) / (half_width * half_width)))


def proportion_half_width(p: float, n: int, confidence: float = 0.95) -> float:
    """Half-width of a proportion p on n records, about 0.006 at p = 0.96 and n = 4000."""
    return z_value(confidence) * math.sqrt(p * (1.0 - p) / n)


def wald_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wald interval of a proportion, kept for comparison with the Wilson interval."""
    p = k / n
    h = proportion_half_width(p, n, confidence)
    return max(0.0, p - h), min(1.0, p + h)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval of k successes in n trials."""
    z = z_value(confidence)
    z2 = z * z
    p = k / n
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def jeffreys_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Jeffreys interval, the equal-tailed interval of the Beta(k + 1/2, n - k + 1/2) posterior.

    The lower end is zero at k = 0 and the upper end is one at k = n.
    """
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5))
    return low, high


def zero_error_upper_bound(n: int, confidence: float = 0.95) -> float:
    """Upper confidence bound on the error rate when n sampled records hold no error.

    The exact one-sided bound is 1 - (1 - confidence) to the power 1/n, about 3/n at
    95 percent confidence.
    """
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def minimum_observed_rate(threshold: float, n: int, confidence: float = 0.95,
                          interval: Callable[[int, int, float], Tuple[float, float]] = wilson_interval) -> float:
    """Smallest observed rate on n records whose interval lower bound reaches a threshold.

    The judgement of stage 5 compares the lower bound with the threshold, so a
    threshold of 0.96 on 4000 records needs an observed rate of about 0.966.
    """
    low, high = 0, n
    if interval(n, n, confidence)[0] < threshold:
        raise ValueError("No observed rate on this sample size reaches the threshold")
    while low < high:
        mid = (low + high) // 2
        if interval(mid, n, confidence)[0] >= threshold:
            high = mid
        else:
            low = mid + 1
    return low / n


def rogan_gladen(p_hat: float, sensitivity: float, specificity: float) -> float:
    """Corrected error rate (p_hat + s_p - 1) divided by (s_e + s_p - 1).

    The correction needs s_e + s_p above one. The result can fall outside [0, 1] on
    finite samples and is returned as computed.
    """
    denominator = sensitivity + specificity - 1.0
    if denominator <= 0.0:
        raise ValueError("The detector has no discriminating power, s_e + s_p must exceed one")
    return (p_hat + specificity - 1.0) / denominator


def detector_rates(flags: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    """Sensitivity and specificity of a detector on audited records.

    A rate without any record in its denominator is NaN.
    """
    flags = np.asarray(flags, bool)
    truth = np.asarray(truth, bool)
    se = float(flags[truth].mean()) if truth.any() else float("nan")
    sp = float((~flags[~truth]).mean()) if (~truth).any() else float("nan")
    return se, sp


def stratified_audit(n_records: int, n: int, strata: Optional[Sequence] = None, seed: int = 0) -> np.ndarray:
    """Draw n record positions, proportional to the stratum sizes and without replacement.

    Proportional allocation keeps the raw shares of the audit unbiased for sensitivity
    and specificity. Seats left by rounding go to the largest remainders.
    """
    rng = np.random.default_rng(seed)
    if n > n_records:
        raise ValueError("The audit is larger than the dataset")
    if strata is None:
        return np.sort(rng.choice(n_records, n, replace=False))
    strata = np.asarray(strata)
    names, sizes = np.unique(strata, return_counts=True)
    quota = n * sizes / n_records
    take = np.floor(quota).astype(int)
    for i in np.argsort(take - quota)[: n - take.sum()]:
        take[i] += 1
    chosen = [rng.choice(np.where(strata == name)[0], t, replace=False) for name, t in zip(names, take)]
    return np.sort(np.concatenate(chosen))


@dataclass(frozen=True)
class CalibratedRate:
    """Output of Algorithm 1."""

    estimate: float
    low: float
    high: float
    p_hat: float
    sensitivity: float
    specificity: float
    method: str


def calibrate_from_audit(flags: np.ndarray, audit_index: np.ndarray, audit_truth: np.ndarray,
                         n_boot: int = 1000, confidence: float = 0.95, seed: int = 0) -> CalibratedRate:
    """Lines 1 and 3 to 10 of Algorithm 1 for an audit that is already labelled.

    The detector ran on all records and gave flags. The audit holds the truth of the
    records at audit_index. When s_e + s_p is at most one, or a rate is undefined, the
    function returns the audit error rate with its Wilson interval. Otherwise it
    returns the corrected rate, cut to [0, 1], with a percentile bootstrap interval
    that resamples both the audit and all records.
    """
    flags = np.asarray(flags, bool)
    audit_index = np.asarray(audit_index)
    audit_truth = np.asarray(audit_truth, bool)
    p_hat = float(flags.mean())
    audit_flags = flags[audit_index]
    se, sp = detector_rates(audit_flags, audit_truth)
    if not (se + sp > 1.0):
        k, n = int(audit_truth.sum()), len(audit_truth)
        low, high = wilson_interval(k, n, confidence)
        return CalibratedRate(k / n, low, high, p_hat, se, sp, "audit error rate, Wilson interval")
    estimate = float(np.clip(rogan_gladen(p_hat, se, sp), 0.0, 1.0))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(audit_truth), len(audit_truth))
        se_b, sp_b = detector_rates(audit_flags[pick], audit_truth[pick])
        p_b = rng.binomial(len(flags), p_hat) / len(flags)
        if se_b + sp_b > 1.0:
            draws.append((p_b + sp_b - 1.0) / (se_b + sp_b - 1.0))
    alpha = 1.0 - confidence
    if not draws:
        k, n = int(audit_truth.sum()), len(audit_truth)
        low, high = wilson_interval(k, n, confidence)
        return CalibratedRate(k / n, low, high, p_hat, se, sp, "audit error rate, Wilson interval")
    low, high = np.clip(np.quantile(draws, [alpha / 2, 1 - alpha / 2]), 0.0, 1.0)
    return CalibratedRate(estimate, float(low), float(high), p_hat, se, sp, "Rogan and Gladen, bootstrap interval")


def calibrate_error_rate(flags: np.ndarray, oracle: Callable[[np.ndarray], np.ndarray], n: int,
                         strata: Optional[Sequence] = None, n_boot: int = 1000,
                         confidence: float = 0.95, seed: int = 0) -> CalibratedRate:
    """Algorithm 1, sampling calibration of a ratio sub-indicator.

    The flags are the output of the detector on all records of the dataset version.
    The function draws n records by the strata and asks the oracle, a human reviewer or
    the source records, for their truth. The oracle maps positions to a boolean array
    that marks the wrong records.
    """
    flags = np.asarray(flags, bool)
    audit = stratified_audit(len(flags), n, strata, seed)
    return calibrate_from_audit(flags, audit, np.asarray(oracle(audit), bool), n_boot, confidence, seed)


def bootstrap_metric_interval(y_true, y_pred, metric: Callable, n_boot: int = 200,
                              confidence: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval of a metric over the test records.

    A metric that is not a mean over records, such as Macro-F1, gets its interval from
    this function.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y_true), len(y_true))
        draws.append(metric(y_true[pick], y_pred[pick]))
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
