"""Finite-population error-rate intervals from an audit without replacement.

The target is the error fraction of the *fixed, current dataset*.  Detector
flags define strata before the audit.  Within each stratum the audited items
must be a simple random sample without replacement, conditional on its size.
This includes poststratifying an independent simple random audit.  It does not
cover convenience samples, sampling based on unseen truth, or extrapolation
to future records.  Audit labels are assumed correct.

Each stratum's unknown integer error count is bounded by equal-tailed
hypergeometric inversion.  Bonferroni over the nonempty population strata
gives simultaneous coverage of at least ``confidence``.  Adding these count
bounds therefore gives the same coverage for the dataset error fraction.
No independence between the stratum intervals is needed for the union bound.
Discreteness and Bonferroni can make the interval conservative.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import operator
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import hypergeom


METHOD = "finite_population_hypergeometric_bonferroni_v1"


def _count(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a nonnegative integer, not bool".format(name))
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError("{} must be a nonnegative integer".format(name)) from exc
    if result < 0:
        raise ValueError("{} must be nonnegative".format(name))
    return int(result)


def _confidence(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("confidence must lie strictly between zero and one")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a finite number") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return value


@lru_cache(maxsize=262144)
def _invert_count(k, n, population, tail_alpha):
    if n == 0:
        return 0, population
    if n == population:
        return k, k
    # Counts outside this range are incompatible with the observed sample.
    first, last = k, population - n + k
    # P_K(X >= k) increases with the population error count K.
    left, right = first, last
    while left < right:
        middle = (left + right) // 2
        if hypergeom.sf(k - 1, population, middle, n) >= tail_alpha:
            right = middle
        else:
            left = middle + 1
    low = left
    # P_K(X <= k) decreases with K.  sf avoids subtraction in the other tail.
    left, right = first, last
    while left < right:
        middle = (left + right + 1) // 2
        if hypergeom.cdf(k, population, middle, n) >= tail_alpha:
            left = middle
        else:
            right = middle - 1
    return low, left


def hypergeometric_count_interval(k: int, n: int, population: int,
                                  confidence: float = 0.95) -> Tuple[int, int]:
    """Bound the unknown number of errors among ``population`` fixed items.

    ``k`` errors are observed in a uniform sample of ``n`` distinct items.
    Both endpoints are inclusive integers.  No sample returns [0, population]
    and a census returns [k, k].  The calculation uses scipy's hypergeometric
    cdf/sf and monotone integer bisection; repeated inputs are cached.
    """
    k, n = _count(k, "k"), _count(n, "n")
    population = _count(population, "population")
    confidence = _confidence(confidence)
    if k > n or n > population:
        raise ValueError("counts must satisfy 0 <= k <= n <= population")
    return _invert_count(k, n, population, (1.0 - confidence) / 2.0)


@dataclass(frozen=True)
class FinitePopulationRate:
    estimate: Optional[float]
    low: float
    high: float
    confidence: float
    method: str
    population_sizes: Tuple[int, ...]
    sample_sizes: Tuple[int, ...]
    error_counts: Tuple[int, ...]
    count_intervals: Tuple[Tuple[int, int], ...]
    unsampled_strata: Tuple[int, ...]
    estimate_reason: str


def stratified_error_interval(error_counts: Sequence[int],
                              sample_sizes: Sequence[int],
                              population_sizes: Sequence[int],
                              confidence: float = 0.95) -> FinitePopulationRate:
    """Estimate a fixed dataset's error fraction and bound it by design.

    Strata and their population sizes must be known before inspecting audit
    truth.  The error-count bound remains valid if a nonempty stratum has no
    sampled item: that stratum contributes [0, N_h], not an inferred zero.
    The point estimate is sum_h (N_h/N)*(k_h/n_h).  With an unsampled stratum,
    it falls back to the observed pooled sample fraction and explicitly marks
    that fallback; it is not claimed design-unbiased.  With no audit at all,
    the point estimate is None and the interval is [0, 1].
    """
    counts = tuple(_count(x, "error count") for x in error_counts)
    samples = tuple(_count(x, "sample size") for x in sample_sizes)
    populations = tuple(_count(x, "population size") for x in population_sizes)
    confidence = _confidence(confidence)
    if not counts or not len(counts) == len(samples) == len(populations):
        raise ValueError("the three nonempty sequences must have equal lengths")
    if any(k > n or n > size for k, n, size in zip(counts, samples, populations)):
        raise ValueError("each stratum must satisfy 0 <= k_h <= n_h <= N_h")
    total = sum(populations)
    if total == 0:
        raise ValueError("the total population must be nonempty")
    active = sum(size > 0 for size in populations)
    # Allocate alpha by population strata, never by observed labels or success.
    tail_alpha = (1.0 - confidence) / (2.0 * active)
    intervals = tuple(_invert_count(k, n, size, tail_alpha)
                      for k, n, size in zip(counts, samples, populations))
    unsampled = tuple(i for i, (n, size) in enumerate(zip(samples, populations))
                      if size > 0 and n == 0)
    if sum(samples) == 0:
        estimate, reason = None, "no_audit"
    elif unsampled:
        estimate = sum(counts) / sum(samples)
        reason = "pooled_sample_fallback_unsampled_population_stratum"
    else:
        estimate = sum((size / total) * (k / n)
                       for k, n, size in zip(counts, samples, populations) if size)
        reason = "population_weighted_stratum_proportions"
    return FinitePopulationRate(
        estimate, sum(a for a, _ in intervals) / total,
        sum(b for _, b in intervals) / total, confidence, METHOD,
        populations, samples, counts, intervals, unsampled, reason)


def detector_interval_from_counts(tp: int, fn: int, fp: int, tn: int,
                                 population_positive: int,
                                 population_negative: int,
                                 confidence: float = 0.95) -> FinitePopulationRate:
    """Two detector strata: flagged (tp, fp) and unflagged (fn, tn).

    Confusion counts contain only audited items.  The two population counts
    come from scanning detector flags on the full dataset, without its truth.
    """
    tp, fn, fp, tn = (_count(x, name) for x, name in
                      ((tp, "tp"), (fn, "fn"), (fp, "fp"), (tn, "tn")))
    return stratified_error_interval((tp, fn), (tp + fp, fn + tn),
                                     (population_positive, population_negative),
                                     confidence)


def _binary_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("{} must be a one-dimensional binary array".format(name))
    return values.astype(bool)


def calibrate_from_audit(flags, audit_index, audit_truth,
                         confidence: float = 0.95) -> FinitePopulationRate:
    """Convenience wrapper accepting full detector flags and audit labels only.

    Audit indices must be distinct.  The sampling design assumptions above
    cannot be verified from indices alone and remain the caller's obligation.
    """
    flags = _binary_vector(flags, "flags")
    truth = _binary_vector(audit_truth, "audit_truth")
    index = np.asarray(audit_index)
    if index.ndim != 1 or len(index) != len(truth):
        raise ValueError("audit_index and audit_truth must be equally long vectors")
    index = np.asarray([_count(x, "audit index") for x in index], dtype=np.int64)
    if len(index) != len(np.unique(index)) or (len(index) and index.max() >= len(flags)):
        raise ValueError("audit indices must be unique and within the population")
    selected = flags[index]
    return detector_interval_from_counts(
        int((selected & truth).sum()), int((~selected & truth).sum()),
        int((selected & ~truth).sum()), int((~selected & ~truth).sum()),
        int(flags.sum()), int((~flags).sum()), confidence)


def interval_cache_info():
    """Diagnostics for repeated validation; does not change inference."""
    return _invert_count.cache_info()._asdict()
