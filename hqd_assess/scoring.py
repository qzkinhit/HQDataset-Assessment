"""Scores of the standard and the compensation bound, Proposition 2 of the survey.

The dimension scores follow the score formula of GB/T 48562-2026 as published in
the public draft and in the technical document TC609-5-2025-04. Every ratio is a
sub-indicator in [0, 1] and every dimension passes at 90 points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .claim import AssessmentLevel as L

PASS_LINE = 90.0


@dataclass(frozen=True)
class SubIndicator:
    """A sub-indicator of the standard with the level its calculation reaches and the level it needs."""

    code: str
    name: str
    unit: str
    given: L
    needed: Tuple[L, ...]
    required: str

    @property
    def lowest_needed(self) -> L:
        """Lowest level that supports the conclusion of the sub-indicator."""
        return min(self.needed)


def _si(code, name, unit, given, needed, required):
    return SubIndicator(code, name, unit, given, needed, required)


_DOC = "Fields checked against data and lineage"
#: The 17 sub-indicators of GB/T 48562-2026 with their assessment levels.
SUB_INDICATORS = {
    "0101": _si("0101", "Documentation completeness, aspect 1", "Aspects", L.L1, (L.L2,), _DOC),
    "0102": _si("0102", "Documentation completeness, aspect 2", "Aspects", L.L1, (L.L2,), _DOC),
    "0103": _si("0103", "Documentation completeness, aspect 3", "Aspects", L.L1, (L.L2,), _DOC),
    "0104": _si("0104", "Documentation completeness, aspect 4", "Aspects", L.L1, (L.L2,), _DOC),
    "0201": _si("0201", "Format conformity", "Records", L.L3, (L.L3,), "Full scan, accepted missing range"),
    "0202": _si("0202", "Safety conformity", "Records", L.L3, (L.L3,), "Sample size, non-compensatory threshold"),
    "0203": _si("0203", "Annotation conformity", "Records", L.L2, (L.L3,), "Sample judged by hand, with an interval"),
    "0204": _si("0204", "Structural completeness", "Records", L.L3, (L.L3,), "Full scan, accepted missing range"),
    "0205": _si("0205", "Content authenticity", "Records", L.L1, (L.L2,), "Lineage compared, sample traced back"),
    "0206": _si("0206", "Content consistency", "Records", L.L3, (L.L3,), "Detector stated, its error corrected"),
    "0207": _si("0207", "Type consistency", "Records", L.L3, (L.L3,), "Detector stated, its error corrected"),
    "0208": _si("0208", "Cleanliness", "Records", L.L3, (L.L4,), "Proxy signals compared with a task score"),
    "0301": _si("0301", "Content diversity", "Dataset", L.L3, (L.L4,), "Coverage by usage, subgroups tested"),
    "0302": _si("0302", "Scale completeness", "Dataset", L.L3, (L.L5,), "Learning curves, independent units"),
    "0303": _si("0303", "Timeliness", "Records", L.L2, (L.L4,), "Task test on a later period"),
    "0304": _si("0304", "Annotation accuracy", "Records", L.L3, (L.L3,), "Sampled ground truth with an interval"),
    "0305": _si("0305", "Model fit", "Dataset", L.L4, (L.L4, L.L5), "Baseline, model, split, intervention"),
}

#: The three dimensions of the standard and the prefix of their sub-indicator codes.
DIMENSIONS = {"documentation": "01", "data quality": "02", "model application": "03"}


def dimension_size(dimension: str) -> int:
    """Number of sub-indicators in a dimension of the standard."""
    prefix = DIMENSIONS[dimension]
    return sum(code.startswith(prefix) for code in SUB_INDICATORS)


def _weights(weights: Optional[Sequence[float]], k: int) -> np.ndarray:
    if weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(weights, float)
    if len(w) != k:
        raise ValueError("One weight per sub-indicator is required")
    if not math.isclose(float(w.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("The weights must sum to one")
    return w


def dimension_score(values: Iterable[float], weights: Optional[Sequence[float]] = None,
                    truncate: bool = True) -> float:
    """Dimension score P = 100 times the weighted sum of the sub-indicators.

    Equal weights apply when no weights are given. A ratio above one is cut to one
    unless truncate is false, in which case an excess in one ratio compensates a
    shortfall in another ratio.
    """
    s = np.asarray(list(values), float)
    if truncate:
        s = np.minimum(s, 1.0)
    return float(100.0 * np.dot(_weights(weights, len(s)), s))


def data_quality_score(rate_missing: float, rate_domain: float, rate_value: float,
                       other_items: int = 5) -> float:
    """Data quality score of a table with a dirty and a clean version.

    Missing values enter structural completeness, out-of-domain values enter format
    conformity and wrong values inside the domain enter content authenticity. The
    other five items of the dimension cannot be read from the two tables and count as
    full marks. The rates are shares of records.
    """
    items = [1.0 - rate_missing, 1.0 - rate_domain, 1.0 - rate_value] + [1.0] * other_items
    return dimension_score(items)


def model_fit_value(performance: float, expected: float, higher_is_better: bool = True) -> float:
    """Model fit s_m, the ratio of the performance in the target scene to the expected one, inverted for a measure where lower is better."""
    return performance / expected if higher_is_better else expected / performance


def model_fit_factor(s_m: float) -> float:
    """Factor F_m of the model application score, exp(s_m - 1) below one and one otherwise."""
    return math.exp(s_m - 1.0) if s_m < 1.0 else 1.0


def model_application_score(s_m: float, others: Iterable[float] = ()) -> float:
    """Model application score P = 100 F_m times the mean of the other sub-indicators."""
    rest = [min(float(v), 1.0) for v in others]
    mean = sum(rest) / len(rest) if rest else 1.0
    return 100.0 * model_fit_factor(s_m) * mean


def passes(score: float, line: float = PASS_LINE) -> bool:
    """Whether a dimension score reaches the pass line, up to the rounding of floating point sums."""
    return score >= line - 1e-9


def compensation_budget(k: int, line: float = PASS_LINE) -> float:
    """Total shortfall that k equally weighted items may carry at the pass line.

    Proposition 2 states that P reaches 90 if and only if the shortfalls 1 - s_i sum
    to at most 0.1 k.
    """
    return (1.0 - line / 100.0) * k


def within_compensation_budget(values: Iterable[float], weights: Optional[Sequence[float]] = None,
                               line: float = PASS_LINE) -> bool:
    """Right side of the equivalence in Proposition 2.

    With equal weights the shortfalls sum to at most 0.1 k. With general weights the weighted
    shortfalls sum to at most 0.1.
    """
    s = np.asarray(list(values), float)
    w = _weights(weights, len(s))
    return float(np.dot(w, 1.0 - s)) <= 1.0 - line / 100.0 + 1e-12


def single_item_floor(k: int, line: float = PASS_LINE) -> float:
    """Lowest value of one item when the other k - 1 equally weighted items are full.

    The floor is 1 - 0.1 k, which gives 0.2 for the 8 items of the data quality
    dimension and 0.6 for the 4 items of the documentation dimension.
    """
    return max(0.0, 1.0 - compensation_budget(k, line))


def weighted_item_floor(weight: float, line: float = PASS_LINE) -> float:
    """Lowest value of an item of a given weight when the other items are full.

    The floor is 1 - 0.1 divided by the weight, and an item with a weight of at most
    0.1 can reach zero while the dimension still passes.
    """
    return max(0.0, 1.0 - (1.0 - line / 100.0) / weight)


def model_fit_threshold(line: float = PASS_LINE) -> float:
    """Value of s_m under which the model application dimension fails for any other items.

    The value is 1 + ln 0.9, about 0.895, because the mean of the other items is at
    most one.
    """
    return 1.0 + math.log(line / 100.0)


def expected_performance_band(score_low: float, score_high: float, line: float = PASS_LINE) -> Tuple[float, float]:
    """Expected performances for which the verdict depends only on the protocol.

    Two protocols give the task scores score_low and score_high on one dataset. For
    any expected performance inside the returned band, with the other items at full
    marks, the protocol with score_low fails and the protocol with score_high passes.
    """
    threshold = model_fit_threshold(line)
    return score_low / threshold, score_high / threshold


def score_range(values: Sequence[float], weights: Optional[Sequence[float]] = None,
                perturbation: float = 0.1) -> Tuple[float, float]:
    """Exact range of the score when each weight moves by a relative perturbation.

    Each weight stays within perturbation of its value in relative terms and the
    weights still sum to one. The extremes come from moving the free weight to the
    largest or to the smallest values first.
    """
    s = np.minimum(np.asarray(values, float), 1.0)
    w = _weights(weights, len(s))
    low, high = w * (1.0 - perturbation), w * (1.0 + perturbation)

    def extreme(order: np.ndarray) -> float:
        free = 1.0 - low.sum()
        out = low.copy()
        for i in order:
            add = min(free, high[i] - low[i])
            out[i] += add
            free -= add
        return float(100.0 * np.dot(out, s))

    order = np.argsort(s)
    return extreme(order), extreme(order[::-1])
