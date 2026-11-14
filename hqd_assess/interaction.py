"""Non-additivity of quality dimensions and paired comparison, Algorithm 4 of docs/framework.md."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats


def interaction_quantity(raw, single: Iterable, joint):
    """Interaction I = (y_J - y_0) minus the sum of (y_k - y_0) over the error types.

    The inputs are task scores on the raw table, after each single repair and after the
    joint repair. Scalars, fractions and aligned arrays all work.
    """
    total = 0
    for score in single:
        total = total + (score - raw)
    return (joint - raw) - total


def one_sample_t(values: Sequence[float]) -> float:
    """Two-sided p value of a one-sample t test against zero, one for constant zero input.

    On the interaction values of aligned runs this test equals a paired t test of the joint gain
    against the sum of the single gains.
    """
    values = np.asarray(values, float)
    if len(values) < 2 or np.all(values == 0):
        return 1.0
    p = float(stats.ttest_1samp(values, 0.0).pvalue)
    return p if np.isfinite(p) else 1.0


def benjamini_yekutieli(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini and Yekutieli adjusted p values, valid under arbitrary dependence."""
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    adjusted = np.minimum(1.0, p[order] * m * harmonic / np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m)
    out[order] = adjusted
    return out


@dataclass(frozen=True)
class InteractionResult:
    """Output of Algorithm 4 for one dataset and model."""

    mean: float
    p: float
    p_by: float
    verdict: str
    joint_gain: float
    single_gain_sum: float
    runs: int
    active_types: int


def interaction_test(experiments: Mapping[Any, Mapping[str, Any]], alpha: float = 0.05, min_runs: int = 3,
                     min_active: int = 2, tolerance: float = 1e-9) -> Dict[Any, InteractionResult]:
    """Algorithm 4, the interaction test of quality dimensions.

    Each experiment holds aligned scores per run under raw, joint and single, where single
    maps an error type to its scores. An experiment enters the test when it has min_runs
    runs and at least min_active error types whose single repair changes the score. All
    comparisons are corrected together. A positive mean after correction is synergy, a
    negative mean is antagonism, and the other cases report no interaction.
    """
    kept = []
    for key, e in experiments.items():
        raw = np.asarray(e["raw"], float)
        single = {k: np.asarray(v, float) for k, v in e["single"].items()}
        joint = np.asarray(e["joint"], float)
        if len(raw) < min_runs:
            continue
        active = sum(float(np.mean(np.abs(v - raw))) > tolerance for v in single.values())
        if active < min_active:
            continue
        i = interaction_quantity(raw, single.values(), joint)
        kept.append((key, i, one_sample_t(i), float(np.mean(joint - raw)),
                     float(np.mean(sum(v - raw for v in single.values()))), len(raw), active))
    adjusted = benjamini_yekutieli([row[2] for row in kept])
    out = {}
    for (key, i, p, joint_gain, single_sum, runs, active), q in zip(kept, adjusted):
        mean = float(np.mean(i))
        verdict = "none"
        if q < alpha:
            verdict = "synergy" if mean > 0 else "antagonism"
        out[key] = InteractionResult(mean, p, float(q), verdict, joint_gain, single_sum, runs, active)
    return out


@dataclass(frozen=True)
class PairedComparison:
    """Paired comparison of an action against the raw table."""

    delta: float
    p: float
    q: float
    flag: str


def paired_flags(comparisons: Mapping[Any, Tuple[Sequence[float], Sequence[float]]], alpha: float = 0.05,
                 min_pairs: int = 3) -> Dict[Any, PairedComparison]:
    """Flags of the actions against the raw table after correction, following CleanML.

    Each comparison holds the action scores and the raw scores of the same runs. The flag
    is P for positive, S for not significant and N for negative. The p values come from a
    paired t test and are corrected together by Benjamini and Yekutieli.
    """
    rows = []
    for key, (action, raw) in comparisons.items():
        action, raw = np.asarray(action, float), np.asarray(raw, float)
        ok = np.isfinite(action) & np.isfinite(raw)
        difference = action[ok] - raw[ok]
        if len(difference) < min_pairs:
            continue
        p = 1.0 if np.abs(difference).sum() == 0 else float(stats.ttest_rel(action[ok], raw[ok]).pvalue)
        rows.append((key, float(difference.mean()), p if np.isfinite(p) else 1.0))
    adjusted = benjamini_yekutieli([row[2] for row in rows])
    out = {}
    for (key, delta, p), q in zip(rows, adjusted):
        flag = "S" if q >= alpha else ("P" if delta > 0 else "N")
        out[key] = PairedComparison(delta, p, float(q), flag)
    return out


def constant_predictor_risk(values: Sequence, truth=0):
    """Risk of the training mean as a constant predictor under squared loss.

    All true values equal truth, so the risk is the squared distance of the mean from it.
    Fractions keep the result exact.
    """
    mean = sum(values) / len(values)
    return (mean - truth) ** 2


def additivity_counterexample() -> Dict[str, Fraction]:
    """An analytic example where two error types cancel on the raw table.

    All 11 true values are 0. Eight records are correct, record a holds +2 outside the value
    range, and records b and c hold -1 inside the range. Repairing a or repairing b and c
    lowers the share of wrong records and raises the risk from 0 to 4/121, while repairing
    all three returns the risk to 0. The interaction returned here is measured on the
    risk and equals -8/121, and on the utility, the negative risk, it is 8/121.
    """
    clean = [Fraction(0)] * 8
    a, b_c = [Fraction(2)], [Fraction(-1)] * 2
    raw = clean + a + b_c
    repair_a = clean + [Fraction(0)] + b_c
    repair_b_c = clean + a + [Fraction(0)] * 2
    repair_all = [Fraction(0)] * 11
    risk = {name: constant_predictor_risk(v) for name, v in
            (("raw", raw), ("repair_a", repair_a), ("repair_b_c", repair_b_c), ("repair_all", repair_all))}
    single_sum = (risk["repair_a"] - risk["raw"]) + (risk["repair_b_c"] - risk["raw"])
    return dict(risk, share_wrong_raw=Fraction(3, 11), share_wrong_repair_a=Fraction(2, 11),
                single_sum=single_sum, joint=risk["repair_all"] - risk["raw"],
                interaction=interaction_quantity(risk["raw"], (risk["repair_a"], risk["repair_b_c"]), risk["repair_all"]))
