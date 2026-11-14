"""Choosing a dataset version, Algorithm 3 of docs/framework.md, with the static rule and the hindsight best."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def accuracy(y_true, y_pred) -> float:
    """Share of correct predictions, a mean of single-record scores."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of the per-class F1 over the classes present in truth or prediction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.union1d(y_true, y_pred):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination, one for a perfect fit on a constant target."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total


def resolution_for_sample_size(n: int, q: float = 0.1, n_candidates: int = 20, delta: float = 0.05) -> float:
    """Precision gamma that n held-out records give to the gain estimates of all candidates at once.

    N candidates are compared with the reference on n independent records. The metric
    is a mean of single-record scores, the paired difference lies in [-1, 1] and its
    variance is at most q. By the Bernstein inequality and a union bound over the N
    candidates, all N estimates lie within gamma / 2 of their true values with
    probability at least 1 - delta when n is at least
    (8 q / gamma^2 + 8 / (3 gamma)) ln(2 N / delta). The function returns the smallest
    such gamma for the given n. Macro-F1 and R^2 get their precision from
    gain_precision_bootstrap.
    """
    per_log = n / math.log(2.0 * n_candidates / delta)
    return (8.0 / 3.0 + math.sqrt((8.0 / 3.0) ** 2 + 32.0 * q * per_log)) / (2.0 * per_log)


def gain_precision_bootstrap(predictions: Mapping[Any, np.ndarray], y_true, metric: Callable, reference: Any,
                             n_boot: int = 200, delta: float = 0.05, seed: int = 0) -> float:
    """Bootstrap counterpart of resolution_for_sample_size for any metric.

    The function resamples the records and measures how far the gains of all candidates
    move at once. It returns twice the 1 - delta quantile of the largest movement, so
    that all estimates lie within half of the result with probability about 1 - delta.
    """
    y_true = np.asarray(y_true)
    others = [name for name in predictions if name != reference]
    if not others:
        return 0.0
    ref = np.asarray(predictions[reference])
    preds = {name: np.asarray(predictions[name]) for name in others}
    point = {name: metric(y_true, preds[name]) - metric(y_true, ref) for name in others}
    rng = np.random.default_rng(seed)
    largest = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y_true), len(y_true))
        base = metric(y_true[pick], ref[pick])
        largest.append(max(abs(metric(y_true[pick], preds[name][pick]) - base - point[name]) for name in others))
    return 2.0 * float(np.quantile(largest, 1.0 - delta))


def gain_interval(pred, reference_pred, y_true, metric: Callable, n_boot: int = 200, confidence: float = 0.95,
                  seed: int = 0) -> Tuple[float, float]:
    """Paired percentile bootstrap interval of the gain of one version over the reference."""
    pred, reference_pred, y_true = np.asarray(pred), np.asarray(reference_pred), np.asarray(y_true)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(y_true), len(y_true))
        draws.append(metric(y_true[pick], pred[pick]) - metric(y_true[pick], reference_pred[pick]))
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def split_holdout(n_records: int, seed: int = 0, validation_share: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Line 1 of Algorithm 3, a random split of the held-out set into validation and test parts."""
    order = np.random.default_rng(seed).permutation(n_records)
    cut = int(n_records * validation_share)
    return order[:cut], order[cut:]


@dataclass(frozen=True)
class SelectionResult:
    """Output of Algorithm 3."""

    selected: Any
    gain: float
    gain_low: float
    gain_high: float
    precision: float
    precision_method: str
    validation_scores: Dict[Any, float]
    test_scores: Dict[Any, float]


def select_version(candidates: Iterable[Any], fit_predict: Callable[[Any], np.ndarray], y_true, metric: Callable,
                   reference: Any, n: int, seed: int = 0, validation_share: float = 0.5, per_record_metric: bool = False,
                   q: Optional[float] = None, delta: float = 0.05, n_boot: int = 200) -> SelectionResult:
    """Algorithm 3, the choice of a dataset version by a task test.

    The candidates include the reference version. The fit_predict function trains a model
    of the class on one candidate and returns its predictions on the whole held-out set
    with truth y_true. The candidate with the best score on n validation records is
    selected, the earliest one on a tie, and its gain over the reference is measured on
    the test part with a paired bootstrap interval.

    The precision gamma(n) comes from resolution_for_sample_size when per_record_metric is true. The
    variance bound q then defaults to the largest share of validation records on which a
    candidate and the reference predict differently. A metric that does not average over
    records gets its precision from the bootstrap.
    """
    y_true = np.asarray(y_true)
    names = list(candidates)
    if reference not in names:
        raise ValueError("The candidates must include the reference version")
    predictions = {name: np.asarray(fit_predict(name)) for name in names}
    validation, test = split_holdout(len(y_true), seed, validation_share)
    used = validation[:n]
    validation_scores = {name: metric(y_true[used], predictions[name][used]) for name in names}
    selected = max(names, key=validation_scores.get)
    test_scores = {name: metric(y_true[test], predictions[name][test]) for name in names}
    gain = test_scores[selected] - test_scores[reference]
    if per_record_metric and len(names) > 1:
        if q is None:
            q = max(float(np.mean(predictions[name][used] != predictions[reference][used]))
                    for name in names if name != reference)
        precision = resolution_for_sample_size(len(used), q, len(names) - 1, delta)
        method = "Bernstein bound"
    else:
        on_used = {name: predictions[name][used] for name in names}
        precision = gain_precision_bootstrap(on_used, y_true[used], metric, reference, n_boot, delta, seed)
        method = "bootstrap"
    low, high = gain_interval(predictions[selected][test], predictions[reference][test], y_true[test], metric, n_boot, 1.0 - delta, seed)
    return SelectionResult(selected, gain, low, high, precision, method, validation_scores, test_scores)


def static_rule(states: Mapping[Any, Mapping[str, float]]) -> Any:
    """The choice of a rule that reads only static scores, the version with the fewest errors.

    Each state holds the record shares rate_missing, rate_domain, rate_value and rate_label
    and the training size n_train. Fewer feature errors win, then fewer label errors, then
    more training records, then the earliest state.
    """
    def order(name):
        s = states[name]
        return (s["rate_missing"] + s["rate_domain"] + s["rate_value"], s.get("rate_label", 0.0), -s.get("n_train", 0))
    return min(states, key=order)


def hindsight_best(test_scores: Mapping[Any, float]) -> Any:
    """The version with the best test score, an upper reference that no rule can read."""
    return max(test_scores, key=test_scores.get)


def validation_score_at(validation: Mapping[str, float], n: int) -> float:
    """Score on the largest recorded validation sample of at most n records.

    A table with fewer records than n uses all it has, and the smallest sample stands in
    when every sample exceeds n.
    """
    sizes = sorted(int(size) for size in validation)
    use = max([size for size in sizes if size <= n] or sizes[:1])
    return validation[str(use)]


def replay_rules(runs: Iterable[Mapping[Any, Mapping[str, Any]]], sizes: Sequence[int], reference: Any = ("keep", 0.0),
                 min_candidates: int = 14) -> Dict[Any, List[float]]:
    """Replay the decision rules on recorded runs and return the gains of each rule.

    Each run maps a candidate to its record with the test score in test, the validation
    scores by sample size in val, the error rates and n_train. Runs without the reference
    or with fewer than min_candidates candidates are left out. The keys of the result are
    static, hindsight and the validation sizes.
    """
    out: Dict[Any, List[float]] = {"static": [], "hindsight": [], **{n: [] for n in sizes}}
    for arms in runs:
        if reference not in arms or len(arms) < min_candidates:
            continue
        base = arms[reference]["test"]
        out["static"].append(arms[static_rule(arms)]["test"] - base)
        out["hindsight"].append(max(r["test"] for r in arms.values()) - base)
        for n in sizes:
            best = max(arms.values(), key=lambda r: validation_score_at(r["val"], n))
            out[n].append(best["test"] - base)
    return out
