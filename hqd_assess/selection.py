"""Choosing a dataset version, Theorems 1 and 2 and Proposition 3 of the survey and Algorithm 3 of docs/framework.md."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def zero_loss_rule_exists(evidence: Mapping[Hashable, Hashable], best: Mapping[Hashable, Iterable[Any]]) -> bool:
    """Condition of Theorem 1 (1) for a decision rule with zero loss in every usage situation.

    The mapping evidence gives the assessment result E(w) of each usage situation w, and
    best gives the set J*(w) of its best versions. A rule that reads only the assessment
    result has zero loss in every situation if and only if, for every result e, the
    situations with E(w) = e share a best version.
    """
    common: Dict[Hashable, set] = {}
    for situation, result in evidence.items():
        versions = set(best[situation])
        common[result] = versions if result not in common else common[result] & versions
    return all(common.values())


def two_version_gaps(differences: Iterable[float]) -> Tuple[float, float]:
    """The quantities a_+ and a_- of Theorem 1 (2) from the utility differences of two versions.

    The differences are mu_a(w) - mu_b(w) over the usage situations w that give one
    assessment result. The function returns their largest value a_+ and the negative
    of their smallest value a_-.
    """
    d = np.asarray(list(differences), float)
    return float(d.max()), float(-d.min())


def minimax_loss(a_plus: float, a_minus: float, randomized: bool = True) -> float:
    """Smallest worst-case expected loss of a choice between two versions, Theorem 1 (2).

    Over the usage situations that give one assessment result, the utility of D_a minus
    that of D_b is at most a_plus and at least -a_minus. A rule that picks D_a with
    probability u has the worst-case expected loss max((1 - u) a_plus, u a_minus), whose
    smallest value is a_plus a_minus / (a_plus + a_minus). A rule that does not randomize
    has the smallest value min(a_plus, a_minus). When a_plus or a_minus is at most zero,
    one version is best in every situation and the loss is zero, as in Theorem 1 (1).
    """
    a_plus, a_minus = max(float(a_plus), 0.0), max(float(a_minus), 0.0)
    if a_plus == 0.0 or a_minus == 0.0:
        return 0.0
    if not randomized:
        return min(a_plus, a_minus)
    return a_plus * a_minus / (a_plus + a_minus)


def minimax_choice_probability(a_plus: float, a_minus: float) -> float:
    """Probability a_plus / (a_plus + a_minus) of picking D_a that attains the loss of Theorem 1 (2).

    The probability is one when D_a is never worse, that is a_minus at most zero, and
    zero when D_b is never worse.
    """
    a_plus, a_minus = max(float(a_plus), 0.0), max(float(a_minus), 0.0)
    if a_minus == 0.0:
        return 1.0
    if a_plus == 0.0:
        return 0.0
    return a_plus / (a_plus + a_minus)


def static_rule_expected_loss(gamma: float) -> float:
    """Minimax loss of Theorem 1 (2) at a_+ = a_- = gamma, which equals half of gamma."""
    return minimax_loss(gamma, gamma)


def task_test_sample_size(gamma: float, q: float = 0.1, n_candidates: int = 20, delta: float = 0.05) -> int:
    """Held-out records that suffice for all gain estimates, Proposition 3 (1).

    N candidates are compared with the reference on n independent records. The metric
    is a mean of single-record scores, the paired difference lies in [-1, 1] and its
    variance is at most q. With n at least (8 q / gamma^2 + 8 / (3 gamma)) ln(2 N / delta),
    all N estimates lie within gamma / 2 of their true values with probability at least
    1 - delta, so gains that differ by gamma can be told apart. With q = 0.1, N = 20,
    delta = 0.05 and gamma = 0.02 the size is 14261. The bound applies to metrics that
    average over records. Macro-F1 and R^2 get their precision from
    gain_precision_bootstrap. The choice of a version alone needs fewer records, as
    selection_sample_size gives.
    """
    log_term = math.log(2.0 * n_candidates / delta)
    return int(math.ceil((8.0 * q / gamma ** 2 + 8.0 / (3.0 * gamma)) * log_term))


def selection_sample_size(gamma: float, q: float = 0.1, n_candidates: int = 20, delta: float = 0.05) -> int:
    """Validation records that suffice to choose a version within gamma of the best, Proposition 3 (2).

    The N candidates are scored on the same n records, the best one exceeds candidate j
    by d_j in expected score, and the single-record score difference of the two has
    variance q_j at most q. The chosen candidate falls more than gamma below the best
    with probability at most the sum over d_j > gamma of
    exp(-n d_j^2 / (2 q_j + 4 d_j / 3)). Each of these at most N - 1 terms stays below
    delta / (N - 1) when n is at least (2 q / gamma^2 + 4 / (3 gamma)) ln((N - 1) / delta),
    and the function returns this size. With q = 0.1, N = 20, delta = 0.05 and
    gamma = 0.02 it is 3367, against 14261 from task_test_sample_size for estimating
    all gains to the same gamma.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if n_candidates < 2:
        return 0
    log_term = math.log((n_candidates - 1) / delta)
    return int(math.ceil((2.0 * q / gamma ** 2 + 4.0 / (3.0 * gamma)) * log_term))


def wrong_choice_bound(n: int, gaps: Sequence[float], variances: Sequence[float], gamma: float = 0.0) -> float:
    """Bound of Proposition 3 (2) on the probability that the chosen version falls more than gamma below the best.

    The gaps hold d_j, the expected score of the best candidate minus that of candidate j,
    and the variances hold q_j, the variance of their single-record score difference. The
    candidate with the highest mean on n validation records is chosen. The bound is the
    sum of exp(-n d_j^2 / (2 q_j + 4 d_j / 3)) over the candidates with d_j above gamma,
    and a sum above one says nothing.
    """
    if len(gaps) != len(variances):
        raise ValueError("One variance per gap is required")
    return float(sum(math.exp(-n * d * d / (2.0 * v + 4.0 * d / 3.0)) for d, v in zip(gaps, variances) if d > gamma))


def coverage_minimax_loss(alpha: float, n_target: int = 0) -> float:
    """Minimax loss of Theorem 2 (1) when the test records miss a region of the target scene.

    The target scene puts probability alpha on an input region from which no test record
    comes, and the models of two candidate versions predict the same outside the region and
    differently everywhere in it. However many test records come from outside the region,
    the minimax loss of the choice is alpha / 2. With n_target further labelled records
    drawn independently from the target scene it is alpha / 2 times (1 - alpha) to the
    power n_target.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return 0.5 * alpha * (1.0 - alpha) ** n_target


def shifted_gain_lower_bound(estimate: float, epsilon: float, beta: float) -> float:
    """Lower bound of Theorem 2 (2) on the gain in a target scene that differs from the test distribution.

    Single-record score differences lie in [-1, 1], the gain estimate on the test
    distribution Q is within epsilon of the gain on Q with probability at least 1 - delta,
    and the target scene is within total variation distance beta of Q on labelled
    records. With the same probability the gain in the target scene is at least
    estimate - epsilon - 2 beta, and the constant 2 cannot be improved.
    """
    return estimate - epsilon - 2.0 * beta


def resolution_for_sample_size(n: int, q: float = 0.1, n_candidates: int = 20, delta: float = 0.05) -> float:
    """Smallest gamma that n records guarantee under Proposition 3 (1), the inverse of the bound."""
    per_log = n / math.log(2.0 * n_candidates / delta)
    return (8.0 / 3.0 + math.sqrt((8.0 / 3.0) ** 2 + 32.0 * q * per_log)) / (2.0 * per_log)


def gain_precision_bootstrap(predictions: Mapping[Any, np.ndarray], y_true, metric: Callable, reference: Any,
                             n_boot: int = 200, delta: float = 0.05, seed: int = 0) -> float:
    """Bootstrap counterpart of the gamma of Proposition 3 (1) for any metric.

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

    The precision gamma(n) comes from Proposition 3 (1) when per_record_metric is true. The
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
        method = "Proposition 3"
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
