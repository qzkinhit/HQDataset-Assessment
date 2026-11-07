"""Independence of training and test data, Algorithm 2 of docs/framework.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np


def group_overlap(train_keys, test_keys) -> float:
    """Share of test items that share a key value with a training item.

    The key can be a run, a time period, a source or a near-duplicate group. The value
    is rho_b of Algorithm 2.
    """
    test_keys = np.asarray(test_keys)
    if len(test_keys) == 0:
        return 0.0
    return float(np.isin(test_keys, np.unique(np.asarray(train_keys))).mean())


def sliding_window_starts(n_rows: int, length: int, stride: int = 1) -> np.ndarray:
    """Start rows of the sliding windows of one run."""
    return np.arange(0, n_rows - length + 1, stride)


def adjacent_window_overlap(length: int, stride: int = 1) -> float:
    """Share of rows that a window shares with the next window of its run."""
    return max(0.0, (length - stride) / length)


def window_row_overlap(train_runs, train_starts, test_runs, test_starts, length: int) -> Dict[str, float]:
    """Row overlap between the test windows and the training windows of the same runs.

    The function returns the mean share of test window rows that also appear in some
    training window, and the share of test windows with at least one such row.
    """
    train_runs, train_starts = np.asarray(train_runs), np.asarray(train_starts)
    test_runs, test_starts = np.asarray(test_runs), np.asarray(test_starts)
    shares = np.zeros(len(test_runs), float)
    for run in np.unique(test_runs):
        members = test_runs == run
        starts = train_starts[train_runs == run]
        if len(starts) == 0:
            continue
        top = int(max(starts.max(), test_starts[members].max())) + length + 1
        cover = np.zeros(top, int)
        np.add.at(cover, starts, 1)
        np.add.at(cover, starts + length, -1)
        covered = np.concatenate(([0], np.cumsum(np.cumsum(cover) > 0)))
        s = test_starts[members]
        shares[members] = (covered[s + length] - covered[s]) / length
    return dict(rows_also_in_train=float(shares.mean()), windows_with_any_overlap=float((shares > 0).mean()))


def group_split(keys, test_share: float = 0.3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split items by key so that training and test items share no key value."""
    keys = np.asarray(keys)
    values = np.random.default_rng(seed).permutation(np.unique(keys))
    if len(values) < 2:
        raise ValueError("A grouped split needs at least two key values")
    n_test = max(1, min(len(values) - 1, int(round(test_share * len(values)))))
    in_test = np.isin(keys, values[:n_test])
    return np.where(~in_test)[0], np.where(in_test)[0]


@dataclass(frozen=True)
class IndependenceResult:
    """Output of Algorithm 2 for one group key.

    The gap is the score on the given split minus the score on the grouped split. It is
    None when the key has no overlap and the grouped split was not needed.
    """

    key: str
    overlap: float
    score_given: Optional[float]
    score_grouped: Optional[float]
    gap: Optional[float]

    @property
    def score_to_report(self) -> Optional[float]:
        """The score that a claim reports, the one on the grouped split when there is overlap."""
        return self.score_given if self.gap is None else self.score_grouped


def check_split_independence(train_index, test_index, group_keys: Mapping[str, np.ndarray],
                             evaluate: Callable[[np.ndarray, np.ndarray], float],
                             resplit: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
                             test_share: float = 0.3, seed: int = 0) -> Dict[str, IndependenceResult]:
    """Algorithm 2, independence check of a split.

    The group_keys map each key name to the key value of every item. The evaluate
    function trains the models of the class on the training positions and returns the
    metric on the test positions. A key with overlap gets a grouped split, taken from
    resplit when given and from group_split otherwise, and the gap of the two scores.
    """
    train_index, test_index = np.asarray(train_index), np.asarray(test_index)
    results: Dict[str, IndependenceResult] = {}
    given = None
    for name, keys in group_keys.items():
        keys = np.asarray(keys)
        overlap = group_overlap(keys[train_index], keys[test_index])
        if overlap > 0:
            if given is None:
                given = float(evaluate(train_index, test_index))
            tr, te = resplit[name] if resplit and name in resplit else group_split(keys, test_share, seed)
            grouped = float(evaluate(np.asarray(tr), np.asarray(te)))
            results[name] = IndependenceResult(name, overlap, given, grouped, given - grouped)
        else:
            results[name] = IndependenceResult(name, 0.0, given, None, None)
    return results
