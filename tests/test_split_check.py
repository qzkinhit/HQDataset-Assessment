"""Overlap of training and test data and Algorithm 2."""
import numpy as np
import pytest

from hqd_assess.split_check import (adjacent_window_overlap, check_split_independence, group_overlap, group_split, sliding_window_starts,
                                    window_row_overlap)


def test_group_overlap():
    assert group_overlap([1, 2, 3], [3, 4, 4, 5]) == pytest.approx(0.25)
    assert group_overlap(["a"], ["b"]) == 0.0
    assert group_overlap([1], [1, 1]) == 1.0
    assert group_overlap([1], []) == 0.0


def test_sliding_windows():
    assert sliding_window_starts(10, 4).tolist() == list(range(7))
    assert sliding_window_starts(10, 4, stride=3).tolist() == [0, 3, 6]
    assert adjacent_window_overlap(20) == pytest.approx(0.95)
    assert adjacent_window_overlap(20, 20) == 0.0
    assert adjacent_window_overlap(20, 30) == 0.0


def test_window_overlap_on_hand_made_windows():
    # training windows of run 0 cover rows 0 to 3 and rows 10 to 13
    result = window_row_overlap([0, 0], [0, 10], [0, 0, 0, 1], [2, 4, 8, 0], 4)
    # window at 2 shares rows 2 and 3, window at 4 shares none, window at 8 shares 10 and 11, run 1 has no training window
    assert result["windows_with_any_overlap"] == pytest.approx(0.5)
    assert result["rows_also_in_train"] == pytest.approx((0.5 + 0 + 0.5 + 0) / 4)


def test_random_split_of_stride_one_windows_overlaps_almost_everywhere():
    starts = sliding_window_starts(480, 20)
    runs = np.zeros(len(starts), int)
    order = np.random.default_rng(0).permutation(len(starts))
    cut = int(0.7 * len(order))
    result = window_row_overlap(runs[order[:cut]], starts[order[:cut]], runs[order[cut:]], starts[order[cut:]], 20)
    assert result["windows_with_any_overlap"] == 1.0
    assert result["rows_also_in_train"] > 0.99


def test_group_split_shares_no_key():
    keys = np.repeat(np.arange(10), 5)
    train, test = group_split(keys, 0.3, seed=2)
    assert not set(keys[train]) & set(keys[test])
    assert len(set(keys[test])) == 3
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(50))
    with pytest.raises(ValueError):
        group_split(np.zeros(4), 0.3)


class Evaluator:
    """Returns a high score when test items share a run with the training items."""

    def __init__(self, run):
        self.run = run
        self.calls = 0

    def __call__(self, train, test):
        self.calls += 1
        return 0.5 + 0.4 * group_overlap(self.run[train], self.run[test])


def test_algorithm_2_measures_the_gap_of_a_key_with_overlap():
    run = np.repeat(np.arange(4), 25)
    order = np.random.default_rng(0).permutation(100)
    train, test = order[:70], order[70:]
    evaluate = Evaluator(run)
    result = check_split_independence(train, test, {"run": run}, evaluate, seed=1)["run"]
    assert result.overlap == 1.0
    assert result.score_given == pytest.approx(0.9)
    assert result.score_grouped == pytest.approx(0.5)
    assert result.gap == pytest.approx(0.4)
    assert result.score_to_report == result.score_grouped
    assert evaluate.calls == 2


def test_algorithm_2_trains_nothing_for_a_key_without_overlap():
    run = np.repeat(np.arange(4), 25)
    train, test = np.where(run < 3)[0], np.where(run == 3)[0]
    evaluate = Evaluator(run)
    result = check_split_independence(train, test, {"run": run}, evaluate)["run"]
    assert result.overlap == 0.0
    assert result.gap is None
    assert evaluate.calls == 0


def test_algorithm_2_uses_a_given_grouped_split_and_each_key():
    run = np.repeat(np.arange(4), 25)
    source = np.tile(np.arange(5), 20)
    order = np.random.default_rng(0).permutation(100)
    evaluate = Evaluator(run)
    official = (np.where(run < 2)[0], np.where(run >= 2)[0])
    results = check_split_independence(order[:70], order[70:], {"run": run, "source": source}, evaluate, resplit={"run": official}, seed=3)
    assert results["run"].score_grouped == pytest.approx(0.5)
    assert set(results) == {"run", "source"}
    assert results["source"].overlap == 1.0
