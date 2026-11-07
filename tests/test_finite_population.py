"""Design-based coverage and boundary tests; runnable with unittest or pytest."""

from fractions import Fraction
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hqd_assess.finite_population import (
    calibrate_from_audit, detector_interval_from_counts,
    hypergeometric_count_interval, stratified_error_interval,
)


def sampling_probabilities(N, K, n):
    return [(k, Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n)))
            for k in range(max(0, n - N + K), min(K, n) + 1)]


class FinitePopulationTests(unittest.TestCase):
    def test_single_stratum_exact_coverage_and_complement_symmetry(self):
        for N in range(1, 13):
            for n in range(N + 1):
                bounds = {k: hypergeometric_count_interval(k, n, N) for k in range(n + 1)}
                for k, (low, high) in bounds.items():
                    other = bounds[n - k]
                    self.assertEqual((N - high, N - low), other)
                for K in range(N + 1):
                    coverage = sum((p for k, p in sampling_probabilities(N, K, n)
                                    if bounds[k][0] <= K <= bounds[k][1]), Fraction())
                    self.assertGreaterEqual(coverage, Fraction(95, 100), (N, K, n))

    def test_two_strata_exact_conditional_coverage(self):
        sizes = (9, 10)
        for samples in itertools.product(*(range(N + 1) for N in sizes)):
            bounds = {counts: stratified_error_interval(counts, samples, sizes)
                      for counts in itertools.product(*(range(n + 1) for n in samples))}
            for errors in itertools.product(*(range(N + 1) for N in sizes)):
                coverage = Fraction()
                laws = [sampling_probabilities(N, K, n) for N, K, n in zip(sizes, errors, samples)]
                for (k1, p1), (k0, p0) in itertools.product(*laws):
                    r = bounds[k1, k0]
                    if sum(a for a, _ in r.count_intervals) <= sum(errors) <= sum(b for _, b in r.count_intervals):
                        coverage += p1 * p0
                self.assertGreaterEqual(coverage, Fraction(95, 100), (samples, errors))

    def test_srs_random_stratum_allocation_coverage(self):
        sizes = (9, 10)
        for errors in itertools.product(range(10), range(11)):
            for n in range(20):
                coverage = Fraction()
                for n1, allocation in sampling_probabilities(19, 9, n):
                    samples = (n1, n - n1)
                    laws = [sampling_probabilities(N, K, nh) for N, K, nh in zip(sizes, errors, samples)]
                    for (k1, p1), (k0, p0) in itertools.product(*laws):
                        r = stratified_error_interval((k1, k0), samples, sizes)
                        if sum(a for a, _ in r.count_intervals) <= sum(errors) <= sum(b for _, b in r.count_intervals):
                            coverage += allocation * p1 * p0
                self.assertGreaterEqual(coverage, Fraction(95, 100), (errors, n))

    def test_census_empty_stratum_and_no_audit(self):
        self.assertEqual(hypergeometric_count_interval(0, 0, 8), (0, 8))
        self.assertEqual(hypergeometric_count_interval(3, 8, 8), (3, 3))
        self.assertEqual(hypergeometric_count_interval(0, 0, 0), (0, 0))
        census = stratified_error_interval((3, 1), (5, 4), (5, 4))
        self.assertEqual((census.low, census.high), (4 / 9, 4 / 9))
        none = stratified_error_interval((0, 0), (0, 0), (5, 4))
        self.assertEqual((none.estimate, none.low, none.high), (None, 0, 1))
        partial = stratified_error_interval((0, 2), (0, 4), (6, 4))
        self.assertEqual(partial.count_intervals, ((0, 6), (2, 2)))
        self.assertEqual((partial.low, partial.high, partial.estimate), (0.2, 0.8, 0.5))
        self.assertEqual(partial.unsampled_strata, (0,))
        self.assertIn("fallback", partial.estimate_reason)
        one = stratified_error_interval((0, 2), (0, 10), (0, 100))
        self.assertEqual(one.count_intervals[1], hypergeometric_count_interval(2, 10, 100))

    def test_perfect_sample_is_not_proof_of_a_perfect_detector(self):
        # Audit labels agree perfectly with flags.  Unseen mismatches remain
        # possible, so a non-census interval must not collapse to a point.
        r = detector_interval_from_counts(15, 0, 0, 35, 3000, 7000)
        self.assertAlmostEqual(r.estimate, 0.3)
        self.assertLess(r.low, 0.3)
        self.assertGreater(r.high, 0.3)

    def test_audit_api_and_design_weighted_point_estimate(self):
        flags = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        r = calibrate_from_audit(flags, [0, 1, 4, 5], [1, 1, 0, 1])
        self.assertAlmostEqual(r.estimate, 0.2 * 1 + 0.8 * 0.5)
        self.assertEqual(r, detector_interval_from_counts(2, 1, 0, 1, 2, 8))
        self.assertEqual(calibrate_from_audit(flags, [], []).estimate, None)

    def test_invalid_inputs_are_rejected(self):
        for values in ((2, 1, 10), (1, 2, 1), (-1, 2, 10), (1.5, 2, 10), (True, 2, 10)):
            with self.assertRaises(ValueError):
                hypergeometric_count_interval(*values)
        for confidence in (0, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                hypergeometric_count_interval(1, 2, 10, confidence)
        with self.assertRaises(ValueError):
            stratified_error_interval((0,), (0,), (0,))
        with self.assertRaises(ValueError):
            stratified_error_interval((0,), (0, 1), (5,))
        for index in ([0, 0], [0, 2], [0, -1], [0, 1.0]):
            with self.assertRaises(ValueError):
                calibrate_from_audit([0, 1], index, [0, 1])
        with self.assertRaises(ValueError):
            calibrate_from_audit([0, 2], [0], [0])


if __name__ == "__main__":
    unittest.main()
