"""Algorithm 5, the judgement with veto items."""
import pytest

from hqd_assess import Estimate, AssessmentLevel, QualityClaim, Requirement, UsageContext, judge
from hqd_assess.sampling import wilson_interval
from hqd_assess.scoring import PASS_LINE, dimension_score, score_range

U = UsageContext("task", "accuracy", "D0", ("rf",), "scene")


def claim(measure, value, low=None, high=None, level=AssessmentLevel.L3):
    return QualityClaim("D", "dimension", measure, Estimate(value, low, high), level, U)


def req(measure, value, low=None, high=None, level=AssessmentLevel.L3, need=AssessmentLevel.L3, threshold=0.9):
    return Requirement(claim(measure, value, low, high, level), need, threshold)


def test_passes_when_all_conditions_hold():
    verdict = judge([req("license", 1.0, level=AssessmentLevel.L2, need=AssessmentLevel.L2)], [req("format", 0.98, 0.95, 0.99), req("labels", 0.95, 0.92, 0.97)])
    assert verdict.passed
    assert verdict.failed_veto is None
    assert verdict.score == pytest.approx(96.5)
    assert verdict.score_low <= verdict.score <= verdict.score_high


def test_a_failed_veto_condition_cannot_be_compensated():
    others = [req(f"item{i}", 1.0) for i in range(7)]
    safety = req("safety", 0.2, level=AssessmentLevel.L3, threshold=0.96)
    assert dimension_score([0.2] + [1.0] * 7) >= PASS_LINE
    verdict = judge([safety], others)
    assert not verdict.passed
    assert verdict.failed_veto is safety
    assert verdict.score is None
    assert verdict.records == ()


def test_a_veto_item_also_needs_its_level():
    declared = req("license", 1.0, level=AssessmentLevel.L1, need=AssessmentLevel.L2)
    assert judge([declared], []).failed_veto is declared


def test_other_claims_record_both_kinds_of_issues():
    verdict = judge([], [req("declared only", 0.99, level=AssessmentLevel.L1), req("low", 0.8), req("fine", 0.95)])
    assert not verdict.passed
    insufficient, below, fine = verdict.records
    assert insufficient.insufficient_level and not insufficient.below_threshold
    assert below.below_threshold and not below.insufficient_level
    assert fine.ok
    assert verdict.score is not None


def test_the_lower_bound_makes_the_sample_size_part_of_the_verdict():
    def sampled(k, n):
        low, high = wilson_interval(k, n)
        return req("safety", k / n, low, high, threshold=0.96)

    assert judge([sampled(3865, 4000)], []).passed
    assert not judge([sampled(3840, 4000)], []).passed
    assert not judge([sampled(966, 1000)], []).passed


def test_a_high_level_does_not_replace_the_threshold():
    assert not judge([], [req("model fit", 0.85, level=AssessmentLevel.L5, need=AssessmentLevel.L4)]).passed


def test_weight_range_is_exact_on_a_hand_made_case():
    low, high = score_range([1.0, 0.0], [0.5, 0.5], perturbation=0.2)
    assert low == pytest.approx(40.0)
    assert high == pytest.approx(60.0)
    assert score_range([0.7, 0.7, 0.7], perturbation=0.1) == pytest.approx((70.0, 70.0))


def test_weights_enter_the_score():
    verdict = judge([], [req("a", 1.0), req("b", 0.9)], weights=[0.75, 0.25], weight_perturbation=0.0)
    assert verdict.score == pytest.approx(97.5)
    assert verdict.score_low == pytest.approx(97.5)
