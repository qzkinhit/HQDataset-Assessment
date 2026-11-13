"""Claims, assessment levels and usage contexts, Definitions 1 to 5."""
import dataclasses

import pytest

from hqd_assess import AssessmentLevel, DatasetVersion, Estimate, QualityClaim, UsageContext, minimum_level, task_gain


def usage():
    return UsageContext("fault diagnosis", "Macro-F1", "split by run", ("rf", "gb"), "base mode", "low", "full scan")


def test_levels_are_ordered():
    assert AssessmentLevel.L1 < AssessmentLevel.L2 < AssessmentLevel.L3 < AssessmentLevel.L4 < AssessmentLevel.L5
    assert [level.label for level in AssessmentLevel] == ["declared", "matched", "measured", "task-tested", "interventional"]
    assert AssessmentLevel.INTERVENTIONAL is AssessmentLevel.L5
    assert all(level.supports for level in AssessmentLevel)


def test_names_of_the_levels():
    assert AssessmentLevel.DECLARED is AssessmentLevel.L1
    assert AssessmentLevel.MATCHED is AssessmentLevel.L2
    assert AssessmentLevel.MEASURED is AssessmentLevel.L3
    assert AssessmentLevel.TASK_TESTED is AssessmentLevel.L4
    assert not hasattr(AssessmentLevel, "E1")


def test_claim_is_a_six_tuple_in_the_order_of_the_definition():
    estimate = Estimate.full_scan(0.841)
    claim = QualityClaim("TEP test set", "annotation accuracy", "file-name label accuracy", estimate, AssessmentLevel.L3, usage())
    assert claim.as_tuple() == ("TEP test set", "annotation accuracy", "file-name label accuracy", estimate, AssessmentLevel.L3, usage())
    text = str(claim)
    assert text.startswith("<TEP test set, annotation accuracy, file-name label accuracy, 0.841 (full scan), L3, (")
    assert "reference split by run" in text and "rf and gb" in text


def test_estimate_lower_bound():
    assert Estimate(0.9).lower == 0.9
    assert Estimate(0.9, 0.85, 0.95).lower == 0.85
    assert Estimate.full_scan(0.5).lower == 0.5
    with pytest.raises(ValueError):
        Estimate(0.9, 0.95, 0.85)
    assert "[0.850, 0.950, Wilson]" in str(Estimate(0.9, 0.85, 0.95, "Wilson"))


def test_version_holds_the_split():
    a = DatasetVersion(records=[1, 2], labels=[0, 1], metadata={"source": "x"}, split="random")
    b = dataclasses.replace(a, split="by run")
    assert a != b


def test_gain_is_a_difference_of_utilities():
    assert task_gain(0.8, 0.7) == pytest.approx(0.1)


def test_purpose_sets_the_minimum_level():
    assert minimum_level("compliance") == AssessmentLevel.L2
    assert minimum_level("selection") == AssessmentLevel.L4
    assert minimum_level("improvement") == AssessmentLevel.L5
