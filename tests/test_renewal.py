"""Algorithm 6, the re-assessment after a change."""
import pytest

from hqd_assess import Estimate, AssessmentLevel, QualityClaim, UsageContext, renew, stages_to_redo, trigger_counts, violation_trigger
from hqd_assess.renewal import STAGES, compare_claims

U = UsageContext("task", "accuracy", "D0", ("rf",), "scene")


def claim(measure, value):
    return QualityClaim("D", "dimension", measure, Estimate(value), AssessmentLevel.L4, U)


@pytest.mark.parametrize("event, stages", [
    ("data_version", (2, 3, 4, 5)), ("scene", (2, 3, 4, 5)), ("labels", (3, 4, 5)), ("target_model", (4, 5)), ("reference", (4, 5)), (None, ()),
    (["labels", "reference"], (3, 4, 5)), (["scene", "labels"], (2, 3, 4, 5)),
])
def test_stages_per_change_event(event, stages):
    assert stages_to_redo(event) == stages


def test_unknown_event_is_rejected():
    with pytest.raises(ValueError):
        stages_to_redo("weather")


def test_six_stages():
    assert sorted(STAGES) == [1, 2, 3, 4, 5, 6]


def test_violation_trigger():
    assert violation_trigger([5.0, 5.5, 9.0, 8.9, 2.0], 2.0) == [2, 4]
    assert violation_trigger([5.0], 2.0) == []
    assert violation_trigger([5.0, 7.0], 2.0) == []


def test_trigger_counts_on_a_hand_made_series():
    rates = [5.0, 5.5, 9.0, 9.1, 4.0, 4.2]
    scores = [0.80, 0.79, 0.60, 0.70, 0.71, 0.72]
    assert trigger_counts(rates, scores) == dict(triggered=2, triggered_hits=1, rest=3, rest_hits=1)


def test_renew_redoes_stage_4_on_a_trigger_and_compares_with_the_last_version():
    old = [claim("score", 0.8)]
    seen = {}

    def redo(stages, recent):
        seen.update(stages=stages, recent=recent)
        return [claim("score", 0.7)]

    out = renew(old, None, [5.0, 9.0], 2.0, redo, valid_until="2026-12")
    assert out.triggered_at == (1,)
    assert out.stages == (4,)
    assert seen == dict(stages=(4,), recent=True)
    assert out.comparison[0]["change"] == pytest.approx(-0.1)
    assert out.valid_until == "2026-12"


def test_renew_unions_the_event_and_the_trigger():
    assert renew([], "labels", [5.0, 9.0], 2.0).stages == (3, 4, 5)
    assert renew([], "scene", [5.0, 5.1], 2.0).stages == (2, 3, 4, 5)


def test_renew_keeps_the_claims_without_a_change():
    old = [claim("score", 0.8)]
    out = renew(old, None, [5.0, 5.1], 2.0, redo=lambda *_: pytest.fail("no stage should run"))
    assert out.stages == ()
    assert out.claims == tuple(old)
    assert out.comparison[0]["change"] == 0.0


def test_a_new_claim_has_no_previous_value():
    rows = compare_claims([], [claim("score", 0.8)])
    assert rows[0]["previous"] is None
    assert rows[0]["change"] is None


def test_renew_keeps_the_claims_that_no_stage_touches():
    fit, license_claim = claim("score", 0.8), claim("license", 1.0)
    out = renew([fit, license_claim], "target_model", [5.0, 5.1], 2.0, lambda stages, recent: [claim("score", 0.7)])
    assert [c.measure for c in out.claims] == ["score", "license"]
    assert out.claims[1] is license_claim
    changes = {row["claim"].measure: row["change"] for row in out.comparison}
    assert changes["score"] == pytest.approx(-0.1)
    assert changes["license"] == 0.0
