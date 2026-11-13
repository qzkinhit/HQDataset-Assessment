"""Graded judgement, stage 5 of the procedure and Algorithm 5 of docs/framework.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .claim import AssessmentLevel, QualityClaim
from .scoring import dimension_score, score_range


@dataclass(frozen=True)
class Requirement:
    """A claim with the level l* and the threshold tau it has to reach."""

    claim: QualityClaim
    required_level: AssessmentLevel
    threshold: float


@dataclass(frozen=True)
class ClaimRecord:
    """Entry of the claim list, with the issues found for the claim."""

    requirement: Requirement
    insufficient_level: bool
    below_threshold: bool

    @property
    def ok(self) -> bool:
        """Whether the claim has the required level and its lower bound reaches the threshold."""
        return not (self.insufficient_level or self.below_threshold)


@dataclass(frozen=True)
class Judgement:
    """Output of Algorithm 5."""

    passed: bool
    failed_veto: Optional[Requirement]
    records: Tuple[ClaimRecord, ...]
    score: Optional[float]
    score_low: Optional[float]
    score_high: Optional[float]


def _record(requirement: Requirement) -> ClaimRecord:
    claim = requirement.claim
    return ClaimRecord(requirement, claim.level < requirement.required_level,
                       claim.estimate.lower < requirement.threshold)


def judge(veto: Sequence[Requirement], others: Sequence[Requirement], weights: Optional[Sequence[float]] = None,
          weight_perturbation: float = 0.1) -> Judgement:
    """Algorithm 5, the judgement of a dataset under one usage context.

    The dataset passes if and only if all veto items hold and every other claim
    reaches its level with a lower bound at or above its threshold. A veto item
    condition that fails ends the judgement and cannot be compensated. For the other claims
    the function records a level below the required one and values below the threshold, computes the
    score P from their point estimates, and gives the range of P under weight perturbation.
    """
    for requirement in veto:
        if not _record(requirement).ok:
            return Judgement(False, requirement, (), None, None, None)
    records: List[ClaimRecord] = [_record(r) for r in others]
    values = [r.claim.estimate.value for r in others]
    score = low = high = None
    if values:
        score = dimension_score(values, weights)
        low, high = score_range(values, weights, weight_perturbation)
    return Judgement(all(record.ok for record in records), None, tuple(records), score, low, high)
