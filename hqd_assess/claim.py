"""Quality claims and their parts, Definitions 1 to 5 of the survey.

A quality claim is a six-tuple of object, dimension, measure, estimate with
uncertainty, assessment level and usage context. The level records how a conclusion
is obtained and is independent of the quality dimension. A higher level does not
replace a lower one, so a model score says nothing about a license.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Tuple


class AssessmentLevel(IntEnum):
    """The five assessment levels of Definition 5, ordered by the conclusion they support.

    The levels record how a conclusion is obtained. L1 is a statement of the provider in its
    documents. L2 matches a statement or a rule against the data, logs and lineage, with the result
    match or mismatch. L3 computes a defined estimator on the data and gives its uncertainty. L4
    trains a given model on the dataset and measures task utility on held-out data. L5 changes one
    aspect of the data with the other conditions fixed and compares the downstream results.
    """

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5

    DECLARED = 1
    MATCHED = 2
    MEASURED = 3
    TASK_TESTED = 4
    INTERVENTIONAL = 5

    @property
    def label(self) -> str:
        """English name of the level."""
        return _LABELS[self.value]

    @property
    def supports(self) -> str:
        """The conclusion a claim at this level can support."""
        return _SUPPORTS[self.value]


_LABELS = {1: "declared", 2: "matched", 3: "measured", 4: "task-tested", 5: "interventional"}
_SUPPORTS = {
    1: "the provider states a property in its documents",
    2: "the documents or rules agree with the data, logs and lineage",
    3: "a defined estimator on the data lies in an interval",
    4: "a model trained on the dataset reaches a score in the given scene",
    5: "a change of one aspect of the data produces a downstream gain",
}

#: Lowest assessment level per purpose of the assessment, stage 1 of the procedure.
PURPOSE_LEVEL = {
    "compliance": AssessmentLevel.L2,
    "selection": AssessmentLevel.L4,
    "improvement": AssessmentLevel.L5,
}


def minimum_level(purpose: str) -> AssessmentLevel:
    """Return the lowest level a claim needs for a purpose.

    Compliance decides whether a dataset may be used and needs L2 or L3, with veto items that
    other claims cannot offset. Selection decides which candidate fits and needs L4 on one held-out set and one
    model class. Improvement decides which change works and needs L5.
    """
    return PURPOSE_LEVEL[purpose]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its interval and the method that produced them.

    A full scan has no sampling error, so its interval collapses to the value.
    """

    value: float
    low: Optional[float] = None
    high: Optional[float] = None
    method: str = ""

    def __post_init__(self) -> None:
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("The lower end of the interval exceeds the upper end")

    @classmethod
    def full_scan(cls, value: float) -> "Estimate":
        """Estimate from a scan of all records."""
        return cls(value, value, value, "full scan")

    @property
    def lower(self) -> float:
        """Lower bound used by the judgement of Algorithm 5."""
        return self.value if self.low is None else self.low

    def __str__(self) -> str:
        if self.low is None or self.high is None or self.low == self.high:
            note = f" ({self.method})" if self.method else ""
            return f"{self.value:.3f}{note}"
        note = f", {self.method}" if self.method else ""
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}{note}]"


@dataclass(frozen=True)
class UsageContext:
    """Usage context U = (T, F, S, B) of Definition 2.

    T is the task with its metric and reference version, F the downstream model class
    with its training procedure, S the target scene and B the available budget.
    """

    task: str
    metric: str
    reference: str
    model_class: Tuple[str, ...]
    scene: str
    risk: str = ""
    budget: Any = None

    def __str__(self) -> str:
        models = " and ".join(self.model_class)
        parts = [self.task, self.metric, f"reference {self.reference}", models, self.scene]
        if self.risk:
            parts.append(f"risk {self.risk}")
        if self.budget is not None:
            parts.append(f"budget {self.budget}")
        return ", ".join(parts)


@dataclass(frozen=True)
class DatasetVersion:
    """Data version D = (R, L, M, split) of Definition 1.

    The split belongs to the version, so changing the split changes the version.
    """

    records: Any
    labels: Any = None
    metadata: Any = None
    split: Any = None
    name: str = ""


@dataclass(frozen=True)
class QualityClaim:
    """Quality claim of Definition 4, the six-tuple (o, d, m, estimate, l, U)."""

    object: str
    dimension: str
    measure: str
    estimate: Estimate
    level: AssessmentLevel
    usage: UsageContext

    def as_tuple(self) -> tuple:
        """Return the six parts in the order of Definition 4."""
        return (self.object, self.dimension, self.measure, self.estimate, self.level, self.usage)

    def __str__(self) -> str:
        return (f"<{self.object}, {self.dimension}, {self.measure}, {self.estimate}, "
                f"{self.level.name}, ({self.usage})>")


def task_gain(utility: float, reference_utility: float) -> float:
    """Gain of a version over the reference version, Definition 3."""
    return utility - reference_utility
