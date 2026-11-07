"""Re-assessment after a change, stage 6 of the procedure and Algorithm 6 of docs/framework.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .claim import QualityClaim

#: The six stages of the procedure.
STAGES = {
    1: "state the purpose and list the claims",
    2: "fix materials and sampling plan before reading results",
    3: "measure ratios with uncertainty and check split independence",
    4: "test the task on held-out data and intervene",
    5: "judge with admission conditions and a claim list",
    6: "record the conditions for re-assessment",
}

_EVENT_ALIASES = {
    "data": "data", "data_version": "data", "version": "data", "scene": "data",
    "labels": "labels", "label": "labels",
    "model": "model", "target_model": "model", "reference": "model",
}


def stages_to_redo(event: Union[str, Iterable[str], None]) -> Tuple[int, ...]:
    """Stages that a change event requires, in the order of tests of Algorithm 6.

    A change of the data version or the scene redoes stages 2 to 5, a change of the labels
    redoes stages 3 to 5, and a change of the target model or the reference redoes stages 4
    and 5. An event that touches several aspects follows the widest of them.
    """
    if event is None:
        return ()
    aspects = {event} if isinstance(event, str) else set(event)
    unknown = aspects - set(_EVENT_ALIASES)
    if unknown:
        raise ValueError(f"Unknown change events {sorted(unknown)}")
    kinds = {_EVENT_ALIASES[a] for a in aspects}
    if "data" in kinds:
        return (2, 3, 4, 5)
    if "labels" in kinds:
        return (3, 4, 5)
    if "model" in kinds:
        return (4, 5)
    return ()


def violation_trigger(series: Sequence[float], threshold: float) -> List[int]:
    """Positions t where the monitored series moves by more than the threshold from t - 1."""
    return [t for t in range(1, len(series)) if abs(series[t] - series[t - 1]) > threshold]


def trigger_counts(rates: Sequence[float], scores: Sequence[float], rate_threshold: float = 2.0,
                   score_threshold: float = 0.05) -> Dict[str, int]:
    """How often a rule violation trigger coincides with a change of the task score.

    The series hold the monthly violation rate in points and the task score of a fixed model.
    The result counts the triggered months, those among them with a score change above the
    threshold, the other months, and those among them with such a change.
    """
    triggered = triggered_hits = rest = rest_hits = 0
    for t in range(1, len(rates)):
        moved = abs(scores[t] - scores[t - 1]) > score_threshold
        if abs(rates[t] - rates[t - 1]) > rate_threshold:
            triggered += 1
            triggered_hits += moved
        else:
            rest += 1
            rest_hits += moved
    return dict(triggered=triggered, triggered_hits=int(triggered_hits), rest=rest, rest_hits=int(rest_hits))


def compare_claims(previous: Iterable[QualityClaim], current: Iterable[QualityClaim]) -> List[Dict[str, Any]]:
    """Paired comparison with the last version, one row per object, dimension and measure."""
    before = {(c.object, c.dimension, c.measure): c for c in previous}
    rows = []
    for claim in current:
        key = (claim.object, claim.dimension, claim.measure)
        old = before.get(key)
        rows.append(dict(claim=claim, previous=old,
                         change=None if old is None else claim.estimate.value - old.estimate.value))
    return rows


@dataclass(frozen=True)
class Renewal:
    """Output of Algorithm 6."""

    triggered_at: Tuple[int, ...]
    stages: Tuple[int, ...]
    claims: Tuple[QualityClaim, ...]
    comparison: Tuple[Dict[str, Any], ...]
    valid_until: Any


def renew(previous: Sequence[QualityClaim], event: Union[str, Iterable[str], None], series: Sequence[float],
          threshold: float, redo: Optional[Callable[[Tuple[int, ...], bool], Sequence[QualityClaim]]] = None,
          valid_until: Any = None) -> Renewal:
    """Algorithm 6, the re-assessment after a change.

    A move of the monitored series above the threshold redoes stage 4 on recent data. A change
    event redoes the stages of stages_to_redo. The redo function receives the stages and a flag
    that says whether recent data triggered the run, and it returns the new claims. Claims that
    no stage touches stay in place. The result holds the paired comparison with the last version.
    """
    triggered = tuple(violation_trigger(series, threshold))
    stages = set(stages_to_redo(event))
    if triggered:
        stages.add(4)
    ordered = tuple(sorted(stages))
    claims = tuple(previous)
    if redo is not None and ordered:
        new = tuple(redo(ordered, bool(triggered)))
        redone = {(c.object, c.dimension, c.measure) for c in new}
        claims = new + tuple(c for c in previous if (c.object, c.dimension, c.measure) not in redone)
    return Renewal(triggered, ordered, claims, tuple(compare_claims(previous, claims)), valid_until)
