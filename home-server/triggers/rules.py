"""Hardcoded scenario rules.

Each rule is a pure function of (DetectionResult, ContextManager.state)
and returns a TriggerCandidate or None. Pure functions = trivially
unit-testable, no side effects, no time travel.

Rules are intentionally cheap — they run on every frame at 5 FPS. They
do NOT call the LLM. They merely flag candidates for the gate to escalate.

The kitchen-scene approach: pre-trained YOLO does not know "kettle"
specifically (COCO classes only). So we use the broader signal of
"person present with kitchen objects" as the candidate trigger and let
the LLM disambiguate the specific activity from the actual frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional


# COCO classes that indicate kitchen activity. YOLOv10s pre-trained on
# COCO recognises these out-of-the-box; specific objects like kettles or
# medicine bottles are confirmed by the LLM downstream.
KITCHEN_OBJECTS: frozenset[str] = frozenset({
    "cup", "bottle", "bowl", "microwave", "oven",
    "refrigerator", "sink", "dining table", "spoon", "knife", "fork",
})

# How long someone must be absent from a kitchen scene before we
# escalate to an LLM call. Tuned for "phone call interruption" not
# "quick bathroom trip" — see decision history in Notion.
KITCHEN_ABANDONMENT_THRESHOLD = timedelta(seconds=180)

# Minimum frames a kitchen scene must be observed before we consider
# it a real activity (filters out brief glances).
MIN_KITCHEN_OBSERVATION_FRAMES = 3

# Window during which a second medication event is treated as a
# double-dose risk. 4h is the typical between-dose interval for the
# demo persona; tune per medication type later.
MEDICATION_DOUBLE_DOSE_THRESHOLD = timedelta(hours=4)

# A bottle must be visible for this many consecutive frames before we
# treat it as "intentionally handled" rather than passing through view.
MIN_BOTTLE_OBSERVATION_FRAMES = 3


class CandidateKind(str, Enum):
    """Each kind maps to a downstream handler in the trigger gate."""
    KITCHEN_ABANDONED = "kitchen_abandoned"                   # cooking, then absent > threshold
    MEDICATION_OBSERVED = "medication_observed"               # bottle handled, no recent dose
    MEDICATION_DOUBLE_DOSE_RISK = "medication_double_dose_risk"  # bottle handled within threshold


@dataclass
class TriggerCandidate:
    """A candidate moment flagged by a local rule for LLM escalation.

    Carries enough context for the LLM prompt builder to assemble a
    precise question; the LLM still gets the full ContextManager summary
    on top of this.
    """
    kind: CandidateKind
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Per-frame YOLO output, normalised to what the rules need.

    `objects` is a list of label strings from the latest frame. We do not
    need bounding boxes for any current rule — the detector module is
    free to keep them but the rules ignore them.
    """
    timestamp: datetime
    objects: list[str]
    person_present: bool


# --- Rule: kitchen abandonment ------------------------------------------

def kitchen_abandoned_rule(
    detection: DetectionResult,
    state: Any,                          # ContextManager.state
    *,
    threshold: timedelta = KITCHEN_ABANDONMENT_THRESHOLD,
    now: datetime | None = None,
) -> Optional[TriggerCandidate]:
    """Fire if a kitchen activity was started and the person has now been gone too long.

    State requirements:
      - state.current_activity == "kitchen_activity"
      - state.activity_start is set
      - person not in current frame
      - last seen at activity_start + (frames * frame_interval) — but we
        approximate by using activity_start as the reference. A future
        refinement can track person_last_seen separately.
    """
    if state.current_activity != "kitchen_activity":
        return None
    if state.activity_start is None:
        return None
    if detection.person_present:
        return None

    now = now or datetime.now(timezone.utc)
    elapsed = now - state.activity_start
    if elapsed < threshold:
        return None

    return TriggerCandidate(
        kind=CandidateKind.KITCHEN_ABANDONED,
        reason=f"kitchen activity started {int(elapsed.total_seconds())}s ago, person absent",
        detail={
            "objects_at_start": [],   # populated by gate from event history if useful
            "elapsed_seconds": int(elapsed.total_seconds()),
        },
    )


def detect_kitchen_scene(detection: DetectionResult) -> bool:
    """True if frame shows a person with at least one kitchen object."""
    if not detection.person_present:
        return False
    return any(obj in KITCHEN_OBJECTS for obj in detection.objects)


# --- Rule: medication ---------------------------------------------------

def medication_rule(
    detection: DetectionResult,
    state: Any,                           # ContextManager.state
    *,
    threshold: timedelta = MEDICATION_DOUBLE_DOSE_THRESHOLD,
    now: datetime | None = None,
) -> Optional[TriggerCandidate]:
    """Fire when a bottle has been observed with a person present.

    This rule does NOT decide whether the bottle is medication or whether
    she is actually taking a pill — that is the LLM's job (it sees the
    frame). The rule only flags the moment as worth examining.

    Returns:
      - MEDICATION_DOUBLE_DOSE_RISK when last_medication_time is within
        the threshold (recent dose already recorded).
      - MEDICATION_OBSERVED otherwise (no recent dose; LLM may ask for
        voice confirmation and log the event via markDone tool).
    """
    if not detection.person_present:
        return None
    if "bottle" not in detection.objects:
        return None

    now = now or datetime.now(timezone.utc)
    last = state.last_medication_time

    if last is not None and (now - last) < threshold:
        minutes_since = int((now - last).total_seconds() // 60)
        return TriggerCandidate(
            kind=CandidateKind.MEDICATION_DOUBLE_DOSE_RISK,
            reason=f"medication taken {minutes_since} min ago, bottle in frame",
            detail={
                "minutes_since_dose": minutes_since,
                "threshold_hours": int(threshold.total_seconds() // 3600),
            },
        )

    return TriggerCandidate(
        kind=CandidateKind.MEDICATION_OBSERVED,
        reason="bottle handled by person, no recent medication recorded",
        detail={"has_prior_dose": last is not None},
    )
