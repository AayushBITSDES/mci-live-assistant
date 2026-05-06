"""Harness tests for TriggerGate + rules + RiskWindow lifecycle.

Pure tests, no I/O. The audit-trail emitter is captured into a list so
tests can assert exactly which events the gate produced.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from context.manager import ContextManager
from triggers.gate import TriggerGate
from triggers.risk_window import close_risk_window_event, open_risk_window_event
from triggers.rules import (
    KITCHEN_ABANDONMENT_THRESHOLD,
    MEDICATION_DOUBLE_DOSE_THRESHOLD,
    MIN_BOTTLE_OBSERVATION_FRAMES,
    MIN_KITCHEN_OBSERVATION_FRAMES,
    CandidateKind,
    DetectionResult,
    detect_kitchen_scene,
    medication_rule,
)


def _det(ts: datetime, objects, person: bool) -> DetectionResult:
    return DetectionResult(timestamp=ts, objects=list(objects), person_present=person)


@pytest.fixture
def gate_with_emitter():
    cm = ContextManager()
    emitted: list[dict] = []
    gate = TriggerGate(cm, emit_event=emitted.append)
    return cm, gate, emitted


# --- detect_kitchen_scene ------------------------------------------------

def test_kitchen_scene_requires_person() -> None:
    ts = datetime.now(timezone.utc)
    assert detect_kitchen_scene(_det(ts, ["cup", "bowl"], person=False)) is False


def test_kitchen_scene_requires_at_least_one_kitchen_object() -> None:
    ts = datetime.now(timezone.utc)
    assert detect_kitchen_scene(_det(ts, ["chair"], person=True)) is False
    assert detect_kitchen_scene(_det(ts, ["cup"], person=True)) is True


def test_detection_result_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        DetectionResult(timestamp=datetime.now(), objects=[], person_present=False)


def test_detection_result_default_recognized_names_empty() -> None:
    ts = datetime.now(timezone.utc)
    d = DetectionResult(timestamp=ts, objects=[], person_present=False)
    assert d.recognized_names == []


# --- Activity opens after MIN frames ------------------------------------

def test_kitchen_activity_opens_after_min_frames(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))
    assert cm.state.current_activity == "kitchen_activity"
    assert any(e["event_type"] == "activity_started" for e in emitted)


def test_kitchen_activity_does_not_open_before_min_frames(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES - 1):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))
    assert cm.state.current_activity is None


def test_counter_resets_when_kitchen_scene_breaks(gate_with_emitter) -> None:
    cm, gate, _ = gate_with_emitter
    base = datetime.now(timezone.utc)
    # Two kitchen frames, then a non-kitchen frame, then two more
    gate.process_frame(_det(base, ["cup"], person=True))
    gate.process_frame(_det(base + timedelta(seconds=0.2), ["cup"], person=True))
    gate.process_frame(_det(base + timedelta(seconds=0.4), ["chair"], person=True))  # break
    gate.process_frame(_det(base + timedelta(seconds=0.6), ["cup"], person=True))
    gate.process_frame(_det(base + timedelta(seconds=0.8), ["cup"], person=True))
    # Counter reset, only 2 consecutive frames again -> not enough
    assert cm.state.current_activity is None


# --- Abandonment fires after threshold ----------------------------------

def test_kitchen_abandoned_fires_after_180s_absence(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    # Open activity
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup", "bowl"], person=True))

    # Person leaves; before threshold -> no candidate
    candidates = gate.process_frame(_det(base + timedelta(seconds=60), [], person=False))
    assert candidates == []

    # After threshold -> candidate fires
    after = base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5)
    candidates = gate.process_frame(_det(after, [], person=False))
    assert len(candidates) == 1
    assert candidates[0].kind == CandidateKind.KITCHEN_ABANDONED
    assert "kitchen_abandoned" in [w.scenario for w in cm.state.open_risk_windows]


def test_abandonment_does_not_double_fire_when_window_already_open(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))

    after = base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5)
    first = gate.process_frame(_det(after, [], person=False))
    later = after + timedelta(seconds=10)
    second = gate.process_frame(_det(later, [], person=False))

    # First call past threshold: one candidate + one window-open event
    assert len(first) == 1 and first[0].kind == CandidateKind.KITCHEN_ABANDONED
    # Second call: NO candidate (window already open) — guards the LLM
    # dispatcher from burning ~300 calls per 60s of absence
    assert second == []
    open_events = [e for e in emitted if e["event_type"] == "risk_window_opened" and e.get("scenario") == "kitchen_abandoned"]
    assert len(open_events) == 1


def test_kitchen_candidate_silenced_for_every_subsequent_frame(gate_with_emitter) -> None:
    """Regression test for the kitchen_candidate-on-every-frame bug.

    At 5 FPS, once the threshold is crossed the rule keeps firing on each
    frame. The gate must suppress the candidate (not just the event)
    while a window is already open.
    """
    cm, gate, _ = gate_with_emitter
    base = datetime.now(timezone.utc)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))

    # Cross the abandonment threshold, then keep sending absent frames at 5 FPS
    start = base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=1)
    candidates_per_frame = []
    for i in range(20):                     # 20 frames @ 5 FPS = 4 seconds of absence
        ts = start + timedelta(seconds=i * 0.2)
        candidates_per_frame.append(gate.process_frame(_det(ts, [], person=False)))

    # Exactly one frame produces a candidate; the other 19 are silent
    nonempty = [c for c in candidates_per_frame if c]
    assert len(nonempty) == 1
    assert nonempty[0][0].kind == CandidateKind.KITCHEN_ABANDONED


def test_kitchen_activity_ends_after_abandonment_so_next_session_starts_fresh(gate_with_emitter) -> None:
    """Regression test for the never-transitions-out lifecycle bug.

    Before the fix, current_activity stayed set forever after the first
    abandonment, so the next time a person was absent (even hours later)
    the rule would instantly fire because elapsed = now - original_start
    was already past threshold. After the fix, the gate emits
    activity_ended when abandonment fires, and subsequent kitchen
    sessions start with a fresh activity_start.
    """
    cm, gate, _ = gate_with_emitter
    base = datetime.now(timezone.utc)

    # Session 1: open + abandon
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))
    gate.process_frame(_det(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5), [], person=False))

    # After the candidate fires, activity has ended in state
    assert cm.state.current_activity is None
    assert cm.state.activity_start is None

    # Externally close the abandonment window (e.g. user voice-dismissal)
    open_window = next(w for w in cm.state.open_risk_windows if w.scenario == CandidateKind.KITCHEN_ABANDONED.value) \
        if cm.state.open_risk_windows else None
    assert open_window is not None
    cm.record_event({
        "event_type": "risk_window_closed",
        "window_id": open_window.id,
        "status": "resolved",
        "timestamp": (base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=10)).isoformat(),
    })

    # Session 2 starts much later (an hour gap)
    later_base = base + timedelta(hours=1)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(later_base + timedelta(seconds=i * 0.2), ["cup"], person=True))
    assert cm.state.current_activity == "kitchen_activity"
    assert cm.state.activity_start is not None
    # activity_start should be inside the SECOND session, not stuck at base
    assert cm.state.activity_start >= later_base

    # Person leaves immediately — should NOT fire abandonment because we
    # just opened the activity. Without the fix, elapsed would have been
    # ~1 hour and the rule would fire on the very next absent frame.
    candidates = gate.process_frame(_det(later_base + timedelta(seconds=2), [], person=False))
    assert candidates == []


# --- Audit-trail invariant ----------------------------------------------

def test_every_state_change_is_emitted(gate_with_emitter) -> None:
    """Whatever the gate writes to ContextManager must also be in the event stream."""
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
        gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["cup"], person=True))

    after = base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5)
    gate.process_frame(_det(after, [], person=False))

    # Replay from the emitted events into a fresh ContextManager
    replay = ContextManager()
    for e in emitted:
        replay.record_event(e)
    assert replay.snapshot() == cm.snapshot()


# --- RiskWindow event helpers --------------------------------------------

def test_open_risk_window_event_has_required_fields() -> None:
    e = open_risk_window_event("stove_unattended")
    assert e["event_type"] == "risk_window_opened"
    assert e["scenario"] == "stove_unattended"
    assert "window_id" in e and "timestamp" in e


def test_close_risk_window_event_carries_status() -> None:
    e = close_risk_window_event("rw-1", "resolved")
    assert e["event_type"] == "risk_window_closed"
    assert e["window_id"] == "rw-1"
    assert e["status"] == "resolved"


# --- Medication rule (pure) ---------------------------------------------

def test_medication_rule_silent_without_bottle() -> None:
    cm = ContextManager()
    ts = datetime.now(timezone.utc)
    assert medication_rule(_det(ts, ["cup"], person=True), cm.state) is None


def test_medication_rule_silent_without_person() -> None:
    cm = ContextManager()
    ts = datetime.now(timezone.utc)
    assert medication_rule(_det(ts, ["bottle"], person=False), cm.state) is None


def test_medication_rule_observed_when_no_prior_dose() -> None:
    cm = ContextManager()
    ts = datetime.now(timezone.utc)
    candidate = medication_rule(_det(ts, ["bottle"], person=True), cm.state, now=ts)
    assert candidate is not None
    assert candidate.kind == CandidateKind.MEDICATION_OBSERVED
    assert candidate.detail["has_prior_dose"] is False


def test_medication_rule_double_dose_within_threshold() -> None:
    cm = ContextManager()
    earlier = datetime.now(timezone.utc) - timedelta(hours=1)
    cm.record_event({"event_type": "medication_taken", "timestamp": earlier.isoformat()})

    now = datetime.now(timezone.utc)
    candidate = medication_rule(_det(now, ["bottle"], person=True), cm.state, now=now)
    assert candidate is not None
    assert candidate.kind == CandidateKind.MEDICATION_DOUBLE_DOSE_RISK
    assert candidate.detail["minutes_since_dose"] == 60
    assert candidate.detail["threshold_hours"] == 4


def test_medication_rule_observed_after_threshold_passed() -> None:
    cm = ContextManager()
    long_ago = datetime.now(timezone.utc) - MEDICATION_DOUBLE_DOSE_THRESHOLD - timedelta(minutes=10)
    cm.record_event({"event_type": "medication_taken", "timestamp": long_ago.isoformat()})

    now = datetime.now(timezone.utc)
    candidate = medication_rule(_det(now, ["bottle"], person=True), cm.state, now=now)
    assert candidate is not None
    assert candidate.kind == CandidateKind.MEDICATION_OBSERVED
    assert candidate.detail["has_prior_dose"] is True


# --- Medication via the gate (debounced + escalation) ------------------

def test_gate_does_not_fire_medication_on_single_frame(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    ts = datetime.now(timezone.utc)
    candidates = gate.process_frame(_det(ts, ["bottle"], person=True))
    assert candidates == []  # below MIN_BOTTLE_OBSERVATION_FRAMES


def test_gate_fires_medication_observed_after_min_frames(gate_with_emitter) -> None:
    cm, gate, emitted = gate_with_emitter
    base = datetime.now(timezone.utc)
    candidates = []
    for i in range(MIN_BOTTLE_OBSERVATION_FRAMES):
        candidates = gate.process_frame(_det(base + timedelta(seconds=i * 0.2), ["bottle"], person=True))
    assert any(c.kind == CandidateKind.MEDICATION_OBSERVED for c in candidates)
    assert any(w.scenario == "medication_observed" for w in cm.state.open_risk_windows)


def test_gate_fires_double_dose_when_recent_dose_recorded(gate_with_emitter) -> None:
    cm, gate, _ = gate_with_emitter
    earlier = datetime.now(timezone.utc) - timedelta(hours=2)
    cm.record_event({"event_type": "medication_taken", "timestamp": earlier.isoformat()})

    now = datetime.now(timezone.utc)
    candidates = []
    for i in range(MIN_BOTTLE_OBSERVATION_FRAMES):
        candidates = gate.process_frame(_det(now + timedelta(seconds=i * 0.2), ["bottle"], person=True))
    assert any(c.kind == CandidateKind.MEDICATION_DOUBLE_DOSE_RISK for c in candidates)


def test_gate_bottle_counter_resets_when_bottle_leaves_frame(gate_with_emitter) -> None:
    cm, gate, _ = gate_with_emitter
    base = datetime.now(timezone.utc)
    gate.process_frame(_det(base, ["bottle"], person=True))
    gate.process_frame(_det(base + timedelta(seconds=0.2), ["bottle"], person=True))
    gate.process_frame(_det(base + timedelta(seconds=0.4), ["chair"], person=True))   # bottle gone
    candidates = gate.process_frame(_det(base + timedelta(seconds=0.6), ["bottle"], person=True))
    assert candidates == []  # counter reset; only 1 consecutive bottle frame
