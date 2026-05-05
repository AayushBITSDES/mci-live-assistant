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
    gate.process_frame(_det(after, [], person=False))
    later = after + timedelta(seconds=10)
    gate.process_frame(_det(later, [], person=False))

    # Only one risk_window_opened event
    open_events = [e for e in emitted if e["event_type"] == "risk_window_opened" and e.get("scenario") == "kitchen_abandoned"]
    assert len(open_events) == 1


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
