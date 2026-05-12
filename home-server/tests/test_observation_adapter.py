from __future__ import annotations

from datetime import datetime, timedelta, timezone

from demo.observations import ObservationAdapter
from demo.orchestrator import DemoOrchestrator
from triggers.rules import DetectionResult


def _detection(ts: datetime, *, objects=None, names=None) -> DetectionResult:
    return DetectionResult(
        timestamp=ts,
        objects=list(objects or []),
        person_present=bool(names),
        recognized_names=list(names or []),
    )


def test_medicine_observation_maps_to_delayed_demo_reminder() -> None:
    demo = DemoOrchestrator(medicine_reminder_delay_seconds=30)
    demo.set_visitor_name("Maya")
    adapter = ObservationAdapter()
    start = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    assert adapter.apply(demo, _detection(start, objects=["medicine_bottle"])) == []
    assert adapter.apply(demo, _detection(start + timedelta(seconds=29))) == []
    messages = adapter.apply(demo, _detection(start + timedelta(seconds=30)))

    assert messages[0]["type"] == "nudge"
    assert messages[0]["scenario"] == "medicine_pending"
    assert demo.state.medicine_state == "pending"


def test_stove_observation_maps_to_timer_escalation() -> None:
    demo = DemoOrchestrator(stove_first_reminder_seconds=30, stove_escalation_seconds=60)
    adapter = ObservationAdapter()
    start = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    assert adapter.apply(demo, _detection(start, objects=["stove"])) == []
    first = adapter.apply(demo, _detection(start + timedelta(seconds=30)))
    second = adapter.apply(demo, _detection(start + timedelta(seconds=60)))

    assert any(m["type"] == "nudge" for m in first)
    assert any(m["type"] == "caregiver_alert" for m in second)
    assert demo.state.stove_state == "escalated"


def test_stove_observation_resolves_when_user_returns() -> None:
    demo = DemoOrchestrator(stove_first_reminder_seconds=30, stove_escalation_seconds=60)
    adapter = ObservationAdapter()
    start = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    assert adapter.apply(demo, _detection(start, objects=["stove"])) == []
    assert adapter.apply(demo, _detection(start + timedelta(seconds=5))) == []
    messages = adapter.apply(demo, _detection(start + timedelta(seconds=6), objects=["stove"]))

    assert messages[0]["type"] == "assistant_reply"
    assert demo.state.stove_state == "off"


def test_face_observation_maps_to_one_cue() -> None:
    demo = DemoOrchestrator()
    adapter = ObservationAdapter()
    ts = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    first = adapter.apply(demo, _detection(ts, names=["Anya"]))
    second = adapter.apply(demo, _detection(ts + timedelta(seconds=1), names=["Anya"]))

    assert first[0]["type"] == "assistant_reply"
    assert "Anya" in first[0]["sentence"]
    assert second == []
