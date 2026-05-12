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


def test_face_observation_refires_after_rearm_gap() -> None:
    # A visitor who leaves the frame and returns later should hear the
    # cue again. Continuous presence should NOT re-fire even past the
    # rearm window, because last-seen keeps getting refreshed.
    demo = DemoOrchestrator(face_cue_rearm_seconds=10)
    adapter = ObservationAdapter(face_cue_rearm_seconds=10)
    t0 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    first = adapter.apply(demo, _detection(t0, names=["Anya"]))
    still_inside = adapter.apply(demo, _detection(t0 + timedelta(seconds=5), names=["Anya"]))
    # Continuous sightings up to and past the rearm window: still no re-fire.
    continuous = adapter.apply(demo, _detection(t0 + timedelta(seconds=12), names=["Anya"]))
    # Now Anya is absent for >10s. Frames at t=20s, t=21s have no recognized
    # names — the dict isn't touched. At t=33s she comes back: last-seen was
    # t=12s, gap is 21s > 10s, so the cue should re-fire.
    _absent_a = adapter.apply(demo, _detection(t0 + timedelta(seconds=20)))
    _absent_b = adapter.apply(demo, _detection(t0 + timedelta(seconds=21)))
    rearm = adapter.apply(demo, _detection(t0 + timedelta(seconds=33), names=["Anya"]))

    assert first and first[0]["type"] == "assistant_reply"
    assert still_inside == []
    assert continuous == []
    assert rearm and rearm[0]["type"] == "assistant_reply"


def test_reset_clears_face_dedup() -> None:
    # An operator-initiated reset should let cues fire again on the very
    # next frame, even if the rearm window has not yet elapsed.
    demo = DemoOrchestrator(face_cue_rearm_seconds=600)
    adapter = ObservationAdapter(face_cue_rearm_seconds=600)
    t0 = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)

    first = adapter.apply(demo, _detection(t0, names=["Anya"]))
    blocked = adapter.apply(demo, _detection(t0 + timedelta(seconds=1), names=["Anya"]))

    adapter.reset()
    demo.reset_face_cues()
    after_reset = adapter.apply(demo, _detection(t0 + timedelta(seconds=2), names=["Anya"]))

    assert first and first[0]["type"] == "assistant_reply"
    assert blocked == []
    assert after_reset and after_reset[0]["type"] == "assistant_reply"
