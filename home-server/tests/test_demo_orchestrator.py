from __future__ import annotations

from app.main import _messages_for_demo_action
from app.models import DemoActionMessage
from demo.orchestrator import DemoOrchestrator


def test_medicine_pending_reminds_by_name_and_resolves_on_done() -> None:
    demo = DemoOrchestrator()
    demo.set_visitor_name("Aayush")

    messages = demo.trigger_medicine_pending(source="operator")

    assert demo.state.visitor_name == "Aayush"
    assert demo.state.medicine_state == "pending"
    assert messages[0]["type"] == "nudge"
    assert "Aayush" in messages[0]["sentence"]
    assert "vitamin" in messages[0]["sentence"].lower()

    resolved = demo.handle_voice_tool("markDone", {"task": "vitamin"})

    assert demo.state.medicine_state == "done"
    assert resolved[0]["type"] == "assistant_reply"
    assert "marked" in resolved[0]["sentence"].lower()


def test_stove_ignores_escalate_to_caregiver_on_third_ignore() -> None:
    demo = DemoOrchestrator()
    demo.set_visitor_name("Shanta")

    first = demo.trigger_stove_on(source="operator")
    second = demo.record_stove_ignored(reason="dismissTemporarily")
    third = demo.record_stove_ignored(reason="overlay_closed")
    fourth = demo.record_stove_ignored(reason="auto_dismiss")

    assert first[0]["type"] == "nudge"
    assert demo.state.stove_ignored_count == 3
    assert not any(m["type"] == "caregiver_alert" for m in second)
    assert not any(m["type"] == "caregiver_alert" for m in third)

    caregiver_alerts = [m for m in fourth if m["type"] == "caregiver_alert"]
    assert len(caregiver_alerts) == 1
    assert caregiver_alerts[0]["risk_type"] == "stove_on"
    assert caregiver_alerts[0]["visitor_name"] == "Shanta"
    assert caregiver_alerts[0]["ignored_count"] == 3
    assert demo.state.stove_state == "escalated"


def test_caregiver_acknowledgement_marks_alert_seen() -> None:
    demo = DemoOrchestrator()
    demo.trigger_stove_on(source="operator")
    demo.record_stove_ignored(reason="one")
    demo.record_stove_ignored(reason="two")
    messages = demo.record_stove_ignored(reason="three")
    alert_id = next(m["alert_id"] for m in messages if m["type"] == "caregiver_alert")

    ack = demo.acknowledge_caregiver_alert(alert_id)

    assert ack["type"] == "caregiver_ack"
    assert ack["alert_id"] == alert_id
    assert demo.state.caregiver_alerts[alert_id]["status"] == "acknowledged"


def test_face_cue_uses_profile_and_cooldown() -> None:
    demo = DemoOrchestrator(
        face_profiles={
            "Aayush": {
                "relationship": "the exhibition guide",
                "cue": "Ask him about how the headset notices routines.",
            }
        }
    )

    first = demo.trigger_face_cue("Aayush")
    second = demo.trigger_face_cue("Aayush")

    assert first[0]["type"] == "assistant_reply"
    assert "Aayush" in first[0]["sentence"]
    assert "headset" in first[0]["sentence"]
    assert second == []


def test_default_face_profile_supports_exhibition_guide() -> None:
    demo = DemoOrchestrator()

    first = demo.trigger_face_cue("Aayush")

    assert first[0]["type"] == "assistant_reply"
    assert "Aayush" in first[0]["sentence"]


def test_non_command_voice_gets_contextual_assistant_reply() -> None:
    demo = DemoOrchestrator()
    demo.set_visitor_name("Maya")
    demo.trigger_medicine_pending(source="operator")

    reply = demo.handle_voice_conversation("what was I doing?")

    assert reply[0]["type"] == "assistant_reply"
    assert "Maya" in reply[0]["sentence"]
    assert "vitamin" in reply[0]["sentence"].lower()


def test_demo_action_only_counts_stove_scenario() -> None:
    demo = DemoOrchestrator()
    demo.set_visitor_name("Shanta")
    demo.trigger_stove_on(source="test")

    unrelated = _messages_for_demo_action(demo, DemoActionMessage(
        device_id="edge-demo",
        action="nudge_closed",
        scenario="medicine_pending",
    ))
    first = _messages_for_demo_action(demo, DemoActionMessage(
        device_id="edge-demo",
        action="nudge_closed",
        scenario="stove_on",
    ))
    second = _messages_for_demo_action(demo, DemoActionMessage(
        device_id="edge-demo",
        action="nudge_auto_dismiss",
        scenario="stove_on",
    ))
    third = _messages_for_demo_action(demo, DemoActionMessage(
        device_id="edge-demo",
        action="nudge_closed",
        scenario="stove_on",
    ))

    assert unrelated == []
    assert first == []
    assert second == []
    assert any(m["type"] == "nudge" for m in third)
    assert any(m["type"] == "caregiver_alert" for m in third)
    assert demo.state.stove_ignored_count == 3
