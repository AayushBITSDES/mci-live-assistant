from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


DEFAULT_FACE_PROFILES: dict[str, dict[str, str]] = {
    "Aayush": {
        "relationship": "the person guiding you today",
        "cue": "You can ask him how the headset notices routines.",
    },
}


@dataclass
class DemoState:
    visitor_name: str | None = None
    medicine_state: str = "idle"
    stove_state: str = "off"
    stove_ignored_count: int = 0
    caregiver_alerts: dict[str, dict[str, Any]] = field(default_factory=dict)


class DemoOrchestrator:
    """Small in-memory orchestrator for demo exhibit scenarios."""

    def __init__(
        self,
        *,
        face_profiles: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.state = DemoState()
        self._face_profiles = dict(DEFAULT_FACE_PROFILES if face_profiles is None else face_profiles)
        self._face_cue_shown: set[str] = set()
        self._alert_seq = 0

    def set_visitor_name(self, name: str) -> None:
        self.state.visitor_name = name

    def trigger_medicine_pending(self, *, source: str) -> list[dict[str, Any]]:
        _ = source
        self.state.medicine_state = "pending"
        name = self.state.visitor_name or "there"
        sentence = (
            f"Hi {name}, it is time for your vitamin. "
            "Please take your vitamin when you are ready."
        )
        return [_nudge(sentence, scenario="medicine_pending")]

    def handle_voice_tool(self, tool: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if tool == "markDone" and payload.get("task") == "vitamin":
            self.state.medicine_state = "done"
            return [
                {
                    "type": "assistant_reply",
                    "sentence": "Got it, your vitamin is marked done.",
                    "ts": time.time(),
                }
            ]
        return []

    def trigger_stove_on(self, *, source: str) -> list[dict[str, Any]]:
        _ = source
        self.state.stove_state = "on"
        self.state.stove_ignored_count = 0
        name = self.state.visitor_name or "there"
        sentence = f"{name}, the stove is on. Please check the stove."
        return [_nudge(sentence, priority="safety", scenario="stove_on")]

    def record_stove_ignored(self, *, reason: str) -> list[dict[str, Any]]:
        _ = reason
        if self.state.stove_state not in {"on", "reminded"}:
            return []
        self.state.stove_ignored_count += 1
        out: list[dict[str, Any]] = []
        self.state.stove_state = "reminded"
        if self.state.stove_ignored_count == 3:
            alert_id = f"alert-{uuid.uuid4().hex[:12]}"
            visitor = self.state.visitor_name or "unknown"
            out.append(_nudge(
                f"{visitor}, I am also letting your caregiver know the stove is still on.",
                priority="safety",
                scenario="stove_on",
            ))
            alert = {
                "type": "caregiver_alert",
                "alert_id": alert_id,
                "risk_type": "stove_on",
                "visitor_name": visitor,
                "ignored_count": 3,
                "ts": time.time(),
            }
            out.append(alert)
            self.state.caregiver_alerts[alert_id] = {"status": "open", "risk_type": "stove_on"}
            self.state.stove_state = "escalated"
        return out

    def acknowledge_caregiver_alert(self, alert_id: str) -> dict[str, Any]:
        if alert_id in self.state.caregiver_alerts:
            self.state.caregiver_alerts[alert_id]["status"] = "acknowledged"
        return {"type": "caregiver_ack", "alert_id": alert_id, "ts": time.time()}

    def trigger_face_cue(self, name: str) -> list[dict[str, Any]]:
        if name in self._face_cue_shown:
            return []
        profile = self._face_profiles.get(name)
        if not profile:
            return []
        self._face_cue_shown.add(name)
        relationship = profile.get("relationship", "visitor")
        cue = profile.get("cue", "")
        sentence = f"{name} is {relationship}. {cue}"
        return [{"type": "assistant_reply", "sentence": sentence, "ts": time.time()}]

    def handle_voice_conversation(self, utterance: str) -> list[dict[str, Any]]:
        _ = utterance
        name = self.state.visitor_name or "there"
        if self.state.medicine_state == "pending":
            sentence = (
                f"{name}, you were about to take your vitamin. "
                "Say mark done when you finish the vitamin."
            )
            return [{"type": "assistant_reply", "sentence": sentence, "ts": time.time()}]
        return [
            {
                "type": "assistant_reply",
                "sentence": f"{name}, I am here if you need anything.",
                "ts": time.time(),
            }
        ]


def _nudge(sentence: str, *, priority: str = "quality", scenario: str) -> dict[str, Any]:
    return {
        "type": "nudge",
        "nudge_id": str(uuid.uuid4()),
        "sentence": sentence,
        "priority": priority,
        "audio_b64": None,
        "auto_dismiss_seconds": 10,
        "scenario": scenario,
        "ts": time.time(),
    }
