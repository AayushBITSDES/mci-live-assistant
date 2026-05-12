from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_FACE_PROFILES: dict[str, dict[str, str]] = {
    "Anya": {
        "relationship": "your caretaker today",
        "cue": "You can ask her what your plan is for the next few minutes.",
    },
}


@dataclass
class DemoState:
    visitor_name: str | None = None
    medicine_state: str = "idle"
    medicine_last_seen_at: datetime | None = None
    stove_state: str = "off"
    stove_ignored_count: int = 0
    stove_started_at: datetime | None = None
    stove_first_reminded_at: datetime | None = None
    caregiver_alerts: dict[str, dict[str, Any]] = field(default_factory=dict)


class DemoOrchestrator:
    """Small in-memory orchestrator for demo exhibit scenarios."""

    def __init__(
        self,
        *,
        face_profiles: dict[str, dict[str, str]] | None = None,
        medicine_reminder_delay_seconds: float = 30.0,
        stove_first_reminder_seconds: float = 30.0,
        stove_escalation_seconds: float = 60.0,
    ) -> None:
        self.state = DemoState()
        self._face_profiles = dict(DEFAULT_FACE_PROFILES if face_profiles is None else face_profiles)
        self._face_cue_shown: set[str] = set()
        self._alert_seq = 0
        self._medicine_delay = timedelta(seconds=medicine_reminder_delay_seconds)
        self._stove_first_delay = timedelta(seconds=stove_first_reminder_seconds)
        self._stove_escalation_delay = timedelta(seconds=stove_escalation_seconds)

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

    def record_medicine_bottle_seen(self, *, now: datetime | None = None) -> None:
        now = _aware(now)
        if self.state.medicine_state in {"idle", "done"}:
            self.state.medicine_state = "handled"
        self.state.medicine_last_seen_at = now

    def check_medicine_reminder(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        now = _aware(now)
        if self.state.medicine_state != "handled":
            return []
        if self.state.medicine_last_seen_at is None:
            return []
        if now - self.state.medicine_last_seen_at < self._medicine_delay:
            return []
        return self.trigger_medicine_pending(source="recognizer")

    def handle_voice_tool(self, tool: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        task = str(payload.get("task", "")).strip().lower()
        if tool == "markDone" and _looks_like_medicine(task):
            self.state.medicine_state = "done"
            return [
                {
                    "type": "assistant_reply",
                    "sentence": "Got it, your vitamin is marked done.",
                    "ts": time.time(),
                }
            ]
        if tool == "markDone" and _looks_like_stove_resolution(task):
            self.state.stove_state = "off"
            self.state.stove_started_at = None
            self.state.stove_first_reminded_at = None
            return [
                {
                    "type": "assistant_reply",
                    "sentence": "Got it, the stove risk is marked resolved.",
                    "ts": time.time(),
                }
            ]
        return []

    def trigger_stove_on(self, *, source: str) -> list[dict[str, Any]]:
        _ = source
        self.state.stove_state = "reminded"
        self.state.stove_ignored_count = 0
        self.state.stove_started_at = _aware(None)
        self.state.stove_first_reminded_at = self.state.stove_started_at
        name = self.state.visitor_name or "there"
        sentence = f"{name}, the stove is on. Please check the stove."
        return [_nudge(sentence, priority="safety", scenario="stove_on")]

    def record_stove_interaction(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        now = _aware(now)
        if self.state.stove_state == "off":
            self.state.stove_state = "active"
            self.state.stove_ignored_count = 0
            self.state.stove_started_at = now
            self.state.stove_first_reminded_at = None
        return []

    def check_stove_timers(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        now = _aware(now)
        if self.state.stove_state == "active":
            started = self.state.stove_started_at
            if started is not None and now - started >= self._stove_first_delay:
                self.state.stove_state = "reminded"
                self.state.stove_first_reminded_at = now
                name = self.state.visitor_name or "there"
                return [_nudge(
                    f"{name}, the stove may still be on. Please check it now.",
                    priority="safety",
                    scenario="stove_on",
                )]
        if self.state.stove_state == "reminded":
            reminded_at = self.state.stove_first_reminded_at or self.state.stove_started_at
            if reminded_at is not None and now - reminded_at >= (
                self._stove_escalation_delay - self._stove_first_delay
            ):
                return self._escalate_stove()
        return []

    def record_stove_ignored(self, *, reason: str) -> list[dict[str, Any]]:
        _ = reason
        if self.state.stove_state not in {"active", "reminded"}:
            return []
        self.state.stove_ignored_count += 1
        self.state.stove_state = "reminded"
        if self.state.stove_first_reminded_at is None:
            self.state.stove_first_reminded_at = _aware(None)
        if self.state.stove_ignored_count >= 2:
            return self._escalate_stove()
        return []

    def _escalate_stove(self) -> list[dict[str, Any]]:
        if self.state.stove_state == "escalated":
            return []
        alert_id = f"alert-{uuid.uuid4().hex[:12]}"
        visitor = self.state.visitor_name or "unknown"
        out: list[dict[str, Any]] = [
            _nudge(
                f"{visitor}, I am also letting your caregiver know the stove may still be on.",
                priority="safety",
                scenario="stove_on",
            )
        ]
        alert = {
            "type": "caregiver_alert",
            "alert_id": alert_id,
            "risk_type": "stove_on",
            "visitor_name": visitor,
            "ignored_count": max(self.state.stove_ignored_count, 2),
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


def _aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _looks_like_medicine(task: str) -> bool:
    return any(k in task for k in ("vitamin", "medic", "pill", "tablet", "dose", "meds"))


def _looks_like_stove_resolution(task: str) -> bool:
    return any(k in task for k in ("stove", "burner", "gas", "cooking", "tea"))
