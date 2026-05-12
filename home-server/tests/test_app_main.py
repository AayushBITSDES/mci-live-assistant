"""Lifespan + WS routing tests for the FastAPI app.

Covers BOTH operating modes:
  - debug-stub: app.state.heavy is None, every Nth frame emits the
    hardcoded "Pipeline OK" nudge so wire-format can be smoke-tested
    without ML deps.
  - live: app.state.heavy is set (via a fake Heavy in tests), each WS
    connection spins up a Session that runs the real frame -> gate ->
    LLM -> nudge path.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import DEBUG_NUDGE_EVERY_N_FRAMES, create_app
from app.pipeline import Heavy
from context.event_log import read_events_since
from detection.types import FullDetectionResult, ObjectDetection
from llm.base import CommandResult, NudgeResult, ToolCall
from triggers.rules import (
    KITCHEN_ABANDONMENT_THRESHOLD,
    MIN_KITCHEN_OBSERVATION_FRAMES,
    DetectionResult as RuleInput,
)


def _isolated_settings(tmp_path: Path, *, fresh_start: bool = True) -> Settings:
    """Build a Settings instance pointed at tmp_path so the test never
    touches the real ./storage tree."""
    return Settings(
        fresh_start=fresh_start,
        active_llm_provider="grok",
        xai_api_key="",
        google_api_key="",
        sarvam_api_key="",
        asr_provider="off",
        tts_provider="browser",
        storage_root=tmp_path / "storage",
        events_dir=tmp_path / "storage" / "events",
        faces_dir=tmp_path / "storage" / "faces",
        feedback_dir=tmp_path / "storage" / "feedback",
        nudges_dir=tmp_path / "storage" / "nudges",
        onboarding_file=tmp_path / "storage" / "onboarding.json",
    )


def test_create_app_lifespan_initializes_context_manager(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.main.settings", _isolated_settings(tmp_path, fresh_start=True))
    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.context_manager is not None


def test_health_endpoint_reports_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.main.settings", _isolated_settings(tmp_path, fresh_start=True))
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "provider" in body


# --- WebSocket routing ---------------------------------------------------

def _frame_msg(i: int) -> dict:
    return {
        "type": "frame",
        "device_id": "test",
        "surface": "desktop",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_b64": "abc",
        "width": 640,
        "height": 480,
    }


def test_ws_ping_returns_pong(monkeypatch, tmp_path: Path) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "ack"
            assert payload["message"] == "pong"


def test_ws_emits_debug_nudge_every_n_frames(monkeypatch, tmp_path: Path) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream?device_id=t&surface=desktop") as ws:
            # Send N-1 frames -> no nudge yet
            for i in range(DEBUG_NUDGE_EVERY_N_FRAMES - 1):
                ws.send_text(json.dumps(_frame_msg(i)))

            # Send the Nth -> debug nudge fires
            ws.send_text(json.dumps(_frame_msg(DEBUG_NUDGE_EVERY_N_FRAMES)))
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "nudge"
            assert "Pipeline OK" in payload["sentence"]
            assert payload["priority"] == "quality"


def test_ws_invalid_message_type_acks_with_unknown(monkeypatch, tmp_path: Path) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text("not-json-at-all")
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "ack"
            assert payload["message"].startswith("unknown:")


def test_ws_status_message_logged_silently(monkeypatch, tmp_path: Path) -> None:
    """Status messages should be accepted but not produce a reply."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({
                "type": "status",
                "device_id": "test",
                "mic_active": False,
                "camera_active": True,
            }))
            # Then send a ping — if status was malformed-handled, this still pongs
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_text()
            assert json.loads(data)["type"] == "ack"


# --- Live-pipeline integration ------------------------------------------

class _RecordingProcessor:
    """Scripted processor used to exercise app/main.py's live branch."""

    def __init__(self, scripted: list[tuple[list[str], bool]]) -> None:
        self._scripted = list(scripted)

    async def process(self, frame_bytes, timestamp=None):
        objects, person = (self._scripted.pop(0) if self._scripted else ([], False))
        full = FullDetectionResult(
            objects=[
                ObjectDetection(label=o, confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))
                for o in objects
            ]
            + (
                [ObjectDetection(label="person", confidence=0.95, bbox=(0.0, 0.0, 1.0, 1.0))]
                if person and "person" not in objects
                else []
            ),
        )
        ts = timestamp or datetime.now(timezone.utc)
        reduced = RuleInput(
            timestamp=ts,
            objects=full.object_labels,
            person_present=full.person_present,
            recognized_names=[],
        )
        return full, reduced


class _RecordingLLM:
    provider_name = "fake"

    def __init__(self, sentence: str, *, voice_tool: ToolCall | None = None) -> None:
        self._sentence = sentence
        self._voice_tool = voice_tool
        self.calls: list[dict] = []
        self.voice_calls: list[str] = []

    async def generate_nudge(self, *, image_b64, context_summary, persona="shanta"):
        self.calls.append({"image_b64": image_b64, "summary": context_summary})
        return NudgeResult(
            sentence=self._sentence,
            should_nudge=True,
            validation=None,
            raw_text=self._sentence,
            provider="fake",
        )

    async def handle_voice_command(self, *, transcript, context_summary):
        self.voice_calls.append(transcript)
        return CommandResult(
            tool=self._voice_tool,
            raw_text="" if self._voice_tool is None else self._voice_tool.name,
            provider="fake",
        )


class _StubWhisper:
    """Returns a fixed transcript for any audio bytes."""

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript
        self.last_content_type: str | None = None

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str | None = None,
    ) -> str:
        self.last_content_type = content_type
        return self._transcript


def _b64_jpeg() -> str:
    """Smallest possible JPEG-shaped base64 string the validators accept."""
    return base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")


def _frame_at(ts: datetime) -> dict:
    return {
        "type": "frame",
        "device_id": "live-test",
        "surface": "desktop",
        "timestamp": ts.isoformat(),
        "image_b64": _b64_jpeg(),
        "width": 640,
        "height": 480,
    }


def test_ws_live_mode_emits_real_nudge_when_gate_fires(
    monkeypatch, tmp_path: Path
) -> None:
    """End-to-end: app.state.heavy is set, frames are crafted so the gate
    fires kitchen abandonment, the fake LLM responds, the WS receives
    a nudge whose sentence came from the LLM (not the debug stub)."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    sentence = "Did you finish your tea?"
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES + [([], False)]
    )
    fake_heavy = Heavy(
        processor=_RecordingProcessor(scripted),  # type: ignore[arg-type]
        llm=_RecordingLLM(sentence),               # type: ignore[arg-type]
        tts=None,                                  # text-only nudge
    )

    app = create_app()
    with TestClient(app) as client:
        # Lifespan ran; override Heavy with our fake AFTER startup so
        # graceful degradation can't accidentally null it out.
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=live-test") as ws:
            base = datetime.now(timezone.utc)
            # Open kitchen activity
            for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
                ws.send_text(json.dumps(_frame_at(base + timedelta(seconds=i * 0.2))))
            # Trip the abandonment threshold
            ws.send_text(json.dumps(
                _frame_at(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5))
            ))

            # The pipeline worker is async; the nudge arrives shortly.
            data = ws.receive_text()
            payload = json.loads(data)

    assert payload["type"] == "nudge"
    assert payload["sentence"] == sentence       # came from the LLM, not the stub
    assert payload["priority"] == "safety"       # kitchen_abandoned -> safety
    assert payload["audio_b64"] is None          # TTS disabled in this test


def test_ws_live_mode_quiet_frames_yield_no_nudges(monkeypatch, tmp_path: Path) -> None:
    """Live mode with frames that never trigger a rule: zero nudges go out,
    and the debug-stub doesn't sneak in as a fallback."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    # Frames showing only a chair: never trips kitchen scene or medication.
    scripted = [(["chair"], True)] * 10
    fake_heavy = Heavy(
        processor=_RecordingProcessor(scripted),  # type: ignore[arg-type]
        llm=_RecordingLLM("should never be called"),  # type: ignore[arg-type]
        tts=None,
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=quiet-test") as ws:
            base = datetime.now(timezone.utc)
            for i in range(10):
                ws.send_text(json.dumps(_frame_at(base + timedelta(seconds=i * 0.2))))
            # Send a ping; the *only* reply should be the pong (no stray nudges)
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_text()
            payload = json.loads(data)

    assert payload["type"] == "ack"
    assert payload["message"] == "pong"
    # The LLM should never have been called either
    assert fake_heavy.llm.calls == []            # type: ignore[attr-defined]


def test_health_reports_mode_live_when_heavy_set(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = Heavy(
            processor=_RecordingProcessor([]),  # type: ignore[arg-type]
            llm=_RecordingLLM("x"),             # type: ignore[arg-type]
            tts=None,
        )
        response = client.get("/health")
        body = response.json()
        assert body["mode"] == "live"


def test_health_reports_mode_debug_stub_when_heavy_missing(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        # Lifespan runs Heavy.from_settings which will fail on the dev box
        # (no ultralytics/insightface installed) and leave heavy=None.
        response = client.get("/health")
        body = response.json()
        assert body["mode"] == "debug-stub"
        assert app.state.heavy is None


def test_cors_allows_lan_vite_origin(monkeypatch, tmp_path: Path) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        response = client.options(
            "/demo/operator/event",
            headers={
                "Origin": "http://192.168.1.42:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://192.168.1.42:5173"


def test_ws_live_mode_audio_chunk_dispatches_mark_done(
    monkeypatch, tmp_path: Path
) -> None:
    """End-to-end audio: client sends AudioChunkMessage, server runs Whisper,
    LLM returns markDone, the dispatcher writes medication_taken to the
    audit log AND updates app.state.context_manager."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    llm = _RecordingLLM(
        sentence="(no frames in this test)",
        voice_tool=ToolCall(name="markDone", arguments={"task": "medication"}),
    )
    fake_heavy = Heavy(
        processor=_RecordingProcessor([]),       # type: ignore[arg-type]
        llm=llm,                                 # type: ignore[arg-type]
        tts=None,
        whisper=_StubWhisper("I already took my pill"),  # type: ignore[arg-type]
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=audio-test") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "audio-test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"\x00\x01\x02 fake-webm").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 3000,
            }))
            voice_data = ws.receive_text()
            voice_payload = json.loads(voice_data)
            reply_payload = json.loads(ws.receive_text())

            # Round-trip a ping so we know the audio message has been
            # fully processed by the server's WS loop before we inspect state.
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_text()
            payload = json.loads(data)

        assert voice_payload["type"] == "voice_command"
        assert voice_payload["transcript"] == "I already took my pill"
        assert voice_payload["tool"] == "markDone"
        assert reply_payload["type"] == "assistant_reply"
        assert "marked done" in reply_payload["sentence"]
        assert payload["type"] == "ack" and payload["message"] == "pong"
        # The LLM was asked about the transcript
        assert llm.voice_calls == ["I already took my pill"]
        # ContextManager reflects the medication
        assert app.state.context_manager.state.last_medication_time is not None

    # Audit log gained both voice_command_received AND medication_taken events
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    events = list(read_events_since(cutoff, events_dir=test_settings.events_dir))
    event_kinds = {e.get("event_type") for e in events}
    assert "voice_command_received" in event_kinds
    assert "medication_taken" in event_kinds
    assert "task_marked_done" in event_kinds


def test_ws_live_mode_audio_toggle_camera_sends_edge_control(
    monkeypatch, tmp_path: Path
) -> None:
    """Voice privacy commands should become real edge controls, not just logs."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    fake_heavy = Heavy(
        processor=_RecordingProcessor([]),       # type: ignore[arg-type]
        llm=_RecordingLLM(
            sentence="(no frames in this test)",
            voice_tool=ToolCall(name="toggleCamera", arguments={"state": "off"}),
        ),                                      # type: ignore[arg-type]
        tts=None,
        whisper=_StubWhisper("camera off"),     # type: ignore[arg-type]
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=privacy-test") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "privacy-test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"\x00\x01\x02 fake-webm").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 3000,
            }))
            voice_payload = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "ping"}))
            control_payload = json.loads(ws.receive_text())

    assert voice_payload["type"] == "voice_command"
    assert voice_payload["tool"] == "toggleCamera"
    assert control_payload["type"] == "edge_control"
    assert control_payload["target"] == "camera"
    assert control_payload["action"] == "off"


def test_ws_live_mode_audio_toggle_audio_sends_edge_control(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    llm = _RecordingLLM(sentence="(no frames in this test)")
    fake_heavy = Heavy(
        processor=_RecordingProcessor([]),       # type: ignore[arg-type]
        llm=llm,                                 # type: ignore[arg-type]
        tts=None,
        whisper=_StubWhisper("audio off"),       # type: ignore[arg-type]
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=audio-control-test") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "audio-control-test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"\x00\x01\x02 fake-webm").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 3000,
            }))
            voice_payload = json.loads(ws.receive_text())
            control_payload = json.loads(ws.receive_text())

    assert voice_payload["type"] == "voice_command"
    assert voice_payload["tool"] == "toggleAudio"
    assert control_payload["type"] == "edge_control"
    assert control_payload["target"] == "audio"
    assert control_payload["action"] == "off"
    assert llm.voice_calls == []


def test_ws_live_mode_audio_raw_text_falls_back_to_assistant_reply(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    llm = _RecordingLLM(sentence="(no frames in this test)")
    fake_heavy = Heavy(
        processor=_RecordingProcessor([]),       # type: ignore[arg-type]
        llm=llm,                                 # type: ignore[arg-type]
        tts=None,
        whisper=_StubWhisper("talk to me"),      # type: ignore[arg-type]
    )

    async def raw_text_command(*, transcript, context_summary):
        llm.voice_calls.append(transcript)
        return CommandResult(
            tool=None,
            raw_text="I am here with you.",
            provider="fake",
        )

    llm.handle_voice_command = raw_text_command  # type: ignore[method-assign]

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=raw-reply-test") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "raw-reply-test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"\x00\x01\x02 fake-webm").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 3000,
            }))
            voice_payload = json.loads(ws.receive_text())
            reply_payload = json.loads(ws.receive_text())

    assert voice_payload["type"] == "voice_command"
    assert voice_payload["tool"] is None
    assert reply_payload["type"] == "assistant_reply"
    assert reply_payload["sentence"] == "I am here with you."


def test_ws_live_mode_audio_assistant_reply_sends_reply_message(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)

    fake_heavy = Heavy(
        processor=_RecordingProcessor([]),       # type: ignore[arg-type]
        llm=_RecordingLLM(
            sentence="(no frames in this test)",
            voice_tool=ToolCall(
                name="assistantReply",
                arguments={"sentence": "That is Aayush; ask him about the headset."},
            ),
        ),                                      # type: ignore[arg-type]
        tts=None,
        whisper=_StubWhisper("who is that"),    # type: ignore[arg-type]
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.heavy = fake_heavy
        with client.websocket_connect("/ws/stream?device_id=reply-test") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "reply-test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"\x00\x01\x02 fake-webm").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 3000,
            }))
            voice_payload = json.loads(ws.receive_text())
            reply_payload = json.loads(ws.receive_text())

    assert voice_payload["type"] == "voice_command"
    assert voice_payload["tool"] == "assistantReply"
    assert reply_payload["type"] == "assistant_reply"
    assert "Aayush" in reply_payload["sentence"]


def test_ws_audio_ignored_in_debug_stub_mode(monkeypatch, tmp_path: Path) -> None:
    """Without a Session (no Heavy), audio is parsed-and-discarded silently —
    no crash, no nudge, no audit-log noise."""
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()
    with TestClient(app) as client:
        # Heavy stays None (lifespan's graceful degradation on this Mac)
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "device_id": "test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_b64": base64.b64encode(b"x").decode("ascii"),
                "sample_rate": 16000,
                "duration_ms": 1000,
            }))
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_text()
            payload = json.loads(data)
            assert payload["type"] == "ack" and payload["message"] == "pong"


# --- Exhibition demo routes ----------------------------------------------

def test_demo_operator_medicine_event_returns_named_nudge(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()

    with TestClient(app) as client:
        client.post("/demo/operator/event", json={"event": "visitor_name", "name": "Maya"})
        response = client.post("/demo/operator/event", json={"event": "medicine_pending"})

    assert response.status_code == 200
    body = response.json()
    assert body["messages"][0]["type"] == "nudge"
    assert "Maya" in body["messages"][0]["sentence"]
    assert "vitamin" in body["messages"][0]["sentence"].lower()


def test_demo_stove_second_ignore_pushes_caregiver_alert_and_acknowledges(
    monkeypatch, tmp_path: Path
) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/caregiver") as caregiver:
            client.post("/demo/operator/event", json={"event": "visitor_name", "name": "Shanta"})
            client.post("/demo/operator/event", json={"event": "stove_on"})
            client.post("/demo/operator/event", json={"event": "stove_ignored", "reason": "later"})
            response = client.post("/demo/operator/event", json={"event": "stove_ignored", "reason": "closed"})

            pushed = json.loads(caregiver.receive_text())
            alert_id = pushed["alert_id"]
            ack = client.post(f"/demo/caregiver/alerts/{alert_id}/ack")

    assert response.status_code == 200
    assert pushed["type"] == "caregiver_alert"
    assert pushed["risk_type"] == "stove_on"
    assert pushed["visitor_name"] == "Shanta"
    assert pushed["ignored_count"] == 2
    assert ack.status_code == 200
    assert ack.json()["type"] == "caregiver_ack"


def test_demo_html_apps_are_served(monkeypatch, tmp_path: Path) -> None:
    test_settings = _isolated_settings(tmp_path, fresh_start=True)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("context.event_log.settings", test_settings)
    app = create_app()

    with TestClient(app) as client:
        caregiver = client.get("/caregiver")
        operator = client.get("/operator")

    assert caregiver.status_code == 200
    assert "Caregiver" in caregiver.text
    assert operator.status_code == 200
    assert "Operator" in operator.text
