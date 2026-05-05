"""Lifespan + WS routing tests for the FastAPI app."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import DEBUG_NUDGE_EVERY_N_FRAMES, create_app


def _isolated_settings(tmp_path: Path, *, fresh_start: bool = True) -> Settings:
    """Build a Settings instance pointed at tmp_path so the test never
    touches the real ./storage tree."""
    return Settings(
        fresh_start=fresh_start,
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
