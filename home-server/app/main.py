"""FastAPI entrypoint: WebSocket stream + health check.

The WS handler receives FrameMessage / AudioChunkMessage / StatusMessage
JSON from edge clients, appends events to the JSONL audit trail, and
emits NudgeMessage replies. When LLM API keys are present, every Nth
frame goes to the configured provider; otherwise a debug nudge fires
periodically so the wire format can be smoke-tested without credits.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AckMessage,
    AudioChunkMessage,
    FrameMessage,
    NudgeMessage,
    NudgePriority,
    StatusMessage,
)
from context.event_log import append_event
from context.manager import ContextManager
from context.replay import replay_recent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Frame interval at which a debug nudge is emitted when no LLM is
# configured. At 5 FPS this is ~6 seconds, enough time to read it.
DEBUG_NUDGE_EVERY_N_FRAMES = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    cm = ContextManager()
    if settings.fresh_start:
        logger.info("Fresh start: skipping JSONL replay (clean demo state)")
    else:
        replayed = replay_recent(cm, lookback_hours=settings.replay_lookback_hours)
        logger.info("Replayed %d recent events into ContextManager", replayed)
    app.state.context_manager = cm
    logger.info(
        "Home server ready | provider=%s | fps=%d | api_key=%s",
        settings.active_llm_provider,
        settings.target_fps,
        "set" if (settings.xai_api_key or settings.google_api_key) else "missing",
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MCI Home Server",
        version="0.1.0",
        description="Always-on home brain for the contextual AI prototype.",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "provider": settings.active_llm_provider}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        """Receive frames + audio + status, emit nudges back."""
        await ws.accept()
        device = ws.query_params.get("device_id", "unknown")
        surface = ws.query_params.get("surface", "desktop")
        logger.info("WS connected | device=%s | surface=%s", device, surface)

        cm: ContextManager = ws.app.state.context_manager
        frame_counter = 0

        try:
            while True:
                raw = await ws.receive_text()
                msg_type, parsed = _route_message(raw)

                if msg_type == "frame":
                    frame_counter += 1
                    nudge = await _maybe_emit_debug_nudge(frame_counter, cm)
                    if nudge is not None:
                        await ws.send_text(nudge.model_dump_json())

                elif msg_type == "audio":
                    # Phase 5 wires real ASR here. For now: ack only.
                    pass

                elif msg_type == "status":
                    logger.info(
                        "WS status | mic=%s | camera=%s",
                        parsed.mic_active,
                        parsed.camera_active,
                    )

                elif msg_type == "ping":
                    ack = AckMessage(message="pong", server_time=datetime.now(timezone.utc))
                    await ws.send_text(ack.model_dump_json())

                else:
                    # Unknown shape -> echo for debugging
                    ack = AckMessage(
                        message=f"unknown:{(raw or '')[:60]}",
                        server_time=datetime.now(timezone.utc),
                    )
                    await ws.send_text(ack.model_dump_json())

        except WebSocketDisconnect:
            logger.info("WS disconnected | device=%s", device)

    return app


# --- Helpers --------------------------------------------------------------

def _route_message(raw: str):
    """Parse a raw WS text into one of the known Pydantic models.

    Returns ("frame"|"audio"|"status"|"ping"|"unknown", parsed_or_None).
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "unknown", None

    if not isinstance(payload, dict):
        return "unknown", None

    kind = payload.get("type")
    try:
        if kind == "frame":
            return "frame", FrameMessage.model_validate(payload)
        if kind == "audio":
            return "audio", AudioChunkMessage.model_validate(payload)
        if kind == "status":
            return "status", StatusMessage.model_validate(payload)
        if kind == "ping":
            return "ping", payload
    except ValidationError as exc:
        logger.warning("Invalid %s message: %s", kind, exc)
        return "unknown", None

    return "unknown", payload


async def _maybe_emit_debug_nudge(frame_counter: int, cm: ContextManager):
    """Periodic hardcoded nudge so edge devs can smoke-test without API keys.

    Real LLM dispatch lands once `app.state.context_manager` is wired into
    the gate + processor + LLM client (next phase). For now this proves
    the wire format end-to-end.
    """
    if frame_counter % DEBUG_NUDGE_EVERY_N_FRAMES != 0:
        return None

    nudge = NudgeMessage(
        nudge_id=str(uuid.uuid4()),
        sentence=f"Pipeline OK — frame #{frame_counter} round-trip.",
        priority=NudgePriority.QUALITY,
        auto_dismiss_seconds=settings.nudge_auto_dismiss_seconds,
    )

    # Audit-log the debug emission so it shows up in the dashboard
    await asyncio.to_thread(
        append_event,
        {
            "event_type": "debug_nudge_emitted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nudge_id": nudge.nudge_id,
            "frame_counter": frame_counter,
        },
    )

    return nudge


app = create_app()
