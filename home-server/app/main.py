"""FastAPI entrypoint: WebSocket stream + health check.

The WS handler receives FrameMessage / AudioChunkMessage / StatusMessage
JSON from edge clients, runs the live pipeline (vision -> gate -> LLM ->
TTS), and emits NudgeMessage replies.

## Two operating modes

1. **Live pipeline** -- `app.state.heavy` is non-None. Each WS connection
   gets its own `Session` (gate + candidate queue + worker). Frames flow
   through the real ML stack and produce real LLM-generated nudges.

2. **Debug stub** -- `app.state.heavy` is None (graceful degradation when
   ML deps are missing on the dev box). Every Nth frame emits a hardcoded
   `Pipeline OK` nudge so the wire format can be smoke-tested without
   needing torch / ultralytics / API keys.

The two modes share the routing layer; only the body of the "frame"
branch differs. Audio is parsed-and-acked in both modes pending the
React client growing a MediaRecorder send path.
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
from app.pipeline import Heavy, Session
from context.event_log import append_event
from context.manager import ContextManager
from context.replay import replay_recent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Frame interval at which a debug nudge is emitted when the live pipeline
# is unavailable. At 5 FPS this is ~6 seconds, enough time to read it.
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

    # Try to build the live pipeline. Missing ML deps / weights / API
    # keys fall back to the debug stub without crashing the server.
    app.state.heavy = Heavy.from_settings(settings)

    logger.info(
        "Home server ready | provider=%s | fps=%d | mode=%s | api_key=%s",
        settings.active_llm_provider,
        settings.target_fps,
        "live" if app.state.heavy is not None else "debug-stub",
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
        live = getattr(app.state, "heavy", None) is not None
        return {
            "status": "ok",
            "provider": settings.active_llm_provider,
            "mode": "live" if live else "debug-stub",
        }

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        """Receive frames + audio + status, emit nudges back."""
        await ws.accept()
        device = ws.query_params.get("device_id", "unknown")
        surface = ws.query_params.get("surface", "desktop")
        logger.info("WS connected | device=%s | surface=%s", device, surface)

        cm: ContextManager = ws.app.state.context_manager
        heavy: Heavy | None = getattr(ws.app.state, "heavy", None)

        # Build a Session iff the live pipeline is available. The on_nudge
        # callback is closed over `ws` so the pipeline can stay
        # transport-agnostic.
        session: Session | None = None
        if heavy is not None:
            async def on_nudge(nudge: NudgeMessage) -> None:
                try:
                    await ws.send_text(nudge.model_dump_json())
                except (WebSocketDisconnect, RuntimeError):
                    # Connection closed mid-flight; drop the nudge silently.
                    # RuntimeError covers "Cannot call 'send' once a close
                    # message has been sent." from starlette.
                    pass

            session = Session(
                heavy=heavy,
                context=cm,
                on_nudge=on_nudge,
                auto_dismiss_seconds=settings.nudge_auto_dismiss_seconds,
            )
            await session.start()

        frame_counter = 0

        try:
            while True:
                raw = await ws.receive_text()
                msg_type, parsed = _route_message(raw)

                if msg_type == "frame":
                    frame_counter += 1
                    if session is not None:
                        # Live pipeline. Vision + gate run synchronously here;
                        # the LLM + TTS happen on the session's worker task
                        # and arrive via the on_nudge callback later.
                        await session.handle_frame(parsed)
                    else:
                        nudge = await _maybe_emit_debug_nudge(frame_counter)
                        if nudge is not None:
                            await ws.send_text(nudge.model_dump_json())

                elif msg_type == "audio":
                    # In live mode we run the chunk through Whisper +
                    # voice-command tool dispatch. In debug-stub mode (no
                    # Heavy on dev box) we just parse-and-discard so the
                    # protocol stays clean.
                    if session is not None:
                        await session.handle_audio(parsed)

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
                    ack = AckMessage(
                        message=f"unknown:{(raw or '')[:60]}",
                        server_time=datetime.now(timezone.utc),
                    )
                    await ws.send_text(ack.model_dump_json())

        except WebSocketDisconnect:
            logger.info("WS disconnected | device=%s", device)
        finally:
            if session is not None:
                await session.close()

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


async def _maybe_emit_debug_nudge(frame_counter: int):
    """Periodic hardcoded nudge for wire-format smoke tests.

    Only used when graceful degradation kicked in (live Heavy unavailable).
    """
    if frame_counter % DEBUG_NUDGE_EVERY_N_FRAMES != 0:
        return None

    nudge = NudgeMessage(
        nudge_id=str(uuid.uuid4()),
        sentence=f"Pipeline OK — frame #{frame_counter} round-trip.",
        priority=NudgePriority.QUALITY,
        auto_dismiss_seconds=settings.nudge_auto_dismiss_seconds,
    )

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
