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
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AckMessage,
    AssistantReplyMessage,
    AudioChunkMessage,
    DemoActionMessage,
    EdgeControlMessage,
    FrameMessage,
    NudgeMessage,
    NudgePriority,
    StatusMessage,
    VoiceCommandMessage,
)
from app.pipeline import Heavy, Session
from context.event_log import append_event
from context.manager import ContextManager
from context.replay import replay_recent
from demo.orchestrator import DemoOrchestrator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_HOME_SERVER_ROOT = Path(__file__).resolve().parent.parent


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
    app.state.demo_orchestrator = DemoOrchestrator()
    app.state.edge_sockets = set()
    app.state.caregiver_sockets = set()

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|[0-9.]+|[a-zA-Z0-9.-]+\.local):5173",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        live = getattr(app.state, "heavy", None) is not None
        return {
            "status": "ok",
            "provider": settings.active_llm_provider,
            "mode": "live" if live else "debug-stub",
        }

    @app.get("/caregiver", response_class=HTMLResponse)
    async def caregiver_page() -> str:
        return _read_demo_html("caregiver/index.html")

    @app.get("/operator", response_class=HTMLResponse)
    async def operator_page() -> str:
        return _read_demo_html("operator/index.html")

    @app.post("/demo/operator/event")
    async def demo_operator_event(payload: dict[str, Any]) -> dict[str, Any]:
        """Hidden exhibition control endpoint for deterministic demo beats."""
        demo: DemoOrchestrator = app.state.demo_orchestrator
        event = str(payload.get("event", "")).strip()
        source = str(payload.get("source", "operator")).strip() or "operator"
        messages: list[dict[str, Any]] = []

        if event == "visitor_name":
            demo.set_visitor_name(str(payload.get("name", "")).strip())
        elif event == "medicine_pending":
            messages = demo.trigger_medicine_pending(source=source)
        elif event == "stove_on":
            messages = demo.trigger_stove_on(source=source)
        elif event == "stove_ignored":
            messages = demo.record_stove_ignored(
                reason=str(payload.get("reason", "operator")).strip() or "operator"
            )
        elif event == "face_cue":
            messages = demo.trigger_face_cue(str(payload.get("name", "")).strip())
        elif event == "voice_conversation":
            messages = demo.handle_voice_conversation(str(payload.get("utterance", "")))
        elif event == "voice_tool":
            args = payload.get("arguments")
            messages = demo.handle_voice_tool(
                str(payload.get("tool", "")).strip(),
                args if isinstance(args, dict) else {},
            )
        else:
            messages = [{
                "type": "ack",
                "message": f"unknown_demo_event:{event}",
                "server_time": datetime.now(timezone.utc).isoformat(),
            }]

        await _broadcast_caregiver_messages(app, messages)
        await _broadcast_edge_messages(app, messages)
        return {"messages": messages}

    @app.post("/demo/caregiver/alerts/{alert_id}/ack")
    async def acknowledge_caregiver_alert(alert_id: str) -> dict[str, Any]:
        demo: DemoOrchestrator = app.state.demo_orchestrator
        ack = demo.acknowledge_caregiver_alert(alert_id)
        await asyncio.to_thread(
            append_event,
            {
                "event_type": "caregiver_alert_acknowledged",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": alert_id,
            },
        )
        return ack

    @app.websocket("/ws/caregiver")
    async def caregiver_stream(ws: WebSocket) -> None:
        """Caregiver app stream for escalated exhibition alerts."""
        await ws.accept()
        sockets: set[WebSocket] = ws.app.state.caregiver_sockets
        sockets.add(ws)
        try:
            while True:
                raw = await ws.receive_text()
                msg_type, _parsed = _route_message(raw)
                if msg_type == "ping":
                    await ws.send_text(AckMessage(
                        message="pong",
                        server_time=datetime.now(timezone.utc),
                    ).model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            sockets.discard(ws)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        """Receive frames + audio + status, emit nudges back."""
        await ws.accept()
        device = ws.query_params.get("device_id", "unknown")
        surface = ws.query_params.get("surface", "desktop")
        logger.info("WS connected | device=%s | surface=%s", device, surface)
        ws.app.state.edge_sockets.add(ws)

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
                        result = await session.handle_audio(parsed)
                        if result is not None:
                            transcript, command = result
                            await ws.send_text(VoiceCommandMessage(
                                transcript=transcript,
                                tool=command.tool.name if command.tool else None,
                                raw=command.raw_text,
                                provider=command.provider,
                            ).model_dump_json())
                            control = _edge_control_from_tool(command.tool)
                            if control is not None:
                                await ws.send_text(control.model_dump_json())
                            reply = _assistant_reply_from_tool(command.tool)
                            if reply is not None:
                                await ws.send_text(reply.model_dump_json())

                elif msg_type == "status":
                    logger.info(
                        "WS status | mic=%s | camera=%s",
                        parsed.mic_active,
                        parsed.camera_active,
                    )

                elif msg_type == "demo_action":
                    demo: DemoOrchestrator = ws.app.state.demo_orchestrator
                    messages = _messages_for_demo_action(demo, parsed)
                    await _broadcast_caregiver_messages(ws.app, messages)
                    await _broadcast_edge_messages(ws.app, messages)

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
            ws.app.state.edge_sockets.discard(ws)
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
        if kind == "demo_action":
            return "demo_action", DemoActionMessage.model_validate(payload)
        if kind == "ping":
            return "ping", payload
    except ValidationError as exc:
        logger.warning("Invalid %s message: %s", kind, exc)
        return "unknown", None

    return "unknown", payload


def _read_demo_html(relative_path: str) -> str:
    path = _HOME_SERVER_ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "<!doctype html><title>MCI Demo</title><h1>Demo page unavailable</h1>"


def _messages_for_demo_action(
    demo: DemoOrchestrator,
    action: DemoActionMessage,
) -> list[dict[str, Any]]:
    """Apply edge UI feedback only to the scenario that emitted the nudge."""
    if action.scenario == "stove_on":
        return demo.record_stove_ignored(reason=action.action)
    return []


def _edge_control_from_tool(tool) -> EdgeControlMessage | None:
    """Map voice-command tools to browser media controls owned by the edge."""
    if tool is None:
        return None
    state = str((tool.arguments or {}).get("state", "")).strip().lower()
    if state not in {"on", "off"}:
        return None
    if tool.name == "toggleMic":
        return EdgeControlMessage(target="mic", action=state)
    if tool.name == "toggleCamera":
        return EdgeControlMessage(target="camera", action=state)
    return None


def _assistant_reply_from_tool(tool) -> AssistantReplyMessage | None:
    if tool is None or tool.name != "assistantReply":
        return None
    sentence = str((tool.arguments or {}).get("sentence", "")).strip()
    if not sentence:
        return None
    return AssistantReplyMessage(sentence=sentence)


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


async def _broadcast_caregiver_messages(app: FastAPI, messages: list[dict[str, Any]]) -> None:
    """Send caregiver_alert messages to every connected caregiver app."""
    sockets: set[WebSocket] = getattr(app.state, "caregiver_sockets", set())
    if not sockets:
        return

    caregiver_messages = [m for m in messages if m.get("type") == "caregiver_alert"]
    if not caregiver_messages:
        return

    stale: list[WebSocket] = []
    for ws in list(sockets):
        for message in caregiver_messages:
            try:
                await ws.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                stale.append(ws)
                break
    for ws in stale:
        sockets.discard(ws)


async def _broadcast_edge_messages(
    app: FastAPI,
    messages: list[dict[str, Any]],
    *,
    exclude: WebSocket | None = None,
) -> None:
    """Send demo nudges/replies/controls to connected edge clients."""
    sockets: set[WebSocket] = getattr(app.state, "edge_sockets", set())
    if not sockets:
        return

    edge_messages = [
        m for m in messages
        if m.get("type") in {"nudge", "assistant_reply", "edge_control"}
    ]
    if not edge_messages:
        return

    stale: list[WebSocket] = []
    for ws in list(sockets):
        if ws is exclude:
            continue
        for message in edge_messages:
            try:
                await ws.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                stale.append(ws)
                break
    for ws in stale:
        sockets.discard(ws)


app = create_app()
