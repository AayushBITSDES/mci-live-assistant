"""FastAPI entrypoint: WebSocket stream + health check.

Phase 1 = echo only. Real processing lands in subsequent phases.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.config import settings
from app.models import AckMessage
from context.manager import ContextManager
from context.replay import replay_recent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


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
    logger.info("Home server ready | provider=%s | fps=%d", settings.active_llm_provider, settings.target_fps)
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
        """Phase 1 echo endpoint.

        Phase 5 will multiplex video + audio frames over this same connection;
        for now we just acknowledge each text message so edge clients can
        verify connectivity end-to-end.
        """
        await ws.accept()
        device = ws.query_params.get("device_id", "unknown")
        logger.info("WS connected | device=%s", device)
        try:
            while True:
                raw = await ws.receive_text()
                ack = AckMessage(message=f"echo:{raw[:60]}", server_time=datetime.now(timezone.utc))
                await ws.send_text(ack.model_dump_json())
        except WebSocketDisconnect:
            logger.info("WS disconnected | device=%s", device)

    return app


app = create_app()
