"""Pydantic models for the wire format between edge and home server.

Every WebSocket message validates against one of these schemas.
Adding a new field? Add it here first, the rest of the codebase is type-safe.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Edge -> Server ---

class FrameMessage(BaseModel):
    """A single video frame from an edge device."""
    type: Literal["frame"] = "frame"
    device_id: str
    surface: Literal["mobile", "quest", "desktop"]
    timestamp: datetime
    image_b64: str = Field(..., description="Base64 JPEG, target 480p")
    width: int
    height: int


class AudioChunkMessage(BaseModel):
    """A short audio chunk from the edge mic."""
    type: Literal["audio"] = "audio"
    device_id: str
    timestamp: datetime
    audio_b64: str = Field(..., description="Base64 PCM/Opus chunk")
    sample_rate: int = 16000
    duration_ms: int


class StatusMessage(BaseModel):
    """Edge status updates: kill switch toggles, connection lifecycle."""
    type: Literal["status"] = "status"
    device_id: str
    mic_active: bool
    camera_active: bool


# --- Server -> Edge ---

class NudgePriority(str, Enum):
    SAFETY = "safety"          # immediate; high-confidence local rule
    QUALITY = "quality"        # standard LLM-generated nudge


class NudgeMessage(BaseModel):
    """One nudge sent back to edge for display + audio playback."""
    type: Literal["nudge"] = "nudge"
    nudge_id: str
    sentence: str = Field(..., description="One full sentence, never a fragment")
    priority: NudgePriority
    audio_b64: Optional[str] = Field(
        default=None,
        description="Piper TTS output, base64 PCM/WAV. None if audio disabled.",
    )
    auto_dismiss_seconds: int = 10


class AckMessage(BaseModel):
    """Generic ack/echo for handshake + debugging."""
    type: Literal["ack"] = "ack"
    message: str
    server_time: datetime
