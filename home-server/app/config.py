"""Centralised configuration loaded from .env with sane defaults.

Single source of truth for all paths, thresholds, and provider selection.
Tests should override by instantiating Settings(...) directly rather than
mutating the global instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve `.env` next to the `home-server/` package root so keys load
# correctly whether you start uvicorn from `home-server/` or the repo root.
_HOME_SERVER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_HOME_SERVER_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider selection ---
    active_llm_provider: Literal["grok", "gemini"] = "grok"
    xai_api_key: str = ""
    google_api_key: str = ""

    # --- Exhibition speech/provider selection ---
    # Sarvam is the exhibition default; browser means "send text only and
    # let the edge use speechSynthesis".
    asr_provider: Literal["sarvam", "whisper", "off"] = "sarvam"
    tts_provider: Literal["sarvam", "piper", "browser", "off"] = "sarvam"
    sarvam_api_key: str = ""
    sarvam_stt_model: str = "saaras:v3"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_language_code: str = "en-IN"

    # --- Vision provider selection ---
    vision_provider: Literal["opencv", "yolo"] = "opencv"

    # --- Models ---
    # Exhibition default favors latency over perfect transcription. Use
    # WHISPER_MODEL=medium/large-v3 on a GPU box when accuracy matters more.
    whisper_model: str = "tiny"
    piper_voice: str = "en_US-amy-medium"
    yolo_model: str = "yolov10s.pt"
    insightface_model: str = "buffalo_l"

    # --- Performance ---
    target_fps: int = 5
    frame_width: int = 640
    frame_height: int = 480
    nudge_auto_dismiss_seconds: int = 10

    # --- Demo behaviour ---
    # If True, skip JSONL replay on startup so each demo session begins
    # with a clean ContextManager. Production stays False (replay restores
    # recent context after a crash). Flip via FRESH_START=1 in .env.
    fresh_start: bool = False
    replay_lookback_hours: float = 2.0
    medicine_reminder_delay_seconds: float = 30.0
    stove_first_reminder_seconds: float = 30.0
    stove_escalation_seconds: float = 60.0

    # --- Paths: defaults under home-server/; .env relative paths anchor there too ---
    storage_root: Path = Field(default=_HOME_SERVER_ROOT / "storage")
    reference_dir: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "reference")
    events_dir: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "events")
    faces_dir: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "faces")
    feedback_dir: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "feedback")
    nudges_dir: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "nudges")
    onboarding_file: Path = Field(default=_HOME_SERVER_ROOT / "storage" / "onboarding.json")

    @model_validator(mode="after")
    def _anchor_relative_paths(self) -> Settings:
        """Paths from .env like ./storage stay under home-server/, not CWD."""

        def anchor(p: Path) -> Path:
            return p if p.is_absolute() else (_HOME_SERVER_ROOT / p).resolve()

        self.storage_root = anchor(self.storage_root)
        self.reference_dir = anchor(self.reference_dir)
        self.events_dir = anchor(self.events_dir)
        self.faces_dir = anchor(self.faces_dir)
        self.feedback_dir = anchor(self.feedback_dir)
        self.nudges_dir = anchor(self.nudges_dir)
        self.onboarding_file = anchor(self.onboarding_file)
        return self

    def ensure_dirs(self) -> None:
        """Create all storage directories if they do not exist."""
        for path in (
            self.storage_root,
            self.reference_dir,
            self.reference_dir / "medicine_bottle",
            self.reference_dir / "stove",
            self.reference_dir / "faces" / "Anya",
            self.events_dir,
            self.faces_dir,
            self.feedback_dir,
            self.nudges_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Global instance — import this everywhere; tests pass their own.
settings = Settings()
