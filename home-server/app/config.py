"""Centralised configuration loaded from .env with sane defaults.

Single source of truth for all paths, thresholds, and provider selection.
Tests should override by instantiating Settings(...) directly rather than
mutating the global instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
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

    # --- Models ---
    whisper_model: str = "medium"
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

    # --- Paths (resolved relative to home-server/) ---
    storage_root: Path = Field(default=Path("./storage"))
    events_dir: Path = Field(default=Path("./storage/events"))
    faces_dir: Path = Field(default=Path("./storage/faces"))
    feedback_dir: Path = Field(default=Path("./storage/feedback"))
    nudges_dir: Path = Field(default=Path("./storage/nudges"))
    onboarding_file: Path = Field(default=Path("./storage/onboarding.json"))

    def ensure_dirs(self) -> None:
        """Create all storage directories if they do not exist."""
        for path in (
            self.storage_root,
            self.events_dir,
            self.faces_dir,
            self.feedback_dir,
            self.nudges_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Global instance — import this everywhere; tests pass their own.
settings = Settings()
