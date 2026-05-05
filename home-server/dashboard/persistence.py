"""File-based persistence for onboarding + runtime settings.

Both Streamlit pages and the FastAPI server read from these files; they
are the source of truth for *user-editable* state. Compile-time defaults
live in `app/config.py` (and stay there); these files override at runtime.

We use plain JSON: human-readable for debugging, safe to load, trivial
to diff in git if the user ever wants to commit a fixture.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- Onboarding ----------------------------------------------------------

@dataclass
class EmergencyContact:
    name: str
    relationship: str
    phone: str


@dataclass
class OnboardingProfile:
    """Persona profile collected via the Onboarding page.

    Matches Design Decision #9 (emergency contacts, weight, height, age)
    plus a free-form persona slug used by the LLM prompt builder.
    """
    persona: str = "shanta"
    user_name: str = "Shanta"
    age: int | None = None
    mci_status: str = "mild"            # none | mild | moderate
    weight_kg: float | None = None
    height_cm: float | None = None
    emergency_contacts: list[EmergencyContact] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OnboardingProfile":
        contacts_raw = data.get("emergency_contacts", []) or []
        contacts = [
            EmergencyContact(
                name=c.get("name", ""),
                relationship=c.get("relationship", ""),
                phone=c.get("phone", ""),
            )
            for c in contacts_raw
            if isinstance(c, dict)
        ]
        return cls(
            persona=data.get("persona", "shanta"),
            user_name=data.get("user_name", "Shanta"),
            age=_coerce_int(data.get("age")),
            mci_status=data.get("mci_status", "mild"),
            weight_kg=_coerce_float(data.get("weight_kg")),
            height_cm=_coerce_float(data.get("height_cm")),
            emergency_contacts=contacts,
            notes=data.get("notes", ""),
        )


def load_onboarding(path: Path) -> OnboardingProfile:
    """Load the onboarding profile, falling back to defaults if missing or corrupt."""
    if not path.exists():
        return OnboardingProfile()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load onboarding (%s); using defaults", exc)
        return OnboardingProfile()
    if not isinstance(data, dict):
        return OnboardingProfile()
    return OnboardingProfile.from_dict(data)


def save_onboarding(profile: OnboardingProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


# --- Runtime settings ----------------------------------------------------

@dataclass
class RuntimeSettings:
    """User-editable knobs surfaced on the Settings page.

    These override the .env-loaded compile-time defaults. The server
    reads this file on startup; live edits require restart (we surface
    that warning in the Settings UI).
    """
    active_llm_provider: str = "grok"          # grok | gemini
    yolo_confidence_threshold: float = 0.5
    face_similarity_threshold: float = 0.45
    target_fps: int = 5
    fresh_start: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeSettings":
        return cls(
            active_llm_provider=str(data.get("active_llm_provider", "grok")).lower(),
            yolo_confidence_threshold=_coerce_float(data.get("yolo_confidence_threshold")) or 0.5,
            face_similarity_threshold=_coerce_float(data.get("face_similarity_threshold")) or 0.45,
            target_fps=_coerce_int(data.get("target_fps")) or 5,
            fresh_start=bool(data.get("fresh_start", False)),
        )


def load_runtime_settings(path: Path) -> RuntimeSettings:
    if not path.exists():
        return RuntimeSettings()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load runtime settings (%s); using defaults", exc)
        return RuntimeSettings()
    if not isinstance(data, dict):
        return RuntimeSettings()
    return RuntimeSettings.from_dict(data)


def save_runtime_settings(settings: RuntimeSettings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)


# --- Coercion helpers ---------------------------------------------------

def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
