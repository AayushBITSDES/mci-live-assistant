"""Tests for dashboard.persistence (onboarding + runtime settings)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.persistence import (
    EmergencyContact,
    OnboardingProfile,
    RuntimeSettings,
    load_onboarding,
    load_runtime_settings,
    save_onboarding,
    save_runtime_settings,
)


# --- OnboardingProfile ---------------------------------------------------

def test_load_onboarding_returns_defaults_when_missing(tmp_path: Path) -> None:
    profile = load_onboarding(tmp_path / "missing.json")
    assert profile == OnboardingProfile()


def test_round_trip_onboarding(tmp_path: Path) -> None:
    path = tmp_path / "onboarding.json"
    original = OnboardingProfile(
        user_name="Shanta",
        age=67,
        mci_status="mild",
        weight_kg=58.0,
        height_cm=160.0,
        emergency_contacts=[
            EmergencyContact(name="Anjali", relationship="daughter", phone="+91-9999"),
        ],
        notes="Lives alone; sister visits Thursdays.",
    )
    save_onboarding(original, path)
    loaded = load_onboarding(path)
    assert loaded == original


def test_corrupt_onboarding_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "onboarding.json"
    path.write_text("not json{")
    assert load_onboarding(path) == OnboardingProfile()


def test_onboarding_handles_legacy_int_strings(tmp_path: Path) -> None:
    """The Streamlit form may submit "67" instead of 67; coerce gracefully."""
    path = tmp_path / "onboarding.json"
    path.write_text(json.dumps({"user_name": "Shanta", "age": "67", "weight_kg": "58.0"}))
    profile = load_onboarding(path)
    assert profile.age == 67
    assert profile.weight_kg == 58.0


def test_onboarding_invalid_age_becomes_none(tmp_path: Path) -> None:
    path = tmp_path / "onboarding.json"
    path.write_text(json.dumps({"age": "not-a-number"}))
    assert load_onboarding(path).age is None


def test_emergency_contacts_skip_malformed_entries(tmp_path: Path) -> None:
    path = tmp_path / "onboarding.json"
    path.write_text(json.dumps({
        "emergency_contacts": [
            {"name": "Anjali", "relationship": "daughter", "phone": "+91"},
            "not-a-dict",                # ignored
            {"name": "Ravi"},            # accepted with empty fields
        ]
    }))
    profile = load_onboarding(path)
    assert len(profile.emergency_contacts) == 2


# --- RuntimeSettings -----------------------------------------------------

def test_default_runtime_settings() -> None:
    rs = RuntimeSettings()
    assert rs.active_llm_provider == "grok"
    assert rs.yolo_confidence_threshold == 0.5
    assert rs.face_similarity_threshold == 0.45
    assert rs.target_fps == 5
    assert rs.fresh_start is False


def test_round_trip_runtime_settings(tmp_path: Path) -> None:
    path = tmp_path / "runtime_settings.json"
    original = RuntimeSettings(
        active_llm_provider="gemini",
        yolo_confidence_threshold=0.6,
        face_similarity_threshold=0.5,
        target_fps=10,
        fresh_start=True,
    )
    save_runtime_settings(original, path)
    loaded = load_runtime_settings(path)
    assert loaded == original


def test_runtime_settings_provider_lowercased(tmp_path: Path) -> None:
    path = tmp_path / "runtime_settings.json"
    path.write_text(json.dumps({"active_llm_provider": "GROK"}))
    assert load_runtime_settings(path).active_llm_provider == "grok"


def test_runtime_settings_corrupt_file_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "runtime_settings.json"
    path.write_text("not json")
    assert load_runtime_settings(path) == RuntimeSettings()
