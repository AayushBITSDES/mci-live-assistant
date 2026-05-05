"""Factory tests — confirms the Streamlit-facing seam works."""
from __future__ import annotations

import pytest

from llm.factory import build_client


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        build_client("anthropic", xai_api_key="x", google_api_key="x")


def test_grok_requires_xai_key() -> None:
    with pytest.raises(ValueError):
        build_client("grok", xai_api_key="", google_api_key="g")


def test_gemini_requires_google_key() -> None:
    with pytest.raises(ValueError):
        build_client("gemini", xai_api_key="x", google_api_key="")


def test_grok_returns_grok_client() -> None:
    from llm.grok_client import GrokClient
    c = build_client("grok", xai_api_key="x")
    assert isinstance(c, GrokClient)
    assert c.provider_name == "grok"


def test_gemini_returns_gemini_client(monkeypatch) -> None:
    """Gemini constructor imports google.genai lazily; we patch it to avoid the dep."""
    import sys
    import types as _types

    fake_genai = _types.SimpleNamespace(Client=lambda api_key: object())
    fake_google = _types.ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    from llm.gemini_client import GeminiClient
    c = build_client("gemini", google_api_key="g")
    assert isinstance(c, GeminiClient)
    assert c.provider_name == "gemini"
