"""GeminiClient harness tests with a mocked google-genai client.

Same shape as GrokClient tests so the two providers prove equivalent
behaviour from the same suite.

We monkeypatch `google.genai.types` so the SDK does not need to be
installed for unit tests. Live integration with the real SDK lives in
`tests/test_gemini_live.py` (skipped without GOOGLE_API_KEY).
"""
from __future__ import annotations

import sys
import types as _types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fake_google_genai(monkeypatch):
    """Inject a fake `google.genai.types` so imports inside GeminiClient succeed."""
    fake_types = _types.SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda data, mime_type: ("part", mime_type, data)),
        GenerateContentConfig=lambda **kwargs: kwargs,
        Tool=lambda function_declarations: ("tool", function_declarations),
        FunctionDeclaration=lambda name, description, parameters: (
            "fn", name, description, parameters
        ),
    )
    fake_genai = _types.ModuleType("google.genai")
    fake_genai.types = fake_types  # type: ignore[attr-defined]
    fake_genai.Client = lambda api_key: object()  # type: ignore[attr-defined]
    fake_google = _types.ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    yield


from llm.gemini_client import GeminiClient  # noqa: E402  (after fixture)


def _mock_response(text: str = "", function_call=None) -> SimpleNamespace:
    parts = []
    if function_call is not None:
        parts.append(SimpleNamespace(function_call=function_call))
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(text=text, candidates=[candidate])


def _mock_client(response) -> MagicMock:
    client = MagicMock()
    client.models = MagicMock()
    client.models.generate_content = MagicMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_no_nudge_response() -> None:
    response = _mock_response(text="NO_NUDGE")
    client = GeminiClient(api_key="x", client=_mock_client(response))
    result = await client.generate_nudge(
        image_b64="aGVsbG8=",
        context_summary="[Recent Context]",
    )
    assert result.should_nudge is False
    assert result.provider == "gemini"


@pytest.mark.asyncio
async def test_valid_nudge_parses_sentence() -> None:
    response = _mock_response(text="The kettle has been boiling for four minutes.")
    client = GeminiClient(api_key="x", client=_mock_client(response))
    result = await client.generate_nudge(
        image_b64="aGVsbG8=",
        context_summary="[Recent Context]",
    )
    assert result.should_nudge is True
    assert result.sentence == "The kettle has been boiling for four minutes."


@pytest.mark.asyncio
async def test_voice_command_extracts_function_call() -> None:
    fc = SimpleNamespace(name="dismissTemporarily", args={})
    response = _mock_response(text="", function_call=fc)
    client = GeminiClient(api_key="x", client=_mock_client(response))
    result = await client.handle_voice_command(
        transcript="remind me later",
        context_summary="[Recent Context]",
    )
    assert result.tool is not None
    assert result.tool.name == "dismissTemporarily"


@pytest.mark.asyncio
async def test_voice_command_no_function_call_returns_none() -> None:
    response = _mock_response(text="")
    client = GeminiClient(api_key="x", client=_mock_client(response))
    result = await client.handle_voice_command(
        transcript="hmm",
        context_summary="[Recent Context]",
    )
    assert result.tool is None


def test_constructor_requires_api_key_or_client() -> None:
    with pytest.raises(ValueError):
        GeminiClient(api_key="")
