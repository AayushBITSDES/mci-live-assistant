"""GrokClient harness tests with a mocked AsyncOpenAI client.

Validates our wrapper layer: prompt assembly, response parsing, tool
extraction. Does NOT hit the real xAI API — that would need network +
keys + cost. The wrapper is what we actually wrote, so it is what we
test most heavily.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.grok_client import GrokClient


def _mock_chat_response(content: str, tool_calls=None) -> SimpleNamespace:
    """Build a minimal stand-in for openai.types.ChatCompletion."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _mock_client(response) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_no_nudge_response_returns_should_nudge_false() -> None:
    response = _mock_chat_response("NO_NUDGE")
    client = GrokClient(api_key="x", client=_mock_client(response))
    result = await client.generate_nudge(
        image_b64="aGVsbG8=",
        context_summary="[Recent Context]",
    )
    assert result.should_nudge is False
    assert result.sentence is None
    assert result.provider == "grok"


@pytest.mark.asyncio
async def test_valid_nudge_response_parses_sentence_and_validates() -> None:
    response = _mock_chat_response("The kettle has been boiling for four minutes.")
    client = GrokClient(api_key="x", client=_mock_client(response))
    result = await client.generate_nudge(
        image_b64="aGVsbG8=",
        context_summary="[Recent Context]",
    )
    assert result.should_nudge is True
    assert result.sentence == "The kettle has been boiling for four minutes."
    assert result.validation is not None
    assert result.validation.valid is True


@pytest.mark.asyncio
async def test_voice_command_extracts_tool_call() -> None:
    fake_tool = SimpleNamespace(
        function=SimpleNamespace(name="markDone", arguments=json.dumps({"task": "medication"})),
    )
    response = _mock_chat_response("", tool_calls=[fake_tool])
    client = GrokClient(api_key="x", client=_mock_client(response))
    result = await client.handle_voice_command(
        transcript="I already took my pills",
        context_summary="[Recent Context]",
    )
    assert result.tool is not None
    assert result.tool.name == "markDone"
    assert result.tool.arguments == {"task": "medication"}


@pytest.mark.asyncio
async def test_voice_command_no_tool_returns_none() -> None:
    response = _mock_chat_response("")
    client = GrokClient(api_key="x", client=_mock_client(response))
    result = await client.handle_voice_command(
        transcript="hmmmm",
        context_summary="[Recent Context]",
    )
    assert result.tool is None


@pytest.mark.asyncio
async def test_malformed_tool_arguments_yields_empty_dict() -> None:
    fake_tool = SimpleNamespace(
        function=SimpleNamespace(name="markDone", arguments="not-json"),
    )
    response = _mock_chat_response("", tool_calls=[fake_tool])
    client = GrokClient(api_key="x", client=_mock_client(response))
    result = await client.handle_voice_command(
        transcript="done",
        context_summary="[Recent Context]",
    )
    assert result.tool is not None
    assert result.tool.arguments == {}


def test_constructor_requires_api_key_or_client() -> None:
    with pytest.raises(ValueError):
        GrokClient(api_key="")
