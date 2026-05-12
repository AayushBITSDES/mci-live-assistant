"""Opt-in live Grok tool-calling contract tests.

Run only when you explicitly want to spend a few real API calls:

    RUN_LIVE_LLM_TESTS=1 XAI_API_KEY=... pytest tests/test_grok_live_voice_contract.py
"""
from __future__ import annotations

import os

import pytest

from llm.grok_client import GrokClient


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1" or not os.getenv("XAI_API_KEY"),
    reason="live Grok tests require RUN_LIVE_LLM_TESTS=1 and XAI_API_KEY",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "expected_tool"),
    [
        ("I already took my vitamin", "markDone"),
        ("Remind me later please", "dismissTemporarily"),
        ("That reminder is wrong", "flagWrong"),
        ("Never remind me about the stove again", "closeForever"),
        ("What was I doing?", "assistantReply"),
    ],
)
async def test_grok_live_voice_commands_pick_expected_tools(
    transcript: str,
    expected_tool: str,
) -> None:
    client = GrokClient(api_key=os.environ["XAI_API_KEY"])

    result = await client.handle_voice_command(
        transcript=transcript,
        context_summary=(
            "[Recent Context]\n"
            "- User: Shanta\n"
            "- Current risk: vitamin reminder pending\n"
            "- Stove risk may be active if the transcript mentions stove\n"
        ),
    )

    assert result.tool is not None
    assert result.tool.name == expected_tool
