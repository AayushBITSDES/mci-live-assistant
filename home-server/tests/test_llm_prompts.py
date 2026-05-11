"""Unit tests for prompt assembly. Verifies design-decision rules survive."""
from __future__ import annotations

import pytest

from llm.prompts import (
    SHANTA_SYSTEM_PROMPT,
    VOICE_COMMAND_SYSTEM_PROMPT,
    build_nudge_prompt,
    build_voice_command_prompt,
)


def test_system_prompt_states_one_sentence_rule() -> None:
    assert "ONE full sentence" in SHANTA_SYSTEM_PROMPT
    assert "15 words" in SHANTA_SYSTEM_PROMPT


def test_system_prompt_states_silence_default() -> None:
    assert "Silence" in SHANTA_SYSTEM_PROMPT or "silent" in SHANTA_SYSTEM_PROMPT.lower()


def test_system_prompt_specifies_no_nudge_sentinel() -> None:
    assert "NO_NUDGE" in SHANTA_SYSTEM_PROMPT


def test_system_prompt_describes_name_usage_rule() -> None:
    assert "Shanta" in SHANTA_SYSTEM_PROMPT
    # Should mention safety vs quality-of-life distinction
    assert "safety" in SHANTA_SYSTEM_PROMPT.lower()


def test_unsupported_persona_raises() -> None:
    with pytest.raises(ValueError):
        build_nudge_prompt("[Recent Context]", persona="sanjay")


def test_nudge_prompt_includes_context_block() -> None:
    summary = "[Recent Context]\n- Current activity: making_tea (started 4 min ago)"
    prompt = build_nudge_prompt(summary)
    assert summary in prompt
    assert "current frame" in prompt.lower()


def test_voice_prompt_quotes_transcript_verbatim() -> None:
    prompt = build_voice_command_prompt("I already took my pills", "[Recent Context]")
    assert '"I already took my pills"' in prompt


def test_voice_command_system_prompt_lists_tool_mappings() -> None:
    for keyword in ["markDone", "dismissTemporarily", "flagWrong", "closeForever", "toggleMic", "toggleCamera"]:
        assert keyword in VOICE_COMMAND_SYSTEM_PROMPT
