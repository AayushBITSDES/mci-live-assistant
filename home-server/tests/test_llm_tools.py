"""Unit tests for tool schemas. No network."""
from __future__ import annotations

from llm.tools import SNOOZE_DURATION_SECONDS, TOOL_DEFINITIONS, to_openai_format, tool_names


def test_six_tools_defined() -> None:
    expected = {"markDone", "dismissTemporarily", "flagWrong", "closeForever", "toggleMic", "toggleCamera"}
    assert set(tool_names()) == expected


def test_snooze_duration_is_5_minutes() -> None:
    assert SNOOZE_DURATION_SECONDS == 300


def test_openai_format_wraps_each_tool() -> None:
    formatted = to_openai_format()
    assert len(formatted) == len(TOOL_DEFINITIONS)
    for entry in formatted:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn


def test_required_fields_present() -> None:
    by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
    assert by_name["markDone"]["parameters"]["required"] == ["task"]
    assert by_name["closeForever"]["parameters"]["required"] == ["category"]
    assert by_name["toggleMic"]["parameters"]["required"] == ["state"]


def test_toggle_state_enum_values() -> None:
    by_name = {t["name"]: t for t in TOOL_DEFINITIONS}
    mic_state = by_name["toggleMic"]["parameters"]["properties"]["state"]
    assert mic_state["enum"] == ["on", "off"]
