"""Regression-style checks for issues discussed in code review.

These tests document how the code behaves today in edge cases that matter
for demos and production-like configuration. Update expectations when the
underlying behavior is intentionally changed.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from starlette.middleware.cors import CORSMiddleware

from app.main import _NAME_PREFIXES, create_app
from demo.orchestrator import DemoOrchestrator
from llm.grok_client import GrokClient


def test_grok_fallback_substring_no_triggers_flag_wrong_on_casual_phrase() -> None:
    """Plain-text fallback uses ``"no" in text`` — substrings like *know* match.

    When the model returns conversational content without a structured tool,
    this path can mis-classify chit-chat as ``flagWrong`` and downstream code
    will emit ``nudge_flagged_wrong`` events.
    """
    tool = GrokClient._fallback_from_text("I know what you mean", "")
    assert tool is not None
    assert tool.name == "flagWrong"


def test_stove_escalation_shorter_than_first_reminder_escalates_immediately() -> None:
    """If escalation delay < first reminder delay, the wait window is negative.

    ``now - reminded_at >= (negative timedelta)`` is true for any positive
    elapsed time, so the demo can jump straight to caregiver alert right after
    entering the *reminded* state.
    """
    demo = DemoOrchestrator(
        stove_first_reminder_seconds=30,
        stove_escalation_seconds=20,
    )
    start = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    demo.record_stove_interaction(now=start)
    assert demo.check_stove_timers(now=start + timedelta(seconds=30))
    out = demo.check_stove_timers(now=start + timedelta(seconds=31))
    assert any(m.get("type") == "caregiver_alert" for m in out)


def test_cors_allow_origin_regex_accepts_public_ipv4_on_port_5173() -> None:
    """The ``[0-9.]+`` segment matches any IPv4-shaped host, not only RFC1918.

    With ``allow_credentials=True``, this is the effective browser trust rule
    for credentialed dev origins when the server is reachable from that host.
    """
    app = create_app()
    mw = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    pat = mw.kwargs["allow_origin_regex"]
    r = re.compile(pat)
    assert r.match("http://203.0.113.1:5173") is not None


def test_name_prefix_tuple_order_diverges_from_longest_first_comment() -> None:
    """``_extract_name_from_transcript`` assumes longest-first prefix stripping.

    The shorter ``hi my name is `` appears before the longer
    ``hello my name is ``; today no prefix is a leading substring of another,
    but the ordering does not satisfy a strict longest-first contract.
    """
    lengths = [len(p) for p in _NAME_PREFIXES]
    assert lengths != sorted(lengths, reverse=True)
    hi = _NAME_PREFIXES.index("hi my name is ")
    hello = _NAME_PREFIXES.index("hello my name is ")
    assert hi < hello
    assert len(_NAME_PREFIXES[hi]) < len(_NAME_PREFIXES[hello])
