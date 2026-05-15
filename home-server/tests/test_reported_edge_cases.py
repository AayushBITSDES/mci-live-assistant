"""Regression checks for review fixes (Grok fallback, stove timers, CORS, names)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from starlette.middleware.cors import CORSMiddleware

from app.main import _NAME_PREFIXES, create_app
from demo.orchestrator import DemoOrchestrator
from llm.grok_client import GrokClient


def test_grok_fallback_casual_phrase_with_know_is_not_flag_wrong() -> None:
    """*Know* must not satisfy a bare ``no`` substring match."""
    tool = GrokClient._fallback_from_text("I know what you mean", "")
    assert tool is not None
    assert tool.name == "assistantReply"
    assert "know" in tool.arguments.get("sentence", "").lower()


def test_grok_fallback_no_more_still_maps_to_close_forever() -> None:
    """``no more`` must stay ``closeForever`` after reordering checks."""
    tool = GrokClient._fallback_from_text("no more stove reminders", "")
    assert tool is not None
    assert tool.name == "closeForever"
    assert tool.arguments.get("category") == "stove_reminder"


def test_grok_fallback_standalone_no_still_flag_wrong() -> None:
    tool = GrokClient._fallback_from_text("no, that reminder is wrong", "")
    assert tool is not None
    assert tool.name == "flagWrong"


def test_stove_misconfigured_escalation_does_not_fire_immediately() -> None:
    demo = DemoOrchestrator(
        stove_first_reminder_seconds=30,
        stove_escalation_seconds=20,
    )
    start = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    demo.record_stove_interaction(now=start)
    assert demo.check_stove_timers(now=start + timedelta(seconds=30))
    out = demo.check_stove_timers(now=start + timedelta(seconds=31))
    assert not any(m.get("type") == "caregiver_alert" for m in out)
    assert demo.state.stove_state == "reminded"


def test_cors_allow_origin_regex_rejects_public_documentation_ip() -> None:
    app = create_app()
    mw = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    pat = mw.kwargs["allow_origin_regex"]
    r = re.compile(pat)
    assert r.match("http://203.0.113.1:5173") is None


def test_cors_allow_origin_regex_accepts_rfc1918_lan() -> None:
    app = create_app()
    mw = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    pat = mw.kwargs["allow_origin_regex"]
    r = re.compile(pat)
    assert r.match("http://192.168.1.42:5173") is not None
    assert r.match("http://10.0.0.1:5173") is not None


def test_name_prefix_tuple_is_strictly_longest_first() -> None:
    lengths = [len(p) for p in _NAME_PREFIXES]
    assert lengths == sorted(lengths, reverse=True)
