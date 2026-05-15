"""Grok-4.3 client via xAI's OpenAI-compatible endpoint.

Requires XAI_API_KEY. The OpenAI Python SDK is reused with a custom
base_url — same call shape as GPT-4o, so the codebase only needs to
know about one client pattern.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from llm.base import NO_NUDGE, CommandResult, LLMClient, NudgeResult, ToolCall
from llm.prompts import (
    SHANTA_SYSTEM_PROMPT,
    VOICE_COMMAND_SYSTEM_PROMPT,
    build_nudge_prompt,
    build_voice_command_prompt,
)
from llm.tools import to_openai_format
from llm.validator import validate_nudge

logger = logging.getLogger(__name__)

GROK_MODEL = "grok-4.3"
GROK_BASE_URL = "https://api.x.ai/v1"


class GrokClient(LLMClient):
    provider_name = "grok"

    def __init__(self, api_key: str, *, model: str = GROK_MODEL, client: Optional[AsyncOpenAI] = None) -> None:
        if not api_key and client is None:
            raise ValueError("GrokClient requires either api_key or a pre-built client")
        self._model = model
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=GROK_BASE_URL)

    async def generate_nudge(
        self,
        *,
        image_b64: str,
        context_summary: str,
        persona: str = "shanta",
    ) -> NudgeResult:
        user_prompt = build_nudge_prompt(context_summary, persona)
        # Grok-4.3 is a reasoning model; "low" effort skips the heavy
        # think-step. Sent via extra_body so older models that ignore the
        # param still work.
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SHANTA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_tokens=80,
            temperature=0.4,
            extra_body={"reasoning_effort": "none"},
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_nudge_response(raw)

    async def handle_voice_command(
        self,
        *,
        transcript: str,
        context_summary: str,
    ) -> CommandResult:
        user_prompt = build_voice_command_prompt(transcript, context_summary)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": VOICE_COMMAND_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=to_openai_format(),
            tool_choice="auto",
            max_tokens=120,
            temperature=0.0,
            extra_body={"reasoning_effort": "none"},
        )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        tool = self._extract_tool_call(choice.message)
        if tool is None and raw:
            tool = self._fallback_from_text(raw, transcript)
        return CommandResult(tool=tool, raw_text=raw, provider=self.provider_name)

    # --- Parsing helpers --------------------------------------------------

    def _parse_nudge_response(self, raw: str) -> NudgeResult:
        if not raw or raw.upper().startswith(NO_NUDGE):
            return NudgeResult(
                sentence=None,
                should_nudge=False,
                validation=None,
                raw_text=raw,
                provider=self.provider_name,
            )
        validation = validate_nudge(raw)
        if validation.warnings:
            logger.warning("Grok nudge validation warnings: %s | text=%r", validation.warnings, raw)
        return NudgeResult(
            sentence=raw,
            should_nudge=True,
            validation=validation,
            raw_text=raw,
            provider=self.provider_name,
        )

    @staticmethod
    def _extract_tool_call(message: Any) -> Optional[ToolCall]:
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return None
        first = tool_calls[0]
        try:
            args = json.loads(first.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return ToolCall(name=first.function.name, arguments=args)

    # The exact set of tools the server knows how to dispatch. Anything
    # else extracted from a fallback text parse is a hallucination — we
    # refuse to forge a call rather than dispatch a bogus tool. Source
    # of truth: ``app.pipeline.Session._dispatch_tool``.
    _KNOWN_TOOLS: frozenset[str] = frozenset({
        "markDone",
        "dismissTemporarily",
        "flagWrong",
        "closeForever",
        "toggleMic",
        "toggleCamera",
        "toggleAudio",
        "assistantReply",
    })

    @classmethod
    def _canonical_tool_name(cls, candidate: str) -> Optional[str]:
        """Return the canonical tool name for ``candidate`` or None."""
        if not candidate:
            return None
        if candidate in cls._KNOWN_TOOLS:
            return candidate
        lowered = candidate.lower()
        for known in cls._KNOWN_TOOLS:
            if known.lower() == lowered:
                return known
        return None

    @classmethod
    def _fallback_from_text(cls, raw: str, transcript: str) -> Optional[ToolCall]:
        import re
        # XML-style
        if "<tool_call" in raw or "<function_call" in raw:
            m = re.search(r'<(?:tool_call|function_call)[^>]*name=["\']([^"\']+)["\']', raw)
            if not m:
                m = re.search(r'name=["\']([^"\']+)["\'][^>]*>', raw)
            if m:
                name = cls._canonical_tool_name(m.group(1))
                if name is not None:
                    args = {}
                    for p in re.finditer(r'<parameter name=["\']([^"\']+)["\']>(.*?)</parameter>', raw, re.DOTALL):
                        args[p.group(1)] = p.group(2).strip()
                    return ToolCall(name=name, arguments=args)

        # Plain-text style: "call assistantReply with sentence is You were..."
        # Only honour the match if the captured word is a real tool —
        # otherwise we'd happily forge calls to "to" / "you" / "for".
        m = re.search(r'(?:call|tool)\s+(\w+)', raw, re.IGNORECASE)
        if m:
            name = cls._canonical_tool_name(m.group(1))
            if name is not None:
                args = {}
                for p in re.finditer(r'(\w+)\s+is\s+([^,\n]+)', raw):
                    args[p.group(1)] = p.group(2).strip()
                return ToolCall(name=name, arguments=args)

        text = (raw or "").lower() + " " + (transcript or "").lower()
        # Longer / more specific phrases first so e.g. "no more" maps to
        # closeForever, not a standalone-word ``no`` match for flagWrong.
        if any(k in text for k in ("never", "stop reminding", "don't remind", "no more")):
            category = "stove_reminder" if "stove" in text else "medicine_reminder"
            return ToolCall(name="closeForever", arguments={"category": category})
        if any(k in text for k in ("remind me later", "not now", "in a bit")):
            return ToolCall(name="dismissTemporarily", arguments={})
        if any(k in text for k in ("wrong", "not right", "incorrect")) or re.search(
            r"\bno\b", text
        ):
            reason = transcript or raw
            return ToolCall(name="flagWrong", arguments={"reason": reason})

        # Pure conversational text from the LLM: surface it as a spoken
        # assistant reply rather than dropping it on the floor. This is
        # what makes the assistant feel like it can actually chat.
        cleaned = (raw or "").strip()
        if cleaned and len(cleaned) <= 280:
            return ToolCall(name="assistantReply", arguments={"sentence": cleaned})
        return None
