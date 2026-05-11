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
        )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        tool = self._extract_tool_call(choice.message)
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
