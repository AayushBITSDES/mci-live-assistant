"""Gemini 3 Flash client via Google AI Studio (google-genai SDK).

Vision-capable, supports tool calling. Image input as base64 inline data
(decoded to bytes for the SDK). Same NudgeResult/CommandResult shape as
GrokClient — drop-in swap.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Optional

from llm.base import NO_NUDGE, CommandResult, LLMClient, NudgeResult, ToolCall
from llm.prompts import (
    SHANTA_SYSTEM_PROMPT,
    VOICE_COMMAND_SYSTEM_PROMPT,
    build_nudge_prompt,
    build_voice_command_prompt,
)
from llm.tools import TOOL_DEFINITIONS
from llm.validator import validate_nudge

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3-flash"


class GeminiClient(LLMClient):
    provider_name = "gemini"

    def __init__(self, api_key: str, *, model: str = GEMINI_MODEL, client: Optional[Any] = None) -> None:
        if not api_key and client is None:
            raise ValueError("GeminiClient requires either api_key or a pre-built client")
        self._model = model
        if client is not None:
            self._client = client
        else:
            from google import genai  # type: ignore
            self._client = genai.Client(api_key=api_key)

    async def generate_nudge(
        self,
        *,
        image_b64: str,
        context_summary: str,
        persona: str = "shanta",
    ) -> NudgeResult:
        user_prompt = build_nudge_prompt(context_summary, persona)
        image_bytes = base64.b64decode(image_b64)

        # google-genai is sync; run in thread to keep the asyncio caller responsive.
        def _call() -> Any:
            from google.genai import types  # type: ignore
            return self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    user_prompt,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SHANTA_SYSTEM_PROMPT,
                    max_output_tokens=80,
                    temperature=0.4,
                ),
            )

        response = await asyncio.to_thread(_call)
        raw = (getattr(response, "text", "") or "").strip()
        return self._parse_nudge_response(raw)

    async def handle_voice_command(
        self,
        *,
        transcript: str,
        context_summary: str,
    ) -> CommandResult:
        user_prompt = build_voice_command_prompt(transcript, context_summary)

        def _call() -> Any:
            from google.genai import types  # type: ignore
            tool = types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"],
                )
                for t in TOOL_DEFINITIONS
            ])
            return self._client.models.generate_content(
                model=self._model,
                contents=[user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=VOICE_COMMAND_SYSTEM_PROMPT,
                    tools=[tool],
                    max_output_tokens=120,
                    temperature=0.0,
                ),
            )

        response = await asyncio.to_thread(_call)
        raw = (getattr(response, "text", "") or "").strip()
        tool = self._extract_tool_call(response)
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
            logger.warning("Gemini nudge validation warnings: %s | text=%r", validation.warnings, raw)
        return NudgeResult(
            sentence=raw,
            should_nudge=True,
            validation=validation,
            raw_text=raw,
            provider=self.provider_name,
        )

    @staticmethod
    def _extract_tool_call(response: Any) -> Optional[ToolCall]:
        # google-genai surfaces function calls on candidates[0].content.parts
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or [] if content else []
            for part in parts:
                fn = getattr(part, "function_call", None)
                if fn is not None:
                    args = dict(getattr(fn, "args", {}) or {})
                    return ToolCall(name=fn.name, arguments=args)
        return None
