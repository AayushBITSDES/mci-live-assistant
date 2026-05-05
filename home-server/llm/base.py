"""Abstract LLMClient interface + result dataclasses.

Both GrokClient and GeminiClient implement this interface. Swapping
providers is then a single config change in Streamlit Settings — no
calling code knows or cares which one is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.validator import ValidationResult


# Special sentinel returned when the LLM decides no nudge is warranted.
NO_NUDGE = "NO_NUDGE"


@dataclass
class NudgeResult:
    """Outcome of a single nudge-generation call."""
    sentence: Optional[str]                 # None when NO_NUDGE
    should_nudge: bool                      # True iff sentence is set
    validation: Optional[ValidationResult]  # None when should_nudge is False
    raw_text: str                           # Original LLM output for logging
    provider: str                           # "grok" | "gemini"


@dataclass
class ToolCall:
    """One tool invocation produced by the LLM."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandResult:
    """Outcome of a voice-command interpretation call."""
    tool: Optional[ToolCall]
    raw_text: str
    provider: str


class LLMClient(ABC):
    """All providers must implement this interface."""

    provider_name: str = "unknown"

    @abstractmethod
    async def generate_nudge(
        self,
        *,
        image_b64: str,
        context_summary: str,
        persona: str = "shanta",
    ) -> NudgeResult:
        """Decide whether to nudge given the current frame + recent context."""

    @abstractmethod
    async def handle_voice_command(
        self,
        *,
        transcript: str,
        context_summary: str,
    ) -> CommandResult:
        """Map a transcribed voice command to one of the canonical tools."""
