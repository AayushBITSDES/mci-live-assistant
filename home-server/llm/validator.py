"""Soft validation for LLM nudge output.

Pure functions, no I/O, fully unit-testable. Returns a structured result
so callers can decide whether to pass through, retry, or log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Design Decision: one full sentence, max 15 words.
MAX_WORDS = 15
SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


@dataclass
class ValidationResult:
    valid: bool
    warnings: list[str]
    word_count: int
    sentence_count: int


def validate_nudge(text: str) -> ValidationResult:
    """Check a candidate nudge against the locked output rules.

    Soft validation: returns warnings rather than rejecting. Caller decides
    whether to retry, truncate, or pass through. We pass through by default
    because a slightly-too-long nudge is better than no nudge at all.
    """
    warnings: list[str] = []
    cleaned = text.strip()

    if not cleaned:
        return ValidationResult(valid=False, warnings=["empty_nudge"], word_count=0, sentence_count=0)

    # Word count
    words = cleaned.split()
    if len(words) > MAX_WORDS:
        warnings.append(f"too_long: {len(words)} words (max {MAX_WORDS})")

    # Sentence count: count terminal punctuation. A single trailing period is fine.
    matches = SENTENCE_END.findall(cleaned)
    sentence_count = max(1, len(matches))
    if sentence_count > 1:
        warnings.append(f"too_many_sentences: {sentence_count} (expected 1)")

    # Fragment check: must end in terminal punctuation OR be a clear short statement.
    if not cleaned[-1] in ".!?":
        warnings.append("missing_terminal_punctuation")

    return ValidationResult(
        valid=len(warnings) == 0,
        warnings=warnings,
        word_count=len(words),
        sentence_count=sentence_count,
    )
