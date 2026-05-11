"""Unit tests for the nudge validator. No network, no API keys."""
from __future__ import annotations

from llm.validator import MAX_WORDS, validate_nudge


def test_valid_short_sentence_passes() -> None:
    r = validate_nudge("The kettle has been boiling for four minutes.")
    assert r.valid
    assert r.warnings == []
    assert r.word_count == 8
    assert r.sentence_count == 1


def test_too_long_flags_warning_but_passes_through() -> None:
    text = "This is " + "very " * 20 + "long sentence."
    r = validate_nudge(text)
    assert not r.valid
    assert any("too_long" in w for w in r.warnings)


def test_too_many_sentences_flags_warning() -> None:
    r = validate_nudge("First. Second.")
    assert not r.valid
    assert any("too_many_sentences" in w for w in r.warnings)


def test_missing_terminal_punctuation_flags() -> None:
    r = validate_nudge("Tea is ready when you are")
    assert not r.valid
    assert "missing_terminal_punctuation" in r.warnings


def test_empty_input_is_invalid() -> None:
    r = validate_nudge("")
    assert not r.valid
    assert "empty_nudge" in r.warnings


def test_exactly_15_words_is_valid() -> None:
    text = " ".join(["w"] * MAX_WORDS) + "."
    r = validate_nudge(text)
    assert r.valid


def test_16_words_is_invalid() -> None:
    text = " ".join(["w"] * (MAX_WORDS + 1)) + "."
    r = validate_nudge(text)
    assert not r.valid
