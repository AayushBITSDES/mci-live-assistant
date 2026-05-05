"""WhisperTranscriber tests with a fake faster-whisper model."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import pytest

from audio.asr import WhisperTranscriber


class _FakeWhisper:
    """Stand-in for faster_whisper.WhisperModel."""

    def __init__(self, segments: Iterable[SimpleNamespace], info: object | None = None) -> None:
        self._segments = list(segments)
        self._info = info or SimpleNamespace()
        self.calls: list[dict] = []

    def transcribe(self, audio, *, language, beam_size, vad_filter):
        self.calls.append({
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        })
        return iter(self._segments), self._info


def _segment(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def test_single_segment_returns_clean_text() -> None:
    model = _FakeWhisper(segments=[_segment(" I already took my pills.")])
    asr = WhisperTranscriber(model=model)
    assert asr.transcribe(b"fake-audio") == "I already took my pills."


def test_multiple_segments_are_joined() -> None:
    model = _FakeWhisper(segments=[_segment("Remind me later"), _segment("please.")])
    asr = WhisperTranscriber(model=model)
    assert asr.transcribe(b"fake-audio") == "Remind me later please."


def test_empty_segments_returns_empty_string() -> None:
    model = _FakeWhisper(segments=[])
    asr = WhisperTranscriber(model=model)
    assert asr.transcribe(b"fake-audio") == ""


def test_whitespace_only_segments_are_dropped() -> None:
    model = _FakeWhisper(segments=[_segment("  "), _segment(""), _segment(" hello.")])
    asr = WhisperTranscriber(model=model)
    assert asr.transcribe(b"fake-audio") == "hello."


def test_transcribe_uses_configured_language() -> None:
    model = _FakeWhisper(segments=[_segment("hi")])
    asr = WhisperTranscriber(language="en", model=model)
    asr.transcribe(b"fake-audio")
    assert model.calls[-1]["language"] == "en"


def test_transcribe_enables_vad_filter() -> None:
    """VAD (voice activity detection) skips silent regions — important
    for an always-on mic that mostly hears nothing."""
    model = _FakeWhisper(segments=[])
    asr = WhisperTranscriber(model=model)
    asr.transcribe(b"fake-audio")
    assert model.calls[-1]["vad_filter"] is True


def test_transcribe_uses_beam_size_one_for_speed() -> None:
    """beam_size=1 trades a tiny accuracy hit for a big latency win —
    appropriate for short voice commands."""
    model = _FakeWhisper(segments=[])
    asr = WhisperTranscriber(model=model)
    asr.transcribe(b"fake-audio")
    assert model.calls[-1]["beam_size"] == 1
