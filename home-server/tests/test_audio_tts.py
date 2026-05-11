"""PiperSynthesizer tests with a fake PiperVoice."""
from __future__ import annotations

import io
import wave

import pytest

from audio.tts import PiperSynthesizer


class _FakeVoice:
    """Stand-in for piper.PiperVoice. synthesize() writes a fixed PCM
    payload to the given wave_write object."""

    def __init__(self, *, sample_rate: int = 22050, payload: bytes = b"\x00\x01" * 100) -> None:
        self._sample_rate = sample_rate
        self._payload = payload
        self.calls: list[str] = []

    def synthesize(self, text: str, wav_writer):
        self.calls.append(text)
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(self._sample_rate)
        wav_writer.writeframes(self._payload)


def _read_wav(wav_bytes: bytes):
    """Decode the produced WAV and return (channels, sampwidth, framerate, frames)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.readframes(wav.getnframes()),
        )


def test_synthesize_returns_valid_wav_bytes() -> None:
    voice = _FakeVoice(payload=b"\x00\x01\x02\x03" * 50)
    tts = PiperSynthesizer(voice=voice)

    wav_bytes = tts.synthesize("The kettle has been boiling for four minutes.")
    channels, sampwidth, rate, frames = _read_wav(wav_bytes)
    assert channels == 1
    assert sampwidth == 2
    assert rate == 22050
    assert len(frames) == 200


def test_synthesize_passes_text_to_voice() -> None:
    voice = _FakeVoice()
    tts = PiperSynthesizer(voice=voice)
    tts.synthesize("Hello Shanta.")
    assert voice.calls == ["Hello Shanta."]


def test_empty_text_returns_silent_wav() -> None:
    voice = _FakeVoice()
    tts = PiperSynthesizer(voice=voice)
    wav_bytes = tts.synthesize("")
    # No call to the voice — we short-circuit on empty input
    assert voice.calls == []
    # Still a valid WAV (44-byte header, zero frames)
    channels, sampwidth, rate, frames = _read_wav(wav_bytes)
    assert frames == b""


def test_whitespace_only_returns_silent_wav() -> None:
    voice = _FakeVoice()
    tts = PiperSynthesizer(voice=voice)
    wav_bytes = tts.synthesize("   \n  ")
    assert voice.calls == []
    _, _, _, frames = _read_wav(wav_bytes)
    assert frames == b""
