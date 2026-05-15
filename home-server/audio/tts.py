"""Piper TTS wrapper for offline server-side text-to-speech.

Piper produces 16-bit PCM WAV at the voice's native sample rate
(usually 22050 Hz). The wrapper exposes raw WAV bytes — the caller
encodes/transcodes as needed before sending to the edge.

Voice selection: see Piper's voice catalogue. `en_US-amy-medium` is a
warm female voice that aligns with Design Decision #3 ("warm and natural
tone"). Each voice is a `~50MB .onnx` file plus a small JSON config.
"""
from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Piper voice files (.onnx + .onnx.json) live here unless voices_dir is set.
_HOME_SERVER_ROOT = Path(__file__).resolve().parent.parent


class PiperSynthesizer:
    def __init__(
        self,
        voice_name: str = "en_US-amy-medium",
        *,
        voices_dir: Optional[str] = None,
        voice: Optional[Any] = None,
    ) -> None:
        """
        Args:
            voice_name: Piper voice identifier (e.g. "en_US-amy-medium").
            voices_dir: Where the .onnx + config files live; if None,
                Piper falls back to its default search path.
            voice: Pre-built PiperVoice for tests; bypasses the lazy load.
        """
        self._voice_name = voice_name
        if voice is not None:
            self._voice = voice
        else:
            self._voice = self._load_voice(voice_name, voices_dir)

    @staticmethod
    def _load_voice(voice_name: str, voices_dir: Optional[str]) -> Any:
        from piper import PiperVoice  # type: ignore

        base = Path(voices_dir) if voices_dir else _HOME_SERVER_ROOT / "storage" / "piper_voices"
        onnx = base / f"{voice_name}.onnx"
        if not onnx.is_file():
            raise FileNotFoundError(
                f"Piper voice not found: {onnx} (expected {onnx} and {onnx}.json). "
                "Download matching files from rhasspy/piper-voices and place them in "
                f"{base.resolve()}/"
            )
        logger.info("Loading Piper voice: %s (%s)", voice_name, onnx)
        return PiperVoice.load(str(onnx))

    def synthesize(self, text: str) -> bytes:
        """Render `text` to a WAV byte string ready for playback.

        Returns 16-bit mono PCM wrapped in a WAV header. Empty input
        returns an empty WAV (44-byte header only) so callers can always
        play the result without null-checks.
        """
        if not text or not text.strip():
            return _empty_wav()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            # piper >= 1.2: WAV writer API; older test fakes may only implement synthesize().
            synth_wav = getattr(self._voice, "synthesize_wav", None)
            if callable(synth_wav):
                synth_wav(text, wav)
            else:
                self._voice.synthesize(text, wav)
        return buf.getvalue()


def _empty_wav() -> bytes:
    """Minimal 44-byte WAV header with zero samples. Plays as silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"")
    return buf.getvalue()
