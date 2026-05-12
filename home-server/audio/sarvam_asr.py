"""Sarvam Saaras speech-to-text adapter.

The rest of the app expects a synchronous ``transcribe(bytes) -> str``
interface because audio work is pushed through ``asyncio.to_thread``.
This wrapper keeps Sarvam behind that same seam.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_AUDIO_CONTENT_TYPE = "audio/webm"

# Filename hint sent to Sarvam. The MIME header is authoritative, but the
# filename extension is a useful secondary signal for servers that route
# decoding by extension. Keys are the bare type ("audio/X") with codec
# params stripped before lookup.
_FILENAME_BY_TYPE: dict[str, str] = {
    "audio/webm": "audio.webm",
    "audio/ogg": "audio.ogg",
    "audio/mp4": "audio.m4a",
    "audio/mpeg": "audio.mp3",
    "audio/wav": "audio.wav",
    "audio/wave": "audio.wav",
    "audio/x-wav": "audio.wav",
}


def _audio_filename_for(content_type: str) -> str:
    """Pick a filename whose extension matches the given content type."""
    base = content_type.split(";", 1)[0].strip().lower()
    return _FILENAME_BY_TYPE.get(base, "audio.bin")


class SarvamTranscriber:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        language_code: str | None = "en-IN",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("SarvamTranscriber requires api_key or a pre-built client")
        self._api_key = api_key
        self._model = model
        self._mode = mode
        self._language_code = language_code
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SarvamTranscriber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str | None = None,
    ) -> str:
        if not audio_bytes:
            return ""

        # The browser's MediaRecorder chooses one of webm/opus, ogg/opus,
        # or mp4 depending on the platform. Honouring what the caller
        # actually recorded avoids telling Sarvam "this is webm" when it
        # is in fact ogg, which can produce a degraded or empty transcript.
        ct = (content_type or "").strip() or DEFAULT_AUDIO_CONTENT_TYPE
        base_ct = ct.split(";", 1)[0].strip()
        filename = _audio_filename_for(ct)

        data: dict[str, Any] = {
            "model": self._model,
            "mode": self._mode,
        }
        if self._language_code:
            data["language_code"] = self._language_code
        response = self._client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": self._api_key},
            data=data,
            files={"file": (filename, audio_bytes, base_ct)},
        )
        response.raise_for_status()
        payload = response.json()

        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            logger.info("Sarvam STT returned an empty transcript")
        return transcript
