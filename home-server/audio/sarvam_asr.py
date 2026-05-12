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
        self._client = client
        self._timeout = timeout

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

        close_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            data: dict[str, Any] = {
                "model": self._model,
                "mode": self._mode,
            }
            if self._language_code:
                data["language_code"] = self._language_code
            response = client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self._api_key},
                data=data,
                files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            if close_client:
                client.close()

        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            logger.info("Sarvam STT returned an empty transcript")
        return transcript
