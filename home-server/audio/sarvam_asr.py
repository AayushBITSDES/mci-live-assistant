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
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SarvamTranscriber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

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
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
        )
        response.raise_for_status()
        payload = response.json()

        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            logger.info("Sarvam STT returned an empty transcript")
        return transcript
