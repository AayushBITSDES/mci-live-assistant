"""Sarvam Bulbul text-to-speech adapter."""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamSynthesizer:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "bulbul:v3",
        speaker: str = "shubh",
        language_code: str = "en-IN",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("SarvamSynthesizer requires api_key or a pre-built client")
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._language_code = language_code
        self._client = client
        self._timeout = timeout

    def synthesize(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""

        close_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            response = client.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "target_language_code": self._language_code,
                    "model": self._model,
                    "speaker": self._speaker,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        finally:
            if close_client:
                client.close()

        audios = payload.get("audios")
        if not isinstance(audios, list) or not audios:
            logger.warning("Sarvam TTS response did not include audios")
            return b""
        first = str(audios[0] or "")
        if not first:
            return b""
        return base64.b64decode(first)
