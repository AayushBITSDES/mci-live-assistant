from __future__ import annotations

import base64
import json

import httpx

from audio.sarvam_asr import SARVAM_STT_URL, SarvamTranscriber
from audio.sarvam_tts import SARVAM_TTS_URL, SarvamSynthesizer


def test_sarvam_asr_posts_multipart_and_reads_transcript() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("api-subscription-key")
        body = request.read()
        seen["body"] = body
        return httpx.Response(200, json={"transcript": "I took my vitamin."})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = SarvamTranscriber(
        api_key="sarvam-key",
        model="saaras:v3",
        language_code="en-IN",
        client=client,
    )

    assert asr.transcribe(b"fake-audio") == "I took my vitamin."
    assert seen["url"] == SARVAM_STT_URL
    assert seen["key"] == "sarvam-key"
    assert b'name="model"' in seen["body"]
    assert b"saaras:v3" in seen["body"]
    assert b'name="mode"' in seen["body"]
    assert b"transcribe" in seen["body"]


def test_sarvam_asr_empty_audio_short_circuits() -> None:
    asr = SarvamTranscriber(api_key="x")
    assert asr.transcribe(b"") == ""


def test_sarvam_tts_posts_json_and_decodes_first_audio() -> None:
    wav = b"RIFFfake-wave"
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("api-subscription-key")
        seen["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(200, json={"audios": [base64.b64encode(wav).decode("ascii")]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tts = SarvamSynthesizer(
        api_key="sarvam-key",
        model="bulbul:v3",
        speaker="shubh",
        language_code="en-IN",
        client=client,
    )

    assert tts.synthesize("Please check the stove.") == wav
    assert seen["url"] == SARVAM_TTS_URL
    assert seen["key"] == "sarvam-key"
    assert seen["json"]["text"] == "Please check the stove."
    assert seen["json"]["target_language_code"] == "en-IN"
    assert seen["json"]["model"] == "bulbul:v3"
    assert seen["json"]["speaker"] == "shubh"


def test_sarvam_tts_empty_text_short_circuits() -> None:
    tts = SarvamSynthesizer(api_key="x")
    assert tts.synthesize("  ") == b""
