from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from audio.sarvam_asr import SARVAM_STT_URL, SarvamTranscriber
from audio.sarvam_tts import SARVAM_TTS_URL, SarvamSynthesizer


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


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
    # No explicit content_type -> falls back to the webm default.
    assert b"audio/webm" in seen["body"]
    assert b"audio.webm" in seen["body"]


def test_sarvam_asr_honours_provided_content_type() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"transcript": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = SarvamTranscriber(api_key="x", client=client)

    asr.transcribe(b"bytes", content_type="audio/ogg;codecs=opus")

    body = seen["body"]
    # Codec param preserved in Content-Type; filename derived from base type.
    assert b"audio/ogg;codecs=opus" in body
    assert b"audio.ogg" in body
    assert b"audio/webm" not in body


def test_sarvam_asr_unknown_content_type_uses_bin_filename() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"transcript": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = SarvamTranscriber(api_key="x", client=client)

    asr.transcribe(b"bytes", content_type="audio/unknown-thing")

    assert b"audio/unknown-thing" in seen["body"]
    assert b"audio.bin" in seen["body"]


def test_sarvam_asr_empty_audio_short_circuits() -> None:
    asr = SarvamTranscriber(api_key="x")
    assert asr.transcribe(b"") == ""


def test_sarvam_asr_reuses_owned_http_client(monkeypatch) -> None:
    instances: list[Any] = []

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout
            self.posts = 0
            self.closed = False
            instances.append(self)

        def post(self, *args, **kwargs):
            self.posts += 1
            return _FakeResponse({"transcript": f"transcript-{self.posts}"})

        def close(self):
            self.closed = True

    monkeypatch.setattr("audio.sarvam_asr.httpx.Client", FakeClient)
    asr = SarvamTranscriber(api_key="x")

    assert asr.transcribe(b"one") == "transcript-1"
    assert asr.transcribe(b"two") == "transcript-2"
    assert len(instances) == 1
    assert instances[0].posts == 2
    assert instances[0].closed is False

    asr.close()
    assert instances[0].closed is True


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


def test_sarvam_tts_reuses_owned_http_client(monkeypatch) -> None:
    wav = base64.b64encode(b"RIFF").decode("ascii")
    instances: list[Any] = []

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout
            self.posts = 0
            self.closed = False
            instances.append(self)

        def post(self, *args, **kwargs):
            self.posts += 1
            return _FakeResponse({"audios": [wav]})

        def close(self):
            self.closed = True

    monkeypatch.setattr("audio.sarvam_tts.httpx.Client", FakeClient)
    tts = SarvamSynthesizer(api_key="x")

    assert tts.synthesize("one") == b"RIFF"
    assert tts.synthesize("two") == b"RIFF"
    assert len(instances) == 1
    assert instances[0].posts == 2
    assert instances[0].closed is False

    tts.close()
    assert instances[0].closed is True
