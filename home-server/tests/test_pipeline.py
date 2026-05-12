"""Harness tests for app.pipeline.Session.

The Session owns the live frame -> gate -> LLM -> nudge path. These tests
use fakes for everything heavy:
  - _FakeProcessor returns a scripted DetectionResult per frame
  - _FakeLLM returns a scripted NudgeResult / can be made to raise
  - _FakeTTS returns scripted WAV bytes

We test the orchestration: worker survival, queue-everything semantics,
TTS-disabled fallback, and the priority mapping for safety scenarios.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from app.config import Settings
from app.models import AudioChunkMessage, FrameMessage, NudgeMessage, NudgePriority
from app.pipeline import MAX_QUEUE_DEPTH, Heavy, Session
from context.event_log import read_events_since
from context.manager import ContextManager
from detection.types import FullDetectionResult, ObjectDetection
from llm.base import CommandResult, NudgeResult, ToolCall
from triggers.rules import (
    KITCHEN_ABANDONMENT_THRESHOLD,
    MIN_BOTTLE_OBSERVATION_FRAMES,
    MIN_KITCHEN_OBSERVATION_FRAMES,
    CandidateKind,
    DetectionResult as RuleInput,
)


# --- Isolation -----------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    """Redirect every JSONL write to a per-test tmp dir.

    `context.event_log` reads `settings.events_dir` at call time, so we
    rebind that module attribute. Without this fixture, pipeline tests
    would pollute ./storage/events/<today>.jsonl on every run.
    """
    s = Settings(
        events_dir=tmp_path / "events",
        storage_root=tmp_path / "storage",
        faces_dir=tmp_path / "faces",
        feedback_dir=tmp_path / "feedback",
        nudges_dir=tmp_path / "nudges",
        onboarding_file=tmp_path / "storage" / "onboarding.json",
    )
    s.ensure_dirs()
    monkeypatch.setattr("context.event_log.settings", s)
    yield s


# --- Fakes ----------------------------------------------------------------

class _FakeProcessor:
    """Scripted FrameProcessor stand-in.

    Each call to `process` pops the next entry from `scripted` (a list of
    (objects, person_present) tuples). When the list is empty, returns
    an empty scene.
    """

    def __init__(self, scripted: list[tuple[list[str], bool]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[bytes] = []

    async def process(
        self, frame_bytes: bytes, timestamp: Optional[datetime] = None
    ) -> tuple[FullDetectionResult, RuleInput]:
        self.calls.append(frame_bytes)
        if self._scripted:
            objects, person = self._scripted.pop(0)
        else:
            objects, person = [], False

        full = FullDetectionResult(
            objects=[
                ObjectDetection(label=o, confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))
                for o in objects
            ]
            + (
                [ObjectDetection(label="person", confidence=0.95, bbox=(0.0, 0.0, 1.0, 1.0))]
                if person and "person" not in objects
                else []
            ),
        )
        ts = timestamp or datetime.now(timezone.utc)
        reduced = RuleInput(
            timestamp=ts,
            objects=full.object_labels,
            person_present=full.person_present,
            recognized_names=[],
        )
        return full, reduced


@dataclass
class _Call:
    image_b64: str
    summary: str


class _FakeLLM:
    """Scripted LLMClient stand-in.

    `results` is a list popped one-per-call. If an entry is an Exception,
    it's raised instead of returned (lets us test worker survival).
    `command_results` does the same for handle_voice_command.
    """
    provider_name = "fake"

    def __init__(
        self,
        results: Optional[list] = None,
        command_results: Optional[list] = None,
    ) -> None:
        self._results = list(results or [])
        self._command_results = list(command_results or [])
        self.calls: list[_Call] = []
        self.voice_calls: list[str] = []

    async def generate_nudge(
        self, *, image_b64: str, context_summary: str, persona: str = "shanta"
    ) -> NudgeResult:
        self.calls.append(_Call(image_b64=image_b64, summary=context_summary))
        if not self._results:
            return NudgeResult(
                sentence="default nudge",
                should_nudge=True,
                validation=None,
                raw_text="default nudge",
                provider="fake",
            )
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def handle_voice_command(
        self, *, transcript: str, context_summary: str
    ) -> CommandResult:
        self.voice_calls.append(transcript)
        if not self._command_results:
            return CommandResult(tool=None, raw_text="", provider="fake")
        result = self._command_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeWhisper:
    """Scripted ASR stand-in. transcript is returned for every call."""

    def __init__(self, transcript: str = "", *, raise_on_call: bool = False) -> None:
        self._transcript = transcript
        self._raise = raise_on_call
        self.calls: list[bytes] = []

    def transcribe(self, audio_bytes: bytes) -> str:
        self.calls.append(audio_bytes)
        if self._raise:
            raise RuntimeError("whisper blew up")
        return self._transcript


class _FakeTTS:
    """Returns a tiny WAV-shaped byte string per synthesize call."""

    def __init__(self, raise_on_call: bool = False) -> None:
        self._raise = raise_on_call
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self._raise:
            raise RuntimeError("synthesise blew up")
        return b"RIFF\x00\x00\x00\x00WAVEfake-pcm"


# --- Helpers --------------------------------------------------------------

_EMPTY_B64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")  # 4-byte JPEG-ish


def _frame(ts: datetime, image_b64: str = _EMPTY_B64) -> FrameMessage:
    return FrameMessage(
        device_id="test-dev",
        surface="desktop",
        timestamp=ts,
        image_b64=image_b64,
        width=640,
        height=480,
    )


def _heavy(
    processor,
    llm: _FakeLLM,
    tts: Optional[_FakeTTS] = None,
    whisper: Optional[_FakeWhisper] = None,
) -> Heavy:
    return Heavy(processor=processor, llm=llm, tts=tts, whisper=whisper)  # type: ignore[arg-type]


def _audio_msg(payload: bytes = b"fake-webm-bytes") -> AudioChunkMessage:
    return AudioChunkMessage(
        device_id="test-dev",
        timestamp=datetime.now(timezone.utc),
        audio_b64=base64.b64encode(payload).decode("ascii"),
        sample_rate=16000,
        duration_ms=3000,
    )


async def _wait_for(predicate, *, timeout: float = 1.0, poll: float = 0.01) -> None:
    """Spin-wait helper. Fails the test if predicate doesn't become true."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll)
    raise AssertionError(f"predicate never became true within {timeout}s")


# --- Happy paths ---------------------------------------------------------

@pytest.mark.asyncio
async def test_kitchen_abandonment_fires_nudge_via_pipeline() -> None:
    """Frames push activity to 'kitchen_activity', threshold elapses,
    candidate is queued, worker calls LLM + TTS, on_nudge fires."""
    # Open kitchen with MIN_KITCHEN_OBSERVATION_FRAMES frames of person+cup,
    # then a single frame past threshold with no person.
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES
        + [([], False)]
    )
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM(results=[
        NudgeResult(
            sentence="Your tea is still on the stove.",
            should_nudge=True,
            validation=None,
            raw_text="Your tea is still on the stove.",
            provider="fake",
        ),
    ])
    tts = _FakeTTS()
    nudges: list[NudgeMessage] = []

    async def on_nudge(n: NudgeMessage) -> None:
        nudges.append(n)

    cm = ContextManager()
    session = Session(heavy=_heavy(processor, llm, tts), context=cm, on_nudge=on_nudge)
    await session.start()
    try:
        base = datetime.now(timezone.utc)
        for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(base + timedelta(seconds=i * 0.2)))
        # Jump past the abandonment threshold
        await session.handle_frame(
            _frame(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5))
        )

        await _wait_for(lambda: len(nudges) == 1)
    finally:
        await session.close()

    nudge = nudges[0]
    assert nudge.sentence == "Your tea is still on the stove."
    assert nudge.priority == NudgePriority.SAFETY        # kitchen_abandoned -> safety
    assert nudge.audio_b64 is not None                   # TTS produced bytes
    assert base64.b64decode(nudge.audio_b64).startswith(b"RIFF")
    assert len(llm.calls) == 1
    assert "[Recent Context]" in llm.calls[0].summary


@pytest.mark.asyncio
async def test_quiet_gate_makes_no_llm_call() -> None:
    """Frames with nothing interesting -> no LLM calls, no nudges."""
    scripted = [(["chair"], True)] * 5
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM()
    nudges: list[NudgeMessage] = []
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    try:
        base = datetime.now(timezone.utc)
        for i in range(5):
            await session.handle_frame(_frame(base + timedelta(seconds=i * 0.2)))
        # Give the worker a chance to pick up anything (it shouldn't)
        await asyncio.sleep(0.05)
    finally:
        await session.close()

    assert llm.calls == []
    assert nudges == []


# --- Worker survival -----------------------------------------------------

@pytest.mark.asyncio
async def test_worker_survives_llm_exception(caplog) -> None:
    """First candidate makes the LLM raise. Worker survives. Second
    candidate (queued behind it) is processed normally."""
    # Two scripted scenarios: kitchen abandoned, then medication observed.
    # Build a frame sequence that produces both candidates.
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES        # open kitchen
        + [([], False)]                                            # abandonment fires
        + [(["bottle"], True)] * MIN_BOTTLE_OBSERVATION_FRAMES    # medication fires
    )
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM(results=[
        RuntimeError("transient API failure"),
        NudgeResult(
            sentence="It's time for your medicine.",
            should_nudge=True,
            validation=None,
            raw_text="ok",
            provider="fake",
        ),
    ])
    nudges: list[NudgeMessage] = []
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    caplog.set_level(logging.ERROR)
    try:
        base = datetime.now(timezone.utc)
        idx = 0
        for _ in range(MIN_KITCHEN_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(base + timedelta(seconds=idx * 0.2)))
            idx += 1
        # Abandonment frame (past threshold)
        abandon_ts = base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5)
        await session.handle_frame(_frame(abandon_ts))
        idx += 1
        # Medication frames (well after, person handling bottle)
        med_base = abandon_ts + timedelta(seconds=10)
        for k in range(MIN_BOTTLE_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(med_base + timedelta(seconds=k * 0.2)))

        # The second candidate eventually arrives; the first was swallowed.
        await _wait_for(lambda: len(nudges) == 1)
    finally:
        await session.close()

    assert len(llm.calls) == 2                       # both were attempted
    assert nudges[0].sentence == "It's time for your medicine."
    # Worker reported the first failure but didn't propagate it
    assert any("candidate handling failed" in rec.message for rec in caplog.records)


# --- NO_NUDGE / TTS edge cases ------------------------------------------

@pytest.mark.asyncio
async def test_no_nudge_result_does_not_emit() -> None:
    """LLM decides NO_NUDGE -> on_nudge NOT called, no audit event."""
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES + [([], False)]
    )
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM(results=[
        NudgeResult(
            sentence=None,
            should_nudge=False,
            validation=None,
            raw_text="NO_NUDGE",
            provider="fake",
        ),
    ])
    nudges: list[NudgeMessage] = []
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    try:
        base = datetime.now(timezone.utc)
        for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(base + timedelta(seconds=i * 0.2)))
        await session.handle_frame(
            _frame(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5))
        )
        await _wait_for(lambda: len(llm.calls) == 1)
        # Worker should not have called on_nudge
        await asyncio.sleep(0.05)
    finally:
        await session.close()

    assert nudges == []


@pytest.mark.asyncio
async def test_nudge_emitted_with_no_tts_when_tts_missing() -> None:
    """heavy.tts is None -> nudge.audio_b64 is None, but the nudge is
    still emitted so the client speech-synthesis fallback can take over."""
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES + [([], False)]
    )
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM(results=[
        NudgeResult(
            sentence="Did you finish your tea?",
            should_nudge=True, validation=None, raw_text="ok", provider="fake",
        ),
    ])
    nudges: list[NudgeMessage] = []
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm, tts=None), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    try:
        base = datetime.now(timezone.utc)
        for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(base + timedelta(seconds=i * 0.2)))
        await session.handle_frame(
            _frame(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5))
        )
        await _wait_for(lambda: len(nudges) == 1)
    finally:
        await session.close()

    assert nudges[0].audio_b64 is None


@pytest.mark.asyncio
async def test_tts_failure_emits_text_only_nudge() -> None:
    """tts.synthesize raises -> nudge still goes out, audio_b64=None."""
    scripted = (
        [(["cup"], True)] * MIN_KITCHEN_OBSERVATION_FRAMES + [([], False)]
    )
    processor = _FakeProcessor(scripted=scripted)
    llm = _FakeLLM(results=[
        NudgeResult(
            sentence="Tea's ready.",
            should_nudge=True, validation=None, raw_text="ok", provider="fake",
        ),
    ])
    bad_tts = _FakeTTS(raise_on_call=True)
    nudges: list[NudgeMessage] = []
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm, tts=bad_tts), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    try:
        base = datetime.now(timezone.utc)
        for i in range(MIN_KITCHEN_OBSERVATION_FRAMES):
            await session.handle_frame(_frame(base + timedelta(seconds=i * 0.2)))
        await session.handle_frame(
            _frame(base + KITCHEN_ABANDONMENT_THRESHOLD + timedelta(seconds=5))
        )
        await _wait_for(lambda: len(nudges) == 1)
    finally:
        await session.close()

    assert nudges[0].audio_b64 is None
    assert nudges[0].sentence == "Tea's ready."


# --- Frame-side fault tolerance -----------------------------------------

@pytest.mark.asyncio
async def test_bad_image_b64_skips_frame_without_crash(caplog) -> None:
    processor = _FakeProcessor(scripted=[])
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm), context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    caplog.set_level(logging.WARNING)
    try:
        # Hand-craft a FrameMessage with non-base64 image_b64.
        bad = FrameMessage(
            device_id="x", surface="desktop",
            timestamp=datetime.now(timezone.utc),
            image_b64="not!!base64!!", width=1, height=1,
        )
        await session.handle_frame(bad)
    finally:
        await session.close()

    assert processor.calls == []
    assert any("bad image_b64" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_processor_exception_does_not_crash_session(caplog) -> None:
    class _ExplodingProcessor:
        async def process(self, frame_bytes, timestamp=None):
            raise RuntimeError("YOLO went sideways")

    cm = ContextManager()
    llm = _FakeLLM()
    session = Session(
        heavy=_heavy(_ExplodingProcessor(), llm),  # type: ignore[arg-type]
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    caplog.set_level(logging.ERROR)
    try:
        await session.handle_frame(_frame(datetime.now(timezone.utc)))
    finally:
        await session.close()

    assert llm.calls == []
    assert any("FrameProcessor raised" in rec.message for rec in caplog.records)


# --- Queue overflow ------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_drops_oldest_on_overflow(caplog) -> None:
    """If MAX_QUEUE_DEPTH+1 candidates are enqueued while the worker is
    stuck on the first one, the oldest pending is dropped (newest kept)."""
    block_first = asyncio.Event()
    release_first = asyncio.Event()

    class _BlockingLLM(_FakeLLM):
        async def generate_nudge(self, *, image_b64, context_summary, persona="shanta"):
            self.calls.append(_Call(image_b64=image_b64, summary=context_summary))
            if len(self.calls) == 1:
                block_first.set()
                await release_first.wait()
            return NudgeResult(
                sentence=f"nudge #{len(self.calls)}",
                should_nudge=True, validation=None, raw_text="ok", provider="fake",
            )

    # We bypass the gate here and push candidates directly to test the
    # bounded-queue behaviour in isolation.
    processor = _FakeProcessor(scripted=[])
    llm = _BlockingLLM()
    cm = ContextManager()
    nudges: list[NudgeMessage] = []
    session = Session(
        heavy=_heavy(processor, llm), context=cm,
        on_nudge=lambda n: _append_async(nudges, n),
    )
    await session.start()
    try:
        from triggers.rules import TriggerCandidate
        first = TriggerCandidate(
            kind=CandidateKind.MEDICATION_OBSERVED, reason="first", detail={},
        )
        # Push the first; worker picks it up and blocks.
        session._enqueue_or_drop_oldest(first, "img-1")  # type: ignore[attr-defined]
        await block_first.wait()

        # Now fill the queue past capacity with distinct candidates.
        for i in range(MAX_QUEUE_DEPTH + 3):
            c = TriggerCandidate(
                kind=CandidateKind.MEDICATION_OBSERVED,
                reason=f"r{i}",
                detail={"i": i},
            )
            session._enqueue_or_drop_oldest(c, f"img-{i+2}")  # type: ignore[attr-defined]

        # Release the blocker; worker drains what's left in order.
        release_first.set()
        await _wait_for(lambda: len(nudges) >= 1 + MAX_QUEUE_DEPTH, timeout=2.0)
    finally:
        await session.close()

    # 1 (first) + MAX_QUEUE_DEPTH (rest of queue) survived
    assert len(nudges) == 1 + MAX_QUEUE_DEPTH
    assert any("dropped oldest" in rec.message for rec in caplog.records or [])


# --- Heavy.from_settings graceful degradation ---------------------------

def test_heavy_returns_none_when_required_imports_missing(monkeypatch) -> None:
    """Simulate missing torch/ultralytics by sabotaging the import.
    Heavy.from_settings must return None, not raise."""
    import builtins

    real_import = builtins.__import__

    def _no_yolo(name, *args, **kwargs):
        if name in ("detection.yolo", "detection.faces"):
            raise ImportError(f"simulated missing dep for {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_yolo)

    from app.config import Settings
    fake = Settings(vision_provider="yolo", active_llm_provider="grok", xai_api_key="x")
    result = Heavy.from_settings(fake)
    assert result is None


def test_heavy_returns_none_when_llm_key_missing() -> None:
    """ActiveProvider=grok but xai_api_key empty -> Heavy=None."""
    from app.config import Settings
    s = Settings(active_llm_provider="grok", xai_api_key="", google_api_key="")
    # If YOLO/Insightface are also unavailable on the dev box, this
    # short-circuits at the vision step. Either way we expect None.
    result = Heavy.from_settings(s)
    assert result is None


# --- Audio path ---------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_audio_dispatches_mark_done_for_medication(_isolated_storage) -> None:
    """End-to-end audio: bytes -> Whisper -> LLM -> markDone -> ContextManager updated."""
    whisper = _FakeWhisper(transcript="I already took my pills.")
    llm = _FakeLLM(command_results=[
        CommandResult(
            tool=ToolCall(name="markDone", arguments={"task": "medication"}),
            raw_text="markDone({task:'medication'})",
            provider="fake",
        ),
    ])
    processor = _FakeProcessor(scripted=[])
    cm = ContextManager()
    session = Session(
        heavy=_heavy(processor, llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    # Whisper saw the bytes
    assert len(whisper.calls) == 1
    # The LLM was asked about the transcript
    assert llm.voice_calls == ["I already took my pills."]
    # ContextManager now reflects the medication being taken
    assert cm.state.last_medication_time is not None


@pytest.mark.asyncio
async def test_handle_audio_mic_off_uses_local_tool_without_llm(_isolated_storage) -> None:
    """Privacy controls should be deterministic, not dependent on LLM tool choice."""
    whisper = _FakeWhisper(transcript="Mic off.")
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        result = await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert result is not None
    transcript, command = result
    assert transcript == "Mic off."
    assert command.tool is not None
    assert command.tool.name == "toggleMic"
    assert command.tool.arguments == {"state": "off"}
    assert command.provider == "local"
    assert llm.voice_calls == []


@pytest.mark.asyncio
async def test_handle_audio_audio_off_uses_local_tool_without_llm(_isolated_storage) -> None:
    whisper = _FakeWhisper(transcript="audio off")
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        result = await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert result is not None
    _transcript, command = result
    assert command.tool is not None
    assert command.tool.name == "toggleAudio"
    assert command.tool.arguments == {"state": "off"}
    assert llm.voice_calls == []


@pytest.mark.asyncio
async def test_handle_audio_presence_check_uses_local_reply_without_llm(_isolated_storage) -> None:
    whisper = _FakeWhisper(transcript="Are you there?")
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        result = await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert result is not None
    _transcript, command = result
    assert command.tool is not None
    assert command.tool.name == "assistantReply"
    assert "listening" in command.tool.arguments["sentence"]
    assert command.provider == "local"
    assert llm.voice_calls == []


@pytest.mark.asyncio
async def test_handle_audio_mark_done_closes_open_risk_window(_isolated_storage) -> None:
    """markDone should close the most-recently-opened risk window."""
    cm = ContextManager()
    # Seed an open kitchen abandonment window
    cm.record_event({
        "event_type": "risk_window_opened",
        "window_id": "w-1",
        "scenario": CandidateKind.KITCHEN_ABANDONED.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert len(cm.state.open_risk_windows) == 1

    whisper = _FakeWhisper(transcript="I finished the tea")
    llm = _FakeLLM(command_results=[
        CommandResult(
            tool=ToolCall(name="markDone", arguments={"task": "tea"}),
            raw_text="markDone", provider="fake",
        ),
    ])
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    # Window closed; medication time NOT touched (task didn't look medication-shaped)
    assert cm.state.open_risk_windows == []
    assert cm.state.last_medication_time is None


@pytest.mark.asyncio
async def test_handle_audio_empty_transcript_is_silent(_isolated_storage) -> None:
    """Whisper VAD-filtered the chunk down to nothing -> no LLM call."""
    whisper = _FakeWhisper(transcript="   ")    # whitespace-only
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert whisper.calls == [b"fake-webm-bytes"]
    assert llm.voice_calls == []         # the LLM was never asked


@pytest.mark.asyncio
async def test_handle_audio_no_whisper_silently_drops(_isolated_storage) -> None:
    """Heavy.whisper is None (graceful degradation) -> audio is a no-op."""
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=None),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert llm.voice_calls == []


@pytest.mark.asyncio
async def test_handle_audio_whisper_failure_does_not_crash(_isolated_storage, caplog) -> None:
    whisper = _FakeWhisper(raise_on_call=True)
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    caplog.set_level(logging.ERROR)
    try:
        await session.handle_audio(_audio_msg())
        # Subsequent chunk should still be processed
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert llm.voice_calls == []
    assert any("Whisper failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_handle_audio_bad_base64_skipped(_isolated_storage, caplog) -> None:
    whisper = _FakeWhisper(transcript="hi")
    llm = _FakeLLM()
    cm = ContextManager()
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    caplog.set_level(logging.WARNING)
    try:
        bad = AudioChunkMessage(
            device_id="x",
            timestamp=datetime.now(timezone.utc),
            audio_b64="not!!base64!!",
            sample_rate=16000,
            duration_ms=1000,
        )
        await session.handle_audio(bad)
    finally:
        await session.close()

    assert whisper.calls == []
    assert any("bad audio_b64" in rec.message for rec in caplog.records)


# --- Tool dispatch (each tool) -------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_dismiss_temporarily_records_snooze(_isolated_storage) -> None:
    """dismissTemporarily emits nudge_snoozed but leaves the window OPEN."""
    cm = ContextManager()
    cm.record_event({
        "event_type": "risk_window_opened",
        "window_id": "w-9",
        "scenario": CandidateKind.KITCHEN_ABANDONED.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    whisper = _FakeWhisper(transcript="remind me later")
    llm = _FakeLLM(command_results=[
        CommandResult(
            tool=ToolCall(name="dismissTemporarily", arguments={}),
            raw_text="dismissTemporarily", provider="fake",
        ),
    ])
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    # Window stays open — the gate's window-guard IS the snooze
    assert len(cm.state.open_risk_windows) == 1
    # Audit log gained a nudge_snoozed entry
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    events = list(read_events_since(cutoff, events_dir=_isolated_storage.events_dir))
    assert any(e.get("event_type") == "nudge_snoozed" for e in events)


@pytest.mark.asyncio
async def test_dispatch_flag_wrong_logs_only(_isolated_storage) -> None:
    cm = ContextManager()
    whisper = _FakeWhisper(transcript="that's wrong, I never had tea")
    llm = _FakeLLM(command_results=[
        CommandResult(
            tool=ToolCall(name="flagWrong", arguments={"reason": "I never had tea"}),
            raw_text="flagWrong", provider="fake",
        ),
    ])
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    events = list(read_events_since(cutoff, events_dir=_isolated_storage.events_dir))
    assert any(
        e.get("event_type") == "nudge_flagged_wrong" and "tea" in e.get("reason", "")
        for e in events
    )
    # No state change beyond the audit log
    assert cm.state.open_risk_windows == []
    assert cm.state.last_medication_time is None


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_is_logged_and_ignored(_isolated_storage, caplog) -> None:
    cm = ContextManager()
    whisper = _FakeWhisper(transcript="do something weird")
    llm = _FakeLLM(command_results=[
        CommandResult(
            tool=ToolCall(name="iDoNotExist", arguments={"x": 1}),
            raw_text="iDoNotExist", provider="fake",
        ),
    ])
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    caplog.set_level(logging.WARNING)
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    assert any("unknown tool" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_dispatch_no_tool_call_still_logs_transcript(_isolated_storage) -> None:
    """LLM came back without a tool. We still want the transcript in the
    audit log so the dashboard can show 'I heard X but mapped it to nothing.'"""
    cm = ContextManager()
    whisper = _FakeWhisper(transcript="just talking to myself")
    llm = _FakeLLM(command_results=[
        CommandResult(tool=None, raw_text="no mapping", provider="fake"),
    ])
    session = Session(
        heavy=_heavy(_FakeProcessor([]), llm, whisper=whisper),
        context=cm,
        on_nudge=lambda n: _append_async([], n),
    )
    await session.start()
    try:
        await session.handle_audio(_audio_msg())
    finally:
        await session.close()

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    events = list(read_events_since(cutoff, events_dir=_isolated_storage.events_dir))
    voice_events = [e for e in events if e.get("event_type") == "voice_command_received"]
    assert len(voice_events) == 1
    assert voice_events[0]["transcript"] == "just talking to myself"
    assert voice_events[0]["tool"] is None


# --- Helpers --------------------------------------------------------------

async def _append_async(target: list, item) -> None:
    target.append(item)
