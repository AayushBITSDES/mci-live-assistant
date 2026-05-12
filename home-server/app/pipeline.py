"""Pipeline: vision -> gate -> LLM -> TTS -> NudgeMessage.

`Heavy` is a singleton bundle of expensive ML components (YOLO, face
recognizer, FrameProcessor, LLM client, TTS). Built once at lifespan
start. None when graceful degradation kicked in.

`Session` is per-WS state: gate, candidate queue, worker task, outbound
callback. Owns the lifecycle (start/close) so the WS handler can stay
focused on routing messages.

## Dispatch policy: queue-everything

Every TriggerCandidate the gate emits is queued and processed in order.
The bound (MAX_QUEUE_DEPTH) is a safety net against unbounded growth on
a stuck LLM call — when full, the OLDEST is dropped to keep the newest
fresh. In practice the gate's window guard prevents repeated fires of
the same scenario, so the queue rarely holds more than a couple of items.

## Graceful degradation

Any of {torch, ultralytics, insightface, faster-whisper, piper, the LLM
SDK} can be missing on the dev box. `Heavy.from_settings` catches the
import/runtime errors and returns None; the WS handler then falls back
to its debug-nudge stub. TTS is optional independently — if Piper fails
to load but the LLM works, nudges go out as text and the React client
speaks them via `speechSynthesis` (kept as a fallback in audio.ts).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Tuple

from app.config import Settings
from app.models import AudioChunkMessage, FrameMessage, NudgeMessage, NudgePriority
from context.builder import build_summary
from context.event_log import append_event
from context.manager import ContextManager
from llm.base import CommandResult, LLMClient, ToolCall
from llm.tools import SNOOZE_DURATION_SECONDS
from triggers.gate import TriggerGate
from triggers.rules import CandidateKind, TriggerCandidate

logger = logging.getLogger(__name__)


# Max pending candidates per session. At 5 FPS with the gate's
# per-scenario window guard, realistic peak is ~2-3 (kitchen + medication
# overlapping). 16 is generous headroom; oldest is dropped on overflow.
MAX_QUEUE_DEPTH = 16


# Candidate kinds that should surface as safety-priority on the edge.
# Quality is the default for everything else (e.g. medication_observed
# reminders without prior-dose context).
_SAFETY_KINDS = {
    CandidateKind.KITCHEN_ABANDONED.value,
    CandidateKind.MEDICATION_DOUBLE_DOSE_RISK.value,
}


NudgeCallback = Callable[[NudgeMessage], Awaitable[None]]


@dataclass
class Heavy:
    """Shared singleton of heavy components.

    Sessions read but never mutate this. LLM and the frame processor
    are required (if either fails, Heavy returns None and the server
    runs in debug-stub mode). TTS and Whisper are independently
    optional — text-only nudges still work without Piper, and the
    audio path becomes a no-op log without Whisper.
    """
    processor: object
    llm: LLMClient
    tts: Optional[object] = None          # PiperSynthesizer when available
    whisper: Optional[object] = None      # ASR provider when available

    @classmethod
    def from_settings(cls, settings: Settings) -> Optional["Heavy"]:
        """Try to construct everything. Returns None on any required failure.

        Failure modes (each logged at WARNING, server keeps running):
          - Required imports missing -> graceful degradation
          - YOLO weights / Insightface models not on disk
          - LLM API key not set for the selected provider
        TTS failure does not kill Heavy (it's optional).
        """
        try:
            from llm.factory import build_client
        except ImportError as exc:
            logger.warning("Heavy: required imports missing (%s) -> debug stub", exc)
            return None

        try:
            if settings.vision_provider == "opencv":
                from detection.opencv_processor import OpenCVFrameProcessor
                processor = OpenCVFrameProcessor(reference_dir=settings.reference_dir)
            else:
                from detection.faces import FaceRecognizer
                from detection.processor import FrameProcessor
                from detection.yolo import YoloDetector
                yolo = YoloDetector(model_path=settings.yolo_model)
                faces = FaceRecognizer(
                    store_path=settings.faces_dir / "embeddings.json",
                    model_name=settings.insightface_model,
                )
                processor = FrameProcessor(yolo=yolo, faces=faces)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Heavy: vision pipeline failed to load (%s) -> debug stub", exc)
            return None

        try:
            llm = build_client(
                settings.active_llm_provider,
                xai_api_key=settings.xai_api_key,
                google_api_key=settings.google_api_key,
            )
        except (ValueError, ImportError) as exc:
            logger.warning("Heavy: LLM client unavailable (%s) -> debug stub", exc)
            return None

        # TTS is optional; client has speechSynthesis fallback.
        tts: Optional[object] = None
        if settings.tts_provider == "sarvam":
            try:
                from audio.sarvam_tts import SarvamSynthesizer
                tts = SarvamSynthesizer(
                    api_key=settings.sarvam_api_key,
                    model=settings.sarvam_tts_model,
                    speaker=settings.sarvam_tts_speaker,
                    language_code=settings.sarvam_language_code,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.info("Heavy: Sarvam TTS unavailable (%s) -> browser TTS fallback", exc)
        elif settings.tts_provider == "piper":
            try:
                from audio.tts import PiperSynthesizer  # heavy import
                tts = PiperSynthesizer(voice_name=settings.piper_voice)
            except (ImportError, OSError, RuntimeError, TypeError) as exc:
                logger.info("Heavy: Piper TTS unavailable (%s) -> browser TTS fallback", exc)

        # ASR is optional; without it the audio path is a no-op log.
        whisper: Optional[object] = None
        if settings.asr_provider == "sarvam":
            try:
                from audio.sarvam_asr import SarvamTranscriber
                whisper = SarvamTranscriber(
                    api_key=settings.sarvam_api_key,
                    model=settings.sarvam_stt_model,
                    language_code=settings.sarvam_language_code,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.info("Heavy: Sarvam ASR unavailable (%s) -> voice commands disabled", exc)
        elif settings.asr_provider == "whisper":
            try:
                from audio.asr import WhisperTranscriber  # heavy import
                whisper = WhisperTranscriber(model_size=settings.whisper_model)
            except (ImportError, OSError, RuntimeError) as exc:
                logger.info("Heavy: Whisper unavailable (%s) -> voice commands disabled", exc)

        logger.info(
            "Heavy: ready | provider=%s | vision=%s | tts=%s | asr=%s",
            settings.active_llm_provider,
            settings.vision_provider,
            "on" if tts is not None else "off",
            "on" if whisper is not None else "off",
        )
        return cls(processor=processor, llm=llm, tts=tts, whisper=whisper)


class Session:
    """Per-WebSocket state for the live pipeline.

    One Session per connected device. Holds the gate (per-device frame
    counters), a candidate queue, a worker task, and the outbound
    callback the WS handler uses to send nudges. Heavy + ContextManager
    are shared via reference.
    """

    def __init__(
        self,
        *,
        heavy: Heavy,
        context: ContextManager,
        on_nudge: NudgeCallback,
        auto_dismiss_seconds: int = 10,
    ) -> None:
        self._heavy = heavy
        self._context = context
        self._on_nudge = on_nudge
        self._auto_dismiss = auto_dismiss_seconds

        # Gate buffer: the gate's emit_event contract is sync. We capture
        # into a list, then flush to the JSONL log off-thread once
        # process_frame returns. This keeps file I/O out of the
        # frame-handler hot path.
        self._gate_events: list[dict] = []
        self._gate = TriggerGate(context, emit_event=self._gate_events.append)

        # Queue (candidate, image_b64). The image is captured at the
        # moment the rule fires so the LLM sees that frame, not whatever
        # frame the worker happens to drain.
        self._queue: asyncio.Queue[Tuple[TriggerCandidate, str]] = asyncio.Queue(
            maxsize=MAX_QUEUE_DEPTH
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._closed = False

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn the worker. Call once after construction."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        """Cancel the worker. Idempotent. Pending nudges are dropped."""
        if self._closed:
            return
        self._closed = True
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Session: worker raised on shutdown (swallowed)")

    # --- Frame path --------------------------------------------------------

    async def handle_frame(self, frame: FrameMessage):
        """Run vision + gate, queue any candidates. Returns immediately.

        Vision + rule evaluation are synchronous awaits (~30ms on GPU,
        a bit more on CPU). The LLM call + TTS happen later in the
        worker, decoupled from frame arrival.
        """
        if self._closed:
            return None

        try:
            image_bytes = base64.b64decode(frame.image_b64)
        except Exception as exc:
            logger.warning("Session: bad image_b64 (%s) - skipping frame", exc)
            return None

        try:
            _full, reduced = await self._heavy.processor.process(
                image_bytes, timestamp=frame.timestamp
            )
        except Exception:
            logger.exception("Session: FrameProcessor raised - skipping frame")
            return

        try:
            candidates = self._gate.process_frame(reduced)
        except Exception:
            logger.exception("Session: TriggerGate raised - skipping frame")
            self._gate_events.clear()
            return None

        # Flush state-change events to the audit log off-thread.
        await self._flush_gate_events()

        for candidate in candidates:
            self._enqueue_or_drop_oldest(candidate, frame.image_b64)
        return reduced

    # --- Internals ---------------------------------------------------------

    async def _flush_gate_events(self) -> None:
        """Drain the captured gate-event buffer to the JSONL log."""
        if not self._gate_events:
            return
        events = self._gate_events
        self._gate_events = []
        for event in events:
            await asyncio.to_thread(append_event, event)

    def _enqueue_or_drop_oldest(self, candidate: TriggerCandidate, image_b64: str) -> None:
        """Push (candidate, image) to the worker queue.

        Queue-everything semantics: when the queue is full, the OLDEST
        candidate is dropped. Loss only happens during sustained backlog
        which the gate's window guard already makes rare.
        """
        item = (candidate, image_b64)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                dropped, _ = self._queue.get_nowait()
                logger.warning(
                    "Session: queue full - dropped oldest candidate (%s)",
                    dropped.kind.value,
                )
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(item)

    async def _worker_loop(self) -> None:
        """Drain the queue forever. Survives individual call failures."""
        while True:
            try:
                candidate, image_b64 = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._handle_one_candidate(candidate, image_b64)
            except asyncio.CancelledError:
                break
            except Exception:
                # A single LLM hiccup cannot tank the whole session.
                logger.exception("Session: candidate handling failed (worker survives)")

    async def _handle_one_candidate(
        self, candidate: TriggerCandidate, image_b64: str
    ) -> None:
        summary = build_summary(self._context)
        result = await self._heavy.llm.generate_nudge(
            image_b64=image_b64,
            context_summary=summary,
            persona="shanta",
        )
        if not result.should_nudge or not result.sentence:
            logger.info(
                "Session: LLM returned NO_NUDGE for %s",
                candidate.kind.value,
            )
            return

        audio_b64 = await self._maybe_synthesize(result.sentence)

        priority = (
            NudgePriority.SAFETY
            if candidate.kind.value in _SAFETY_KINDS
            else NudgePriority.QUALITY
        )
        nudge = NudgeMessage(
            nudge_id=str(uuid.uuid4()),
            sentence=result.sentence,
            priority=priority,
            audio_b64=audio_b64,
            auto_dismiss_seconds=self._auto_dismiss,
        )

        # Audit log first so the dashboard reflects the emission even if
        # the WS send fails (e.g. device disconnected mid-flight).
        await asyncio.to_thread(
            append_event,
            {
                "event_type": "nudge_emitted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nudge_id": nudge.nudge_id,
                "candidate_kind": candidate.kind.value,
                "priority": priority.value,
                "provider": result.provider,
            },
        )

        try:
            await self._on_nudge(nudge)
        except Exception:
            logger.exception("Session: on_nudge callback raised")

    async def _maybe_synthesize(self, sentence: str) -> Optional[str]:
        """Render sentence via Piper if available; return base64 WAV or None."""
        tts = self._heavy.tts
        if tts is None:
            return None
        try:
            wav_bytes = await asyncio.to_thread(tts.synthesize, sentence)
        except Exception:
            logger.exception("Session: TTS synthesis failed - text-only nudge")
            return None
        return base64.b64encode(wav_bytes).decode("ascii")

    # --- Audio path --------------------------------------------------------

    async def handle_audio(self, audio: AudioChunkMessage) -> Optional[tuple[str, CommandResult]]:
        """Transcribe an audio chunk and dispatch any resulting tool call.

        Independent of the frame path: an audio chunk is its own
        utterance; we transcribe, ask the LLM to map it to a tool, then
        apply the tool's state changes. Failures at any stage are logged
        and the next chunk is processed normally.
        """
        if self._closed:
            return None
        whisper = self._heavy.whisper
        if whisper is None:
            # Voice commands are not available in this build (no Whisper).
            # Acknowledge silently so the protocol stays clean.
            return None

        try:
            audio_bytes = base64.b64decode(audio.audio_b64)
        except Exception as exc:
            logger.warning("Session: bad audio_b64 (%s) - skipping chunk", exc)
            return None
        if not audio_bytes:
            return None

        asr_started = time.perf_counter()
        try:
            transcript = await asyncio.to_thread(whisper.transcribe, audio_bytes)
        except Exception:
            logger.exception("Session: Whisper failed - skipping audio chunk")
            return None
        logger.info("Session: ASR completed in %.2fs", time.perf_counter() - asr_started)

        transcript = (transcript or "").strip()
        if not transcript:
            # VAD inside Whisper filtered the chunk down to silence.
            return None

        logger.info("Session: audio transcript=%r", transcript)
        command = _local_voice_command(transcript)
        if command is None:
            llm_started = time.perf_counter()
            try:
                command = await self._heavy.llm.handle_voice_command(
                    transcript=transcript,
                    context_summary=build_summary(self._context),
                )
            except Exception:
                logger.exception("Session: handle_voice_command failed - dropping audio")
                return None
            logger.info("Session: voice LLM completed in %.2fs", time.perf_counter() - llm_started)

        # Always log the interpretation, even when no tool was returned —
        # it's useful in the dashboard for debugging mis-recognitions.
        await asyncio.to_thread(
            append_event,
            {
                "event_type": "voice_command_received",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transcript": transcript,
                "tool": command.tool.name if command.tool else None,
                "raw": command.raw_text,
                "provider": command.provider,
            },
        )

        if command.tool is not None:
            await self._dispatch_tool(command.tool)
        return transcript, command

    # --- Tool dispatch -----------------------------------------------------

    async def _dispatch_tool(self, tool: ToolCall) -> None:
        """Apply a tool's state changes to ContextManager + emit audit events.

        Unknown tools are logged-and-ignored (forward-compat: an LLM
        might return a tool we haven't wired yet).
        """
        handlers = {
            "markDone": self._dispatch_mark_done,
            "dismissTemporarily": self._dispatch_dismiss_temporarily,
            "flagWrong": self._dispatch_flag_wrong,
            "closeForever": self._dispatch_close_forever,
            "toggleMic": self._dispatch_toggle_log_only,
            "toggleCamera": self._dispatch_toggle_log_only,
            "toggleAudio": self._dispatch_toggle_log_only,
            "assistantReply": self._dispatch_assistant_reply_log_only,
        }
        handler = handlers.get(tool.name)
        if handler is None:
            logger.warning("Session: unknown tool %r (args=%s) - ignored",
                           tool.name, tool.arguments)
            return
        try:
            events = handler(tool.arguments)
        except Exception:
            logger.exception("Session: tool %r raised - no state change", tool.name)
            return
        for event in events:
            self._context.record_event(event)
            await asyncio.to_thread(append_event, event)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _dispatch_mark_done(self, args: dict) -> list[dict]:
        """User confirmed completion. Closes most recent open window +
        logs medication_taken if the task looks medication-shaped."""
        task = str(args.get("task", "")).strip()
        events: list[dict] = []

        # Medication-shaped tasks update last_medication_time.
        if _looks_like_medication(task):
            events.append({
                "event_type": "medication_taken",
                "timestamp": self._now_iso(),
                "source": "voice_command",
            })

        # Close the most recently opened risk window; the user has
        # resolved whatever the nudge was about.
        state = self._context.state
        if state.open_risk_windows:
            latest = max(state.open_risk_windows, key=lambda w: w.start_time)
            events.append({
                "event_type": "risk_window_closed",
                "window_id": latest.id,
                "status": "resolved",
                "timestamp": self._now_iso(),
            })

        events.append({
            "event_type": "task_marked_done",
            "task": task,
            "timestamp": self._now_iso(),
        })
        return events

    def _dispatch_dismiss_temporarily(self, args: dict) -> list[dict]:
        """User said 'remind me later'. Leave the risk window OPEN so the
        gate's window-guard keeps natural re-fires suppressed — that IS
        the snooze. We just record the request for the audit trail.
        """
        state = self._context.state
        window_id: Optional[str] = None
        if state.open_risk_windows:
            latest = max(state.open_risk_windows, key=lambda w: w.start_time)
            window_id = latest.id
        return [{
            "event_type": "nudge_snoozed",
            "window_id": window_id,
            "snooze_seconds": SNOOZE_DURATION_SECONDS,
            "timestamp": self._now_iso(),
        }]

    def _dispatch_flag_wrong(self, args: dict) -> list[dict]:
        return [{
            "event_type": "nudge_flagged_wrong",
            "reason": str(args.get("reason", "")).strip(),
            "timestamp": self._now_iso(),
        }]

    def _dispatch_close_forever(self, args: dict) -> list[dict]:
        return [{
            "event_type": "category_suppressed",
            "category": str(args.get("category", "")).strip(),
            "timestamp": self._now_iso(),
        }]

    def _dispatch_toggle_log_only(self, args: dict) -> list[dict]:
        """toggleMic/toggleCamera by voice: log the request only.

        The kill switches live on the edge client (it owns the
        MediaStream tracks). Honouring a voice toggle requires a
        ControlMessage protocol back to the client; deferred for now.
        """
        return [{
            "event_type": "kill_switch_voice_request",
            "requested_state": str(args.get("state", "")).strip(),
            "timestamp": self._now_iso(),
        }]

    def _dispatch_assistant_reply_log_only(self, args: dict) -> list[dict]:
        return [{
            "event_type": "assistant_reply_requested",
            "sentence": str(args.get("sentence", "")).strip(),
            "timestamp": self._now_iso(),
        }]


# --- Helpers --------------------------------------------------------------

def _local_voice_command(transcript: str) -> Optional[CommandResult]:
    """Deterministic fast path for privacy/audio controls.

    These commands should not depend on a remote model choosing the right
    tool. The transcript is already ASR output, so keep the parser simple
    and conservative.
    """
    words = set(
        transcript.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )
    if not words:
        return None

    if {"there", "hear", "hearing", "hello", "hi"} & words:
        return CommandResult(
            tool=ToolCall(
                name="assistantReply",
                arguments={"sentence": "I am here and listening."},
            ),
            raw_text="assistantReply:presence",
            provider="local",
        )

    if "off" in words or "mute" in words:
        state = "off"
    elif "on" in words or "unmute" in words:
        state = "on"
    else:
        return None

    if {"mic", "microphone"} & words:
        return CommandResult(
            tool=ToolCall(name="toggleMic", arguments={"state": state}),
            raw_text=f"toggleMic:{state}",
            provider="local",
        )
    if {"camera", "video"} & words:
        return CommandResult(
            tool=ToolCall(name="toggleCamera", arguments={"state": state}),
            raw_text=f"toggleCamera:{state}",
            provider="local",
        )
    if {"audio", "sound", "speaker", "speakers"} & words:
        return CommandResult(
            tool=ToolCall(name="toggleAudio", arguments={"state": state}),
            raw_text=f"toggleAudio:{state}",
            provider="local",
        )
    return None


def _looks_like_medication(task: str) -> bool:
    """Heuristic: a 'markDone' task string is medication-shaped if it
    mentions a pill / medicine / dose keyword. We deliberately keep this
    loose so localised phrasing ('my goli', 'morning meds') still maps."""
    t = task.lower()
    return any(
        k in t for k in ("medic", "pill", "dose", "tablet", "meds", "goli", "drug")
    )
