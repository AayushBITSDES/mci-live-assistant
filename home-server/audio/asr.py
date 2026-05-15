"""faster-whisper wrapper for server-side automatic speech recognition.

Loads the whisper model lazily so this module imports without the
ML deps installed. Production: pass `model_size` and let it auto-load.
Tests: pass `model=fake` to skip the import entirely.

Model selection guide:
  tiny      (~75MB)  Fast but error-prone in noise
  small     (~250MB) Decent quality, good for clean speech
  medium    (~1.5GB) Best balance — recommended default
  large-v3  (~3GB)   Slower; only worth it on a 4070+

GPU is auto-detected. On Apple Silicon it uses the Metal Performance
Shaders backend; on NVIDIA it picks CUDA; otherwise CPU.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "medium",
        *,
        language: str = "en",
        device: str = "auto",
        compute_type: str = "int8",
        model: Optional[Any] = None,
    ) -> None:
        self._language = language
        if model is not None:
            self._model = model
        else:
            self._model = self._load_model(model_size, device, compute_type)

    @staticmethod
    def _load_model(model_size: str, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel  # type: ignore
        logger.info("Loading faster-whisper model: %s on %s", model_size, device)
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str | None = None,
    ) -> str:
        """Transcribe a chunk of audio bytes to text.

        Returns the concatenated text from all segments. Empty string if
        no speech is detected. Caller is responsible for passing a
        complete utterance — this is not a streaming interface.

        We accept WAV/PCM/Opus-encoded bytes; faster-whisper auto-detects
        via librosa/soundfile under the hood. ``content_type`` is
        accepted for interface parity with SarvamTranscriber and ignored
        here; the local model sniffs the container itself.
        """
        _ = content_type
        segments, _info = self._model.transcribe(
            io.BytesIO(audio_bytes),
            language=self._language,
            beam_size=1,            # demo speed > marginal accuracy
            vad_filter=True,        # skip silence
        )
        text_parts = [segment.text for segment in segments]
        return " ".join(t.strip() for t in text_parts if t and t.strip())
