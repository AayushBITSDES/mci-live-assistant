"""Per-frame processor: runs YOLO + face recognition, builds the result.

Runs the two models in parallel via `asyncio.to_thread` so a slow face
pass cannot stall object detection (or vice versa).

Returns:
  - FullDetectionResult: rich detail used by Streamlit's live view
  - triggers.DetectionResult: reduced view consumed by TriggerGate
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Tuple

from detection.faces import FaceRecognizer
from detection.types import FullDetectionResult
from detection.yolo import YoloDetector
from triggers.rules import DetectionResult as RuleInput

logger = logging.getLogger(__name__)


class FrameProcessor:
    def __init__(self, yolo: YoloDetector, faces: FaceRecognizer) -> None:
        self._yolo = yolo
        self._faces = faces

    async def process(
        self,
        frame_bytes: bytes,
        timestamp: datetime | None = None,
    ) -> Tuple[FullDetectionResult, RuleInput]:
        """Run both models concurrently, build both views.

        Returns a tuple so callers can keep the rich version for the
        dashboard while feeding the reduced version to the trigger gate.
        """
        ts = timestamp or datetime.now(timezone.utc)

        objects_task = asyncio.to_thread(self._yolo.detect, frame_bytes)
        faces_task = asyncio.to_thread(self._faces.recognize, frame_bytes)
        objects, faces = await asyncio.gather(objects_task, faces_task)

        full = FullDetectionResult(objects=objects, faces=faces)
        reduced = RuleInput(
            timestamp=ts,
            objects=full.object_labels,
            person_present=full.person_present,
        )
        return full, reduced
