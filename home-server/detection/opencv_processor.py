"""FrameProcessor implementation for the M2-friendly exhibition path."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from detection.opencv_recognizers import OpenCVFaceRecognizer, SpecificObjectRecognizer
from detection.types import FullDetectionResult
from triggers.rules import DetectionResult as RuleInput


class OpenCVFrameProcessor:
    def __init__(self, reference_dir: Path) -> None:
        self._objects = SpecificObjectRecognizer({
            "medicine_bottle": reference_dir / "medicine_bottle",
            "stove": reference_dir / "stove",
        })
        self._faces = OpenCVFaceRecognizer(reference_dir / "faces")

    async def process(
        self,
        frame_bytes: bytes,
        timestamp: datetime | None = None,
    ) -> Tuple[FullDetectionResult, RuleInput]:
        ts = timestamp or datetime.now(timezone.utc)
        objects_task = asyncio.to_thread(self._objects.detect, frame_bytes)
        faces_task = asyncio.to_thread(self._faces.recognize, frame_bytes)
        objects, faces = await asyncio.gather(objects_task, faces_task)

        full = FullDetectionResult(objects=objects, faces=faces)
        reduced = RuleInput(
            timestamp=ts,
            objects=full.object_labels,
            person_present=bool(full.faces),
            recognized_names=full.recognized_names,
        )
        return full, reduced
