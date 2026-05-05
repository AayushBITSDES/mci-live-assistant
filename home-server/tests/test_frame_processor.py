"""FrameProcessor tests: composes YOLO + face recognition into both views.

Uses lightweight fakes for YoloDetector and FaceRecognizer; we are
testing the orchestration / data flow, not the underlying ML wrappers
(those have their own tests).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from detection.processor import FrameProcessor
from detection.types import FullDetectionResult, ObjectDetection, FaceMatch


class _FakeYolo:
    def __init__(self, detections: list[ObjectDetection]) -> None:
        self._detections = detections
        self.calls: list[bytes] = []

    def detect(self, frame: bytes) -> list[ObjectDetection]:
        self.calls.append(frame)
        return list(self._detections)


class _FakeFaces:
    def __init__(self, matches: list[FaceMatch]) -> None:
        self._matches = matches

    def recognize(self, frame: bytes) -> list[FaceMatch]:
        return list(self._matches)


@pytest.mark.asyncio
async def test_process_returns_full_and_reduced_results() -> None:
    yolo = _FakeYolo(detections=[
        ObjectDetection(label="person", confidence=0.9, bbox=(0, 0, 100, 200)),
        ObjectDetection(label="bottle", confidence=0.8, bbox=(150, 50, 200, 100)),
    ])
    faces = _FakeFaces(matches=[
        FaceMatch(name="Anjali", similarity=0.7, bbox=(20, 30, 80, 90)),
    ])
    processor = FrameProcessor(yolo, faces)  # type: ignore[arg-type]

    ts = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    full, reduced = await processor.process(b"fake-jpeg", timestamp=ts)

    # Full view keeps richness for the dashboard
    assert isinstance(full, FullDetectionResult)
    assert len(full.objects) == 2
    assert full.recognized_names == ["Anjali"]
    assert full.person_present is True
    assert "bottle" in full.object_labels

    # Reduced view is what the trigger gate consumes
    assert reduced.timestamp == ts
    assert reduced.person_present is True
    assert reduced.objects == ["person", "bottle"]


@pytest.mark.asyncio
async def test_process_with_no_detections() -> None:
    yolo = _FakeYolo(detections=[])
    faces = _FakeFaces(matches=[])
    processor = FrameProcessor(yolo, faces)  # type: ignore[arg-type]

    full, reduced = await processor.process(b"fake-jpeg")
    assert full.objects == []
    assert full.faces == []
    assert reduced.person_present is False
    assert reduced.objects == []
    # Auto-generated timestamp is timezone-aware (matches DetectionResult contract)
    assert reduced.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_full_result_helpers() -> None:
    full = FullDetectionResult(
        objects=[
            ObjectDetection(label="person", confidence=0.9, bbox=(0, 0, 1, 1)),
            ObjectDetection(label="cup", confidence=0.6, bbox=(0, 0, 1, 1)),
        ],
        faces=[FaceMatch(name="Ravi", similarity=0.6, bbox=(0, 0, 1, 1))],
    )
    assert full.person_present is True
    assert full.object_labels == ["person", "cup"]
    assert full.recognized_names == ["Ravi"]


@pytest.mark.asyncio
async def test_full_result_person_present_false_when_only_objects() -> None:
    full = FullDetectionResult(
        objects=[ObjectDetection(label="cup", confidence=0.7, bbox=(0, 0, 1, 1))],
    )
    assert full.person_present is False
