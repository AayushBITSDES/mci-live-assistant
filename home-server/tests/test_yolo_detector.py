"""YoloDetector wrapper tests with a fake Ultralytics model.

We never install ultralytics in this test environment. The wrapper
takes the model object via constructor, so a fake stand-in with the
same shape is enough to exercise the full parsing path.
"""
from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Iterable

import pytest

from detection.types import ObjectDetection
from detection.yolo import YoloDetector


# --- Fake Ultralytics result ----------------------------------------------

class _FakeBoxes:
    """Stand-in for ultralytics.engine.results.Boxes.

    Real `xyxy`/`conf`/`cls` are torch tensors; we use objects that
    expose `.tolist()` so the wrapper's `_to_python_list()` finds them.
    """
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = SimpleNamespace(tolist=lambda: list(xyxy))
        self.conf = SimpleNamespace(tolist=lambda: list(conf))
        self.cls = SimpleNamespace(tolist=lambda: list(cls))


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class _FakeYoloModel:
    """Records calls; returns the configured detections regardless of input."""
    def __init__(self, results: Iterable[_FakeResult]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    def predict(self, image, *, conf, verbose):
        self.calls.append({"conf": conf, "verbose": verbose})
        return self._results


# --- Helpers --------------------------------------------------------------

@pytest.fixture
def jpeg_bytes() -> bytes:
    """Tiny synthetic JPEG so PIL.Image.open() succeeds."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- Tests ---------------------------------------------------------------

def test_detect_parses_single_object(jpeg_bytes: bytes) -> None:
    boxes = _FakeBoxes(xyxy=[[10, 20, 30, 40]], conf=[0.92], cls=[0])
    result = _FakeResult(boxes=boxes, names={0: "person"})
    model = _FakeYoloModel(results=[result])
    yolo = YoloDetector(model=model)

    detections = yolo.detect(jpeg_bytes)
    assert len(detections) == 1
    d = detections[0]
    assert isinstance(d, ObjectDetection)
    assert d.label == "person"
    assert d.confidence == pytest.approx(0.92)
    assert d.bbox == (10.0, 20.0, 30.0, 40.0)


def test_detect_parses_multiple_objects(jpeg_bytes: bytes) -> None:
    boxes = _FakeBoxes(
        xyxy=[[0, 0, 10, 10], [20, 20, 40, 50]],
        conf=[0.6, 0.8],
        cls=[0, 41],
    )
    result = _FakeResult(boxes=boxes, names={0: "person", 41: "cup"})
    yolo = YoloDetector(model=_FakeYoloModel(results=[result]))

    detections = yolo.detect(jpeg_bytes)
    assert {d.label for d in detections} == {"person", "cup"}


def test_unknown_class_id_falls_back_to_string(jpeg_bytes: bytes) -> None:
    boxes = _FakeBoxes(xyxy=[[0, 0, 1, 1]], conf=[0.9], cls=[999])
    result = _FakeResult(boxes=boxes, names={0: "person"})
    yolo = YoloDetector(model=_FakeYoloModel(results=[result]))

    detections = yolo.detect(jpeg_bytes)
    assert detections[0].label == "999"


def test_no_boxes_returns_empty(jpeg_bytes: bytes) -> None:
    result = SimpleNamespace(boxes=None, names={})
    yolo = YoloDetector(model=_FakeYoloModel(results=[result]))
    assert yolo.detect(jpeg_bytes) == []


def test_confidence_threshold_passed_to_model(jpeg_bytes: bytes) -> None:
    model = _FakeYoloModel(results=[])
    yolo = YoloDetector(model=model, confidence_threshold=0.7)
    yolo.detect(jpeg_bytes)
    assert model.calls[-1]["conf"] == 0.7
    assert model.calls[-1]["verbose"] is False


def test_default_threshold_is_half() -> None:
    yolo = YoloDetector(model=_FakeYoloModel(results=[]))
    assert yolo._conf == 0.5
