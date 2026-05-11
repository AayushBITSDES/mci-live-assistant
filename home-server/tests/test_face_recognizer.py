"""FaceRecognizer tests with a fake InsightFace app.

Real Pillow + numpy are installed (lightweight). InsightFace itself is
the heavy dependency — that we replace via the `app` constructor arg so
tests never touch ONNX runtimes or model weights.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest


@pytest.fixture
def jpeg_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeApp:
    """Stand-in for insightface.app.FaceAnalysis.

    Each call to .get() returns the next pre-configured frame's faces.
    A face is a SimpleNamespace with `normed_embedding` and `bbox`.
    """
    def __init__(self, frames: Iterable[list[SimpleNamespace]]) -> None:
        self._iter = iter(list(frames))

    def get(self, _img):
        try:
            return next(self._iter)
        except StopIteration:
            return []


def _face(emb: list[float], bbox: tuple[float, float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(normed_embedding=emb, bbox=list(bbox))


# --- Construction --------------------------------------------------------

def test_constructor_loads_empty_store_when_file_absent(tmp_path: Path) -> None:
    from detection.faces import FaceRecognizer
    store = tmp_path / "embeddings.json"
    rec = FaceRecognizer(store_path=store, app=_FakeApp(frames=[]))
    assert rec.known_names() == []


def test_constructor_loads_existing_store(tmp_path: Path) -> None:
    from detection.faces import FaceRecognizer
    store = tmp_path / "embeddings.json"
    store.write_text(json.dumps({"Anjali": [[1.0, 0.0, 0.0]]}))
    rec = FaceRecognizer(store_path=store, app=_FakeApp(frames=[]))
    assert rec.known_names() == ["Anjali"]


def test_corrupt_store_falls_back_to_empty(tmp_path: Path) -> None:
    from detection.faces import FaceRecognizer
    store = tmp_path / "embeddings.json"
    store.write_text("{not json")
    rec = FaceRecognizer(store_path=store, app=_FakeApp(frames=[]))
    assert rec.known_names() == []


# --- Cosine similarity ----------------------------------------------------

def test_cosine_identity_is_one() -> None:
    from detection.faces import _cosine
    v = [0.6, 0.8]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    from detection.faces import _cosine
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_mismatched_lengths() -> None:
    from detection.faces import _cosine
    assert _cosine([1.0, 0.0], [1.0]) == -1.0


# --- recognize() end to end ---------------------------------------------

def test_recognize_matches_known_name(tmp_path: Path, jpeg_bytes: bytes) -> None:
    from detection.faces import FaceRecognizer

    store = tmp_path / "embeddings.json"
    store.write_text(json.dumps({"Anjali": [[1.0, 0.0, 0.0]]}))

    # The frame contains one face whose embedding equals Anjali's reference
    app = _FakeApp(frames=[[_face([1.0, 0.0, 0.0], (10, 10, 50, 60))]])
    rec = FaceRecognizer(store_path=store, similarity_threshold=0.9, app=app)

    matches = rec.recognize(jpeg_bytes)
    assert len(matches) == 1
    assert matches[0].name == "Anjali"
    assert matches[0].similarity == pytest.approx(1.0)
    assert matches[0].bbox == (10.0, 10.0, 50.0, 60.0)


def test_recognize_drops_below_threshold(tmp_path: Path, jpeg_bytes: bytes) -> None:
    from detection.faces import FaceRecognizer

    store = tmp_path / "embeddings.json"
    store.write_text(json.dumps({"Anjali": [[1.0, 0.0, 0.0]]}))

    # Orthogonal embedding -> similarity 0 -> below 0.45 threshold
    app = _FakeApp(frames=[[_face([0.0, 1.0, 0.0], (0, 0, 1, 1))]])
    rec = FaceRecognizer(store_path=store, similarity_threshold=0.45, app=app)
    assert rec.recognize(jpeg_bytes) == []


def test_recognize_with_empty_store_returns_empty(tmp_path: Path, jpeg_bytes: bytes) -> None:
    from detection.faces import FaceRecognizer

    app = _FakeApp(frames=[[_face([1.0, 0.0, 0.0], (0, 0, 1, 1))]])
    rec = FaceRecognizer(store_path=tmp_path / "embeddings.json", app=app)
    assert rec.recognize(jpeg_bytes) == []


# --- add_face() saves to disk -------------------------------------------

def test_add_face_persists_embedding(tmp_path: Path, jpeg_bytes: bytes) -> None:
    from detection.faces import FaceRecognizer

    photo = tmp_path / "anjali.jpg"
    photo.write_bytes(jpeg_bytes)

    store = tmp_path / "embeddings.json"
    app = _FakeApp(frames=[[_face([0.5, 0.5, 0.5], (0, 0, 100, 100))]])
    rec = FaceRecognizer(store_path=store, app=app)

    added = rec.add_face("Anjali", [photo])
    assert added == 1
    saved = json.loads(store.read_text())
    assert "Anjali" in saved
    assert len(saved["Anjali"]) == 1
    assert saved["Anjali"][0] == [0.5, 0.5, 0.5]


def test_add_face_skips_photo_without_face(tmp_path: Path, jpeg_bytes: bytes) -> None:
    from detection.faces import FaceRecognizer

    photo = tmp_path / "blank.jpg"
    photo.write_bytes(jpeg_bytes)

    store = tmp_path / "embeddings.json"
    app = _FakeApp(frames=[[]])  # no faces detected
    rec = FaceRecognizer(store_path=store, app=app)
    assert rec.add_face("Ravi", [photo]) == 0
    saved = json.loads(store.read_text())
    assert saved == {"Ravi": []}
