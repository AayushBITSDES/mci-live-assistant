"""Lightweight OpenCV recognizers for the exhibition path.

These recognizers are intentionally specific-prop matchers, not general
object detectors. Drop a few clear reference photos into storage/reference
and the matcher looks for those exact visual signatures at low FPS.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from detection.types import Bbox, FaceMatch, ObjectDetection

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SpecificObjectRecognizer:
    def __init__(
        self,
        references: dict[str, Path],
        *,
        min_matches: int = 12,
        max_distance: float = 64.0,
        cv2_module: Any | None = None,
    ) -> None:
        self._cv2 = cv2_module or _load_cv2()
        self._orb = self._cv2.ORB_create(nfeatures=700)
        self._matcher = self._cv2.BFMatcher(self._cv2.NORM_HAMMING, crossCheck=True)
        self._min_matches = min_matches
        self._max_distance = max_distance
        self._refs: dict[str, list[Any]] = {}

        for label, folder in references.items():
            descriptors = self._load_reference_descriptors(folder)
            self._refs[label] = descriptors
            logger.info("OpenCV object refs | label=%s | images=%d", label, len(descriptors))

    def detect(self, frame_bytes: bytes) -> list[ObjectDetection]:
        image = _decode_gray(self._cv2, frame_bytes)
        if image is None:
            return []
        _keypoints, frame_desc = self._orb.detectAndCompute(image, None)
        if frame_desc is None:
            return []

        height, width = image.shape[:2]
        out: list[ObjectDetection] = []
        for label, ref_descriptors in self._refs.items():
            best_count = 0
            for ref_desc in ref_descriptors:
                matches = self._matcher.match(ref_desc, frame_desc)
                good = [m for m in matches if m.distance <= self._max_distance]
                best_count = max(best_count, len(good))
            if best_count >= self._min_matches:
                confidence = min(0.99, best_count / max(self._min_matches * 2, 1))
                out.append(ObjectDetection(
                    label=label,
                    confidence=confidence,
                    bbox=(0.0, 0.0, float(width), float(height)),
                ))
        return out

    def _load_reference_descriptors(self, folder: Path) -> list[Any]:
        descriptors: list[Any] = []
        for path in _image_paths(folder):
            image = self._cv2.imread(str(path), self._cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("OpenCV could not read reference image: %s", path)
                continue
            _keypoints, desc = self._orb.detectAndCompute(image, None)
            if desc is not None:
                descriptors.append(desc)
        return descriptors


class OpenCVFaceRecognizer:
    def __init__(
        self,
        faces_root: Path,
        *,
        min_matches: int = 14,
        max_distance: float = 62.0,
        cv2_module: Any | None = None,
    ) -> None:
        self._cv2 = cv2_module or _load_cv2()
        self._orb = self._cv2.ORB_create(nfeatures=700)
        self._matcher = self._cv2.BFMatcher(self._cv2.NORM_HAMMING, crossCheck=True)
        self._min_matches = min_matches
        self._max_distance = max_distance
        self._refs: dict[str, list[Any]] = {}

        for person_dir in sorted(Path(faces_root).iterdir()) if Path(faces_root).is_dir() else []:
            if not person_dir.is_dir():
                continue
            descriptors = self._load_reference_descriptors(person_dir)
            self._refs[person_dir.name] = descriptors
            logger.info("OpenCV face refs | name=%s | images=%d", person_dir.name, len(descriptors))

    def recognize(self, frame_bytes: bytes) -> list[FaceMatch]:
        image = _decode_gray(self._cv2, frame_bytes)
        if image is None:
            return []
        candidates = self._face_regions(image)
        out: list[FaceMatch] = []
        for crop, bbox in candidates:
            _keypoints, face_desc = self._orb.detectAndCompute(crop, None)
            if face_desc is None:
                continue
            best_name = ""
            best_count = 0
            for name, ref_descriptors in self._refs.items():
                for ref_desc in ref_descriptors:
                    matches = self._matcher.match(ref_desc, face_desc)
                    good = [m for m in matches if m.distance <= self._max_distance]
                    if len(good) > best_count:
                        best_name = name
                        best_count = len(good)
            if best_name and best_count >= self._min_matches:
                similarity = min(0.99, best_count / max(self._min_matches * 2, 1))
                out.append(FaceMatch(name=best_name, similarity=similarity, bbox=bbox))
        return out

    def _load_reference_descriptors(self, folder: Path) -> list[Any]:
        descriptors: list[Any] = []
        for path in _image_paths(folder):
            image = self._cv2.imread(str(path), self._cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("OpenCV could not read face reference image: %s", path)
                continue
            for crop, _bbox in self._face_regions(image):
                _keypoints, desc = self._orb.detectAndCompute(crop, None)
                if desc is not None:
                    descriptors.append(desc)
                break
        return descriptors

    def _face_regions(self, gray_image) -> list[tuple[Any, Bbox]]:
        cascade_path = getattr(self._cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        cascade = self._cv2.CascadeClassifier(cascade_path)
        boxes = [] if cascade.empty() else cascade.detectMultiScale(gray_image, 1.1, 5)
        if len(boxes) == 0:
            return []
        regions = []
        for x, y, w, h in boxes:
            regions.append((
                gray_image[y:y + h, x:x + w],
                (float(x), float(y), float(x + w), float(y + h)),
            ))
        return regions


def _load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV exhibition recognizers require opencv-python-headless"
        ) from exc
    return cv2


def _decode_gray(cv2, frame_bytes: bytes):
    import numpy as np  # type: ignore

    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return image


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
