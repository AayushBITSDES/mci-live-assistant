"""Shared detection types.

Kept in `detection/` so vision modules don't depend on `triggers/`.
`triggers/rules.py` already defines its own `DetectionResult` with the
fields it needs; the processor builds one of those from the richer
output here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# (x1, y1, x2, y2) in pixel coordinates
Bbox = Tuple[float, float, float, float]


@dataclass
class ObjectDetection:
    """One object found by YOLO."""
    label: str
    confidence: float
    bbox: Bbox


@dataclass
class FaceMatch:
    """One recognized face. similarity is cosine similarity in [0, 1]."""
    name: str
    similarity: float
    bbox: Bbox


@dataclass
class FullDetectionResult:
    """Rich per-frame output before reduction to triggers.DetectionResult.

    Kept separate so the processor can keep bboxes + confidences for the
    Streamlit live-view overlay even though the rules layer ignores them.
    """
    objects: list[ObjectDetection] = field(default_factory=list)
    faces: list[FaceMatch] = field(default_factory=list)

    @property
    def person_present(self) -> bool:
        return any(o.label == "person" for o in self.objects)

    @property
    def object_labels(self) -> list[str]:
        """Just the labels — what triggers/rules.py expects."""
        return [o.label for o in self.objects]

    @property
    def recognized_names(self) -> list[str]:
        return [f.name for f in self.faces]
