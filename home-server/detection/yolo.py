"""YOLOv10s wrapper around the Ultralytics package.

Lazy-imports `ultralytics` so this module loads without the heavy ML
deps installed. Production: pass `model_path` and let it auto-load.
Tests: pass `model=fake` and skip the import entirely.

GPU is auto-detected by Ultralytics; on Apple Silicon it picks MPS, on
NVIDIA it picks CUDA, otherwise CPU. No code change needed.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

from detection.types import ObjectDetection

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolov10s.pt",
        *,
        confidence_threshold: float = 0.5,
        model: Optional[Any] = None,
    ) -> None:
        self._conf = confidence_threshold
        if model is not None:
            self._model = model
        else:
            self._model = self._load_model(model_path)

    @staticmethod
    def _load_model(path: str) -> Any:
        # Lazy import — heavy dep, only loaded when actually instantiated.
        from ultralytics import YOLO  # type: ignore
        logger.info("Loading YOLO model: %s", path)
        return YOLO(path)

    def detect(self, frame: bytes) -> list[ObjectDetection]:
        """Run inference on a JPEG byte string. Returns filtered detections.

        Ultralytics accepts a PIL Image, file path, numpy array, or bytes
        via BytesIO. We use bytes -> PIL to avoid a numpy dependency in
        this module (PIL is a transitive dep).
        """
        from PIL import Image  # PIL is a Pillow dep, available with the rest

        image = Image.open(io.BytesIO(frame)).convert("RGB")
        results = self._model.predict(image, conf=self._conf, verbose=False)
        return self._parse_results(results)

    @staticmethod
    def _parse_results(results: Any) -> list[ObjectDetection]:
        """Parse Ultralytics output into our dataclass.

        Ultralytics returns a list of Results, each with `.boxes` carrying
        `xyxy`, `conf`, and `cls`. The model's `.names` dict maps class
        ids to label strings.
        """
        detections: list[ObjectDetection] = []
        for r in results:
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            xyxy = _to_python_list(getattr(boxes, "xyxy", []))
            confs = _to_python_list(getattr(boxes, "conf", []))
            classes = _to_python_list(getattr(boxes, "cls", []))

            for box, conf, cls in zip(xyxy, confs, classes):
                label = names.get(int(cls), str(int(cls)))
                detections.append(
                    ObjectDetection(
                        label=label,
                        confidence=float(conf),
                        bbox=tuple(float(v) for v in box),  # type: ignore[arg-type]
                    )
                )
        return detections


def _to_python_list(value: Any) -> list:
    """Coerce a tensor-like (torch tensor, numpy array, list) to a plain list.

    We avoid importing torch/numpy directly so this module loads without them.
    Anything that has `.tolist()` is converted; otherwise we trust it is iterable.
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
