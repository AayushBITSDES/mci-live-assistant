"""InsightFace wrapper with a simple JSON-based embedding store.

Embeddings are stored in a JSON file: { name: [[float, float, ...], ...] }
where each person can have several reference embeddings (one per photo).
Recognition uses cosine similarity against every stored embedding for
that name and keeps the best match.

JSON instead of pickle: safe to load, human-readable, no exec risk.
A typical embedding is 512 floats (~7KB); a few thousand faces fit in
under a megabyte.
"""
from __future__ import annotations

import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from detection.types import Bbox, FaceMatch

logger = logging.getLogger(__name__)


# Default cosine-similarity threshold. Buffalo_l embeddings normalised to
# unit length give cosine similarity in roughly [-1, 1]; 0.45 is a
# reasonable starting point for "this is the same person."
DEFAULT_SIMILARITY_THRESHOLD = 0.45


class FaceRecognizer:
    def __init__(
        self,
        store_path: Path,
        *,
        model_name: str = "buffalo_l",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        app: Optional[Any] = None,
    ) -> None:
        self._store_path = Path(store_path)
        self._threshold = similarity_threshold
        self._embeddings: dict[str, list[list[float]]] = self._load_store()
        if app is not None:
            self._app = app
        else:
            self._app = self._init_insightface(model_name)

    # --- Init helpers ------------------------------------------------------

    @staticmethod
    def _init_insightface(model_name: str) -> Any:
        from insightface.app import FaceAnalysis  # type: ignore
        logger.info("Initialising InsightFace model: %s", model_name)
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=0)
        return app

    def _load_store(self) -> dict[str, list[list[float]]]:
        if not self._store_path.exists():
            return {}
        try:
            with self._store_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Embeddings store has unexpected shape; starting empty")
                return {}
            # Coerce all values to plain list-of-list-of-float
            return {
                str(k): [list(map(float, v)) for v in vs]
                for k, vs in data.items()
                if isinstance(vs, list)
            }
        except (json.JSONDecodeError, OSError) as exc:
            logger.exception("Failed to load embeddings store: %s", exc)
            return {}

    def save_store(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with self._store_path.open("w", encoding="utf-8") as f:
            json.dump(self._embeddings, f)

    # --- Public API --------------------------------------------------------

    def known_names(self) -> list[str]:
        return sorted(self._embeddings.keys())

    def add_face(self, name: str, image_paths: list[Path]) -> int:
        """Embed every photo for `name` and persist to disk.

        Returns the number of embeddings successfully added (faces detected).
        Skips images where no face is found.
        """
        added = 0
        bucket = self._embeddings.setdefault(name, [])
        for path in image_paths:
            with Path(path).open("rb") as f:
                emb = self._embed_first_face(f.read())
            if emb is None:
                logger.warning("No face detected in %s — skipping", path)
                continue
            bucket.append(emb)
            added += 1
        self.save_store()
        return added

    def recognize(self, frame: bytes) -> list[FaceMatch]:
        """Return all faces in the frame whose best match passes threshold."""
        faces = self._extract_faces(frame)
        matches: list[FaceMatch] = []
        for face_emb, bbox in faces:
            best_name: Optional[str] = None
            best_score = -1.0
            for name, ref_embs in self._embeddings.items():
                for ref in ref_embs:
                    score = _cosine(face_emb, ref)
                    if score > best_score:
                        best_score = score
                        best_name = name
            if best_name is not None and best_score >= self._threshold:
                matches.append(FaceMatch(name=best_name, similarity=best_score, bbox=bbox))
        return matches

    # --- InsightFace plumbing ---------------------------------------------

    def _embed_first_face(self, image_bytes: bytes) -> Optional[list[float]]:
        """Embed the largest face in a single image; return None if no face."""
        faces = self._extract_faces(image_bytes)
        if not faces:
            return None
        # Pick the largest by bounding-box area (most prominent subject)
        faces.sort(key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]), reverse=True)
        return faces[0][0]

    def _extract_faces(self, image_bytes: bytes) -> list[tuple[list[float], Bbox]]:
        """Run InsightFace on bytes; return (embedding, bbox) per face."""
        from PIL import Image
        import numpy as np  # type: ignore

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(pil)
        # InsightFace expects BGR (OpenCV convention)
        bgr = rgb[:, :, ::-1].copy()
        faces = self._app.get(bgr)

        out: list[tuple[list[float], Bbox]] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            bbox_arr = list(getattr(f, "bbox", []))
            bbox: Bbox = (
                float(bbox_arr[0]) if len(bbox_arr) > 0 else 0.0,
                float(bbox_arr[1]) if len(bbox_arr) > 1 else 0.0,
                float(bbox_arr[2]) if len(bbox_arr) > 2 else 0.0,
                float(bbox_arr[3]) if len(bbox_arr) > 3 else 0.0,
            )
            out.append((list(_to_list(emb)), bbox))
        return out


# --- Math helpers ---------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Pure Python so this module has no hard numpy dependency at import time.
    Acceptable cost: a typical embedding is 512 floats — microseconds.
    """
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def _to_list(value: Any) -> list:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
