from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from detection.opencv_recognizers import OpenCVFaceRecognizer, SpecificObjectRecognizer


def _feature_image():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 160), (255, 255, 255), 3)
    cv2.circle(image, (80, 80), 28, (0, 255, 0), -1)
    cv2.line(image, (30, 150), (210, 35), (255, 0, 0), 4)
    cv2.putText(image, "ANYA", (55, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image


def _jpeg_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_specific_object_recognizer_matches_reference_image(tmp_path) -> None:
    ref_dir = tmp_path / "medicine_bottle"
    ref_dir.mkdir()
    cv2.imwrite(str(ref_dir / "bottle.jpg"), _feature_image())

    recognizer = SpecificObjectRecognizer(
        {"medicine_bottle": ref_dir},
        min_matches=4,
        max_distance=80,
    )

    detections = recognizer.detect(_jpeg_bytes(_feature_image()))

    assert [d.label for d in detections] == ["medicine_bottle"]
    assert detections[0].confidence > 0


def test_opencv_face_recognizer_matches_known_person_folder(tmp_path) -> None:
    anya_dir = tmp_path / "faces" / "Anya"
    anya_dir.mkdir(parents=True)
    cv2.imwrite(str(anya_dir / "anya.jpg"), _feature_image())

    recognizer = OpenCVFaceRecognizer(
        tmp_path / "faces",
        min_matches=4,
        max_distance=80,
    )

    matches = recognizer.recognize(_jpeg_bytes(_feature_image()))

    assert [m.name for m in matches] == ["Anya"]
    assert matches[0].similarity > 0
