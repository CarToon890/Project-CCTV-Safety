"""Local face detection and fail-closed face anonymization for web previews."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_FACE_CONFIDENCE = 0.35
DEFAULT_FACE_PADDING = 0.25
DEFAULT_FACE_BLUR_STRENGTH = 0.8


class FaceModelUnavailable(RuntimeError):
    """The local YuNet model is missing or cannot be loaded."""


class YuNetFaceAnonymizer:
    def __init__(
        self,
        weights_dir: Path,
        face_confidence: float = DEFAULT_FACE_CONFIDENCE,
        padding_fraction: float = DEFAULT_FACE_PADDING,
        blur_strength: float = DEFAULT_FACE_BLUR_STRENGTH,
    ):
        self.padding_fraction = padding_fraction
        self.blur_strength = blur_strength
        self.path = Path(weights_dir) / MODEL_FILENAME
        if not self.path.is_file():
            raise FaceModelUnavailable(
                f"Local face model not found: {self.path}. Obtain the OpenCV Zoo YuNet model."
            )
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(self.path), "", (320, 320), face_confidence, 0.3, 5000
            )
        except Exception as exc:  # noqa: BLE001
            raise FaceModelUnavailable(f"Could not load local YuNet face model: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        boxes: list[list[int]] = []
        if faces is None:
            return boxes
        for face in faces:
            x, y, w, h = (float(v) for v in face[:4])
            # Expand around the face so ears, jaw and a small amount of hair are covered.
            pad_x, pad_y = w * self.padding_fraction, h * self.padding_fraction
            x1 = max(0, int(np.floor(x - pad_x)))
            y1 = max(0, int(np.floor(y - pad_y)))
            x2 = min(width, int(np.ceil(x + w + pad_x)))
            y2 = min(height, int(np.ceil(y + h + pad_y)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return boxes

    @staticmethod
    def blur(
        frame: np.ndarray,
        boxes: list[list[int]],
        blur_strength: float = DEFAULT_FACE_BLUR_STRENGTH,
    ) -> np.ndarray:
        """Blur a copy; the array sent to YOLO/X3D is never modified."""
        output = frame.copy()
        height, width = output.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # Strong Gaussian blur; keep kernel odd and proportional to face region.
            k = min(99, max(21, int(round(min(roi.shape[:2]) * blur_strength) | 1)))
            output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return output


def anonymize(anonymizer: YuNetFaceAnonymizer, frame: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    boxes = anonymizer.detect(frame)
    strength = getattr(anonymizer, "blur_strength", DEFAULT_FACE_BLUR_STRENGTH)
    return anonymizer.blur(frame, boxes, strength), boxes


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
