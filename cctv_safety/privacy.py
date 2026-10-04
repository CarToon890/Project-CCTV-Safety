"""Local face detection and fail-closed face anonymization for web previews."""

from __future__ import annotations

from pathlib import Path
from inspect import signature
from time import perf_counter

import cv2
import numpy as np


MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_FACE_CONFIDENCE = 0.35
DEFAULT_FACE_PADDING = 0.25
DEFAULT_FACE_BLUR_STRENGTH = 0.8
MAX_FACE_GAP_FRAMES = 5


class FaceModelUnavailable(RuntimeError):
    """The local YuNet model is missing or cannot be loaded."""


class YuNetFaceAnonymizer:
    def __init__(
        self,
        weights_dir: Path,
        face_confidence: float = DEFAULT_FACE_CONFIDENCE,
        padding_fraction: float = DEFAULT_FACE_PADDING,
        blur_strength: float = DEFAULT_FACE_BLUR_STRENGTH,
        device: str = "cpu",
    ):
        self.padding_fraction = padding_fraction
        self.blur_strength = blur_strength
        self.device = device
        self.path = Path(weights_dir) / MODEL_FILENAME
        if not self.path.is_file():
            raise FaceModelUnavailable(
                f"Local face model not found: {self.path}. Obtain the OpenCV Zoo YuNet model."
            )
        try:
            args = (str(self.path), "", (320, 320), face_confidence, 0.3, 5000)
            if str(device).startswith("cuda"):
                self.detector = cv2.FaceDetectorYN.create(
                    *args, cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
                )
            else:
                self.detector = cv2.FaceDetectorYN.create(*args)
            self._input_size = None
        except Exception as exc:  # noqa: BLE001
            raise FaceModelUnavailable(f"Could not load local YuNet face model: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        height, width = frame.shape[:2]
        input_size = (width, height)
        if input_size != self._input_size:
            self.detector.setInputSize(input_size)
            self._input_size = input_size
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
            # Coarse pixelation removes facial detail more reliably than blur alone.
            block = max(8, int(round(min(roi.shape[:2]) * 0.16 * blur_strength)))
            small_w = max(1, int(np.ceil(roi.shape[1] / block)))
            small_h = max(1, int(np.ceil(roi.shape[0] / block)))
            tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            output[y1:y2, x1:x2] = cv2.resize(
                tiny, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return output


def anonymize(anonymizer: YuNetFaceAnonymizer, frame: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    boxes = anonymizer.detect(frame)
    strength = getattr(anonymizer, "blur_strength", DEFAULT_FACE_BLUR_STRENGTH)
    blur = anonymizer.blur
    # Keep simple test/custom anonymizers compatible with the original two-argument API.
    if len(signature(blur).parameters) < 3:
        safe = blur(frame, boxes)
    else:
        safe = blur(frame, boxes, strength)
    return safe, boxes


def bridge_short_face_gaps(
    frames: list[list[list[int]]], width: int, height_px: int,
    max_gap: int = MAX_FACE_GAP_FRAMES,
) -> list[list[list[int]]]:
    """Interpolate face boxes across brief all-face detection gaps in offline video.

    This stabilizes the privacy mask during motion blur. It deliberately does not
    invent boxes over long gaps or when endpoint faces cannot be matched safely.
    """
    result = [[box[:] for box in frame] for frame in frames]
    height = len(result)
    index = 0
    while index < height:
        if result[index]:
            index += 1
            continue
        gap_start = index
        while index < height and not result[index]:
            index += 1
        gap_end = index
        gap = gap_end - gap_start
        if gap == 0 or gap > max_gap or gap_start == 0 or gap_end >= height:
            continue
        left, right = result[gap_start - 1], result[gap_end]
        if not left or not right:
            continue

        # Greedy one-to-one matching by center displacement, normalized by face size.
        candidates = []
        for li, a in enumerate(left):
            acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            asize = max(1.0, ((a[2] - a[0]) + (a[3] - a[1])) / 2)
            for ri, b in enumerate(right):
                bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                bsize = max(1.0, ((b[2] - b[0]) + (b[3] - b[1])) / 2)
                scale = max(asize, bsize)
                distance = (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale
                size_ratio = max(asize, bsize) / min(asize, bsize)
                if distance <= 3.5 and size_ratio <= 2.5:
                    candidates.append((distance, li, ri))
        used_left, used_right = set(), set()
        matches = []
        for _, li, ri in sorted(candidates):
            if li not in used_left and ri not in used_right:
                used_left.add(li)
                used_right.add(ri)
                matches.append((left[li], right[ri]))

        for a, b in matches:
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                box = [int(round(av + (bv - av) * ratio)) for av, bv in zip(a, b)]
                # Add a small motion margin to interpolated regions.
                pad_x = max(2, int((box[2] - box[0]) * 0.12))
                pad_y = max(2, int((box[3] - box[1]) * 0.12))
                frames_box = [max(0, box[0] - pad_x), max(0, box[1] - pad_y),
                              min(width, box[2] + pad_x), min(height_px, box[3] + pad_y)]
                result[gap_start + offset - 1].append(frames_box)
    return result


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
