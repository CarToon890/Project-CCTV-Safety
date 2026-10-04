"""Stage 1 YOLOv8 detector: runtime contract, loading and per-class thresholded detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from cctv_safety.ppe import Detection
from cctv_safety.schema import CLASS_NAMES


def _names_list(model_names) -> list[str]:
    if isinstance(model_names, dict):
        return [model_names[key] for key in sorted(model_names)]
    return list(model_names)


def _validate_names(model_names) -> None:
    actual_names = _names_list(model_names)
    expected_names = list(CLASS_NAMES)
    if actual_names != expected_names:
        raise ValueError(f"Detector schema mismatch: expected {expected_names}, got {actual_names}")


def validate_runtime_contract(model_names, thresholds: dict) -> None:
    """Model class names must equal CLASS_NAMES and threshold keys must match them exactly."""
    _validate_names(model_names)
    expected_names = list(CLASS_NAMES)
    if set(thresholds) != set(expected_names):
        raise ValueError(
            "Threshold keys must exactly match detector classes: "
            f"expected {expected_names}, got {sorted(thresholds)}"
        )


def load_thresholds(path="configs/thresholds.yaml") -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: thresholds must be a mapping of class name to confidence")
    return {str(name): float(value) for name, value in data.items()}


def load_yolo(path) -> "YOLO":
    from ultralytics import YOLO

    model = YOLO(str(path))
    _validate_names(model.names)
    return model


def detect(model, frame_bgr: np.ndarray, thresholds: dict, *, iou: float = 0.7,
           imgsz: int = 640, max_det: int = 300) -> list[Detection]:
    """Predict at the minimum threshold, keep boxes passing their class threshold, sort by confidence desc."""
    minimum = min(float(value) for value in thresholds.values())
    names = model.names
    detections: list[Detection] = []
    # Ultralytics' predictor already runs under torch inference mode.
    results = model.predict(frame_bgr, conf=minimum, iou=iou, imgsz=imgsz,
                            max_det=max_det, verbose=False)
    for result in results:
        for box in result.boxes:
            class_name = names[int(box.cls.item())]
            confidence = float(box.conf.item())
            if confidence >= float(thresholds[class_name]):
                detections.append(Detection(class_name, confidence, tuple(float(v) for v in box.xyxy[0].tolist())))
    detections = suppress_frame_spanning_persons(detections, frame_bgr.shape[1], frame_bgr.shape[0])
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def suppress_frame_spanning_persons(
    detections: list[Detection], width: int, height: int
) -> list[Detection]:
    """Drop only implausible full-height person boxes overlapping a fall detection.

    Pilot guard for oversized background boxes seen on fallen-person clips. It does
    not alter fall boxes or ordinary person boxes; retraining/validation is still
    needed for a general person detector.
    """
    if width <= 0 or height <= 0:
        return detections
    falls = [item.xyxy for item in detections if item.class_name == "fall"]
    if not falls:
        return detections

    kept = []
    for item in detections:
        if item.class_name != "person":
            kept.append(item)
            continue
        x1, y1, x2, y2 = item.xyxy
        box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        spans_frame = (
            box_width >= width * 0.45
            and box_height >= height * 0.90
            and y1 <= height * 0.03
            and y2 >= height * 0.97
        )
        overlaps_fall = any(_box_iou(item.xyxy, fall) >= 0.20 for fall in falls)
        if not (spans_frame and overlaps_fall):
            kept.append(item)
    return kept


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
