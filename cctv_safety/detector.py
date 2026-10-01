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


def detect(model, frame_bgr: np.ndarray, thresholds: dict) -> list[Detection]:
    """Predict at the minimum threshold, keep boxes passing their class threshold, sort by confidence desc."""
    minimum = min(float(value) for value in thresholds.values())
    names = model.names
    detections: list[Detection] = []
    # Ultralytics' predictor already runs under torch inference mode.
    results = model.predict(frame_bgr, conf=minimum, verbose=False)
    for result in results:
        for box in result.boxes:
            class_name = names[int(box.cls.item())]
            confidence = float(box.conf.item())
            if confidence >= float(thresholds[class_name]):
                detections.append(Detection(class_name, confidence, tuple(float(v) for v in box.xyxy[0].tolist())))
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections
