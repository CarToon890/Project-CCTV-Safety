#!/usr/bin/env python
"""Run detector inference and emit derived PPE-compliance results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_safety.ppe import Detection, assess_ppe
from cctv_safety.schema import CLASS_NAMES


def validate_runtime_contract(model: YOLO, thresholds: dict) -> None:
    actual_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    expected_names = list(CLASS_NAMES)
    if actual_names != expected_names:
        raise ValueError(f"Detector schema mismatch: expected {expected_names}, got {actual_names}")
    if set(thresholds) != set(expected_names):
        raise ValueError(
            "Threshold keys must exactly match detector classes: "
            f"expected {expected_names}, got {sorted(thresholds)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("source")
    parser.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.yaml"))
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    model = YOLO(args.weights)
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    validate_runtime_contract(model, thresholds)
    minimum = min(float(value) for value in thresholds.values())
    for result in model.predict(args.source, conf=minimum, stream=True, save=args.save):
        detections = []
        for box in result.boxes:
            class_name = model.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            if confidence >= float(thresholds[class_name]):
                detections.append(Detection(class_name, confidence, tuple(float(value) for value in box.xyxy[0].tolist())))
        print(json.dumps({
            "image": str(result.path),
            "detections": [item.__dict__ for item in detections],
            "ppe_compliance": assess_ppe(detections, min(thresholds["person"], thresholds["helmet"], thresholds["vest"])),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
