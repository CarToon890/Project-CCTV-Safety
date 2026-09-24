#!/usr/bin/env python
"""Train YOLOv8n and YOLOv8s under the same configuration and compare results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import torch
import yaml
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    augmentation = config.get("augmentations", {})
    for model_name in config["models"]:
        run_name = Path(model_name).stem
        model = YOLO(model_name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        model.train(
            data=config["data"], epochs=config["epochs"], imgsz=config["imgsz"],
            batch=config["batch"], patience=config["patience"], seed=config["seed"],
            workers=config["workers"], device=config["device"], project=str(args.output / "runs"),
            name=run_name, exist_ok=True, **augmentation,
        )
        best = YOLO(args.output / "runs" / run_name / "weights" / "best.pt")
        metrics = best.val(data=config["data"], split="test")
        elapsed = perf_counter() - started
        measured = {int(class_id): position for position, class_id in enumerate(metrics.box.ap_class_index)}
        per_class = {}
        for class_id, name in best.names.items():
            position = measured.get(int(class_id))
            if position is None:
                per_class[name] = {"precision": None, "recall": None, "f1": None, "map50_95": None}
                continue
            precision = float(metrics.box.p[position])
            recall = float(metrics.box.r[position])
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "map50_95": float(metrics.box.maps[int(class_id)]),
            }
        export_path = best.export(format="onnx", imgsz=config["imgsz"], dynamic=True)
        inference_ms = float(metrics.speed.get("inference", 0.0))
        row = {
            "model": run_name,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "training_seconds": elapsed,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "inference_ms_per_image": inference_ms,
            "fps": 1000 / inference_ms if inference_ms else 0,
            "weights_mb": (args.output / "runs" / run_name / "weights" / "best.pt").stat().st_size / 1024**2,
            "onnx": str(export_path),
            "per_class": per_class,
        }
        rows.append(row)
        (args.output / f"{run_name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "map50", "map50_95", "training_seconds", "gpu_peak_mb", "inference_ms_per_image", "fps", "weights_mb", "onnx"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "per_class"} for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
