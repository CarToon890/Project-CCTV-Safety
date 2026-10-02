"""Deterministic, read-only missing-label pilot for the corrected D-Fire dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/dfire_corrected"
OUTPUT = ROOT / "data/processed/dfire_missing_label_pilot"
WEIGHTS = ROOT / "models/weights"
SPLITS = ("train", "val", "test")
SAMPLES_PER_SPLIT = 40
IMGSZ = 640
CONF = 0.25
IOU_MATCH = 0.30
OVERLAY_CAP = 60
WORLD_CLASSES = ["person", "safety helmet", "safety vest"]
CANONICAL_CLASS = {"person": 0, "safety helmet": 1, "safety vest": 2}


def deterministic_sample(items: list[Path], count: int) -> list[Path]:
    """Select evenly spaced paths from a sorted population, including both ends."""
    if len(items) < count:
        raise ValueError(f"Need {count} items, found {len(items)}")
    if count == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def load_labels(image: Path, split: str) -> list[tuple[int, float, float, float, float]]:
    label = DATASET / "labels" / split / f"{image.stem}.txt"
    rows = []
    if not label.exists():
        return rows
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()[:5]
        rows.append((int(cls), float(x), float(y), float(w), float(h)))
    return rows


def xywhn_to_xyxy(box, width: int, height: int):
    _, x, y, w, h = box
    return ((x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_predict(model, paths: list[str], **kwargs):
    synchronize()
    started = time.perf_counter()
    results = model.predict(paths, verbose=False, stream=False, **kwargs)
    synchronize()
    return results, time.perf_counter() - started


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "overlays").mkdir(parents=True)

    manifest = []
    selected = []
    for split in SPLITS:
        paths = sorted((DATASET / "images" / split).glob("*"), key=lambda p: p.name.lower())
        sample = deterministic_sample(paths, SAMPLES_PER_SPLIT)
        for p in sample:
            selected.append((split, p))
            manifest.append({"split": split, "image": p.name, "relative_path": p.relative_to(DATASET).as_posix()})

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    coco = YOLO(WEIGHTS / "yolo11n.pt")
    coco_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    world = YOLOWorld(WEIGHTS / "yolov8s-worldv2.pt")
    world.set_classes(WORLD_CLASSES)
    world_load = time.perf_counter() - t0

    warmup_path = str(selected[0][1])
    _, coco_warmup = timed_predict(coco, [warmup_path], device=0, imgsz=IMGSZ, conf=CONF, half=True)
    _, world_warmup = timed_predict(world, [warmup_path], device=0, imgsz=IMGSZ, conf=CONF, half=True)

    paths = [str(p) for _, p in selected]
    coco_results, coco_time = timed_predict(coco, paths, device=0, imgsz=IMGSZ, conf=CONF, half=True, batch=16)
    world_results, world_time = timed_predict(world, paths, device=0, imgsz=IMGSZ, conf=CONF, half=True, batch=16)

    detections = []
    candidates = []
    candidate_images = set()
    for index, ((split, image_path), coco_result, world_result) in enumerate(zip(selected, coco_results, world_results)):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        labels = load_labels(image_path, split)
        canonical_boxes = {c: [xywhn_to_xyxy(row, width, height) for row in labels if row[0] == c] for c in range(3)}
        image_candidates = []

        model_outputs = [("yolo11n", coco_result, {0: ("person", 0)})]
        world_mapping = {i: (name, CANONICAL_CLASS[name]) for i, name in enumerate(WORLD_CLASSES)}
        model_outputs.append(("yolov8s-worldv2", world_result, world_mapping))
        for model_name, result, mapping in model_outputs:
            if result.boxes is None:
                continue
            for box in result.boxes:
                source_cls = int(box.cls.item())
                if source_cls not in mapping:
                    continue
                class_name, canonical_cls = mapping[source_cls]
                coords = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf.item())
                max_iou = max((iou(coords, existing) for existing in canonical_boxes[canonical_cls]), default=0.0)
                is_candidate = max_iou < IOU_MATCH
                row = {
                    "split": split, "image": image_path.name, "model": model_name,
                    "class_name": class_name, "canonical_class": canonical_cls,
                    "confidence": round(confidence, 6), "max_iou_existing": round(max_iou, 6),
                    "candidate_missing_label": is_candidate,
                    "x1": round(coords[0], 2), "y1": round(coords[1], 2),
                    "x2": round(coords[2], 2), "y2": round(coords[3], 2),
                }
                detections.append(row)
                if is_candidate:
                    candidates.append(row)
                    image_candidates.append(row)
                    candidate_images.add((split, image_path.name))

        if image_candidates and len(list((OUTPUT / "overlays").glob("*.jpg"))) < OVERLAY_CAP:
            for row in image_candidates:
                color = {0: (255, 180, 0), 1: (0, 255, 255), 2: (255, 0, 255)}[row["canonical_class"]]
                p1, p2 = (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"]))
                cv2.rectangle(image, p1, p2, color, 2)
                text = f'{row["model"]}:{row["class_name"]} {row["confidence"]:.2f}'
                cv2.putText(image, text, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(OUTPUT / "overlays" / f"{split}_{image_path.stem}_candidate.jpg"), image)

    fieldnames = ["split", "image", "model", "class_name", "canonical_class", "confidence",
                  "max_iou_existing", "candidate_missing_label", "x1", "y1", "x2", "y2"]
    for filename, rows in (("detections.csv", detections), ("candidates.csv", candidates)):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with (OUTPUT / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "relative_path"])
        writer.writeheader(); writer.writerows(manifest)

    steady_total = coco_time + world_time
    full_count = sum(len(list((DATASET / "images" / split).glob("*"))) for split in SPLITS)
    linear_seconds = steady_total / len(selected) * full_count
    summary = {
        "configuration": {"sample_method": "sorted evenly spaced including endpoints", "images": len(selected),
                          "per_split": SAMPLES_PER_SPLIT, "imgsz": IMGSZ, "confidence": CONF,
                          "iou_existing_match": IOU_MATCH, "device": torch.cuda.get_device_name(0),
                          "models": ["yolo11n.pt", "yolov8s-worldv2.pt"], "world_classes": WORLD_CLASSES},
        "counts": {"detections": len(detections), "candidates": len(candidates),
                   "candidate_images": len(candidate_images),
                   "candidates_by_model_class": dict(Counter(f'{r["model"]}:{r["class_name"]}' for r in candidates)),
                   "overlays": len(list((OUTPUT / "overlays").glob("*.jpg")))},
        "timing_seconds": {"coco_load": coco_load, "world_load_and_set_classes": world_load,
                           "coco_warmup": coco_warmup, "world_warmup": world_warmup,
                           "coco_steady_120": coco_time, "world_steady_120": world_time,
                           "both_models_steady_total": steady_total,
                           "combined_images_per_second": len(selected) / steady_total},
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "full_scan_estimate": {"images": full_count, "linear_seconds": linear_seconds,
                               "cautious_low_seconds": linear_seconds * 1.15,
                               "cautious_high_seconds": linear_seconds * 1.75,
                               "note": "Includes both models; range adds 15%-75% overhead for I/O and variability."},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
