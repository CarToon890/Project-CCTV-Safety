"""Deterministic full missing-label candidate scan over D-Fire corrected dataset.

Runs YOLO11n (person only) and YOLOv8s-WorldV2 (safety helmet and safety vest only)
on CUDA device 0 over all 21,527 images in data/processed/dfire_corrected.
Performs:
- Environment and CUDA verification.
- Verification of existing yolo11n.pt (person) and yolov8s-worldv2.pt in models/weights.
- Full dataset integrity snapshot (hashes/sizes) before inference.
- Batch inference on CUDA device 0 (no training).
- Canonical ground-truth IoU matching (threshold 0.30) against classes 0/1/2.
- Same-class proposal deduplication via documented Non-Maximum Suppression (NMS 0.45).
- Candidate tier classification (HIGH >= 0.70, MEDIUM 0.40-0.70, LOW 0.20-0.40).
- Priority Human QA queue generation.
- Full QA overlays for 100% of HIGH candidate images, plus capped samples for lower tiers.
- Post-run integrity verification ensuring exactly 21,527 images processed and 0 label/raw mutations.
- Human QA workload estimation with explicit assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/dfire_corrected"
OUTPUT = ROOT / "data/processed/dfire_missing_label_full_scan"
WEIGHTS = ROOT / "models/weights"

SPLITS = ("train", "val", "test")
EXPECTED_TOTAL_IMAGES = 21527
IMGSZ = 640
BATCH_SIZE = 16
CHUNK_SIZE = 512
CONF_PREDICT = 0.20
IOU_MATCH_THRESH = 0.30
NMS_IOU_THRESH = 0.45

# Overlay generation limits
OVERLAY_CAP_HIGH = 100000  # Practical infinity: overlay for every HIGH candidate image
OVERLAY_CAP_MEDIUM = 300   # Cap lower tier sample
OVERLAY_CAP_LOW = 200      # Cap lower tier sample

CANONICAL_NAMES = {0: "person", 1: "helmet", 2: "vest"}
WORLD_CLASSES = ["safety helmet", "safety vest"]
WORLD_TO_CANONICAL = {0: (1, "helmet"), 1: (2, "vest")}

# BGR Colors for visualization
CANONICAL_COLORS = {
    0: (0, 165, 255),   # person: Amber / Orange
    1: (0, 255, 255),   # helmet: Yellow
    2: (255, 0, 255),   # vest: Magenta
}
GT_COLORS = {
    0: (0, 255, 0),     # GT person: Green
    1: (0, 255, 0),     # GT helmet: Green
    2: (0, 255, 0),     # GT vest: Green
    4: (0, 0, 255),     # GT fire: Red
    5: (180, 180, 180), # GT smoke: Gray
}


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_yolo11n(weights_dir: Path) -> Path:
    """Ensure existing yolo11n.pt is present in models/weights; fail clearly if missing."""
    target = weights_dir / "yolo11n.pt"
    if not target.exists():
        raise FileNotFoundError(
            f"Required person candidate model weights missing: {target}. "
            "Expected existing models/weights/yolo11n.pt but file was not found. "
            "Automated downloading of unrelated weights is disabled."
        )
    if target.stat().st_size < 1_000_000:
        raise ValueError(
            f"Required model weights at {target} appear invalid or corrupted ({target.stat().st_size} bytes)."
        )
    print(f"[Setup] yolo11n.pt verified at {target} ({target.stat().st_size} bytes)")
    return target


def classify_tier(confidence: float) -> str:
    """Documented candidate confidence ranking: HIGH / MEDIUM / LOW."""
    if confidence >= 0.70:
        return "HIGH"
    elif confidence >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Standard Intersection over Union (IoU) calculation."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_canonical_labels(
    label_path: Path, width: int, height: int
) -> tuple[dict[int, list[tuple[float, float, float, float]]], list[tuple[int, tuple[float, float, float, float]]]]:
    """Loads existing YOLO labels and converts normalized xywh to pixel xyxy boxes.

    Returns:
      canonical_boxes: mapping {0: [xyxy], 1: [xyxy], 2: [xyxy]} for IoU matching.
      all_boxes: all annotations [(class_id, xyxy), ...] for visualization.
    """
    canonical_boxes = {0: [], 1: [], 2: []}
    all_boxes = []
    if not label_path.exists():
        return canonical_boxes, all_boxes

    try:
        content = label_path.read_text(encoding="utf-8")
    except Exception:
        return canonical_boxes, all_boxes

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0.0, (xc - w / 2.0) * width)
        y1 = max(0.0, (yc - h / 2.0) * height)
        x2 = min(float(width), (xc + w / 2.0) * width)
        y2 = min(float(height), (yc + h / 2.0) * height)
        box = (x1, y1, x2, y2)
        all_boxes.append((cls_id, box))
        if cls_id in canonical_boxes:
            canonical_boxes[cls_id].append(box)
    return canonical_boxes, all_boxes


def nms_deduplicate(proposals: list[dict], iou_thresh: float = NMS_IOU_THRESH) -> tuple[list[dict], int]:
    """Deduplicates same-class overlapping proposals on an image using Non-Maximum Suppression (NMS).

    Algorithm:
    1. Group proposals by canonical_class (person=0, helmet=1, vest=2).
    2. Within each canonical class, sort candidates descending by raw confidence.
    3. Greedily select the highest confidence candidate and suppress any subsequent
       proposal of that same class whose IoU >= iou_thresh.
    4. Returns retained candidates and count of suppressed duplicates.
    """
    if not proposals:
        return [], 0

    by_class: dict[int, list[dict]] = defaultdict(list)
    for p in proposals:
        by_class[p["canonical_class"]].append(p)

    retained: list[dict] = []
    suppressed_count = 0

    for cls_id in (0, 1, 2):
        cls_items = by_class.get(cls_id, [])
        if not cls_items:
            continue
        # Sort descending by raw confidence
        cls_sorted = sorted(cls_items, key=lambda x: x["confidence"], reverse=True)
        kept_for_class: list[dict] = []
        for cand in cls_sorted:
            box_a = (cand["x1"], cand["y1"], cand["x2"], cand["y2"])
            is_suppressed = False
            for kept in kept_for_class:
                box_b = (kept["x1"], kept["y1"], kept["x2"], kept["y2"])
                if compute_iou(box_a, box_b) >= iou_thresh:
                    is_suppressed = True
                    break
            if is_suppressed:
                suppressed_count += 1
            else:
                kept_for_class.append(cand)
        retained.extend(kept_for_class)

    return retained, suppressed_count


def draw_overlay(
    image_path: Path,
    candidates: list[dict],
    gt_boxes: list[tuple[int, tuple[float, float, float, float]]],
    output_path: Path,
):
    """Generates and writes visual QA overlay with candidate & ground-truth annotations."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    height, width = image.shape[:2]

    # Draw existing GT boxes first (thin, dashed/distinguishable)
    for cls_id, box in gt_boxes:
        color = GT_COLORS.get(cls_id, (120, 120, 120))
        gx1, gy1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
        gx2, gy2 = min(width - 1, int(round(box[2]))), min(height - 1, int(round(box[3])))
        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), color, 1, cv2.LINE_AA)
        label_name = {0: "GT:person", 1: "GT:helmet", 2: "GT:vest", 4: "GT:fire", 5: "GT:smoke"}.get(
            cls_id, f"GT:{cls_id}"
        )
        cv2.putText(image, label_name, (gx1, max(12, gy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Draw candidate missing-label boxes (thick border + labeled banner)
    for c in candidates:
        color = CANONICAL_COLORS.get(c["canonical_class"], (0, 255, 0))
        x1, y1 = max(0, int(round(c["x1"]))), max(0, int(round(c["y1"])))
        x2, y2 = min(width - 1, int(round(c["x2"]))), min(height - 1, int(round(c["y2"])))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        text = f"[{c['tier']}] {c['class_name']} {c['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1 - 4)
        # Background box for label text
        cv2.rectangle(image, (x1, ty - th - 3), (x1 + tw + 4, ty + baseline - 1), color, -1)
        # Black text on bright background
        cv2.putText(image, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def calculate_qa_workload(candidate_counts_by_tier: dict[str, int]) -> dict:
    """Calculates Human QA workload estimate with stated assumptions.

    Assumptions:
    - High-tier items: Highly distinct candidates, verification & bbox check ~10s per item.
    - Medium-tier items: Ambiguous / occluded context review ~15s per item.
    - Low-tier items: Borderline proposals review ~15s per item.
    - Effective review pace: 50 productive minutes per hour (accounting for fatigue/breaks).
    - Standard working day: 8 hours (400 effective productive minutes / day).
    """
    high_count = candidate_counts_by_tier.get("HIGH", 0)
    med_count = candidate_counts_by_tier.get("MEDIUM", 0)
    low_count = candidate_counts_by_tier.get("LOW", 0)

    # Hours = (count * seconds_per_item) / (50 min/hr * 60 sec/min)
    high_hours = (high_count * 10.0) / 3000.0
    med_hours = (med_count * 15.0) / 3000.0
    low_hours = (low_count * 15.0) / 3000.0
    total_hours = high_hours + med_hours + low_hours

    high_days = high_hours / 8.0
    med_days = med_hours / 8.0
    low_days = low_hours / 8.0
    total_days = total_hours / 8.0

    return {
        "assumptions": {
            "high_tier_seconds_per_item": 10,
            "medium_tier_seconds_per_item": 15,
            "low_tier_seconds_per_item": 15,
            "effective_minutes_per_hour": 50,
            "standard_hours_per_day": 8,
            "hourly_review_rates": {
                "HIGH": 300,
                "MEDIUM": 200,
                "LOW": 200,
            },
        },
        "estimated_hours": {
            "HIGH": round(high_hours, 2),
            "MEDIUM": round(med_hours, 2),
            "LOW": round(low_hours, 2),
            "total": round(total_hours, 2),
        },
        "estimated_person_days": {
            "HIGH": round(high_days, 2),
            "MEDIUM": round(med_days, 2),
            "LOW": round(low_days, 2),
            "total": round(total_days, 2),
        },
        "phased_triage_plan": {
            "phase_1_immediate_roi": f"Review 100% of HIGH tier candidates ({high_count} items, ~{round(high_hours, 1)} hrs / {round(high_days, 1)} days)",
            "phase_2_targeted_ppe": f"Review 100% of safety PPE (helmet/vest) in MEDIUM tier and stratified sample of person",
            "phase_3_quality_audit": "Audit 5% random sample of LOW tier candidates to evaluate false positive rate and model noise boundary",
        },
    }


def main():
    t_start_total = time.perf_counter()
    print("=" * 80)
    print("D-FIRE CORRECTED: FULL MISSING-LABEL CANDIDATE SCAN")
    print("=" * 80)

    # 1. CUDA & Environment Verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on device 0, but torch.cuda.is_available() is False!")
    device_name = torch.cuda.get_device_name(0)
    print(f"[Device] Using CUDA Device 0: {device_name}")
    torch.cuda.reset_peak_memory_stats(0)

    # 2. Output directory preparation (safely replace any old/partial output)
    if OUTPUT.exists():
        print(f"[Output] Cleaning previous/partial output directory at {OUTPUT}...")
        try:
            shutil.rmtree(OUTPUT)
        except Exception:
            time.sleep(0.2)
            shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "overlays").mkdir(parents=True, exist_ok=True)

    # 3. Model Weights Verification
    yolo11n_path = ensure_yolo11n(WEIGHTS)
    worldv2_path = WEIGHTS / "yolov8s-worldv2.pt"
    if not worldv2_path.exists():
        raise FileNotFoundError(f"Required model weights missing: {worldv2_path}")

    # 4. Dataset Discovery & Pre-Scan Integrity Snapshot
    print(f"[Dataset] Scanning images and labels in {DATASET}...")
    all_images: list[tuple[str, Path]] = []
    initial_label_hashes: dict[str, str] = {}
    initial_image_stats: dict[str, tuple[int, int]] = {}

    for split in SPLITS:
        split_img_dir = DATASET / "images" / split
        split_lbl_dir = DATASET / "labels" / split
        if not split_img_dir.is_dir() or not split_lbl_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_img_dir} or {split_lbl_dir}")

        split_images = sorted(
            [p for p in split_img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
            key=lambda p: p.name.lower(),
        )
        for img_path in split_images:
            all_images.append((split, img_path))
            rel_img = str(img_path.relative_to(DATASET))
            initial_image_stats[rel_img] = (img_path.stat().st_size, img_path.stat().st_mtime_ns)

        split_labels = sorted([p for p in split_lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
        for lbl_path in split_labels:
            rel_lbl = str(lbl_path.relative_to(DATASET))
            initial_label_hashes[rel_lbl] = hashlib.sha256(lbl_path.read_bytes()).hexdigest()

    total_images_found = len(all_images)
    print(f"[Dataset] Found exactly {total_images_found} images across splits: "
          f"{Counter(split for split, _ in all_images)}")
    if total_images_found != EXPECTED_TOTAL_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_IMAGES} images, but discovered {total_images_found}!")
    print(f"[Dataset] Pre-scan integrity snapshot completed: {len(initial_label_hashes)} label files hashed.")

    # 5. Model Loading & Setup
    print("[Models] Loading models into memory...")
    t_load_start = time.perf_counter()
    yolo11n = YOLO(str(yolo11n_path))
    world = YOLOWorld(str(worldv2_path))
    world.set_classes(WORLD_CLASSES)
    synchronize()
    model_load_time = time.perf_counter() - t_load_start
    print(f"[Models] Models loaded in {model_load_time:.2f}s: yolo11n (person only), yolov8s-worldv2 ({WORLD_CLASSES})")

    # 6. GPU Warmup
    print("[Warmup] Warming up CUDA execution graphs...")
    t_warmup_start = time.perf_counter()
    warmup_sample = str(all_images[0][1])
    yolo11n.predict([warmup_sample], classes=[0], imgsz=IMGSZ, conf=CONF_PREDICT, half=True, verbose=False, device=0)
    world.predict([warmup_sample], imgsz=IMGSZ, conf=CONF_PREDICT, half=True, verbose=False, device=0)
    synchronize()
    warmup_time = time.perf_counter() - t_warmup_start
    print(f"[Warmup] Warmup completed in {warmup_time:.2f}s")

    # 7. Full Missing-Label Scan
    print("[Scan] Commencing full missing-label inference scan...")
    all_candidates: list[dict] = []
    manifest_rows: list[dict] = []
    pure_gpu_inference_time = 0.0
    postprocessing_time = 0.0

    raw_proposals_total = 0
    nms_suppressed_total = 0

    overlays_count_high = 0
    overlays_count_med = 0
    overlays_count_low = 0

    candidate_images_set = set()
    num_chunks = (total_images_found + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        chunk_slice = all_images[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        chunk_paths = [str(p) for _, p in chunk_slice]

        # A. GPU Inference
        synchronize()
        t_inf_start = time.perf_counter()
        # yolo11n: person only
        yolo_results = yolo11n.predict(
            chunk_paths,
            classes=[0],
            imgsz=IMGSZ,
            conf=CONF_PREDICT,
            half=True,
            batch=BATCH_SIZE,
            verbose=False,
            device=0,
        )
        # yolov8s-worldv2: safety helmet and safety vest only
        world_results = world.predict(
            chunk_paths,
            imgsz=IMGSZ,
            conf=CONF_PREDICT,
            half=True,
            batch=BATCH_SIZE,
            verbose=False,
            device=0,
        )
        synchronize()
        chunk_inference_duration = time.perf_counter() - t_inf_start
        pure_gpu_inference_time += chunk_inference_duration

        # B. Postprocessing, IoU Matching & Deduplication
        t_post_start = time.perf_counter()
        for (split, img_path), yolo_res, world_res in zip(chunk_slice, yolo_results, world_results):
            height, width = yolo_res.orig_shape
            label_file = DATASET / "labels" / split / f"{img_path.stem}.txt"
            canonical_boxes, all_gt_boxes = load_canonical_labels(label_file, width, height)

            image_raw_proposals: list[dict] = []

            # 1. Collect person detections from YOLO11n
            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                for box in yolo_res.boxes:
                    cls_id = int(box.cls.item())
                    if cls_id != 0:
                        continue
                    coords = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf.item())
                    # IoU match against existing canonical person boxes (class 0)
                    existing_boxes = canonical_boxes[0]
                    max_iou = max((compute_iou(tuple(coords), ex) for ex in existing_boxes), default=0.0)
                    if max_iou < IOU_MATCH_THRESH:
                        image_raw_proposals.append({
                            "split": split,
                            "image": img_path.name,
                            "relative_path": str(img_path.relative_to(DATASET)).replace("\\", "/"),
                            "model": "yolo11n",
                            "class_name": "person",
                            "canonical_class": 0,
                            "confidence": round(conf, 6),
                            "tier": classify_tier(conf),
                            "max_iou_existing": round(max_iou, 6),
                            "x1": round(coords[0], 2),
                            "y1": round(coords[1], 2),
                            "x2": round(coords[2], 2),
                            "y2": round(coords[3], 2),
                        })

            # 2. Collect safety helmet & safety vest detections from YOLOv8s-WorldV2
            if world_res.boxes is not None and len(world_res.boxes) > 0:
                for box in world_res.boxes:
                    source_cls = int(box.cls.item())
                    if source_cls not in WORLD_TO_CANONICAL:
                        continue
                    canonical_cls, class_name = WORLD_TO_CANONICAL[source_cls]
                    coords = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf.item())
                    # IoU match against existing canonical boxes for this class
                    existing_boxes = canonical_boxes[canonical_cls]
                    max_iou = max((compute_iou(tuple(coords), ex) for ex in existing_boxes), default=0.0)
                    if max_iou < IOU_MATCH_THRESH:
                        image_raw_proposals.append({
                            "split": split,
                            "image": img_path.name,
                            "relative_path": str(img_path.relative_to(DATASET)).replace("\\", "/"),
                            "model": "yolov8s-worldv2",
                            "class_name": class_name,
                            "canonical_class": canonical_cls,
                            "confidence": round(conf, 6),
                            "tier": classify_tier(conf),
                            "max_iou_existing": round(max_iou, 6),
                            "x1": round(coords[0], 2),
                            "y1": round(coords[1], 2),
                            "x2": round(coords[2], 2),
                            "y2": round(coords[3], 2),
                        })

            raw_proposals_total += len(image_raw_proposals)

            # 3. Deduplicate same-class overlapping proposals via documented NMS
            deduped_candidates, suppressed_count = nms_deduplicate(image_raw_proposals, iou_thresh=NMS_IOU_THRESH)
            nms_suppressed_total += suppressed_count

            # 4. Overlays & Image Tracking
            has_overlay = False
            overlay_rel_path = ""
            highest_tier = "NONE"

            if deduped_candidates:
                candidate_images_set.add((split, img_path.name))
                for c in deduped_candidates:
                    all_candidates.append(c)

                # Determine highest tier present on this image
                if any(c["tier"] == "HIGH" for c in deduped_candidates):
                    highest_tier = "HIGH"
                elif any(c["tier"] == "MEDIUM" for c in deduped_candidates):
                    highest_tier = "MEDIUM"
                else:
                    highest_tier = "LOW"

                # Check overlay quota
                should_make_overlay = False
                if highest_tier == "HIGH" and overlays_count_high < OVERLAY_CAP_HIGH:
                    should_make_overlay = True
                    overlays_count_high += 1
                elif highest_tier == "MEDIUM" and overlays_count_med < OVERLAY_CAP_MEDIUM:
                    should_make_overlay = True
                    overlays_count_med += 1
                elif highest_tier == "LOW" and overlays_count_low < OVERLAY_CAP_LOW:
                    should_make_overlay = True
                    overlays_count_low += 1

                if should_make_overlay:
                    overlay_file = f"{highest_tier}_{split}_{img_path.stem}.jpg"
                    overlay_full = OUTPUT / "overlays" / overlay_file
                    draw_overlay(img_path, deduped_candidates, all_gt_boxes, overlay_full)
                    has_overlay = True
                    overlay_rel_path = f"overlays/{overlay_file}"

            manifest_rows.append({
                "split": split,
                "image": img_path.name,
                "relative_path": str(img_path.relative_to(DATASET)).replace("\\", "/"),
                "status": "processed",
                "candidates_count": len(deduped_candidates),
                "highest_tier": highest_tier,
                "has_overlay": has_overlay,
                "overlay_file": overlay_rel_path,
            })

        postprocessing_time += (time.perf_counter() - t_post_start)

        # Free chunk memory
        del yolo_results
        del world_results

        # Progress reporting
        processed_so_far = min(total_images_found, (chunk_idx + 1) * CHUNK_SIZE)
        elapsed_so_far = time.perf_counter() - t_start_total
        fps_so_far = processed_so_far / max(0.001, elapsed_so_far)
        print(f"[Progress] Chunk {chunk_idx + 1}/{num_chunks}: {processed_so_far}/{total_images_found} images "
              f"({processed_so_far / total_images_found * 100:.1f}%) | "
              f"Candidates so far: {len(all_candidates)} | "
              f"Speed: {fps_so_far:.1f} img/s | "
              f"Elapsed: {elapsed_so_far:.1f}s")

    # 8. Assign candidate IDs & priority ranks for QA Queue
    print("[Post-Scan] Formatting Human QA Queue and saving CSV/JSON outputs...")
    # Sort candidates for QA Queue: HIGH first, then MEDIUM, then LOW. Within tier: descending confidence.
    tier_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_candidates.sort(key=lambda x: (tier_order[x["tier"]], -x["confidence"], x["split"], x["image"]))

    for idx, cand in enumerate(all_candidates, 1):
        cand["candidate_id"] = f"CAND_{idx:06d}"

    # Build human QA queue
    # Build overlay mapping from manifest
    overlay_by_image = {(r["split"], r["image"]): (r["has_overlay"], r["overlay_file"]) for r in manifest_rows}
    qa_queue_rows: list[dict] = []
    for rank, cand in enumerate(all_candidates, 1):
        has_overlay, overlay_file = overlay_by_image.get((cand["split"], cand["image"]), (False, ""))
        qa_queue_rows.append({
            "queue_rank": rank,
            "candidate_id": cand["candidate_id"],
            "tier": cand["tier"],
            "split": cand["split"],
            "image": cand["image"],
            "class_name": cand["class_name"],
            "canonical_class": cand["canonical_class"],
            "confidence": cand["confidence"],
            "max_iou_existing": cand["max_iou_existing"],
            "x1": cand["x1"],
            "y1": cand["y1"],
            "x2": cand["x2"],
            "y2": cand["y2"],
            "has_overlay": has_overlay,
            "overlay_file": overlay_file,
        })

    # Save candidates.csv
    candidate_fieldnames = [
        "candidate_id", "split", "image", "relative_path", "model", "class_name",
        "canonical_class", "confidence", "tier", "max_iou_existing", "x1", "y1", "x2", "y2"
    ]
    with (OUTPUT / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fieldnames)
        writer.writeheader()
        writer.writerows(all_candidates)

    # Save human_qa_queue.csv
    qa_fieldnames = [
        "queue_rank", "candidate_id", "tier", "split", "image", "class_name",
        "canonical_class", "confidence", "max_iou_existing", "x1", "y1", "x2", "y2",
        "has_overlay", "overlay_file"
    ]
    with (OUTPUT / "human_qa_queue.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=qa_fieldnames)
        writer.writeheader()
        writer.writerows(qa_queue_rows)

    # Save scan_manifest.csv
    with (OUTPUT / "scan_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "relative_path", "status", "candidates_count", "highest_tier", "has_overlay", "overlay_file"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 9. Verification of Zero Dataset Mutation
    print("[Integrity] Verifying zero label and image mutations...")
    mutated_labels: list[str] = []
    mutated_images: list[str] = []

    # Check labels
    for split in SPLITS:
        split_lbl_dir = DATASET / "labels" / split
        current_labels = sorted([p for p in split_lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
        for lbl_path in current_labels:
            rel_lbl = str(lbl_path.relative_to(DATASET))
            if rel_lbl not in initial_label_hashes:
                mutated_labels.append(f"Added label: {rel_lbl}")
            else:
                curr_hash = hashlib.sha256(lbl_path.read_bytes()).hexdigest()
                if curr_hash != initial_label_hashes[rel_lbl]:
                    mutated_labels.append(f"Modified label: {rel_lbl}")

    # Check images
    for split in SPLITS:
        split_img_dir = DATASET / "images" / split
        current_imgs = sorted([p for p in split_img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        for img_path in current_imgs:
            rel_img = str(img_path.relative_to(DATASET))
            if rel_img not in initial_image_stats:
                mutated_images.append(f"Added image: {rel_img}")
            else:
                curr_size = img_path.stat().st_size
                if curr_size != initial_image_stats[rel_img][0]:
                    mutated_images.append(f"Modified image: {rel_img}")

    if mutated_labels:
        raise RuntimeError(f"FATAL: Detected {len(mutated_labels)} label mutations! {mutated_labels[:5]}")
    if mutated_images:
        raise RuntimeError(f"FATAL: Detected {len(mutated_images)} image mutations! {mutated_images[:5]}")
    print("[Integrity] VERIFIED: 0 label mutations and 0 image mutations across all files.")

    # 10. Aggregations & Metrics
    total_wall_time = time.perf_counter() - t_start_total
    peak_gpu_mem_bytes = torch.cuda.max_memory_allocated(0)
    peak_gpu_mem_mb = peak_gpu_mem_bytes / (1024 * 1024)

    counts_by_tier = Counter(c["tier"] for c in all_candidates)
    counts_by_class = Counter(c["class_name"] for c in all_candidates)
    counts_by_split = Counter(c["split"] for c in all_candidates)
    counts_by_tier_class = Counter((c["tier"], c["class_name"]) for c in all_candidates)
    counts_by_split_tier = Counter((c["split"], c["tier"]) for c in all_candidates)

    qa_workload = calculate_qa_workload(counts_by_tier)

    # Save human QA workload estimate JSON
    with (OUTPUT / "human_qa_workload_estimate.json").open("w", encoding="utf-8") as f:
        json.dump(qa_workload, f, indent=2)

    total_overlays = overlays_count_high + overlays_count_med + overlays_count_low

    summary = {
        "task": "D-Fire Corrected Full Missing Label Inference Scan",
        "configuration": {
            "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
            "total_images": total_images_found,
            "splits": list(SPLITS),
            "imgsz": IMGSZ,
            "batch_size": BATCH_SIZE,
            "chunk_size": CHUNK_SIZE,
            "conf_predict": CONF_PREDICT,
            "iou_match_threshold": IOU_MATCH_THRESH,
            "nms_iou_threshold": NMS_IOU_THRESH,
            "device": device_name,
            "models": {
                "person": "yolo11n.pt (COCO class 0)",
                "safety_ppe": "yolov8s-worldv2.pt (classes: safety helmet, safety vest)",
            },
            "tier_thresholds": {
                "HIGH": "confidence >= 0.70",
                "MEDIUM": "0.40 <= confidence < 0.70",
                "LOW": "0.20 <= confidence < 0.40",
            },
        },
        "validation_and_integrity": {
            "target_images": EXPECTED_TOTAL_IMAGES,
            "processed_images": len(manifest_rows),
            "raw_images_modified": len(mutated_images),
            "corrected_labels_modified": len(mutated_labels),
            "label_sha256_verified": True,
            "image_size_verified": True,
            "errors_count": 0,
            "errors": [],
        },
        "candidate_counts": {
            "total_candidates": len(all_candidates),
            "total_candidate_images": len(candidate_images_set),
            "raw_proposals_before_nms": raw_proposals_total,
            "nms_suppressed_duplicates": nms_suppressed_total,
            "by_tier": {
                "HIGH": counts_by_tier.get("HIGH", 0),
                "MEDIUM": counts_by_tier.get("MEDIUM", 0),
                "LOW": counts_by_tier.get("LOW", 0),
            },
            "by_class": {
                "person": counts_by_class.get("person", 0),
                "helmet": counts_by_class.get("helmet", 0),
                "vest": counts_by_class.get("vest", 0),
            },
            "by_split": {
                "train": counts_by_split.get("train", 0),
                "val": counts_by_split.get("val", 0),
                "test": counts_by_split.get("test", 0),
            },
            "by_tier_and_class": {
                f"{tier}_{cls}": counts_by_tier_class.get((tier, cls), 0)
                for tier in ("HIGH", "MEDIUM", "LOW")
                for cls in ("person", "helmet", "vest")
            },
            "by_split_and_tier": {
                f"{split}_{tier}": counts_by_split_tier.get((split, tier), 0)
                for split in SPLITS
                for tier in ("HIGH", "MEDIUM", "LOW")
            },
            "overlays_generated": {
                "HIGH": overlays_count_high,
                "MEDIUM": overlays_count_med,
                "LOW": overlays_count_low,
                "total": total_overlays,
            },
        },
        "timing_seconds": {
            "model_load_and_setup": round(model_load_time, 3),
            "warmup": round(warmup_time, 3),
            "pure_gpu_inference": round(pure_gpu_inference_time, 3),
            "postprocessing_and_overlays": round(postprocessing_time, 3),
            "total_wall_time": round(total_wall_time, 3),
            "images_per_second_inference": round(total_images_found / max(0.001, pure_gpu_inference_time), 2),
            "images_per_second_overall": round(total_images_found / max(0.001, total_wall_time), 2),
        },
        "gpu_memory": {
            "peak_allocated_bytes": peak_gpu_mem_bytes,
            "peak_allocated_mb": round(peak_gpu_mem_mb, 2),
        },
        "human_qa_workload_estimate": qa_workload,
    }

    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SCAN COMPLETE - SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
