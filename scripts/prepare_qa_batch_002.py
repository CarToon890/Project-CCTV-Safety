#!/usr/bin/env python
"""
Prepare Human QA Batch 2 for unreviewed D-Fire MEDIUM person candidates.
Deterministically samples 200 candidates with seed 43 and generates:
- 200 context overlays
- 200 zoomed crops
- 13 contact sheets
- qa_queue.csv (verdicts empty)
- batch_manifest.json
- README.md
"""

from __future__ import annotations

import ast
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]

# Authoritative paths
SNAPSHOT_DIR = REPO_ROOT / "data" / "processed" / "dfire_remediated_verified_b001"
MANIFEST_PATH = SNAPSHOT_DIR / "remediation_manifest.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "audit_artifacts" / "dfire" / "human_qa_medium_person_batch_002"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
CROPS_DIR = OUTPUT_DIR / "crops"
CONTACT_SHEETS_DIR = OUTPUT_DIR / "contact_sheets"

CLASS_NAMES = ["person", "helmet", "vest", "fall", "fire", "smoke"]
CLASS_COLORS = {
    0: (0, 255, 0),       # person: green
    1: (0, 255, 255),     # helmet: yellow
    2: (255, 165, 0),     # vest: orange
    3: (128, 0, 128),     # fall: purple
    4: (0, 0, 255),       # fire: red
    5: (255, 0, 0),       # smoke: blue
}
CANDIDATE_COLOR = (255, 0, 128)  # thick magenta


def find_image_file(image_name: str, split: str) -> Path:
    candidates = [
        SNAPSHOT_DIR / "images" / split / image_name,
        REPO_ROOT / "data" / "processed" / "dfire_remediated_verified" / "images" / split / image_name,
        REPO_ROOT / "data" / "raw" / "dfire" / "data" / split / "images" / image_name,
        REPO_ROOT / "data" / "raw" / "dfire" / "images" / split / image_name,
        REPO_ROOT / "data" / "raw" / "dfire" / "data" / image_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback rglob
    matches = list(REPO_ROOT.glob(f"data/**/{image_name}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find image {image_name} for split {split}")


def get_verified_labels(image_stem: str, split: str) -> List[Tuple[int, float, float, float, float]]:
    label_path = SNAPSHOT_DIR / "labels" / split / f"{image_stem}.txt"
    if not label_path.exists():
        label_path = REPO_ROOT / "data" / "processed" / "dfire_remediated_verified" / "labels" / split / f"{image_stem}.txt"
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cid, xc, yc, w, h))
    return boxes


def parse_bbox_tuple(raw: str) -> Tuple[float, float, float, float]:
    return ast.literal_eval(raw)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - interArea
    return interArea / denom if denom > 0 else 0.0


def compute_dir_sha256(dir_path: Path) -> Tuple[int, str]:
    if not dir_path.exists():
        return 0, "MISSING"
    files = sorted([p for p in dir_path.rglob("*") if p.is_file()],
                   key=lambda p: str(p.relative_to(dir_path)).replace("\\", "/"))
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(dir_path)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(f.read_bytes())
    return len(files), h.hexdigest()


def main() -> int:
    print(f"Reading authoritative manifest: {MANIFEST_PATH}")
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found!")
        return 1

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total rows in manifest: {len(all_rows)}")

    # Filter eligible pool
    eligible = []
    for r in all_rows:
        if (
            "Unreviewed" in r["phase"]
            and "MEDIUM" in r["phase"]
            and r["class_name"] == "person"
            and r["status"] == "EXCLUDED"
            and r["action"] == "EXCLUDED_PENDING_GENUINE_VISUAL_QA"
        ):
            eligible.append(r)

    print(f"Eligible unreviewed MEDIUM person candidates: {len(eligible)}")
    if len(eligible) != 1440:
        print(f"WARNING: Expected 1440 eligible candidates, found {len(eligible)}!")

    # Count candidates per image
    image_cand_counts = {}
    for r in eligible:
        img = r["image"]
        image_cand_counts[img] = image_cand_counts.get(img, 0) + 1

    # Extract features for stratification
    processed_candidates = []
    for r in eligible:
        cid = r["candidate_id"]
        split = r["split"]
        img_name = r["image"]
        conf = float(r["confidence"])
        x1, y1, x2, y2 = parse_bbox_tuple(r["original_bbox"])

        # Confidence band
        if conf >= 0.60:
            conf_band = "upper_0.60_0.70"
        elif conf >= 0.50:
            conf_band = "mid_0.50_0.60"
        else:
            conf_band = "lower_0.40_0.50"

        # Image file & dimensions
        img_path = find_image_file(img_name, split)
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        bw = x2 - x1
        bh = y2 - y1
        xc_norm = (x1 + bw / 2.0) / w_img
        yc_norm = (y1 + bh / 2.0) / h_img
        w_norm = bw / w_img
        h_norm = bh / h_img
        norm_bbox = (round(xc_norm, 6), round(yc_norm, 6), round(w_norm, 6), round(h_norm, 6))

        # Size band (normalized area)
        norm_area = w_norm * h_norm
        if norm_area < 0.01:
            size_band = "small_box"
        elif norm_area < 0.08:
            size_band = "medium_box"
        else:
            size_band = "large_box"

        # Border case
        is_border = (x1 <= 3 or y1 <= 3 or x2 >= w_img - 3 or y2 >= h_img - 3 or
                     xc_norm - w_norm / 2 <= 0.005 or yc_norm - h_norm / 2 <= 0.005 or
                     xc_norm + w_norm / 2 >= 0.995 or yc_norm + h_norm / 2 >= 0.995)

        # Extreme aspect ratio (tall person vs horizontal fragment)
        ar = bh / max(bw, 1e-4)
        is_extreme_ar = (ar >= 3.0 or ar <= 0.6)

        # Multi-candidate
        is_multi = image_cand_counts[img_name] > 1

        # Fire / smoke overlap
        verified_boxes = get_verified_labels(Path(img_name).stem, split)
        has_fire_smoke = False
        cand_norm_xyxy = (xc_norm - w_norm / 2, yc_norm - h_norm / 2, xc_norm + w_norm / 2, yc_norm + h_norm / 2)
        for v_cid, v_xc, v_yc, v_w, v_h in verified_boxes:
            if v_cid in (4, 5):  # fire or smoke
                v_xyxy = (v_xc - v_w / 2, v_yc - v_h / 2, v_xc + v_w / 2, v_yc + v_h / 2)
                if compute_iou(cand_norm_xyxy, v_xyxy) > 0.01:
                    has_fire_smoke = True
                    break

        # Geometry stratum designation
        if has_fire_smoke:
            geom_stratum = "smoke_fire_overlap"
        elif is_border:
            geom_stratum = "border_touching"
        elif is_extreme_ar:
            geom_stratum = "extreme_aspect_ratio"
        elif is_multi:
            geom_stratum = "multi_candidate_image"
        else:
            geom_stratum = "standard_geometry"

        processed_candidates.append({
            "candidate_id": cid,
            "raw_row": r,
            "split": split,
            "image": img_name,
            "img_path": img_path,
            "img_dims": (w_img, h_img),
            "confidence": conf,
            "conf_band": conf_band,
            "pixel_bbox": (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
            "normalized_bbox": norm_bbox,
            "size_band": size_band,
            "geom_stratum": geom_stratum,
            "is_border": is_border,
            "is_multi": is_multi,
            "is_extreme_ar": is_extreme_ar,
            "has_fire_smoke": has_fire_smoke,
        })

    # Proportional Stratified Deterministic Sampling (seed=43)
    # Target 200 candidates:
    # Splits proportional to dataset: train ~ 146, test ~ 31, val ~ 23
    # Inside each split, stratify across confidence band and geom_stratum
    target_total = 200
    split_targets = {
        "train": round(target_total * sum(1 for c in processed_candidates if c["split"] == "train") / len(processed_candidates)),
        "test": round(target_total * sum(1 for c in processed_candidates if c["split"] == "test") / len(processed_candidates)),
        "val": 0,
    }
    split_targets["val"] = target_total - split_targets["train"] - split_targets["test"]

    print(f"Target splits: {split_targets}")

    rng = random.Random(43)
    selected_candidates = []

    # Stratified by (split, conf_band, geom_stratum)
    for sp, sp_target in split_targets.items():
        sp_cands = [c for c in processed_candidates if c["split"] == sp]
        # Group by composite key
        groups = {}
        for c in sp_cands:
            key = (c["conf_band"], c["geom_stratum"])
            groups.setdefault(key, []).append(c)

        # Proportional allocation
        sp_selected = []
        allocated = {}
        remaining_target = sp_target
        # First allocate proportionally
        for key, items in sorted(groups.items()):
            # Sort items deterministically by candidate_id before shuffling
            items.sort(key=lambda x: x["candidate_id"])
            shuffled = list(items)
            rng.shuffle(shuffled)
            groups[key] = shuffled

            share = len(items) / len(sp_cands) * sp_target
            alloc = int(math.floor(share))
            allocated[key] = alloc
            sp_selected.extend(shuffled[:alloc])
            groups[key] = shuffled[alloc:]

        # Fill remaining slots using largest fractional remainder
        remainders = []
        for key, items in sorted(groups.items()):
            orig_len = allocated[key] + len(items)
            exact_share = orig_len / len(sp_cands) * sp_target
            frac = exact_share - allocated[key]
            remainders.append((frac, key))
        remainders.sort(key=lambda x: (-x[0], x[1]))

        curr_count = len(sp_selected)
        for frac, key in remainders:
            if curr_count >= sp_target:
                break
            if groups[key]:
                sp_selected.append(groups[key].pop(0))
                curr_count += 1

        selected_candidates.extend(sp_selected)

    print(f"Total selected candidates: {len(selected_candidates)}")
    assert len(selected_candidates) == 200, f"Expected 200, got {len(selected_candidates)}"

    # Sort final queue by confidence DESC, candidate_id ASC
    selected_candidates.sort(key=lambda x: (-x["confidence"], x["candidate_id"]))

    # Prepare directories
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate visual artifacts
    print("Generating overlays, crops, and contact sheets...")
    crop_images_for_sheets = []

    for idx, cand in enumerate(selected_candidates, 1):
        cid = cand["candidate_id"]
        img_path = cand["img_path"]
        w_img, h_img = cand["img_dims"]
        x1, y1, x2, y2 = cand["pixel_bbox"]

        # Read base image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # 1. Context Overlay
        overlay_bgr = img_bgr.copy()
        # Draw verified boxes
        verified_boxes = get_verified_labels(Path(cand["image"]).stem, cand["split"])
        for v_cid, v_xc, v_yc, v_w, v_h in verified_boxes:
            vx1 = int(round((v_xc - v_w / 2) * w_img))
            vy1 = int(round((v_yc - v_h / 2) * h_img))
            vx2 = int(round((v_xc + v_w / 2) * w_img))
            vy2 = int(round((v_yc + v_h / 2) * h_img))
            v_color = CLASS_COLORS.get(v_cid, (200, 200, 200))
            cv2.rectangle(overlay_bgr, (vx1, vy1), (vx2, vy2), v_color, 2)
            lbl = CLASS_NAMES[v_cid] if v_cid < len(CLASS_NAMES) else str(v_cid)
            cv2.putText(overlay_bgr, lbl, (vx1, max(15, vy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, v_color, 1, cv2.LINE_AA)

        # Draw candidate box in thick magenta
        px1, py1, px2, py2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(overlay_bgr, (px1, py1), (px2, py2), (255, 0, 128), 3)
        cand_badge = f"{cid} (MEDIUM {cand['confidence']:.3f})"
        cv2.putText(overlay_bgr, cand_badge, (px1, max(22, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 128), 2, cv2.LINE_AA)

        overlay_out_path = OVERLAYS_DIR / f"{cid}_context_overlay.jpg"
        cv2.imwrite(str(overlay_out_path), overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Zoomed Crop with context margin
        bw = x2 - x1
        bh = y2 - y1
        pad_x = max(24, int(bw * 0.45))
        pad_y = max(24, int(bh * 0.45))
        crop_x1 = max(0, int(round(x1 - pad_x)))
        crop_y1 = max(0, int(round(y1 - pad_y)))
        crop_x2 = min(w_img, int(round(x2 + pad_x)))
        crop_y2 = min(h_img, int(round(y2 + pad_y)))

        crop_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Draw candidate box inside crop
        rel_x1 = px1 - crop_x1
        rel_y1 = py1 - crop_y1
        rel_x2 = px2 - crop_x1
        rel_y2 = py2 - crop_y1
        cv2.rectangle(crop_bgr, (rel_x1, rel_y1), (rel_x2, rel_y2), (255, 0, 128), 2)
        cv2.putText(crop_bgr, f"{cid}", (rel_x1, max(15, rel_y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 128), 1, cv2.LINE_AA)

        crop_out_path = CROPS_DIR / f"{cid}_zoomed_crop.jpg"
        cv2.imwrite(str(crop_out_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Save info for contact sheets
        sheet_num = ((idx - 1) // 16) + 1
        item_in_sheet = ((idx - 1) % 16) + 1
        row_in_sheet = ((item_in_sheet - 1) // 4) + 1
        col_in_sheet = ((item_in_sheet - 1) % 4) + 1
        sheet_ref = f"contact_sheets/contact_sheet_{sheet_num:02d}.jpg (item {item_in_sheet}, row {row_in_sheet}, col {col_in_sheet})"

        cand["overlay_rel_path"] = f"docs/audit_artifacts/dfire/human_qa_medium_person_batch_002/overlays/{cid}_context_overlay.jpg"
        cand["crop_rel_path"] = f"docs/audit_artifacts/dfire/human_qa_medium_person_batch_002/crops/{cid}_zoomed_crop.jpg"
        cand["contact_sheet_reference"] = sheet_ref
        cand["sheet_num"] = sheet_num
        cand["crop_bgr"] = crop_bgr

    # 3. Build 13 Contact Sheets (16 items each, sheet 13 has 8 items)
    print("Assembling contact sheets...")
    for sheet_idx in range(1, 14):
        sheet_items = [c for c in selected_candidates if c["sheet_num"] == sheet_idx]
        # Grid: 4 cols x 4 rows
        cell_w, cell_h = 360, 360
        header_h = 60
        grid_w = cell_w * 4
        grid_h = cell_h * 4 + header_h
        sheet_img = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

        # Header banner
        title = f"D-Fire Human QA Batch 2 - Contact Sheet {sheet_idx:02d} / 13 (Items {(sheet_idx-1)*16 + 1} - {(sheet_idx-1)*16 + len(sheet_items)})"
        cv2.putText(sheet_img, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        for i, cand in enumerate(sheet_items):
            r = i // 4
            c = i % 4
            x_start = c * cell_w
            y_start = header_h + r * cell_h

            crop = cand["crop_bgr"]
            # Resize crop preserving aspect ratio into (cell_w - 20, cell_h - 70)
            target_w = cell_w - 20
            target_h = cell_h - 75
            ch, cw = crop.shape[:2]
            scale = min(target_w / cw, target_h / ch)
            nw, nh = int(round(cw * scale)), int(round(ch * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

            off_x = x_start + (cell_w - nw) // 2
            off_y = y_start + 45 + (target_h - nh) // 2
            sheet_img[off_y:off_y + nh, off_x:off_x + nw] = resized

            # Border around cell
            cv2.rectangle(sheet_img, (x_start + 4, y_start + 4), (x_start + cell_w - 4, y_start + cell_h - 4), (70, 70, 70), 1)

            # Metadata text
            badge_1 = f"#{ (sheet_idx-1)*16 + i + 1 } | {cand['candidate_id']} | conf: {cand['confidence']:.3f}"
            badge_2 = f"{cand['split']} | {cand['image']} | {cand['geom_stratum']} | {cand['size_band']}"
            cv2.putText(sheet_img, badge_1, (x_start + 10, y_start + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(sheet_img, badge_2, (x_start + 10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        sheet_out_path = CONTACT_SHEETS_DIR / f"contact_sheet_{sheet_idx:02d}.jpg"
        cv2.imwrite(str(sheet_out_path), sheet_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 4. Write initial qa_queue.csv (verdicts empty)
    print("Writing initial qa_queue.csv...")
    queue_path = OUTPUT_DIR / "qa_queue.csv"
    fieldnames = [
        "candidate_id", "tier", "split", "image", "confidence",
        "pixel_bbox", "normalized_bbox", "overlay_path", "crop_path",
        "contact_sheet_reference", "human_verdict", "corrected_bbox",
        "reviewer_notes", "reviewer", "reviewed_at"
    ]
    with queue_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in selected_candidates:
            writer.writerow({
                "candidate_id": cand["candidate_id"],
                "tier": "MEDIUM",
                "split": cand["split"],
                "image": cand["image"],
                "confidence": f"{cand['confidence']:.6f}",
                "pixel_bbox": f"({cand['pixel_bbox'][0]:.2f}, {cand['pixel_bbox'][1]:.2f}, {cand['pixel_bbox'][2]:.2f}, {cand['pixel_bbox'][3]:.2f})",
                "normalized_bbox": f"({cand['normalized_bbox'][0]:.6f}, {cand['normalized_bbox'][1]:.6f}, {cand['normalized_bbox'][2]:.6f}, {cand['normalized_bbox'][3]:.6f})",
                "overlay_path": cand["overlay_rel_path"],
                "crop_path": cand["crop_rel_path"],
                "contact_sheet_reference": cand["contact_sheet_reference"],
                "human_verdict": "",
                "corrected_bbox": "",
                "reviewer_notes": "",
                "reviewer": "",
                "reviewed_at": "",
            })

    # 5. Compute source hashes
    print("Computing source dataset directory hashes...")
    raw_cnt, raw_hash = compute_dir_sha256(REPO_ROOT / "data" / "raw" / "dfire" / "data")
    corr_cnt, corr_hash = compute_dir_sha256(REPO_ROOT / "data" / "processed" / "dfire_corrected" / "labels")
    rem_cnt, rem_hash = compute_dir_sha256(REPO_ROOT / "data" / "processed" / "dfire_remediated" / "labels")
    ver_cnt, ver_hash = compute_dir_sha256(REPO_ROOT / "data" / "processed" / "dfire_remediated_verified" / "labels")
    b001_cnt, b001_hash = compute_dir_sha256(REPO_ROOT / "data" / "processed" / "dfire_remediated_verified_b001" / "labels")

    counts_by_split = dict(sorted({s: sum(1 for c in selected_candidates if c["split"] == s) for s in ("train", "test", "val")}.items()))
    counts_by_conf = dict(sorted({b: sum(1 for c in selected_candidates if c["conf_band"] == b) for b in ("upper_0.60_0.70", "mid_0.50_0.60", "lower_0.40_0.50")}.items()))
    counts_by_geom = dict(sorted({g: sum(1 for c in selected_candidates if c["geom_stratum"] == g) for g in set(c["geom_stratum"] for c in selected_candidates)}.items()))
    counts_by_size = dict(sorted({sz: sum(1 for c in selected_candidates if c["size_band"] == sz) for sz in ("small_box", "medium_box", "large_box")}.items()))

    # 6. Write batch_manifest.json
    print("Writing batch_manifest.json...")
    manifest_data = {
        "batch_id": "human_qa_medium_person_batch_002",
        "batch_version": "1.0.0",
        "task_scope": "Preparation of Human QA Batch 2 for unreviewed D-Fire MEDIUM person candidates",
        "trusted_dataset": "data/processed/dfire_remediated_verified_b001",
        "total_unreviewed_pool": len(eligible),
        "batch_size": len(selected_candidates),
        "candidate_selection_rule": (
            "Proportional stratified deterministic sampling across splits (train=146, val=23, test=31), "
            "feature strata (smoke_fire_overlap, extreme_aspect_ratio, border_touching, multi_candidate_image, standard_geometry), "
            "box size bands (small, medium, large), and confidence bands ([0.60, 0.70), [0.50, 0.60), [0.40, 0.50)). "
            "Bins sorted by candidate_id before seed=43 shuffle. Final queue sorted by confidence DESC, candidate_id ASC."
        ),
        "seed_and_ordering": {
            "seed": 43,
            "ordering": "confidence DESC, candidate_id ASC"
        },
        "source_hashes": {
            "data_raw_dfire_data_count": raw_cnt,
            "data_raw_dfire_data_sha256": raw_hash,
            "data_processed_dfire_corrected_labels_count": corr_cnt,
            "data_processed_dfire_corrected_labels_sha256": corr_hash,
            "data_processed_dfire_remediated_labels_count": rem_cnt,
            "data_processed_dfire_remediated_labels_sha256": rem_hash,
            "data_processed_dfire_remediated_verified_labels_count": ver_cnt,
            "data_processed_dfire_remediated_verified_labels_sha256": ver_hash,
            "data_processed_dfire_remediated_verified_b001_labels_count": b001_cnt,
            "data_processed_dfire_remediated_verified_b001_labels_sha256": b001_hash,
        },
        "selected_candidate_ids": [c["candidate_id"] for c in selected_candidates],
        "counts_by_split": counts_by_split,
        "counts_by_confidence_band": counts_by_conf,
        "counts_by_geometry_strata": counts_by_geom,
        "counts_by_size_band": counts_by_size,
        "contact_sheets_count": 13,
        "generation_timestamp": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(),
    }

    manifest_out = OUTPUT_DIR / "batch_manifest.json"
    manifest_out.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. Write README.md
    print("Writing README.md...")
    readme_content = f"""# D-Fire Human QA: Medium Person Batch 002

## 1. Overview & Objectives
- **Batch Identifier:** `human_qa_medium_person_batch_002`
- **Scope:** Exactly **200 MEDIUM-tier person candidates** sampled deterministically from the 1,440 remaining genuinely unreviewed D-Fire population.
- **Trusted Reference Dataset:** `data/processed/dfire_remediated_verified_b001`
- **Target Class:** `person` (canonical class ID `0`)
- **Queue Location:** [qa_queue.csv](file:///C:/Users/USER/Desktop/BUU69/Project-CCTV-Safety-antigravity/docs/audit_artifacts/dfire/human_qa_medium_person_batch_002/qa_queue.csv)

All 200 candidates in this batch currently have blank review verdicts in this initial template. No automated semantic verdict or label mutation has been applied.

---

## 2. Review Artifacts Structure
Reviewers must consult the provided visual artifacts for every candidate:
1. **Contact Sheets:** `contact_sheets/contact_sheet_01.jpg` to `contact_sheets/contact_sheet_13.jpg` (16 items per sheet for sheets 1-12, 8 items on sheet 13 with readable ID, split, image name, confidence, and stratum).
2. **Context Overlays:** `overlays/<candidate_id>_context_overlay.jpg` showing the candidate box in thick magenta (`#FF0080`), with existing verified dataset labels rendered in distinct colors (`person`: green, `helmet`: yellow, `vest`: orange, `fall`: purple, `fire`: red, `smoke`: blue).
3. **Zoomed Diagnostic Crops:** `crops/<candidate_id>_zoomed_crop.jpg` providing high-resolution local context around the candidate detection.

---

## 3. Human Review Instructions & Verdict Taxonomy

Human reviewers must record verdicts strictly using one of the four allowed tokens:

| Verdict | Meaning | When to Use | Action Required |
| :--- | :--- | :--- | :--- |
| **`PASS`** | Confirmed True Positive | The candidate is a genuine, usable person instance (pedestrian, firefighter, civilian) with an accurate bounding box. | Set `human_verdict` to `PASS`. Leave `corrected_bbox` blank. |
| **`FIX`** | True Positive with Inaccurate Box | A genuine person is present, but the proposed candidate bounding box is loose, shifted, truncated, or excludes limbs/head. | Set `human_verdict` to `FIX`. Provide exact corrected coordinates in `corrected_bbox`. |
| **`REMOVE`** | False Positive / Reject | The candidate is not a person. Typical false positives include smoke puffs, flame shapes, tree trunks, fire equipment, shadows, reflections, or vehicles. | Set `human_verdict` to `REMOVE`. Leave `corrected_bbox` blank. |
| **`UNCERTAIN`** | Borderline / Ambiguous | The detection is heavily obscured, severely blurred, in thick smoke haze, or distant to the point of unresolvable ambiguity. | Set `human_verdict` to `UNCERTAIN`. Leave `corrected_bbox` blank. |

### Recording FIX Coordinates
When marking a candidate as `FIX`, the reviewer MUST provide the corrected coordinates in the `corrected_bbox` column in either format:
- **Normalized YOLO Format (Recommended):** `(xc, yc, w, h)` with values in $[0.0, 1.0]$ relative to the full image.
- **Pixel Coordinates Format:** `(x1, y1, x2, y2)` with pixel values $[0, W]$ and $[0, H]$.

### Unreviewed Status
- A **blank** value in `human_verdict` signifies that the candidate has **not yet been reviewed**.
- Do NOT prefill or auto-assign verdicts. Every verdict must originate from verified visual inspection.

### Reviewer Metadata
When completing a review row in `qa_queue.csv`, please fill:
- `reviewer`: Reviewer ID (e.g. `ANTIGRAVITY_VISUAL_QA`).
- `reviewed_at`: ISO 8601 completion timestamp (e.g. `2026-09-26T14:00:00+07:00`).
- `reviewer_notes`: Detailed, image-specific explanation of visual rationale (especially for `FIX`, `REMOVE`, and `UNCERTAIN`).

---

## 4. Batch Selection Summary
- **Candidate Pool:** 1,440 unreviewed MEDIUM persons (`data/processed/dfire_remediated_verified_b001/remediation_manifest.csv`)
- **Sampling Strategy:** Stratified deterministic sampling (`seed=43`)
- **Split Breakdown:** `train`: {counts_by_split.get('train', 0)} | `val`: {counts_by_split.get('val', 0)} | `test`: {counts_by_split.get('test', 0)} (Total: 200)
- **Confidence Bands:** Upper [0.60, 0.70): {counts_by_conf.get('upper_0.60_0.70', 0)} | Mid [0.50, 0.60): {counts_by_conf.get('mid_0.50_0.60', 0)} | Lower [0.40, 0.50): {counts_by_conf.get('lower_0.40_0.50', 0)}
- **Geometry Strata:** {", ".join(f"{k}: {v}" for k, v in counts_by_geom.items())}
- **Box Size Bands:** {", ".join(f"{k}: {v}" for k, v in counts_by_size.items())}
"""
    readme_out = OUTPUT_DIR / "README.md"
    readme_out.write_text(readme_content, encoding="utf-8")

    print(f"Preparation complete! Artifacts written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
