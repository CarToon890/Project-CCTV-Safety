"""Comprehensive 8-Gate Dataset Validator for dfire_remediated_verified_b001.

Validates:
1. 21,527 image-label pairs (1:1 mapping, no orphans, exactly matching split sizes).
2. Class IDs strictly in 0..5 (person, helmet, vest, fall, fire, smoke).
3. Bounding box syntax, bounds [0, 1], and positive non-zero area.
4. Exact final count reconciliation:
   person=1,887; helmet=117; vest=0; fall=0; fire=14,683; smoke=11,854; total=28,541.
5. Split/group leakage: disjoint image sets across train, val, test.
6. Same-class duplication: zero duplicated/redundant boxes within any image.
7. Source immutability & pre-batch snapshot immutability (data/raw, dfire_corrected, dfire_remediated, dfire_remediated_verified).
8. Verdict-to-final-label mapping for all 200 Batch 1 rows:
   - 155 PASS boxes are present in target labels.
   - 1 FIX box (CAND_003322) is present with corrected coordinates.
   - 43 REMOVE boxes and 1 UNCERTAIN box are explicitly absent from target labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Base source directories
RAW_DIR = ROOT / "data/raw/dfire/data"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
VERIFIED_DIR = ROOT / "data/processed/dfire_remediated_verified"

# Target dataset
TARGET_DIR = ROOT / "data/processed/dfire_remediated_verified_b001"
TARGET_IMAGES_DIR = TARGET_DIR / "images"
TARGET_LABELS_DIR = TARGET_DIR / "labels"
TARGET_MANIFEST = TARGET_DIR / "remediation_manifest.csv"
TARGET_DATASET_MANIFEST = TARGET_DIR / "dataset_manifest.json"
TARGET_DATA_YAML = TARGET_DIR / "data.yaml"

# Batch 1 artifacts
BATCH_DIR = ROOT / "docs/audit_artifacts/dfire/human_qa_medium_person_batch_001"
QA_QUEUE_CSV = BATCH_DIR / "qa_queue.csv"
APP_MANIFEST_CSV = BATCH_DIR / "batch_001_application_manifest.csv"
APP_MANIFEST_JSON = BATCH_DIR / "batch_001_application_manifest.json"

SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {
    "train": 17248,
    "val": 1488,
    "test": 2791,
}
EXPECTED_TOTAL_IMAGES = 21527

# Verified expected hashes
EXPECTED_RAW_HASH = "88acdd03c14ba1035d5b3eb3a688fccbf061a69d58e0a57ddf03c12563bb11b2"
EXPECTED_CORR_HASH = "ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da"
EXPECTED_REMED_HASH = "b95bf2f4d7f848e0d76e6c8a0f7116ef945915376666db8a34e5330a7fe0188c"
EXPECTED_VERIF_HASH = "393353fdaff2253a97eb08eac6f4c2adfbbb07940e496c00187ac71f976427b9"

EXPECTED_CLASS_COUNTS = {
    0: 1887,   # person
    1: 117,    # helmet
    2: 0,      # vest
    3: 0,      # fall
    4: 14683,  # fire
    5: 11854,  # smoke
}
EXPECTED_TOTAL_BOXES = 28541


def hash_directory_labels(labels_root: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    file_count = 0
    all_txt = []
    for split in SPLITS:
        s_dir = labels_root / split
        if s_dir.exists():
            for p in sorted(s_dir.iterdir()):
                if p.is_file() and p.suffix.lower() == ".txt":
                    all_txt.append((split, p))

    for split, p in all_txt:
        rel_name = f"{split}/{p.name}"
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hasher.update(f"{rel_name}:{digest}\n".encode("utf-8"))
        file_count += 1

    return hasher.hexdigest(), file_count


def hash_raw_directory(raw_root: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    file_count = 0
    all_imgs = []
    for split in SPLITS:
        s_dir = raw_root / split / "images"
        if s_dir.exists():
            for p in sorted(s_dir.iterdir()):
                if p.is_file():
                    all_imgs.append((split, p))

    for split, p in all_imgs:
        stat = p.stat()
        hasher.update(f"{split}/{p.name}:{stat.st_size}\n".encode("utf-8"))
        file_count += 1

    return hasher.hexdigest(), file_count


def parse_bbox_tuple(s: str) -> tuple[float, float, float, float]:
    clean = s.strip().strip("()").replace(",", " ")
    parts = [float(p) for p in clean.split() if p.strip()]
    return parts[0], parts[1], parts[2], parts[3]


def compute_iou_xywh(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    axc, ayc, aw, ah = box_a
    bxc, byc, bw, bh = box_b
    ax1, ay1, ax2, ay2 = axc - aw / 2.0, ayc - ah / 2.0, axc + aw / 2.0, ayc + ah / 2.0
    bx1, by1, bx2, by2 = bxc - bw / 2.0, byc - bh / 2.0, bxc + bw / 2.0, byc + bh / 2.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0.0 else 0.0


def main():
    start_time = time.perf_counter()
    print("=" * 80)
    print("RUNNING 8-GATE VALIDATOR ON DFIRE_REMEDIATED_VERIFIED_B001")
    print("=" * 80)

    errors = []

    # -------------------------------------------------------------------------
    # GATE 1: 21,527 IMAGE-LABEL PAIRS (1:1 MAPPING)
    # -------------------------------------------------------------------------
    print("\n[Gate 1] 21,527 Image-Label Pairs (1:1 Mapping)...")
    split_images = {}
    split_labels = {}
    total_images = 0
    total_labels = 0

    for split in SPLITS:
        s_img_dir = TARGET_IMAGES_DIR / split
        s_lbl_dir = TARGET_LABELS_DIR / split

        if not s_img_dir.exists():
            errors.append(f"Missing images directory: {s_img_dir}")
            continue
        if not s_lbl_dir.exists():
            errors.append(f"Missing labels directory: {s_lbl_dir}")
            continue

        imgs = {p.stem: p for p in s_img_dir.iterdir() if p.is_file()}
        lbls = {p.stem: p for p in s_lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        split_images[split] = imgs
        split_labels[split] = lbls
        total_images += len(imgs)
        total_labels += len(lbls)

        print(f"  {split}: {len(imgs)} images, {len(lbls)} labels (expected {EXPECTED_SPLIT_COUNTS[split]})")

        if len(imgs) != EXPECTED_SPLIT_COUNTS[split]:
            errors.append(f"{split} image count mismatch: {len(imgs)} != {EXPECTED_SPLIT_COUNTS[split]}")
        if len(lbls) != EXPECTED_SPLIT_COUNTS[split]:
            errors.append(f"{split} label count mismatch: {len(lbls)} != {EXPECTED_SPLIT_COUNTS[split]}")

        # Check 1:1 pairing
        missing_lbls = set(imgs.keys()) - set(lbls.keys())
        missing_imgs = set(lbls.keys()) - set(imgs.keys())
        if missing_lbls:
            errors.append(f"{split} images missing labels: {len(missing_lbls)} (sample: {list(missing_lbls)[:3]})")
        if missing_imgs:
            errors.append(f"{split} labels missing images: {len(missing_imgs)} (sample: {list(missing_imgs)[:3]})")

    print(f"  Total pairs: {total_images} images, {total_labels} labels (expected {EXPECTED_TOTAL_IMAGES})")
    if total_images != EXPECTED_TOTAL_IMAGES or total_labels != EXPECTED_TOTAL_IMAGES:
        errors.append(f"Total pair count mismatch: {total_images} imgs, {total_labels} lbls vs {EXPECTED_TOTAL_IMAGES}")

    # -------------------------------------------------------------------------
    # GATE 2 & 3: CLASS IDS (0..5), BBOX SYNTAX, BOUNDS & NONZERO AREA
    # GATE 4: EXACT FINAL COUNT RECONCILIATION
    # GATE 6: SAME-CLASS DUPLICATION
    # -------------------------------------------------------------------------
    print("\n[Gates 2, 3, 4, 6] Auditing labels for IDs 0..5, Bbox Bounds, Exact Counts, & Duplication...")
    class_counts = Counter()
    split_class_counts = {s: Counter() for s in SPLITS}
    total_boxes = 0
    syntax_errors = 0
    bounds_errors = 0
    duplicate_errors = 0
    invalid_class_errors = 0

    for split in SPLITS:
        lbl_dict = split_labels.get(split, {})
        for stem, p in lbl_dict.items():
            lines = p.read_text(encoding="utf-8").splitlines()
            boxes_in_file = []
            for line_no, raw_line in enumerate(lines, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split()
                if len(parts) != 5:
                    syntax_errors += 1
                    errors.append(f"Syntax error in {p.name}:{line_no} -> '{raw_line}'")
                    continue

                try:
                    cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    syntax_errors += 1
                    errors.append(f"Value error in {p.name}:{line_no} -> '{raw_line}'")
                    continue

                # Gate 2: Class ID range
                if cid not in range(6):
                    invalid_class_errors += 1
                    errors.append(f"Invalid class ID {cid} in {p.name}:{line_no}")
                    continue

                # Gate 3: Bounds and positive non-zero area
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bounds_errors += 1
                    errors.append(f"Bounds violation in {p.name}:{line_no} -> ({xc}, {yc}, {w}, {h})")

                class_counts[cid] += 1
                split_class_counts[split][cid] += 1
                total_boxes += 1
                boxes_in_file.append((cid, (xc, yc, w, h)))

            # Gate 6: Same-class duplication
            for i in range(len(boxes_in_file)):
                c1, b1 = boxes_in_file[i]
                for j in range(i + 1, len(boxes_in_file)):
                    c2, b2 = boxes_in_file[j]
                    if c1 == c2:
                        iou = compute_iou_xywh(b1, b2)
                        if iou > 0.85:
                            duplicate_errors += 1
                            errors.append(
                                f"Same-class duplicate box in {p.name}: class {c1} boxes {b1} and {b2} have IoU={iou:.4f}"
                            )

    print(f"  Total boxes parsed: {total_boxes}")
    print(f"  Class breakdown:")
    for cid in range(6):
        cname = ["person", "helmet", "vest", "fall", "fire", "smoke"][cid]
        cnt = class_counts.get(cid, 0)
        exp = EXPECTED_CLASS_COUNTS[cid]
        match_str = "OK" if cnt == exp else f"MISMATCH (exp {exp})"
        print(f"    Class {cid} ({cname}): {cnt} -> {match_str}")
        if cnt != exp:
            errors.append(f"Class {cid} count mismatch: {cnt} != {exp}")

    if total_boxes != EXPECTED_TOTAL_BOXES:
        errors.append(f"Total box count mismatch: {total_boxes} != {EXPECTED_TOTAL_BOXES}")

    if syntax_errors:
        errors.append(f"Found {syntax_errors} syntax errors in labels")
    if bounds_errors:
        errors.append(f"Found {bounds_errors} bounds errors in labels")
    if invalid_class_errors:
        errors.append(f"Found {invalid_class_errors} invalid class ID errors")
    if duplicate_errors:
        errors.append(f"Found {duplicate_errors} duplicate same-class boxes")

    # -------------------------------------------------------------------------
    # GATE 5: SPLIT/GROUP LEAKAGE CHECK
    # -------------------------------------------------------------------------
    print("\n[Gate 5] Split/Group Leakage Check...")
    train_stems = set(split_images["train"].keys())
    val_stems = set(split_images["val"].keys())
    test_stems = set(split_images["test"].keys())

    inter_train_val = train_stems & val_stems
    inter_train_test = train_stems & test_stems
    inter_val_test = val_stems & test_stems

    if inter_train_val:
        errors.append(f"Split leakage train & val: {len(inter_train_val)} overlapping images")
    if inter_train_test:
        errors.append(f"Split leakage train & test: {len(inter_train_test)} overlapping images")
    if inter_val_test:
        errors.append(f"Split leakage val & test: {len(inter_val_test)} overlapping images")

    print(f"  Pairwise split overlaps: train-val={len(inter_train_val)}, train-test={len(inter_train_test)}, val-test={len(inter_val_test)}")

    # -------------------------------------------------------------------------
    # GATE 7: SOURCE & BASELINE IMMUTABILITY HASHES
    # -------------------------------------------------------------------------
    print("\n[Gate 7] Source Datasets & Baseline Snapshot Immutability...")
    raw_hash, raw_cnt = hash_raw_directory(RAW_DIR)
    corr_hash, corr_cnt = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_hash, remed_cnt = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_hash, verif_cnt = hash_directory_labels(VERIFIED_DIR / "labels")

    print(f"  data/raw count: {raw_cnt}, hash: {raw_hash}")
    print(f"  dfire_corrected count: {corr_cnt}, hash: {corr_hash}")
    print(f"  dfire_remediated count: {remed_cnt}, hash: {remed_hash}")
    print(f"  dfire_remediated_verified count: {verif_cnt}, hash: {verif_hash}")

    if raw_hash != EXPECTED_RAW_HASH or raw_cnt != EXPECTED_TOTAL_IMAGES:
        errors.append(f"data/raw hash mismatch: {raw_hash}")
    if corr_hash != EXPECTED_CORR_HASH or corr_cnt != EXPECTED_TOTAL_IMAGES:
        errors.append(f"dfire_corrected hash mismatch: {corr_hash}")
    if remed_hash != EXPECTED_REMED_HASH or remed_cnt != EXPECTED_TOTAL_IMAGES:
        errors.append(f"dfire_remediated hash mismatch: {remed_hash}")
    if verif_hash != EXPECTED_VERIF_HASH or verif_cnt != EXPECTED_TOTAL_IMAGES:
        errors.append(f"dfire_remediated_verified hash mismatch: {verif_hash}")

    # -------------------------------------------------------------------------
    # GATE 8: VERDICT-TO-FINAL-LABEL MAPPING & EXPLICIT ABSENCE CHECK
    # -------------------------------------------------------------------------
    print("\n[Gate 8] Verdict-to-Final-Label Mapping & Absence Verification (All 200 Rows)...")
    with open(QA_QUEUE_CSV, "r", encoding="utf-8") as f:
        queue_rows = list(csv.DictReader(f))

    pass_found = 0
    fix_found = 0
    remove_absent = 0
    uncertain_absent = 0

    for r in queue_rows:
        cid = r["candidate_id"].strip()
        v = r["human_verdict"].strip()
        split = r["split"].strip()
        image = r["image"].strip()
        stem = Path(image).stem
        lbl_p = TARGET_LABELS_DIR / split / f"{stem}.txt"

        if not lbl_p.exists():
            errors.append(f"Target label file missing for {cid}: {lbl_p}")
            continue

        label_lines = [line.strip().split() for line in lbl_p.read_text(encoding="utf-8").splitlines() if line.strip()]

        if v == "PASS":
            xc, yc, w, h = parse_bbox_tuple(r["normalized_bbox"])
            target_box = (xc, yc, w, h)
            # Find matching person box in label file
            matched = False
            for parts in label_lines:
                if int(parts[0]) == 0:
                    l_box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    if compute_iou_xywh(target_box, l_box) > 0.95:
                        matched = True
                        break
            if matched:
                pass_found += 1
            else:
                errors.append(f"PASS candidate {cid} bbox {target_box} NOT found in {lbl_p.name}")

        elif v == "FIX":
            xc, yc, w, h = parse_bbox_tuple(r["corrected_bbox"])
            target_box = (xc, yc, w, h)
            matched = False
            for parts in label_lines:
                if int(parts[0]) == 0:
                    l_box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    if compute_iou_xywh(target_box, l_box) > 0.95:
                        matched = True
                        break
            if matched:
                fix_found += 1
            else:
                errors.append(f"FIX candidate {cid} corrected bbox {target_box} NOT found in {lbl_p.name}")

        elif v == "REMOVE":
            xc, yc, w, h = parse_bbox_tuple(r["normalized_bbox"])
            target_box = (xc, yc, w, h)
            # Must NOT be present
            matched = False
            for parts in label_lines:
                if int(parts[0]) == 0:
                    l_box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    if compute_iou_xywh(target_box, l_box) > 0.90:
                        matched = True
                        break
            if not matched:
                remove_absent += 1
            else:
                errors.append(f"REMOVE candidate {cid} bbox {target_box} UNEXPECTEDLY FOUND in {lbl_p.name}")

        elif v == "UNCERTAIN":
            xc, yc, w, h = parse_bbox_tuple(r["normalized_bbox"])
            target_box = (xc, yc, w, h)
            matched = False
            for parts in label_lines:
                if int(parts[0]) == 0:
                    l_box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    if compute_iou_xywh(target_box, l_box) > 0.90:
                        matched = True
                        break
            if not matched:
                uncertain_absent += 1
            else:
                errors.append(f"UNCERTAIN candidate {cid} bbox {target_box} UNEXPECTEDLY FOUND in {lbl_p.name}")

    print(f"  PASS verified present: {pass_found}/155")
    print(f"  FIX verified present: {fix_found}/1")
    print(f"  REMOVE verified absent: {remove_absent}/43")
    print(f"  UNCERTAIN verified absent: {uncertain_absent}/1")

    if pass_found != 155:
        errors.append(f"PASS found count mismatch: {pass_found} != 155")
    if fix_found != 1:
        errors.append(f"FIX found count mismatch: {fix_found} != 1")
    if remove_absent != 43:
        errors.append(f"REMOVE absent count mismatch: {remove_absent} != 43")
    if uncertain_absent != 1:
        errors.append(f"UNCERTAIN absent count mismatch: {uncertain_absent} != 1")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    if errors:
        print("VALIDATION SUMMARY: ISSUES DETECTED")
        for err in errors:
            print(f"  [FAIL] {err}")
        print("=" * 80)
        sys.exit(1)
    else:
        print(f"VALIDATION SUMMARY: ALL 8 GATES PASSED CLEANLY IN {elapsed:.2f}s!")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
