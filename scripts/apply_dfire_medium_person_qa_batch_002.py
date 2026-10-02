"""Apply completed Human QA Batch 2 verdicts safely.

Builds:
  data/processed/dfire_remediated_verified_b002
  Safely created from source data/processed/dfire_remediated_verified_b001:
    - NTFS hardlinks for images (21,527 images)
    - Independent copied labels (21,527 label files)
  Applies:
    - 185 PASS candidates (canonical class 0: person)
    - 1 FIX candidate CAND_001600 with visually corrected coordinates
    - 0 REMOVE candidates (14 candidates safely excluded)

Reconciles exact final counts:
  person: 2,073
  helmet: 117
  vest: 0
  fall: 0
  fire: 14,683
  smoke: 11,854
  total: 28,727

Maintains exclusive ownership, updates heartbeats, produces application manifests,
events, status, and preliminary report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Base source directories
RAW_DIR = ROOT / "data/raw/dfire/data"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
VERIFIED_DIR = ROOT / "data/processed/dfire_remediated_verified"
SOURCE_DIR = ROOT / "data/processed/dfire_remediated_verified_b001"
SOURCE_IMAGES_DIR = SOURCE_DIR / "images"
SOURCE_LABELS_DIR = SOURCE_DIR / "labels"
SOURCE_MANIFEST = SOURCE_DIR / "remediation_manifest.csv"
SOURCE_DATASET_MANIFEST = SOURCE_DIR / "dataset_manifest.json"

# Target dataset directory
TARGET_DIR = ROOT / "data/processed/dfire_remediated_verified_b002"
TARGET_IMAGES_DIR = TARGET_DIR / "images"
TARGET_LABELS_DIR = TARGET_DIR / "labels"
TARGET_MANIFEST = TARGET_DIR / "remediation_manifest.csv"
TARGET_DATASET_MANIFEST = TARGET_DIR / "dataset_manifest.json"
TARGET_DATA_YAML = TARGET_DIR / "data.yaml"

# QA Batch 2 Artifacts
BATCH_DIR = ROOT / "docs/audit_artifacts/dfire/human_qa_medium_person_batch_002"
QA_QUEUE_CSV = BATCH_DIR / "qa_queue.csv"
APP_MANIFEST_CSV = BATCH_DIR / "batch_002_application_manifest.csv"
APP_MANIFEST_JSON = BATCH_DIR / "batch_002_application_manifest.json"
STATUS_JSON = BATCH_DIR / "status.json"
EVENTS_JSONL = BATCH_DIR / "events.jsonl"
FINAL_REPORT_MD = BATCH_DIR / "final_report.md"
REGISTRY_JSON = BATCH_DIR / "task_registry.json"

SPLITS = ("train", "val", "test")
EXPECTED_FILE_COUNT = 21527

# Verified baseline aggregate hashes
EXPECTED_RAW_HASH = "88acdd03c14ba1035d5b3eb3a688fccbf061a69d58e0a57ddf03c12563bb11b2"
EXPECTED_CORR_HASH = "ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da"
EXPECTED_REMED_HASH = "b95bf2f4d7f848e0d76e6c8a0f7116ef945915376666db8a34e5330a7fe0188c"
EXPECTED_VERIF_HASH = "393353fdaff2253a97eb08eac6f4c2adfbbb07940e496c00187ac71f976427b9"

EXPECTED_FINAL_COUNTS = {
    0: 2073,   # person
    1: 117,    # helmet
    2: 0,      # vest
    3: 0,      # fall
    4: 14683,  # fire
    5: 11854,  # smoke
}
EXPECTED_TOTAL_BOXES = 28727


def update_heartbeat(step_name: str, status: str = "RUNNING", detail: str = ""):
    now_iso = datetime.now(timezone.utc).astimezone().isoformat()
    # Update status.json
    status_data = {}
    if STATUS_JSON.exists():
        try:
            status_data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:
            status_data = {}

    status_data["task_id"] = "dfire_batch002_apply"
    status_data["status"] = status
    status_data["owner"] = "dfire_batch002_apply_worker"
    status_data["ownership"] = "EXCLUSIVE"
    status_data["heartbeat"] = now_iso
    status_data["updated_at"] = now_iso
    status_data["current_step"] = step_name
    STATUS_JSON.write_text(json.dumps(status_data, indent=2), encoding="utf-8")

    # Append event to events.jsonl
    event_entry = {
        "timestamp": now_iso,
        "task_id": "dfire_batch002_apply",
        "event": step_name,
        "status": status,
        "detail": detail,
    }
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_entry) + "\n")

    # Update task_registry.json
    if REGISTRY_JSON.exists():
        try:
            reg_data = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
            if "tasks" in reg_data and "dfire_batch002_apply" in reg_data["tasks"]:
                reg_data["tasks"]["dfire_batch002_apply"]["status"] = status
                reg_data["tasks"]["dfire_batch002_apply"]["heartbeat"] = now_iso
                reg_data["tasks"]["dfire_batch002_apply"]["updated_at"] = now_iso
                REGISTRY_JSON.write_text(json.dumps(reg_data, indent=2), encoding="utf-8")
        except Exception:
            pass


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
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox string: {s}")
    return parts[0], parts[1], parts[2], parts[3]


def compute_iou_xywh(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    axc, ayc, aw, ah = box_a
    bxc, byc, bw, bh = box_b
    ax1, ay1, ax2, ay2 = axc - aw / 2.0, ayc - ah / 2.0, axc + aw / 2.0, ayc + ah / 2.0
    bx1, bx1_h, bx2, by2 = bxc - bw / 2.0, byc - bh / 2.0, bxc + bw / 2.0, byc + bh / 2.0
    by1 = bx1_h
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
    run_timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    print("=" * 80)
    print("APPLYING HUMAN QA BATCH 2 VERDICTS TO D-FIRE DATASET")
    print(f"Task ID: dfire_batch002_apply")
    print(f"Timestamp: {run_timestamp}")
    print("=" * 80)

    # Heartbeat before starting
    update_heartbeat("HEARTBEAT_BEFORE_STARTING", "RUNNING", "Pre-application and immutability checks starting")

    # -------------------------------------------------------------------------
    # STEP 1: PRE-CHECK IMMUTABILITY OF TRUSTED DATASETS
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying immutability of source datasets...")
    raw_hash, raw_cnt = hash_raw_directory(RAW_DIR)
    corr_hash, corr_cnt = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_hash, remed_cnt = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_hash, verif_cnt = hash_directory_labels(VERIFIED_DIR / "labels")
    source_b001_hash, source_b001_cnt = hash_directory_labels(SOURCE_LABELS_DIR)

    print(f"  data/raw image count: {raw_cnt}, SHA-256: {raw_hash}")
    print(f"  dfire_corrected label count: {corr_cnt}, SHA-256: {corr_hash}")
    print(f"  dfire_remediated label count: {remed_cnt}, SHA-256: {remed_hash}")
    print(f"  dfire_remediated_verified label count: {verif_cnt}, SHA-256: {verif_hash}")
    print(f"  dfire_remediated_verified_b001 label count: {source_b001_cnt}, SHA-256: {source_b001_hash}")

    if raw_hash != EXPECTED_RAW_HASH or raw_cnt != EXPECTED_FILE_COUNT:
        print(f"FATAL: data/raw hash mismatch: {raw_hash}")
        sys.exit(1)
    if corr_hash != EXPECTED_CORR_HASH or corr_cnt != EXPECTED_FILE_COUNT:
        print(f"FATAL: dfire_corrected hash mismatch: {corr_hash}")
        sys.exit(1)
    if remed_hash != EXPECTED_REMED_HASH or remed_cnt != EXPECTED_FILE_COUNT:
        print(f"FATAL: dfire_remediated hash mismatch: {remed_hash}")
        sys.exit(1)
    if verif_hash != EXPECTED_VERIF_HASH or verif_cnt != EXPECTED_FILE_COUNT:
        print(f"FATAL: dfire_remediated_verified hash mismatch: {verif_hash}")
        sys.exit(1)
    if source_b001_cnt != EXPECTED_FILE_COUNT:
        print(f"FATAL: dfire_remediated_verified_b001 file count mismatch: {source_b001_cnt}")
        sys.exit(1)

    print("  All source baselines verified immutable!")

    # -------------------------------------------------------------------------
    # STEP 2: LOAD AND VERIFY AUTHORITATIVE QA QUEUE
    # -------------------------------------------------------------------------
    print("\n[Step 2] Ingesting and auditing authoritative qa_queue.csv...")
    if not QA_QUEUE_CSV.exists():
        print(f"FATAL: {QA_QUEUE_CSV} not found!")
        sys.exit(1)

    with open(QA_QUEUE_CSV, "r", encoding="utf-8") as f:
        queue_rows = list(csv.DictReader(f))

    print(f"  Loaded {len(queue_rows)} rows from qa_queue.csv")
    if len(queue_rows) != 200:
        print(f"FATAL: Expected exactly 200 rows, found {len(queue_rows)}")
        sys.exit(1)

    verdict_counts = Counter(r["human_verdict"].strip() for r in queue_rows)
    print(f"  Verdict Census: {dict(verdict_counts)}")

    if verdict_counts.get("PASS", 0) != 185:
        print(f"FATAL: Expected exactly 185 PASS, found {verdict_counts.get('PASS', 0)}")
        sys.exit(1)
    if verdict_counts.get("FIX", 0) != 1:
        print(f"FATAL: Expected exactly 1 FIX, found {verdict_counts.get('FIX', 0)}")
        sys.exit(1)
    if verdict_counts.get("REMOVE", 0) != 14:
        print(f"FATAL: Expected exactly 14 REMOVE, found {verdict_counts.get('REMOVE', 0)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3: PRE-APPLICATION VALIDATION & COLLISION CHECKING
    # -------------------------------------------------------------------------
    print("\n[Step 3] Pre-verifying candidate bounding boxes and checking collisions against b001...")
    applied_candidates = []
    excluded_candidates = []
    all_batch_records = []

    for r in queue_rows:
        cid = r["candidate_id"].strip()
        v = r["human_verdict"].strip()
        split = r["split"].strip()
        image = r["image"].strip()
        stem = Path(image).stem

        # Verify source image exists in base verified b001
        img_p = SOURCE_IMAGES_DIR / split / image
        if not img_p.exists():
            print(f"FATAL [BLOCKED]: Source image missing: {img_p}")
            sys.exit(1)

        # Verify source label file exists in base verified b001
        lbl_p = SOURCE_LABELS_DIR / split / f"{stem}.txt"
        if not lbl_p.exists():
            print(f"FATAL [BLOCKED]: Source label missing: {lbl_p}")
            sys.exit(1)

        if v == "PASS":
            bbox_str = r["normalized_bbox"].strip()
            xc, yc, w, h = parse_bbox_tuple(bbox_str)
            if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                print(f"FATAL [BLOCKED]: Candidate {cid} out of bounds: ({xc}, {yc}, {w}, {h})")
                sys.exit(1)

            rec = {
                "candidate_id": cid,
                "split": split,
                "image": image,
                "stem": stem,
                "verdict": "PASS",
                "application_status": "ADDED_PASS",
                "action": "ADD",
                "confidence": float(r["confidence"]),
                "original_bbox": r["normalized_bbox"].strip(),
                "final_bbox": f"({xc:.8f}, {yc:.8f}, {w:.8f}, {h:.8f})",
                "bbox_floats": (xc, yc, w, h),
                "resulting_label_path": f"labels/{split}/{stem}.txt",
                "notes": r["reviewer_notes"].strip(),
                "reviewer": r["reviewer"].strip(),
                "reviewed_at": r["reviewed_at"].strip(),
            }
            applied_candidates.append(rec)
            all_batch_records.append(rec)

        elif v == "FIX":
            if cid != "CAND_001600":
                print(f"FATAL [BLOCKED]: Unexpected FIX candidate {cid}")
                sys.exit(1)
            bbox_str = r["corrected_bbox"].strip()
            if not bbox_str:
                print(f"FATAL [BLOCKED]: FIX candidate {cid} missing corrected_bbox")
                sys.exit(1)
            xc, yc, w, h = parse_bbox_tuple(bbox_str)
            if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                print(f"FATAL [BLOCKED]: FIX candidate {cid} out of bounds: ({xc}, {yc}, {w}, {h})")
                sys.exit(1)

            rec = {
                "candidate_id": cid,
                "split": split,
                "image": image,
                "stem": stem,
                "verdict": "FIX",
                "application_status": "ADDED_FIX",
                "action": "ADD",
                "confidence": float(r["confidence"]),
                "original_bbox": r["normalized_bbox"].strip(),
                "final_bbox": f"({xc:.8f}, {yc:.8f}, {w:.8f}, {h:.8f})",
                "bbox_floats": (xc, yc, w, h),
                "resulting_label_path": f"labels/{split}/{stem}.txt",
                "notes": r["reviewer_notes"].strip(),
                "reviewer": r["reviewer"].strip(),
                "reviewed_at": r["reviewed_at"].strip(),
            }
            applied_candidates.append(rec)
            all_batch_records.append(rec)

        elif v == "REMOVE":
            rec = {
                "candidate_id": cid,
                "split": split,
                "image": image,
                "stem": stem,
                "verdict": "REMOVE",
                "application_status": "EXCLUDED_REMOVE",
                "action": "EXCLUDE",
                "confidence": float(r["confidence"]),
                "original_bbox": r["normalized_bbox"].strip(),
                "final_bbox": "None",
                "bbox_floats": None,
                "resulting_label_path": "NONE",
                "notes": r["reviewer_notes"].strip(),
                "reviewer": r["reviewer"].strip(),
                "reviewed_at": r["reviewed_at"].strip(),
            }
            excluded_candidates.append(rec)
            all_batch_records.append(rec)
        else:
            print(f"FATAL [BLOCKED]: Unexpected verdict {v} for candidate {cid}")
            sys.exit(1)

    print(f"  Applied candidates: {len(applied_candidates)} (185 PASS + 1 FIX)")
    print(f"  Excluded candidates: {len(excluded_candidates)} (14 REMOVE)")

    # Check collisions with existing person labels in dfire_remediated_verified_b001
    collision_errors = []
    for cand in applied_candidates:
        lbl_p = SOURCE_DIR / cand["resulting_label_path"]
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id == 0:  # Existing person
                    ex_box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    iou = compute_iou_xywh(cand["bbox_floats"], ex_box)
                    if iou > 0.50:
                        collision_errors.append(
                            f"Collision: {cand['candidate_id']} on {cand['resulting_label_path']} has IoU={iou:.4f} with existing person box {ex_box}"
                        )

    if collision_errors:
        print(f"FATAL [BLOCKED]: {len(collision_errors)} collision(s) detected:")
        for cerr in collision_errors:
            print(f"  {cerr}")
        sys.exit(1)

    print("  Zero same-class collisions detected. All candidate boxes safe to apply!")

    # -------------------------------------------------------------------------
    # STEP 4: BUILD TARGET DATASET (NTFS Hardlinks for Images, Copied Labels)
    # -------------------------------------------------------------------------
    print(f"\n[Step 4] Building versioned dataset directory: {TARGET_DIR.name}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 4a. Link Images
    print("  Creating NTFS hardlinks for images from b001...")
    img_link_count = 0
    for split in SPLITS:
        src_s_img = SOURCE_IMAGES_DIR / split
        dst_s_img = TARGET_IMAGES_DIR / split
        dst_s_img.mkdir(parents=True, exist_ok=True)
        for img_file in src_s_img.iterdir():
            if img_file.is_file():
                dst_file = dst_s_img / img_file.name
                if dst_file.exists():
                    dst_file.unlink()
                os.link(str(img_file), str(dst_file))
                img_link_count += 1

    print(f"  Hardlinked {img_link_count} images across all splits.")
    if img_link_count != EXPECTED_FILE_COUNT:
        print(f"FATAL: Image link count {img_link_count} != {EXPECTED_FILE_COUNT}")
        sys.exit(1)

    # 4b. Copy Independent Labels
    print("  Copying independent label files from b001...")
    lbl_copy_count = 0
    for split in SPLITS:
        src_s_lbl = SOURCE_LABELS_DIR / split
        dst_s_lbl = TARGET_LABELS_DIR / split
        dst_s_lbl.mkdir(parents=True, exist_ok=True)
        for lbl_file in src_s_lbl.iterdir():
            if lbl_file.is_file() and lbl_file.suffix.lower() == ".txt":
                dst_file = dst_s_lbl / lbl_file.name
                shutil.copyfile(lbl_file, dst_file)
                lbl_copy_count += 1

    print(f"  Copied {lbl_copy_count} independent label files across all splits.")
    if lbl_copy_count != EXPECTED_FILE_COUNT:
        print(f"FATAL: Label copy count {lbl_copy_count} != {EXPECTED_FILE_COUNT}")
        sys.exit(1)

    # Heartbeat immediately after copy
    update_heartbeat("HEARTBEAT_IMMEDIATELY_AFTER_COPY", "RUNNING", f"Hardlinked {img_link_count} images, copied {lbl_copy_count} labels")

    # 4c. Apply the 186 Candidates to Labels
    print("  Applying 186 verified person boxes to copied labels...")
    applied_files = set()
    for cand in applied_candidates:
        dst_lbl_p = TARGET_DIR / cand["resulting_label_path"]
        xc, yc, w, h = cand["bbox_floats"]
        yolo_line = f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n"

        curr_text = dst_lbl_p.read_text(encoding="utf-8")
        if curr_text and not curr_text.endswith("\n"):
            curr_text += "\n"
        new_text = curr_text + yolo_line
        dst_lbl_p.write_text(new_text, encoding="utf-8")
        applied_files.add(dst_lbl_p)

    print(f"  Appended 186 boxes into {len(applied_files)} label files.")

    # Heartbeat after applying labels
    update_heartbeat("HEARTBEAT_AFTER_APPLYING_LABELS", "RUNNING", f"Successfully appended 186 person boxes into {len(applied_files)} label files")

    # -------------------------------------------------------------------------
    # STEP 5: WRITE DATA.YAML & DATASET MANIFEST
    # -------------------------------------------------------------------------
    print("\n[Step 5] Writing data.yaml and dataset manifests...")

    # 5a. data.yaml
    data_yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: person
  1: helmet
  2: vest
  3: fall
  4: fire
  5: smoke
"""
    TARGET_DATA_YAML.write_text(data_yaml_content, encoding="utf-8")
    print(f"  Created {TARGET_DATA_YAML.name}")

    # 5b. Compute exact final counts
    class_counter = Counter()
    split_class_counter = {s: Counter() for s in SPLITS}
    total_lbl_files = 0

    for split in SPLITS:
        s_lbl_dir = TARGET_LABELS_DIR / split
        for lbl_file in s_lbl_dir.iterdir():
            if lbl_file.is_file() and lbl_file.suffix.lower() == ".txt":
                total_lbl_files += 1
                for line in lbl_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        class_counter[cls_id] += 1
                        split_class_counter[split][cls_id] += 1

    total_boxes = sum(class_counter.values())
    print(f"\nTarget Dataset Final Class Counts:")
    for cid in range(6):
        cname = ["person", "helmet", "vest", "fall", "fire", "smoke"][cid]
        cnt = class_counter.get(cid, 0)
        exp = EXPECTED_FINAL_COUNTS.get(cid, 0)
        print(f"  Class {cid} ({cname}): {cnt:,} (expected {exp:,})")
        if cnt != exp:
            print(f"FATAL: Count mismatch for class {cid} ({cname}): {cnt} != {exp}")
            sys.exit(1)

    print(f"  Total Bounding Boxes: {total_boxes:,} (expected {EXPECTED_TOTAL_BOXES:,})")
    if total_boxes != EXPECTED_TOTAL_BOXES:
        print(f"FATAL: Total boxes {total_boxes} != {EXPECTED_TOTAL_BOXES}")
        sys.exit(1)

    # 5c. dataset_manifest.json
    manifest_data = {
        "dataset_name": "dfire_remediated_verified_b002",
        "version": "2.4.0-verified-batch002",
        "date_rebuilt": datetime.now(timezone.utc).astimezone().isoformat(),
        "description": "Trusted snapshot of D-Fire dataset with visual-QA verified labels including Phase 1 HIGH, Phase 2 MEDIUM sample, Human QA Batch 1 MEDIUM persons (155 PASS + 1 FIX applied), and Human QA Batch 2 MEDIUM persons (185 PASS + 1 FIX applied). All unreviewed and excluded candidates safely omitted.",
        "immutable_base": "data/processed/dfire_remediated_verified_b001",
        "base_labels_sha256": source_b001_hash,
        "total_images": EXPECTED_FILE_COUNT,
        "total_boxes": total_boxes,
        "canonical_classes": {str(k): v for k, v in sorted(class_counter.items())},
        "canonical_classes_per_split": {
            s: {str(k): v for k, v in sorted(split_class_counter[s].items())}
            for s in SPLITS
        },
        "phase1_verified_candidates": {
            "census_population": 1569,
            "pass_retained": 1561,
            "fix_adjusted": 1,
            "remove_rejected": 7,
            "total_applied": 1562,
        },
        "phase2_verified_candidates": {
            "reviewed_population": 326,
            "pass_added": 269,
            "fix_adjusted": 1,
            "remove_rejected": 54,
            "uncertain_escalated": 2,
            "total_applied": 270,
        },
        "human_qa_batch_001": {
            "reviewed_population": 200,
            "pass_added": 155,
            "fix_adjusted": 1,
            "remove_rejected": 43,
            "uncertain_escalated": 1,
            "total_applied": 156,
        },
        "human_qa_batch_002": {
            "reviewed_population": 200,
            "pass_added": 185,
            "fix_adjusted": 1,
            "remove_rejected": 14,
            "uncertain_escalated": 0,
            "total_applied": 186,
        },
        "remaining_unreviewed_pool": {
            "remaining_unreviewed_medium_person": 1240,
            "unreviewed_low_person": 1978,
            "unreviewed_low_helmet": 66,
            "total_unreviewed": 3284,
            "status": "EXCLUDED_PENDING_GENUINE_VISUAL_QA",
            "notes": "1,240 unreviewed MEDIUM person candidates and all LOW candidates remain excluded pending future visual QA.",
        },
    }
    TARGET_DATASET_MANIFEST.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"  Created {TARGET_DATASET_MANIFEST.name}")

    # -------------------------------------------------------------------------
    # STEP 6: WRITE APPLICATION MANIFESTS
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating batch_002_application_manifest (CSV & JSON)...")

    # CSV
    with open(APP_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "candidate_id",
            "split",
            "image",
            "confidence",
            "human_verdict",
            "application_status",
            "action",
            "original_bbox",
            "final_bbox",
            "resulting_label_path",
            "reviewer_notes",
            "reviewer",
            "reviewed_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_batch_records:
            writer.writerow({
                "candidate_id": rec["candidate_id"],
                "split": rec["split"],
                "image": rec["image"],
                "confidence": f"{rec['confidence']:.6f}",
                "human_verdict": rec["verdict"],
                "application_status": rec["application_status"],
                "action": rec["action"],
                "original_bbox": rec["original_bbox"],
                "final_bbox": rec["final_bbox"],
                "resulting_label_path": rec["resulting_label_path"],
                "reviewer_notes": rec["notes"],
                "reviewer": rec["reviewer"],
                "reviewed_at": rec["reviewed_at"],
            })
    print(f"  Created {APP_MANIFEST_CSV.name}")

    # JSON
    app_manifest_json_data = {
        "batch_id": "human_qa_medium_person_batch_002",
        "applied_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_queue": str(QA_QUEUE_CSV.relative_to(ROOT)),
        "target_dataset": str(TARGET_DIR.relative_to(ROOT)),
        "summary": {
            "total_candidates": 200,
            "added_pass": 185,
            "added_fix": 1,
            "total_added": 186,
            "excluded_remove": 14,
            "excluded_uncertain": 0,
            "total_excluded": 14,
        },
        "records": [
            {
                "candidate_id": rec["candidate_id"],
                "split": rec["split"],
                "image": rec["image"],
                "confidence": rec["confidence"],
                "human_verdict": rec["verdict"],
                "application_status": rec["application_status"],
                "action": rec["action"],
                "original_bbox": rec["original_bbox"],
                "final_bbox": rec["final_bbox"],
                "resulting_label_path": rec["resulting_label_path"],
            }
            for rec in all_batch_records
        ],
    }
    APP_MANIFEST_JSON.write_text(json.dumps(app_manifest_json_data, indent=2), encoding="utf-8")
    print(f"  Created {APP_MANIFEST_JSON.name}")

    # -------------------------------------------------------------------------
    # STEP 7: SYNCHRONIZE REMEDIATION MANIFEST
    # -------------------------------------------------------------------------
    print("\n[Step 7] Synchronizing remediation_manifest.csv from b001...")
    with open(SOURCE_MANIFEST, "r", encoding="utf-8") as f:
        base_manifest_rows = list(csv.DictReader(f))

    print(f"  Loaded {len(base_manifest_rows)} rows from {SOURCE_MANIFEST.name}")
    batch_map = {r["candidate_id"]: r for r in all_batch_records}

    updated_manifest_rows = []
    batch_updated_count = 0

    for r in base_manifest_rows:
        cid = r["candidate_id"]
        if cid in batch_map:
            b_info = batch_map[cid]
            new_r = dict(r)
            new_r["phase"] = "Batch 2 (MEDIUM)"
            new_r["visual_verdict"] = b_info["verdict"]
            new_r["evidence_source_artifact"] = "docs/audit_artifacts/dfire/human_qa_medium_person_batch_002/qa_queue.csv"
            new_r["action"] = b_info["action"]
            new_r["status"] = "ADDED" if b_info["action"] == "ADD" else "EXCLUDED"
            new_r["final_bbox"] = b_info["final_bbox"]
            new_r["resulting_label_path"] = b_info["resulting_label_path"]
            new_r["notes"] = b_info["notes"]
            updated_manifest_rows.append(new_r)
            batch_updated_count += 1
        else:
            updated_manifest_rows.append(r)

    print(f"  Updated {batch_updated_count} manifest rows for Batch 2.")
    if batch_updated_count != 200:
        print(f"FATAL: Updated manifest rows count {batch_updated_count} != 200")
        sys.exit(1)

    # Check remaining unreviewed MEDIUM persons count
    remaining_unreviewed = [
        r for r in updated_manifest_rows
        if r["phase"] == "Unreviewed (MEDIUM)" and r["class_name"] == "person" and r["status"] == "EXCLUDED"
    ]
    print(f"  Remaining unreviewed MEDIUM persons: {len(remaining_unreviewed)} (expected 1,240)")
    if len(remaining_unreviewed) != 1240:
        print(f"FATAL: Remaining unreviewed MEDIUM persons {len(remaining_unreviewed)} != 1240")
        sys.exit(1)

    # Write target remediation_manifest.csv
    with open(TARGET_MANIFEST, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(base_manifest_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_manifest_rows)
    print(f"  Created {TARGET_MANIFEST.name}")

    # -------------------------------------------------------------------------
    # STEP 8: POST-APPLICATION IMMUTABILITY CHECK
    # -------------------------------------------------------------------------
    print("\n[Step 8] Verifying post-application immutability of all source datasets...")
    raw_post_hash, raw_post_cnt = hash_raw_directory(RAW_DIR)
    corr_post_hash, corr_post_cnt = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_post_hash, remed_post_cnt = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_post_hash, verif_post_cnt = hash_directory_labels(VERIFIED_DIR / "labels")
    b001_post_hash, b001_post_cnt = hash_directory_labels(SOURCE_LABELS_DIR)

    if raw_post_hash != EXPECTED_RAW_HASH or raw_post_cnt != EXPECTED_FILE_COUNT:
        print("FATAL: data/raw was mutated during application!")
        sys.exit(1)
    if corr_post_hash != EXPECTED_CORR_HASH or corr_post_cnt != EXPECTED_FILE_COUNT:
        print("FATAL: dfire_corrected was mutated during application!")
        sys.exit(1)
    if remed_post_hash != EXPECTED_REMED_HASH or remed_post_cnt != EXPECTED_FILE_COUNT:
        print("FATAL: dfire_remediated was mutated during application!")
        sys.exit(1)
    if verif_post_hash != EXPECTED_VERIF_HASH or verif_post_cnt != EXPECTED_FILE_COUNT:
        print("FATAL: dfire_remediated_verified was mutated during application!")
        sys.exit(1)
    if b001_post_hash != source_b001_hash or b001_post_cnt != EXPECTED_FILE_COUNT:
        print("FATAL: dfire_remediated_verified_b001 was mutated during application!")
        sys.exit(1)

    target_labels_hash, target_labels_cnt = hash_directory_labels(TARGET_LABELS_DIR)
    print(f"\nTarget dataset labels hash: {target_labels_hash} ({target_labels_cnt} files)")

    # -------------------------------------------------------------------------
    # STEP 9: WRITE INTERIM FINAL REPORT & UPDATE STATUS
    # -------------------------------------------------------------------------
    report_content = f"""# Final Report: D-Fire Human QA Batch 002 Application

## 1. Task Reconciliation & Resumption
- **Task ID:** `dfire_batch002_apply`
- **Reconciliation Event:** Prior attempt was halted at 2026-09-26 18:48 without script execution; status was reconciled from RUNNING to `INTERRUPTED/BLOCKED`. Resumed under exclusive output ownership.
- **Execution Timestamp:** {run_timestamp}
- **Status:** `APPLICATION_COMPLETED` (Pending Validation)

## 2. Dataset Lineage & Integrity
- **Source Dataset:** `data/processed/dfire_remediated_verified_b001`
- **Target Dataset:** `data/processed/dfire_remediated_verified_b002` (independent copy, non-destructive)
- **Image Linking:** 21,527 NTFS hardlinks created
- **Label Duplication:** 21,527 independent label files copied
- **Target Labels SHA-256:** `{target_labels_hash}`

## 3. Batch 002 Verdict Application Summary
- **Queue Source:** `docs/audit_artifacts/dfire/human_qa_medium_person_batch_002/qa_queue.csv`
- **Reviewed Population:** 200 candidates
- **PASS Applied:** 185
- **FIX Applied:** 1 (`CAND_001600` with corrected coordinates `(0.073333, 0.630564, 0.146667, 0.738872)`)
- **REMOVE Excluded:** 14
- **UNCERTAIN:** 0
- **Total Net Person Additions:** +186

## 4. Reconciled Class Counts
| Class ID | Class Name | Baseline (b001) | Added (Batch 2) | Target Total (b002) | Expected | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | person | 1,887 | +186 | 2,073 | 2,073 | MATCH |
| 1 | helmet | 117 | +0 | 117 | 117 | MATCH |
| 2 | vest | 0 | +0 | 0 | 0 | MATCH |
| 3 | fall | 0 | +0 | 0 | 0 | MATCH |
| 4 | fire | 14,683 | +0 | 14,683 | 14,683 | MATCH |
| 5 | smoke | 11,854 | +0 | 11,854 | 11,854 | MATCH |
| **Total** | | **28,541** | **+186** | **28,727** | **28,727** | **MATCH** |

## 5. Pool Accounting
- **Remaining Unreviewed MEDIUM Persons:** 1,240 (1,440 - 200)
- **Unreviewed LOW Persons:** 1,978
- **Unreviewed LOW Helmets:** 66
- **Total Remaining Excluded Pool:** 3,284
"""
    FINAL_REPORT_MD.write_text(report_content, encoding="utf-8")
    print(f"  Created interim {FINAL_REPORT_MD.name}")

    update_heartbeat("HEARTBEAT_APPLY_COMPLETED", "APPLICATION_COMPLETED", f"Application complete in {time.perf_counter() - start_time:.2f}s; target labels hash: {target_labels_hash}")

    elapsed = time.perf_counter() - start_time
    print(f"\nAPPLICATION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
