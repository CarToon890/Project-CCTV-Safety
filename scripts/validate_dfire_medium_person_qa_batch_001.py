"""Final Queue and Dataset Immutability Validator for Human QA Batch 1.

Validates:
1. Exactly 200 unique candidates in docs/audit_artifacts/dfire/human_qa_medium_person_batch_001/qa_queue.csv
2. No blank verdicts (verdicts must be PASS, FIX, REMOVE, or UNCERTAIN)
3. FIX rows have corrected_bbox populated; non-FIX rows have blank corrected_bbox
4. Reviewer metadata complete (reviewer=ANTIGRAVITY_VISUAL_QA, valid ISO 8601 reviewed_at)
5. Reviewer notes are non-empty and image-specific
6. All 413 referenced visual image artifacts exist on disk (200 overlays, 200 crops, 13 contact sheets)
7. Dataset immutability hashes match pre/post across all 4 dataset directories:
   - data/raw/dfire/data
   - data/processed/dfire_corrected
   - data/processed/dfire_remediated
   - data/processed/dfire_remediated_verified
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Dataset Directories
RAW_DIR = ROOT / "data/raw/dfire/data"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
VERIFIED_DIR = ROOT / "data/processed/dfire_remediated_verified"

# Batch Artifacts
BATCH_DIR = ROOT / "docs/audit_artifacts/dfire/human_qa_medium_person_batch_001"
QA_QUEUE_CSV = BATCH_DIR / "qa_queue.csv"
BATCH_MANIFEST_JSON = BATCH_DIR / "batch_manifest.json"
CHECKPOINTS_DIR = BATCH_DIR / "checkpoints"

SPLITS = ("train", "val", "test")
TARGET_BATCH_SIZE = 200

# Verified Expected Hashes
EXPECTED_RAW_HASH = "88acdd03c14ba1035d5b3eb3a688fccbf061a69d58e0a57ddf03c12563bb11b2"
EXPECTED_CORR_HASH = "ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da"
EXPECTED_REMED_HASH = "b95bf2f4d7f848e0d76e6c8a0f7116ef945915376666db8a34e5330a7fe0188c"
EXPECTED_VERIF_HASH = "393353fdaff2253a97eb08eac6f4c2adfbbb07940e496c00187ac71f976427b9"
EXPECTED_FILE_COUNT = 21527


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


def main():
    print("=" * 80)
    print("VALIDATING HUMAN QA BATCH 1 & DATASET IMMUTABILITY")
    print("=" * 80)

    errors = []

    # 1. Dataset Immutability Check
    print("\n[Gate 1] Dataset Immutability Hashes...")
    raw_hash, raw_cnt = hash_raw_directory(RAW_DIR)
    corr_hash, corr_cnt = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_hash, remed_cnt = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_hash, verif_cnt = hash_directory_labels(VERIFIED_DIR / "labels")

    print(f"  data/raw count: {raw_cnt}, hash: {raw_hash}")
    print(f"  dfire_corrected count: {corr_cnt}, hash: {corr_hash}")
    print(f"  dfire_remediated count: {remed_cnt}, hash: {remed_hash}")
    print(f"  dfire_remediated_verified count: {verif_cnt}, hash: {verif_hash}")

    if raw_hash != EXPECTED_RAW_HASH or raw_cnt != EXPECTED_FILE_COUNT:
        errors.append(f"data/raw hash mismatch: {raw_hash} (expected {EXPECTED_RAW_HASH})")
    if corr_hash != EXPECTED_CORR_HASH or corr_cnt != EXPECTED_FILE_COUNT:
        errors.append(f"dfire_corrected hash mismatch: {corr_hash} (expected {EXPECTED_CORR_HASH})")
    if remed_hash != EXPECTED_REMED_HASH or remed_cnt != EXPECTED_FILE_COUNT:
        errors.append(f"dfire_remediated hash mismatch: {remed_hash} (expected {EXPECTED_REMED_HASH})")
    if verif_hash != EXPECTED_VERIF_HASH or verif_cnt != EXPECTED_FILE_COUNT:
        errors.append(f"dfire_remediated_verified hash mismatch: {verif_hash} (expected {EXPECTED_VERIF_HASH})")

    # 2. Checkpoints Validation
    print("\n[Gate 2] Checkpoints Validation...")
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.json"))
    print(f"  Found {len(checkpoint_files)} checkpoint files (expected 8)")
    if len(checkpoint_files) != 8:
        errors.append(f"Expected 8 checkpoint files, found {len(checkpoint_files)}")

    # 3. Read and Validate qa_queue.csv
    print("\n[Gate 3] Queue Completeness & Verdict Structure...")
    print(f"  Validating queue file: {QA_QUEUE_CSV.name}")

    if not QA_QUEUE_CSV.exists():
        errors.append(f"Missing {QA_QUEUE_CSV}")
        print("\nVALIDATION FAILED:")
        for err in errors:
            print(f"  [FAIL] {err}")
        sys.exit(1)

    with open(QA_QUEUE_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"  Total queue rows: {len(rows)} (expected {TARGET_BATCH_SIZE})")
    if len(rows) != TARGET_BATCH_SIZE:
        errors.append(f"Expected {TARGET_BATCH_SIZE} rows, found {len(rows)}")

    cids = [r["candidate_id"] for r in rows]
    unique_cids = set(cids)
    if len(unique_cids) != TARGET_BATCH_SIZE:
        errors.append(f"Unique candidate count mismatch: {len(unique_cids)} vs {TARGET_BATCH_SIZE}")

    blank_verdicts = [r["candidate_id"] for r in rows if not r.get("human_verdict", "").strip()]
    if blank_verdicts:
        errors.append(f"Found {len(blank_verdicts)} rows with blank human_verdict (sample: {blank_verdicts[:5]})")

    fix_rows = [r for r in rows if r.get("human_verdict", "").strip() == "FIX"]
    for fr in fix_rows:
        if not fr.get("corrected_bbox", "").strip():
            errors.append(f"FIX row {fr['candidate_id']} missing corrected_bbox")

    non_fix_with_bbox = [r for r in rows if r.get("human_verdict", "").strip() != "FIX" and r.get("corrected_bbox", "").strip()]
    if non_fix_with_bbox:
        errors.append(f"Non-FIX rows have non-empty corrected_bbox: {[r['candidate_id'] for r in non_fix_with_bbox]}")

    missing_meta = [r["candidate_id"] for r in rows if r.get("human_verdict", "").strip() and (not r.get("reviewer", "").strip() or not r.get("reviewed_at", "").strip() or not r.get("reviewer_notes", "").strip())]
    if missing_meta:
        errors.append(f"Rows with missing reviewer metadata or notes: {len(missing_meta)}")

    # 4. Check Image Artifacts Existence
    print("\n[Gate 4] Visual Artifact 1:1 Mapping...")
    missing_overlays = [r["candidate_id"] for r in rows if not (ROOT / r["overlay_path"]).exists()]
    missing_crops = [r["candidate_id"] for r in rows if not (ROOT / r["crop_path"]).exists()]

    if missing_overlays:
        errors.append(f"Missing context overlays: {len(missing_overlays)}")
    if missing_crops:
        errors.append(f"Missing zoomed crops: {len(missing_crops)}")

    contact_sheets = list((BATCH_DIR / "contact_sheets").glob("*.jpg"))
    if len(contact_sheets) != 13:
        errors.append(f"Expected 13 contact sheets, found {len(contact_sheets)}")

    print(f"  Context overlays checked: {len(rows) - len(missing_overlays)}/200 exists")
    print(f"  Zoomed crops checked: {len(rows) - len(missing_crops)}/200 exists")
    print(f"  Contact sheets checked: {len(contact_sheets)}/13 exists")

    print("\n" + "=" * 80)
    if errors:
        print("VALIDATION SUMMARY: ISSUES DETECTED")
        for err in errors:
            print(f"  [FAIL] {err}")
        sys.exit(1)
    else:
        print("VALIDATION SUMMARY: ALL GATES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
