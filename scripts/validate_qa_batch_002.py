#!/usr/bin/env python
"""
Validation script for Human QA Batch 2 (D-Fire MEDIUM person candidates).
Validates:
1. Candidate pool & sampling fidelity (200 unique eligible IDs from 1440 pool).
2. Queue completeness (zero blanks, valid verdicts, FIX coordinate completeness, metadata).
3. Visual artifact mapping (200 overlays, 200 crops, 13 contact sheets).
4. Checkpoint coverage (8 checkpoints every 25 rows from 025 to 200).
5. Dataset immutability hashes for raw, dfire_corrected, dfire_remediated, dfire_remediated_verified, and dfire_remediated_verified_b001.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT_DIR = REPO_ROOT / "data" / "processed" / "dfire_remediated_verified_b001"
MANIFEST_PATH = SNAPSHOT_DIR / "remediation_manifest.csv"
BATCH_DIR = REPO_ROOT / "docs" / "audit_artifacts" / "dfire" / "human_qa_medium_person_batch_002"
QUEUE_PATH = BATCH_DIR / "qa_queue.csv"
CHECKPOINTS_DIR = BATCH_DIR / "checkpoints"
OVERLAYS_DIR = BATCH_DIR / "overlays"
CROPS_DIR = BATCH_DIR / "crops"
CONTACT_SHEETS_DIR = BATCH_DIR / "contact_sheets"
BATCH_MANIFEST_PATH = BATCH_DIR / "batch_manifest.json"
README_PATH = BATCH_DIR / "README.md"

# Known authoritative hashes
EXPECTED_HASHES = {
    "data_raw_dfire_data": (
        REPO_ROOT / "data" / "raw" / "dfire" / "data",
        43054,
        "6a4bd07ed422c5f8453016f503751bff8a15840accc543971e2587a40beb628b",
    ),
    "data_processed_dfire_corrected_labels": (
        REPO_ROOT / "data" / "processed" / "dfire_corrected" / "labels",
        21527,
        "336b911ba6d6075137fcc213c434e6e96d99a7449d84c92e33b8f7153719aea0",
    ),
    "data_processed_dfire_remediated_labels": (
        REPO_ROOT / "data" / "processed" / "dfire_remediated" / "labels",
        21527,
        "a3fc8eac0bd1b08beae84f0aae46df44ac73f02701a33382c01d9c627a0c2e43",
    ),
    "data_processed_dfire_remediated_verified_labels": (
        REPO_ROOT / "data" / "processed" / "dfire_remediated_verified" / "labels",
        21527,
        "fe29c3c5cf79c522defe222f15e9d79956ed3e85ed74f058119ea38ec8a17887",
    ),
    "data_processed_dfire_remediated_verified_b001_labels": (
        REPO_ROOT / "data" / "processed" / "dfire_remediated_verified_b001" / "labels",
        21527,
        "8b9a0fca8071646ac0045b007406615fe436ca04194aab69bbade5a2c2e5d85e",
    ),
}


def compute_dir_sha256(dir_path: Path) -> Tuple[int, str]:
    if not dir_path.exists():
        return 0, "MISSING"
    files = sorted(
        [p for p in dir_path.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(dir_path)).replace("\\", "/"),
    )
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(dir_path)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(f.read_bytes())
    return len(files), h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("VALIDATION: D-Fire Human QA Batch 2 (MEDIUM Person Candidates)")
    print("=" * 70)
    all_passed = True

    # 1. Authoritative manifest and eligible pool
    print("\n[CHECK 1] Authoritative Manifest & Eligible Pool...")
    if not MANIFEST_PATH.exists():
        print(f"  FAIL: Manifest missing at {MANIFEST_PATH}")
        return 1

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    eligible_ids = set()
    for r in manifest_rows:
        if (
            "Unreviewed" in r["phase"]
            and "MEDIUM" in r["phase"]
            and r["class_name"] == "person"
            and r["status"] == "EXCLUDED"
            and r["action"] == "EXCLUDED_PENDING_GENUINE_VISUAL_QA"
        ):
            eligible_ids.add(r["candidate_id"])

    print(f"  Total remediation_manifest rows: {len(manifest_rows)}")
    print(f"  Eligible unreviewed MEDIUM person pool: {len(eligible_ids)}")
    if len(eligible_ids) == 1440:
        print("  PASS: Eligible candidate pool matches expected count of 1440.")
    else:
        print(f"  FAIL: Expected 1440 eligible candidates, got {len(eligible_ids)}")
        all_passed = False

    # 2. QA Queue Completeness and Correctness
    print("\n[CHECK 2] QA Queue Rows and Review Completeness...")
    if not QUEUE_PATH.exists():
        print(f"  FAIL: qa_queue.csv missing at {QUEUE_PATH}")
        return 1

    with QUEUE_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        queue_rows = list(reader)

    print(f"  Queue row count: {len(queue_rows)}")
    if len(queue_rows) != 200:
        print(f"  FAIL: Expected 200 rows in queue, found {len(queue_rows)}")
        all_passed = False
    else:
        print("  PASS: Exactly 200 candidate rows in queue.")

    queue_cids = [r["candidate_id"] for r in queue_rows]
    unique_cids = set(queue_cids)
    if len(unique_cids) != 200:
        print(f"  FAIL: Candidate IDs not unique! Unique count: {len(unique_cids)}")
        all_passed = False
    else:
        print("  PASS: All 200 candidate IDs are strictly unique.")

    # All candidate IDs in eligible pool
    ineligible = [cid for cid in queue_cids if cid not in eligible_ids]
    if ineligible:
        print(f"  FAIL: Candidates not in eligible pool: {ineligible}")
        all_passed = False
    else:
        print("  PASS: All 200 selected candidates strictly belong to the 1,440 unreviewed MEDIUM person pool.")

    # Verdicts, notes, reviewer, timestamps
    verdicts = {}
    blank_verdicts = 0
    blank_notes = 0
    blank_reviewer = 0
    blank_reviewed_at = 0
    fix_rows = []
    uncertain_rows = []

    for idx, r in enumerate(queue_rows, 1):
        v = r.get("human_verdict", "").strip()
        notes = r.get("reviewer_notes", "").strip()
        rev = r.get("reviewer", "").strip()
        rev_at = r.get("reviewed_at", "").strip()
        cid = r["candidate_id"]

        if not v:
            blank_verdicts += 1
        else:
            verdicts[v] = verdicts.get(v, 0) + 1

        if not notes:
            blank_notes += 1
        elif len(notes) < 15:
            print(f"  WARN: Row {idx} ({cid}) has very short reviewer_notes: '{notes}'")

        if not rev or rev != "ANTIGRAVITY_VISUAL_QA":
            blank_reviewer += 1

        if not rev_at:
            blank_reviewed_at += 1

        if v == "FIX":
            fix_rows.append(r)
        elif v == "UNCERTAIN":
            uncertain_rows.append(r)

    print(f"  Verdict distribution: {verdicts}")
    if blank_verdicts > 0:
        print(f"  FAIL: Found {blank_verdicts} blank verdicts in queue!")
        all_passed = False
    else:
        print("  PASS: Zero blank verdicts.")

    if blank_notes > 0 or blank_reviewer > 0 or blank_reviewed_at > 0:
        print(f"  FAIL: Metadata completeness failure! Blank notes: {blank_notes}, blank reviewer: {blank_reviewer}, blank reviewed_at: {blank_reviewed_at}")
        all_passed = False
    else:
        print("  PASS: Metadata complete (all rows have reviewer='ANTIGRAVITY_VISUAL_QA', reviewed_at timestamp, and detailed reviewer_notes).")

    allowed_verdicts = {"PASS", "FIX", "REMOVE", "UNCERTAIN"}
    invalid_verdicts = set(verdicts.keys()) - allowed_verdicts
    if invalid_verdicts:
        print(f"  FAIL: Invalid verdict tokens found: {invalid_verdicts}")
        all_passed = False
    else:
        print(f"  PASS: All verdict tokens are valid ({sorted(allowed_verdicts)}).")

    # FIX coordinate validation
    print("\n[CHECK 3] FIX Item Coordinate Completeness...")
    print(f"  Total FIX items: {len(fix_rows)}")
    for fr in fix_rows:
        cid = fr["candidate_id"]
        cbbox = fr.get("corrected_bbox", "").strip()
        if not cbbox:
            print(f"  FAIL: FIX candidate {cid} has empty corrected_bbox!")
            all_passed = False
        else:
            try:
                coords = ast.literal_eval(cbbox)
                if not (isinstance(coords, (tuple, list)) and len(coords) == 4):
                    print(f"  FAIL: FIX candidate {cid} corrected_bbox not a 4-tuple: {cbbox}")
                    all_passed = False
                elif not all(isinstance(x, (int, float)) and x >= 0 for x in coords):
                    print(f"  FAIL: FIX candidate {cid} coordinates invalid: {coords}")
                    all_passed = False
                else:
                    print(f"  PASS: FIX candidate {cid} has valid corrected_bbox: {coords}")
            except Exception as e:
                print(f"  FAIL: FIX candidate {cid} could not parse corrected_bbox '{cbbox}': {e}")
                all_passed = False

    # Check non-FIX items have empty corrected_bbox
    non_fix_with_bbox = [r["candidate_id"] for r in queue_rows if r.get("human_verdict") != "FIX" and r.get("corrected_bbox", "").strip()]
    if non_fix_with_bbox:
        print(f"  FAIL: Non-FIX items have corrected_bbox populated: {non_fix_with_bbox}")
        all_passed = False
    else:
        print("  PASS: All non-FIX items have empty corrected_bbox.")

    # 3. Visual Artifact Mapping
    print("\n[CHECK 4] Visual Artifact Files Mapping...")
    overlays = list(OVERLAYS_DIR.glob("*.jpg"))
    crops = list(CROPS_DIR.glob("*.jpg"))
    sheets = list(CONTACT_SHEETS_DIR.glob("*.jpg"))

    print(f"  Overlays count: {len(overlays)} / 200")
    print(f"  Crops count: {len(crops)} / 200")
    print(f"  Contact sheets count: {len(sheets)} / 13")

    if len(overlays) != 200 or len(crops) != 200 or len(sheets) != 13:
        print("  FAIL: Artifact count mismatch!")
        all_passed = False
    else:
        print("  PASS: Exact visual artifact counts present.")

    # Check that each queue row references an existing overlay and crop
    missing_artifacts = []
    for r in queue_rows:
        ov_p = REPO_ROOT / r["overlay_path"]
        cr_p = REPO_ROOT / r["crop_path"]
        if not ov_p.exists() or ov_p.stat().st_size == 0:
            missing_artifacts.append(f"Overlay missing/empty: {r['overlay_path']}")
        if not cr_p.exists() or cr_p.stat().st_size == 0:
            missing_artifacts.append(f"Crop missing/empty: {r['crop_path']}")

    if missing_artifacts:
        print(f"  FAIL: Missing or empty artifact files: {missing_artifacts[:5]}")
        all_passed = False
    else:
        print("  PASS: All 200 queue rows have verified non-empty context overlays and zoomed crops on disk.")

    # 4. Checkpoints Coverage
    print("\n[CHECK 5] Review Checkpoints Coverage (every 25 rows)...")
    expected_checkpoints = [
        ("checkpoint_025.json", 25),
        ("checkpoint_050.json", 50),
        ("checkpoint_075.json", 75),
        ("checkpoint_100.json", 100),
        ("checkpoint_125.json", 125),
        ("checkpoint_150.json", 150),
        ("checkpoint_175.json", 175),
        ("checkpoint_200.json", 200),
    ]

    for ckpt_name, expected_count in expected_checkpoints:
        ckpt_path = CHECKPOINTS_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"  FAIL: Checkpoint missing: {ckpt_name}")
            all_passed = False
            continue
        try:
            ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            rc = ckpt_data.get("rows_completed")
            if rc != expected_count:
                print(f"  FAIL: {ckpt_name} has rows_completed={rc}, expected {expected_count}")
                all_passed = False
            else:
                print(f"  PASS: {ckpt_name} present and verified (rows_completed={rc}).")
        except Exception as e:
            print(f"  FAIL: {ckpt_name} invalid JSON: {e}")
            all_passed = False

    # 5. Batch Manifest and README
    print("\n[CHECK 6] Batch Manifest & README.md Documentation...")
    if not BATCH_MANIFEST_PATH.exists() or BATCH_MANIFEST_PATH.stat().st_size == 0:
        print(f"  FAIL: batch_manifest.json missing or empty at {BATCH_MANIFEST_PATH}")
        all_passed = False
    else:
        b_manifest = json.loads(BATCH_MANIFEST_PATH.read_text(encoding="utf-8"))
        if b_manifest.get("batch_size") == 200 and b_manifest.get("seed_and_ordering", {}).get("seed") == 43:
            print("  PASS: batch_manifest.json valid, seed=43, batch_size=200.")
        else:
            print(f"  FAIL: batch_manifest.json metadata incorrect: {b_manifest.get('batch_size')}")
            all_passed = False

    if not README_PATH.exists() or README_PATH.stat().st_size == 0:
        print(f"  FAIL: README.md missing or empty at {README_PATH}")
        all_passed = False
    else:
        print("  PASS: README.md documentation present and non-empty.")

    # 6. Source Dataset Immutability Check
    print("\n[CHECK 7] Dataset Immutability Hash Verification...")
    for key, (path, exp_count, exp_hash) in EXPECTED_HASHES.items():
        cnt, sha = compute_dir_sha256(path)
        if cnt == exp_count and sha == exp_hash:
            print(f"  PASS: {key} (count={cnt}, sha256={sha[:16]}...) UNCHANGED")
        else:
            print(f"  FAIL: {key} MUTATED!")
            print(f"    Expected count: {exp_count}, Actual: {cnt}")
            print(f"    Expected sha:   {exp_hash}")
            print(f"    Actual sha:     {sha}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL RESULT: ALL VALIDATION CHECKS PASSED [OK]")
        print("=" * 70)
        return 0
    else:
        print("OVERALL RESULT: VALIDATION CHECKS FAILED [ERROR]")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
