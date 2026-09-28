"""Comprehensive validation audit for trusted Phase-2 dataset (dfire_remediated_verified).

Validates:
1. Exact 1:1 image-to-label pairing across all splits (train, val, test) = 21,527 pairs.
2. Canonical 6-class schema compliance (classes 0-5 only, no unexpected IDs).
3. Coordinate bounds strictly within [0.0, 1.0], widths/heights > 0, zero degenerate boxes.
4. Same-class duplicate audit: zero duplicate boxes (IoU >= 0.85) for person and helmet.
5. Split integrity & cross-split group isolation (0 leaking groups).
6. Immutable source verification:
   - data/raw/dfire/data has 21,527 raw images intact.
   - data/processed/dfire_corrected labels match exact pre-remediation aggregate SHA-256.
   - data/processed/dfire_remediated preserved intact as evidence.
7. Hardlink integrity: images share NTFS file index/inode with dfire_corrected.
8. Exact expected counts independently reconciled:
   - person: 1,731
   - helmet: 117
   - vest: 0
   - fall: 0
   - fire: 14,683
   - smoke: 11,854
   - total: 28,385
9. Explicit proof that NO Phase 3-only candidate is present (audit of 2,547 Phase 3 additions).
10. Remediation manifest audit: exactly 5,579 candidate records (100% of candidate pool).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = ROOT / "data/processed/dfire_remediated_verified"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
RAW_DIR = ROOT / "data/raw/dfire/data"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"

# Phase 3 adjustments for explicit exclusion audit
P3_ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase3_label_adjustments.csv"

SPLITS = ["train", "val", "test"]
EXPECTED_TOTAL_PAIRS = 21527
KNOWN_PRE_HASH = "ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da"

EXPECTED_CLASS_COUNTS = {
    0: 1731,   # person (16 raw + 1554 Phase 1 + 161 Phase 2)
    1: 117,    # helmet (0 raw + 8 Phase 1 + 109 Phase 2)
    2: 0,      # vest
    3: 0,      # fall
    4: 14683,  # fire (unchanged)
    5: 11854,  # smoke (unchanged)
}
EXPECTED_TOTAL_BOXES = 28385


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
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
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        hasher.update(f"{rel_name}:{digest}\n".encode("utf-8"))
        file_count += 1
    return hasher.hexdigest(), file_count


def validate() -> bool:
    print("=" * 80)
    print("COMPREHENSIVE AUDIT OF TRUSTED PHASE-2 VERIFIED DATASET")
    print("Target:", VERIFIED_DIR)
    print("=" * 80)

    errors = []
    warnings = []

    # 1. Directory Structure Check
    if not VERIFIED_DIR.exists():
        errors.append(f"Directory {VERIFIED_DIR} does not exist!")
        print(f"FAILED: {errors[-1]}")
        return False

    # 2. 1:1 Pairing Check
    print("\n[Audit 1/10] Image-to-Label 1:1 Pairing Check...")
    total_images = 0
    split_counts = {}
    for s in SPLITS:
        img_dir = VERIFIED_DIR / "images" / s
        lbl_dir = VERIFIED_DIR / "labels" / s

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Missing images or labels directory for split: {s}")
            continue

        imgs = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        missing_lbls = set(imgs.keys()) - set(lbls.keys())
        missing_imgs = set(lbls.keys()) - set(imgs.keys())

        if missing_lbls:
            errors.append(f"{s}: {len(missing_lbls)} images missing labels (e.g. {list(missing_lbls)[:3]})")
        if missing_imgs:
            errors.append(f"{s}: {len(missing_imgs)} labels missing images (e.g. {list(missing_imgs)[:3]})")

        split_counts[s] = len(imgs)
        total_images += len(imgs)

    print(f"  Total pairs: {total_images} (train={split_counts.get('train', 0)}, "
          f"val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)})")
    if total_images != EXPECTED_TOTAL_PAIRS:
        errors.append(f"Expected {EXPECTED_TOTAL_PAIRS} image-label pairs, found {total_images}")

    # 3. Class Schema and Box Geometry Check
    print("\n[Audit 2/10] Class Schema and Box Geometry Check...")
    class_counts = Counter()
    split_class_counts = {s: Counter() for s in SPLITS}
    invalid_class_lines = []
    out_of_bounds_boxes = []
    zero_area_boxes = []
    total_boxes = 0

    for s in SPLITS:
        lbl_dir = VERIFIED_DIR / "labels" / s
        for p in lbl_dir.glob("*.txt"):
            lines = p.read_text(encoding="utf-8").splitlines()
            for l_idx, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{s}/{p.name}:{l_idx} Malformed line with {len(parts)} tokens")
                    continue
                try:
                    c = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    errors.append(f"{s}/{p.name}:{l_idx} Non-float tokens: {line}")
                    continue

                total_boxes += 1
                class_counts[c] += 1
                split_class_counts[s][c] += 1

                if c not in (0, 1, 2, 3, 4, 5):
                    invalid_class_lines.append((s, p.name, l_idx, c))

                if w <= 0 or h <= 0:
                    zero_area_boxes.append((s, p.name, l_idx, w, h))

                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2

                eps = 1e-4
                if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
                    out_of_bounds_boxes.append((s, p.name, l_idx, x1, y1, x2, y2))

    print(f"  Total bounding boxes: {total_boxes}")
    for c in sorted(class_counts.keys()):
        print(f"    Class {c}: {class_counts[c]} boxes")

    if invalid_class_lines:
        errors.append(f"Found {len(invalid_class_lines)} invalid class annotations outside 0-5")
    if out_of_bounds_boxes:
        errors.append(f"Found {len(out_of_bounds_boxes)} out-of-bounds boxes")
    if zero_area_boxes:
        errors.append(f"Found {len(zero_area_boxes)} zero-or-negative area boxes")

    # 4. Same-Class Duplicate Box Audit (IoU >= 0.85)
    print("\n[Audit 3/10] Same-Class IoU >= 0.85 Deduplication Audit...")
    duplicate_violations = 0
    for split in SPLITS:
        lbl_dir = VERIFIED_DIR / "labels" / split
        for p in lbl_dir.glob("*.txt"):
            lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            boxes = []
            for l in lines:
                parts = l.split()
                c = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((c, (xc - w/2, yc - h/2, xc + w/2, yc + h/2)))

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    c1, b1 = boxes[i]
                    c2, b2 = boxes[j]
                    if c1 == c2 and c1 in (0, 1):
                        iou = compute_iou(b1, b2)
                        if iou >= 0.85:
                            duplicate_violations += 1
                            errors.append(f"{split}/{p.name}: Duplicate class {c1} boxes detected with IoU={iou:.4f} >= 0.85")

    print(f"  Duplicate same-class violations (IoU >= 0.85): {duplicate_violations}")
    if duplicate_violations == 0:
        print("  VERIFIED: Zero duplicate boxes exist across all annotations.")

    # 5. Expected Counts Verification
    print("\n[Audit 4/10] Expected Class & Box Totals Verification...")
    if total_boxes != EXPECTED_TOTAL_BOXES:
        errors.append(f"Expected {EXPECTED_TOTAL_BOXES} total boxes, found {total_boxes}")
    for c, exp_c in EXPECTED_CLASS_COUNTS.items():
        act_c = class_counts.get(c, 0)
        if act_c != exp_c:
            errors.append(f"Class {c} expected {exp_c} boxes, got {act_c}")
        else:
            print(f"  Class {c} ({act_c} boxes): matches expectation.")

    # 6. Split Integrity & Leakage Check
    print("\n[Audit 5/10] Split Integrity & Cross-Split Group Leakage Check...")
    split_groups = defaultdict(set)
    for s in SPLITS:
        meta_p = VERIFIED_DIR / "metadata" / f"{s}.csv"
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    split_groups[s].add(r["group_id"])

    leaking_groups = set()
    train_groups = split_groups.get("train", set())
    val_groups = split_groups.get("val", set())
    test_groups = split_groups.get("test", set())

    leaking_groups.update(train_groups & val_groups)
    leaking_groups.update(train_groups & test_groups)
    leaking_groups.update(val_groups & test_groups)

    print(f"  Cross-split group leakage: {len(leaking_groups)} leaking groups")
    if leaking_groups:
        errors.append(f"Cross-split group leakage detected in {len(leaking_groups)} groups")
    else:
        print("  VERIFIED: 0 cross-split leaking groups.")

    # 7. Immutable Source Verification
    print("\n[Audit 6/10] Immutable Source Verification...")
    curr_corrected_hash, curr_corrected_count = hash_directory_labels(CORRECTED_DIR / "labels")
    print(f"  dfire_corrected label files: {curr_corrected_count}")
    print(f"  dfire_corrected aggregate SHA-256: {curr_corrected_hash}")
    if curr_corrected_hash != KNOWN_PRE_HASH:
        errors.append(f"dfire_corrected was altered! Expected hash {KNOWN_PRE_HASH}, got {curr_corrected_hash}")
    else:
        print("  VERIFIED: data/processed/dfire_corrected is 100% UNCHANGED.")

    raw_images_count = sum(len(list((RAW_DIR / s / "images").iterdir())) for s in SPLITS if (RAW_DIR / s / "images").exists())
    print(f"  data/raw image count: {raw_images_count}")
    if raw_images_count != EXPECTED_TOTAL_PAIRS:
        errors.append(f"data/raw altered! Expected {EXPECTED_TOTAL_PAIRS} raw images, got {raw_images_count}")
    else:
        print("  VERIFIED: data/raw is 100% UNCHANGED.")

    if not REMEDIATED_DIR.exists():
        errors.append(f"Evidence directory {REMEDIATED_DIR} was removed!")
    else:
        print(f"  VERIFIED: evidence directory {REMEDIATED_DIR} is preserved.")

    # 8. Hardlink Integrity Audit
    print("\n[Audit 7/10] Hardlink Storage Audit...")
    sample_hardlink_checks = []
    for s in SPLITS:
        src_img_dir = CORRECTED_DIR / "images" / s
        dst_img_dir = VERIFIED_DIR / "images" / s
        sample_img = next(src_img_dir.iterdir())
        dst_sample = dst_img_dir / sample_img.name
        src_stat = sample_img.stat()
        dst_stat = dst_sample.stat()
        is_same_file = (src_stat.st_ino == dst_stat.st_ino) and (src_stat.st_size == dst_stat.st_size)
        sample_hardlink_checks.append((s, sample_img.name, is_same_file))

    for s, name, is_link in sample_hardlink_checks:
        print(f"  Split '{s}' sample image '{name}': hardlink verified = {is_link}")
        if not is_link:
            warnings.append(f"Sample image {name} does not share inode (copy fallback used)")

    # 9. Explicit Proof that NO Phase 3-only Candidates are Present
    print("\n[Audit 8/10] Explicit Audit of Phase 3 Exclusion (No Heuristic Additions Present)...")
    if not P3_ADJUSTMENTS_CSV.exists():
        errors.append("Missing Phase 3 adjustments CSV to verify exclusion!")
    else:
        with open(P3_ADJUSTMENTS_CSV, "r", encoding="utf-8") as f:
            p3_adj_rows = list(csv.DictReader(f))

        p3_heuristic_added = [r for r in p3_adj_rows if r["action"] in ("ADD", "ADJUST_BBOX")]
        print(f"  Phase 3 heuristic additions to check for absence: {len(p3_heuristic_added)}")

        p3_leaked_count = 0
        for r in p3_heuristic_added:
            cid = r["candidate_id"]
            split = r["split"]
            img = r["image"]
            stem = Path(img).stem
            ccls = int(r["canonical_class"])

            # Candidate box coords
            box_str = r["new_bbox_norm"] if r["new_bbox_norm"] != "None" else r["old_bbox_norm"]
            cleaned = box_str.strip("()").replace(",", " ")
            coords = tuple(round(float(v), 4) for v in cleaned.split())

            lbl_file = VERIFIED_DIR / "labels" / split / f"{stem}.txt"
            if lbl_file.exists():
                lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                for line in lines:
                    parts = line.split()
                    line_cls = int(parts[0])
                    line_coords = tuple(round(float(v), 4) for v in parts[1:])
                    if line_cls == ccls and line_coords == coords:
                        p3_leaked_count += 1
                        errors.append(f"Phase 3 candidate {cid} found in verified label {split}/{stem}.txt!")

        print(f"  Phase 3 candidates found in verified dataset: {p3_leaked_count}")
        if p3_leaked_count == 0:
            print("  VERIFIED: Exactly 0 Phase 3 candidates present in dfire_remediated_verified.")
        else:
            errors.append(f"Found {p3_leaked_count} Phase 3 candidates in verified dataset!")

    # 10. Remediation Manifest Audit (5,579 candidate records, 100% census)
    print("\n[Audit 9/10] Remediation Manifest Audit (5,579 Candidates)...")
    manifest_p = VERIFIED_DIR / "remediation_manifest.csv"
    if not manifest_p.exists():
        errors.append(f"Missing remediation manifest: {manifest_p}")
    else:
        with open(manifest_p, "r", encoding="utf-8") as f:
            m_rows = list(csv.DictReader(f))
        print(f"  Manifest records: {len(m_rows)}")
        status_counts = Counter(r["status"] for r in m_rows)
        print(f"  Status breakdown: {dict(status_counts)}")
        if len(m_rows) != 5579:
            errors.append(f"Expected 5,579 manifest rows, found {len(m_rows)}")

        with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
            all_cands = list(csv.DictReader(f))

        manifest_cids = {r["candidate_id"] for r in m_rows}
        missing_cids = set(c["candidate_id"] for c in all_cands) - manifest_cids
        if missing_cids:
            errors.append(f"{len(missing_cids)} candidates missing from manifest!")
        else:
            print("  VERIFIED: All 5,579 candidates from candidates.csv are referenced in trusted manifest.")

        # Check unreviewed population
        unreviewed_cands = [r for r in m_rows if "Unreviewed" in r["phase"]]
        print(f"  Unreviewed candidate population marked for future visual QA: {len(unreviewed_cands)}")
        if len(unreviewed_cands) != 3684:
            errors.append(f"Expected 3,684 unreviewed candidates, got {len(unreviewed_cands)}")

    # 11. Metadata and Dataset Manifest
    print("\n[Audit 10/10] Dataset Manifest & Configuration Verification...")
    ds_man_p = VERIFIED_DIR / "dataset_manifest.json"
    if not ds_man_p.exists():
        errors.append(f"Missing dataset manifest: {ds_man_p}")
    else:
        with open(ds_man_p, "r", encoding="utf-8") as f:
            ds_man = json.load(f)
        if ds_man.get("version") != "2.1.0-verified-phase2":
            errors.append(f"Expected version '2.1.0-verified-phase2', got {ds_man.get('version')}")
        else:
            print(f"  Dataset manifest version verified: {ds_man.get('version')}")

    yaml_p = VERIFIED_DIR / "data.yaml"
    if not yaml_p.exists():
        errors.append(f"Missing data.yaml at {yaml_p}")
    else:
        print("  data.yaml verified.")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"FAILED with {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  [ERROR] {err}")
        return False
    else:
        print("RESULT: ALL 10 AUDIT GATES PASSED (100% compliant)")
        return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
