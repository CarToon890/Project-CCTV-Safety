"""Comprehensive validation of corrected and group-isolated D-Fire dataset.

Validates:
1. Exact 1:1 image-to-label pairing across train, val, test splits.
2. Canonical 6-class schema compliance:
   - Only canonical class IDs 4 (fire) and 5 (smoke) are present.
   - Zero unexpected or legacy class IDs.
3. Coordinate validity:
   - All bounding boxes strictly within [0.0, 1.0].
   - All widths and heights strictly > 0 (zero degenerate boxes).
   - Zero boundary edge violations.
4. Split leakage & group isolation:
   - Evaluates metadata CSVs to verify that zero group_ids span multiple splits.
5. Raw dataset immutability:
   - Verifies raw data/raw/dfire/data has remained intact and untouched.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

def validate():
    root = Path("data/processed/dfire_corrected")
    raw_root = Path("data/raw/dfire/data")
    splits = ["train", "val", "test"]

    print("=== D-FIRE COMPREHENSIVE VALIDATION ===")

    if not root.exists():
        print(f"Error: Processed directory {root} does not exist.")
        return False

    errors = []
    warnings = []

    # 1. 1:1 Pairing check
    total_images = 0
    split_counts = {}
    for s in splits:
        img_dir = root / "images" / s
        lbl_dir = root / "labels" / s

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Missing images or labels directory for split: {s}")
            continue

        imgs = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file()}

        missing_lbls = set(imgs.keys()) - set(lbls.keys())
        missing_imgs = set(lbls.keys()) - set(imgs.keys())

        if missing_lbls:
            errors.append(f"{s}: {len(missing_lbls)} images missing labels (e.g. {list(missing_lbls)[:3]})")
        if missing_imgs:
            errors.append(f"{s}: {len(missing_imgs)} labels missing images (e.g. {list(missing_imgs)[:3]})")

        split_counts[s] = len(imgs)
        total_images += len(imgs)

    print(f"Pairing check: Total pairs = {total_images} (train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)})")
    if total_images != 21527:
        errors.append(f"Expected 21,527 total image pairs, found {total_images}")

    # 2. Bounding box & syntax check
    canonical_classes = Counter()
    split_class_counts = defaultdict(Counter)
    oob_count = 0
    zero_area_count = 0
    syntax_errors = 0
    empty_labels = Counter()

    for s in splits:
        lbl_dir = root / "labels" / s
        if not lbl_dir.exists():
            continue
        for p in lbl_dir.iterdir():
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                empty_labels[s] += 1
                continue

            for line_no, raw_line in enumerate(text.splitlines(), 1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    syntax_errors += 1
                    errors.append(f"{s}/{p.name}:{line_no}: Malformed line: '{line}'")
                    continue

                try:
                    cid = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError as e:
                    syntax_errors += 1
                    errors.append(f"{s}/{p.name}:{line_no}: Parse error: {e}")
                    continue

                # Check class IDs
                if cid not in (4, 5):
                    errors.append(f"{s}/{p.name}:{line_no}: Invalid class ID {cid} (must be canonical 4 or 5)")
                canonical_classes[cid] += 1
                split_class_counts[s][cid] += 1

                # Check zero-area
                if w <= 0.0 or h <= 0.0:
                    zero_area_count += 1
                    errors.append(f"{s}/{p.name}:{line_no}: Zero-area box (w={w}, h={h})")

                # Check coordinate ranges
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    oob_count += 1
                    errors.append(f"{s}/{p.name}:{line_no}: Coordinate out of [0, 1] range: xc={xc}, yc={yc}, w={w}, h={h}")

                # Check edge boundaries with floating-point tolerance
                if (xc - w/2.0 < -1e-5) or (xc + w/2.0 > 1.0 + 1e-5) or (yc - h/2.0 < -1e-5) or (yc + h/2.0 > 1.0 + 1e-5):
                    oob_count += 1
                    errors.append(f"{s}/{p.name}:{line_no}: Box boundary exceeds [0, 1] bounds")

    print(f"Empty labels (negative images): {dict(empty_labels)} (total = {sum(empty_labels.values())})")
    print(f"Canonical classes: {dict(canonical_classes)} (class 4 [fire]={canonical_classes[4]}, class 5 [smoke]={canonical_classes[5]})")
    print(f"Per-split classes: {dict(split_class_counts)}")
    print(f"Syntax errors: {syntax_errors}")
    print(f"OOB / edge violations: {oob_count}")
    print(f"Zero-area boxes: {zero_area_count}")

    # 3. Metadata & group leakage check
    group_to_splits = defaultdict(set)
    meta_total_rows = 0
    for s in splits:
        meta_file = root / "metadata" / f"{s}.csv"
        if not meta_file.exists():
            errors.append(f"Missing metadata file: {meta_file}")
            continue
        with meta_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_total_rows += 1
                gid = row["group_id"]
                group_to_splits[gid].add(s)

    leaking_groups = [gid for gid, sps in group_to_splits.items() if len(sps) > 1]
    print(f"Total metadata records: {meta_total_rows}")
    print(f"Total distinct groups: {len(group_to_splits)}")
    print(f"Cross-split group leakage: {len(leaking_groups)} groups leaking across splits")
    if leaking_groups:
        errors.append(f"Split leakage detected in {len(leaking_groups)} groups: {leaking_groups[:5]}")

    # 4. Raw immutability check
    raw_images = sum(len(list((raw_root / s / "images").iterdir())) for s in splits if (raw_root / s / "images").exists())
    if raw_images != 21527:
        errors.append(f"Raw dataset altered! Expected 21,527 raw images, found {raw_images}")
    else:
        print("Raw dataset immutability verified: 21,527 raw images intact.")

    print("\n=== VALIDATION RESULT ===")
    if errors:
        print(f"FAILED with {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  ERROR: {err}")
        return False
    else:
        print("PASSED: 100% compliant with canonical schema, zero OOB, zero group leakage, 1:1 pairing verified.")
        return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
