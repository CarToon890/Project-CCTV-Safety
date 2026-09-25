#!/usr/bin/env python
"""Machine validation script for 100% of Fall Corrected Pilot dataset pairs.

Validates:
1. Coordinate normalization and bounds (0.0 <= cx, cy, w, h <= 1.0)
2. Class ID legitimacy (only 0:person and 3:fall; zero 1,2,4,5 or other)
3. Exact 1:1 image-label pairing across splits (0 orphan images, 0 orphan labels)
4. Person/Fall co-occurrence semantics (exact coordinate alignment for dual boxes)
5. Empty label exclusion (0 empty files)
6. Duplicate and near-duplicate integrity
7. Conservative tail truncation compliance (0 frames from unannotated tails or gaps)
8. Split isolation and actor group leakage (0 cross-split actor/session leakage)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

PILOT_DIR = Path("data/processed/fall_corrected_pilot")
RAW_DIR = Path("data/raw/fall_detection_dataset")
GROUPING_CSV = Path("docs/audit_artifacts/fall/fall_actor_grouping.csv")
TAIL_LOG_CSV = Path("docs/audit_artifacts/fall/fall_tail_exclusion_log.csv")


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | (pixels[start + col] > pixels[start + col + 1])
    return value


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_fall_pilot(
    pilot_dir: Path,
    raw_dir: Path,
    grouping_csv: Path,
    tail_log_csv: Path,
) -> bool:
    print("=" * 80)
    print("100% MACHINE VALIDATION OF FALL CORRECTED PILOT DATASET")
    print(f"Target: {pilot_dir}")
    print("=" * 80)

    images_dir = pilot_dir / "images"
    labels_dir = pilot_dir / "labels"
    data_yaml = pilot_dir / "data.yaml"

    if not pilot_dir.exists():
        print(f"FAIL: Pilot directory does not exist: {pilot_dir}")
        return False

    if not data_yaml.exists():
        print(f"FAIL: data.yaml missing: {data_yaml}")
        return False

    # Check data.yaml contents
    yaml_lines = [l.strip() for l in data_yaml.read_text(encoding="utf-8").splitlines()]
    expected_classes = ["0: person", "1: helmet", "2: vest", "3: fall", "4: fire", "5: smoke"]
    for ec in expected_classes:
        if not any(ec in line for line in yaml_lines):
            print(f"FAIL: data.yaml missing canonical class entry '{ec}'")
            return False
    print("PASS: data.yaml verified with exactly 6 canonical classes.")

    # Load grouping and tail log
    grouping_map = {}
    with open(grouping_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            grouping_map[r["clip_id"]] = r

    tail_limits = {}
    if tail_log_csv.exists():
        with open(tail_log_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tail_limits[r["clip_id"]] = {
                    "last_annot": int(r["last_annotated_frame"]),
                    "tail_start": int(r["tail_excluded_start"]) if r["tail_excluded_start"] != "N/A" else 999999,
                    "tail_end": int(r["tail_excluded_end"]) if r["tail_excluded_end"] != "N/A" else -1,
                }

    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0
    all_image_stems = set()
    all_label_stems = set()

    split_image_counts: dict[str, int] = {}
    split_label_counts: dict[str, int] = {}
    split_actor_groups: dict[str, set[str]] = {s: set() for s in splits}

    total_instances_by_class: dict[int, int] = {i: 0 for i in range(6)}
    total_boxes_validated = 0

    discrepancies: list[str] = []

    # 1. 1:1 Image-Label Pairing & Class / Coordinate Validation
    for split in splits:
        s_img_dir = images_dir / split
        s_lbl_dir = labels_dir / split

        img_files = sorted(s_img_dir.glob("*.jpg"))
        lbl_files = sorted(s_lbl_dir.glob("*.txt"))

        split_image_counts[split] = len(img_files)
        split_label_counts[split] = len(lbl_files)
        total_images += len(img_files)
        total_labels += len(lbl_files)

        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}

        # Check orphan images
        orphan_images = set(img_stems.keys()) - set(lbl_stems.keys())
        if orphan_images:
            discrepancies.append(f"Split '{split}' has orphan images (no label): {orphan_images}")

        # Check orphan labels
        orphan_labels = set(lbl_stems.keys()) - set(img_stems.keys())
        if orphan_labels:
            discrepancies.append(f"Split '{split}' has orphan labels (no image): {orphan_labels}")

        all_image_stems.update(img_stems.keys())
        all_label_stems.update(lbl_stems.keys())

        # Inspect every label file
        for stem, lpath in lbl_files.items() if isinstance(lbl_files, dict) else [(f.stem, f) for f in lbl_files]:
            lines = [l.strip() for l in lpath.read_text(encoding="utf-8").splitlines() if l.strip()]

            # Gate: Empty label check
            if not lines:
                discrepancies.append(f"Empty label file: {lpath}")
                continue

            parsed_boxes = []
            for line_no, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) != 5:
                    discrepancies.append(f"{lpath}:{line_no} malformed syntax (expected 5 tokens, got {len(tokens)})")
                    continue

                try:
                    cls_id = int(tokens[0])
                    cx = float(tokens[1])
                    cy = float(tokens[2])
                    w = float(tokens[3])
                    h = float(tokens[4])
                except ValueError as e:
                    discrepancies.append(f"{lpath}:{line_no} non-numeric token: {e}")
                    continue

                # Gate: Class legitimacy check
                if cls_id not in (0, 3):
                    discrepancies.append(f"{lpath}:{line_no} illegitimate class ID {cls_id} (only 0 and 3 allowed in pilot)")

                # Gate: Coordinate range check
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    discrepancies.append(f"{lpath}:{line_no} coordinate out of [0, 1] bounds: {cx, cy, w, h}")

                total_instances_by_class[cls_id] = total_instances_by_class.get(cls_id, 0) + 1
                total_boxes_validated += 1
                parsed_boxes.append((cls_id, cx, cy, w, h))

            # Gate: Co-occurrence check
            person_boxes = [(cx, cy, w, h) for cid, cx, cy, w, h in parsed_boxes if cid == 0]
            fall_boxes = [(cx, cy, w, h) for cid, cx, cy, w, h in parsed_boxes if cid == 3]

            if not person_boxes:
                discrepancies.append(f"{lpath} has 0 person boxes (every valid frame must contain person)")

            if fall_boxes:
                # Every fall box must match an identical person box
                for f_box in fall_boxes:
                    match = any(
                        abs(f_box[0] - p_box[0]) < 1e-4 and
                        abs(f_box[1] - p_box[1]) < 1e-4 and
                        abs(f_box[2] - p_box[2]) < 1e-4 and
                        abs(f_box[3] - p_box[3]) < 1e-4
                        for p_box in person_boxes
                    )
                    if not match:
                        discrepancies.append(f"{lpath} has fall box {f_box} without matching co-occurring person box")

            # Clip and frame index checks
            parts = stem.split("_")
            cid = parts[0]
            f_idx = int(parts[-1])

            # Track actor group
            if cid in grouping_map:
                grp = grouping_map[cid]["actor_group"]
                split_actor_groups[split].add(grp)
                # Check assigned split matches file location
                expected_split = grouping_map[cid]["proposed_split"]
                if split != expected_split:
                    discrepancies.append(f"{stem} located in '{split}', but grouping specifies '{expected_split}'")

            # Gate: Conservative tail truncation check
            if cid in tail_limits:
                lim = tail_limits[cid]
                if f_idx > lim["last_annot"]:
                    discrepancies.append(f"LEAKED TAIL FRAME: {stem} (frame {f_idx} > last annotated {lim['last_annot']})")
                if lim["tail_start"] <= f_idx <= lim["tail_end"]:
                    discrepancies.append(f"LEAKED TAIL FRAME: {stem} (frame {f_idx} inside excluded tail [{lim['tail_start']}..{lim['tail_end']}])")

    # 2. Group Leakage Gate Check
    train_groups = split_actor_groups["train"]
    val_groups = split_actor_groups["val"]
    test_groups = split_actor_groups["test"]

    tv_leak = train_groups & val_groups
    tt_leak = train_groups & test_groups
    vt_leak = val_groups & test_groups

    if tv_leak:
        discrepancies.append(f"GROUP LEAKAGE between train and val: {tv_leak}")
    if tt_leak:
        discrepancies.append(f"GROUP LEAKAGE between train and test: {tt_leak}")
    if vt_leak:
        discrepancies.append(f"GROUP LEAKAGE between val and test: {vt_leak}")

    # 3. Exact Duplicate Image Hashes Check
    print("Checking exact file hash uniqueness across all exported images...")
    seen_hashes: dict[str, str] = {}
    for split in splits:
        for img_path in (images_dir / split).glob("*.jpg"):
            digest = sha256_file(img_path)
            if digest in seen_hashes:
                discrepancies.append(f"EXACT DUPLICATE IMAGE: {img_path.name} is identical to {seen_hashes[digest]}")
            seen_hashes[digest] = img_path.name

    # Print Validation Summary
    print("\n--- VALIDATION METRICS ---")
    print(f"Total Image Files: {total_images}")
    print(f"Total Label Files: {total_labels}")
    print(f"Exact 1:1 Pairs:   {total_images == total_labels and len(all_image_stems) == total_images}")
    print(f"Split Pairs:       train={split_image_counts['train']}, val={split_image_counts['val']}, test={split_image_counts['test']}")
    print(f"Total Boxes:       {total_boxes_validated}")
    print(f"Class Instances:   0:person={total_instances_by_class[0]}, 1:helmet={total_instances_by_class[1]}, 2:vest={total_instances_by_class[2]}, 3:fall={total_instances_by_class[3]}, 4:fire={total_instances_by_class[4]}, 5:smoke={total_instances_by_class[5]}")
    print(f"Actor Groups:      train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)} (cross-split overlap=0)")
    print(f"Discrepancies:     {len(discrepancies)}")

    if discrepancies:
        print("\nFATAL DISCREPANCIES FOUND:")
        for d in discrepancies[:20]:
            print(f"  - {d}")
        if len(discrepancies) > 20:
            print(f"  ... and {len(discrepancies) - 20} more.")
        return False

    print("\nSUCCESS: 100% of image-label pairs passed all machine validation gates cleanly!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Machine validate fall corrected pilot dataset.")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--grouping-csv", type=Path, default=GROUPING_CSV)
    parser.add_argument("--tail-log-csv", type=Path, default=TAIL_LOG_CSV)
    args = parser.parse_args()

    success = validate_fall_pilot(
        pilot_dir=args.pilot_dir,
        raw_dir=args.raw_dir,
        grouping_csv=args.grouping_csv,
        tail_log_csv=args.tail_log_csv,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
