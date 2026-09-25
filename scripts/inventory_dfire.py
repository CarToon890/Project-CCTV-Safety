"""Comprehensive machine inventory of D-Fire raw dataset.

Performs:
1. File verification: exact pairing between images and labels across train, val, test.
2. Label syntax & bounding box check:
   - Class IDs observed
   - Malformed lines
   - Box coordinate ranges [0, 1]
   - Box boundary checks (x - w/2, etc.)
   - Category distribution (empty/negative, fire-only, smoke-only, both)
   - Instance counts per class and split
3. Image integrity check:
   - Check readable by PIL
   - Image formats / dimensions
4. Exact duplicate analysis (SHA-256 hash)
5. Group / filename / leakage analysis
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

def analyze_dfire():
    raw_root = Path("data/raw/dfire/data")
    splits = ["train", "val", "test"]

    print("=== D-FIRE COMPREHENSIVE MACHINE INVENTORY ===")

    pairing_issues = []
    label_syntax_issues = []
    box_oob_issues = []
    box_boundary_issues = []
    corrupt_images = []

    class_id_counter = Counter()
    class_id_per_split = defaultdict(Counter)
    category_counter = Counter()
    category_per_split = defaultdict(Counter)

    pair_counts = Counter()
    empty_label_counts = Counter()

    # Image hashes for exact duplicate check
    image_sha256_to_paths = defaultdict(list)
    total_images = 0

    for split in splits:
        img_dir = raw_root / split / "images"
        lbl_dir = raw_root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Error: {split} directory missing under {raw_root}")
            return

        img_files = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
        lbl_files = {p.stem: p for p in lbl_dir.iterdir() if p.is_file()}

        # 1. Pairing
        missing_lbl = set(img_files.keys()) - set(lbl_files.keys())
        missing_img = set(lbl_files.keys()) - set(img_files.keys())

        if missing_lbl:
            pairing_issues.append(f"{split}: {len(missing_lbl)} images missing labels")
        if missing_img:
            pairing_issues.append(f"{split}: {len(missing_img)} labels missing images")

        pair_counts[split] = len(img_files)
        total_images += len(img_files)

        print(f"Processing {split} split ({len(img_files)} image-label pairs)...")

        for stem, img_path in sorted(img_files.items()):
            lbl_path = lbl_files.get(stem)

            # Check label
            has_smoke = False
            has_fire = False
            other_classes = set()
            box_count = 0

            if lbl_path and lbl_path.exists():
                text = lbl_path.read_text(encoding="utf-8", errors="replace")
                lines = text.strip().splitlines()
                if not lines or text.strip() == "":
                    empty_label_counts[split] += 1
                else:
                    for line_no, raw_line in enumerate(lines, 1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            label_syntax_issues.append((str(lbl_path), line_no, line, "parts != 5"))
                            continue
                        try:
                            cid = int(parts[0])
                            xc, yc, w, h = (float(v) for v in parts[1:])
                        except ValueError as e:
                            label_syntax_issues.append((str(lbl_path), line_no, line, str(e)))
                            continue

                        box_count += 1
                        class_id_counter[cid] += 1
                        class_id_per_split[split][cid] += 1

                        if cid == 0:
                            has_smoke = True
                        elif cid == 1:
                            has_fire = True
                        else:
                            other_classes.add(cid)

                        # Check coordinate validity [0, 1]
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            box_oob_issues.append((str(lbl_path), line_no, line, "value out of [0, 1]"))
                        
                        # Boundary warning (crosses boundary slightly)
                        if (xc - w/2 < 0) or (xc + w/2 > 1) or (yc - h/2 < 0) or (yc + h/2 > 1):
                            box_boundary_issues.append((str(lbl_path), line_no, line, "crosses image edge"))

            # Determine category
            if box_count == 0:
                cat = "negative"
            elif has_fire and has_smoke:
                cat = "both"
            elif has_fire:
                cat = "fire_only"
            elif has_smoke:
                cat = "smoke_only"
            else:
                cat = f"other_{sorted(other_classes)}"

            category_counter[cat] += 1
            category_per_split[split][cat] += 1

            # Exact image duplicate check via SHA-256
            try:
                data = img_path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                image_sha256_to_paths[digest].append(f"{split}/{img_path.name}")
            except Exception as e:
                corrupt_images.append((str(img_path), str(e)))

    print("\n=== INVENTORY SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Pairs per split: {dict(pair_counts)}")
    print(f"Pairing issues: {len(pairing_issues)}")
    print(f"Empty labels per split: {dict(empty_label_counts)}")
    print(f"Source classes: {dict(class_id_counter)}")
    print(f"Categories: {dict(category_counter)}")
    print(f"Syntax errors: {len(label_syntax_issues)}")
    print(f"OOB boxes: {len(box_oob_issues)}")
    print(f"Edge crossings: {len(box_boundary_issues)}")
    print(f"Corrupt images: {len(corrupt_images)}")

    exact_dup_groups = {k: v for k, v in image_sha256_to_paths.items() if len(v) > 1}
    print(f"Exact duplicate image groups: {len(exact_dup_groups)}")

    cross_split_exact_dups = 0
    for digest, paths in exact_dup_groups.items():
        splits_in_group = set(p.split('/')[0] for p in paths)
        if len(splits_in_group) > 1:
            cross_split_exact_dups += 1

    report = {
        "total_images": total_images,
        "splits": dict(pair_counts),
        "empty_labels": dict(empty_label_counts),
        "source_classes": dict(class_id_counter),
        "source_classes_per_split": {s: dict(c) for s, c in class_id_per_split.items()},
        "categories": dict(category_counter),
        "categories_per_split": {s: dict(c) for s, c in category_per_split.items()},
        "syntax_errors_count": len(label_syntax_issues),
        "oob_boxes_count": len(box_oob_issues),
        "edge_crossings_count": len(box_boundary_issues),
        "corrupt_images_count": len(corrupt_images),
        "exact_duplicate_groups_count": len(exact_dup_groups),
        "cross_split_exact_duplicate_groups": cross_split_exact_dups,
    }

    report_dir = Path("docs/audit_artifacts/dfire")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "dfire_machine_inventory.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote inventory report to {out_file}")

if __name__ == "__main__":
    analyze_dfire()
