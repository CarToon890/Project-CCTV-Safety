"""Build complete corrected Construction-PPE dataset under data/processed/construction_ppe_corrected.

Strictly preserves raw data as read-only.
Applies:
1. Proposed regroup manifest (1,416 images) eliminating cross-split sequence leakage.
2. Canonical Stage-1 mapping: 0: person, 1: helmet, 2: vest.
3. 100% verified remediations for all 88 defect rows (85 unique images).
4. Generates QA overlays, contact sheets, metadata CSVs, dataset_manifest.json, and remediation_qa_queue.csv.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppe_remediation_data import REMEDIATED_BOXES

RAW_ROOT = Path("C:/Users/USER/Desktop/BUU69/Project-CCTV-Safety/data/raw/construction-ppe")
OUTPUT_DIR = Path("data/processed/construction_ppe_corrected")
REGROUP_MANIFEST = Path("docs/audit_artifacts/construction_ppe/proposed_regroup_manifest.csv")
REMEDIATION_MANIFEST = Path("docs/audit_artifacts/construction_ppe/label_remediation_manifest.csv")
DOCS_ARTIFACTS = Path("docs/audit_artifacts/construction_ppe")

CANONICAL_CLASSES = {0: "person", 1: "helmet", 2: "vest"}
CANONICAL_COLORS = {
    0: ("#00FF00", "person"),
    1: ("#00FFFF", "helmet"),
    2: ("#FFA500", "vest"),
}


def build_dataset() -> None:
    print("==================================================")
    print("   Building Corrected Construction-PPE Dataset    ")
    print("==================================================")

    # 1. Load regroup manifest
    with open(REGROUP_MANIFEST, "r", encoding="utf-8") as f:
        regroup_rows = list(csv.DictReader(f))
    print(f"Loaded regroup manifest: {len(regroup_rows)} images")

    # 2. Load remediation manifest
    with open(REMEDIATION_MANIFEST, "r", encoding="utf-8") as f:
        remed_rows = list(csv.DictReader(f))
    print(f"Loaded remediation manifest: {len(remed_rows)} defect rows ({len(REMEDIATED_BOXES)} unique images)")

    # 3. Setup directories
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "qa_overlays" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "contact_sheets").mkdir(parents=True, exist_ok=True)

    metadata_by_split = defaultdict(list)
    class_counts = Counter()
    split_counts = Counter()
    remediated_count = 0
    unchanged_count = 0

    # 4. Process each image
    for idx, row in enumerate(regroup_rows, 1):
        filename = row["filename"]
        source_split = row["source_split"]
        proposed_split = row["proposed_split"]
        group_id = row["group_id"]
        
        rel_key = f"images/{source_split}/{filename}"
        src_img_path = RAW_ROOT / "images" / source_split / filename
        if not src_img_path.exists():
            raise FileNotFoundError(f"Missing raw image: {src_img_path}")

        dst_img_path = OUTPUT_DIR / "images" / proposed_split / filename
        dst_lbl_path = OUTPUT_DIR / "labels" / proposed_split / f"{Path(filename).stem}.txt"

        # Copy image unchanged
        shutil.copy2(src_img_path, dst_img_path)
        split_counts[proposed_split] += 1

        # Determine label lines
        if rel_key in REMEDIATED_BOXES:
            # Remediated
            remediated_count += 1
            boxes = REMEDIATED_BOXES[rel_key]
            lbl_lines = []
            for cid, xc, yc, w, h in boxes:
                class_counts[cid] += 1
                lbl_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            dst_lbl_path.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")

            # Generate QA overlay for remediated image
            with Image.open(src_img_path) as im:
                im = im.convert("RGB")
                w_px, h_px = im.size
                draw = ImageDraw.Draw(im)
                for cid, xc, yc, bw, bh in boxes:
                    x1 = (xc - bw / 2.0) * w_px
                    y1 = (yc - bh / 2.0) * h_px
                    x2 = (xc + bw / 2.0) * w_px
                    y2 = (yc + bh / 2.0) * h_px
                    color, name = CANONICAL_COLORS.get(cid, ("#FFFFFF", str(cid)))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw.text((x1 + 2, max(0, y1 - 10)), name, fill=color)
                qa_dst = OUTPUT_DIR / "qa_overlays" / proposed_split / f"{Path(filename).stem}_qa.jpg"
                im.save(qa_dst, quality=90)
        else:
            # Unchanged: canonical mapping from raw
            unchanged_count += 1
            src_lbl_path = RAW_ROOT / "labels" / source_split / f"{Path(filename).stem}.txt"
            lbl_lines = []
            seen_boxes = set()
            if src_lbl_path.exists():
                for line in src_lbl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    raw_cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    # Map: 6 (Person) -> 0, 0 (helmet) -> 1, 2 (vest) -> 2
                    canonical_cid = None
                    if raw_cid == 6:
                        canonical_cid = 0
                    elif raw_cid == 0:
                        canonical_cid = 1
                    elif raw_cid == 2:
                        canonical_cid = 2
                    
                    if canonical_cid is not None:
                        canonical_box = (canonical_cid, *coords[:4])
                        if canonical_box in seen_boxes:
                            continue
                        seen_boxes.add(canonical_box)
                        class_counts[canonical_cid] += 1
                        lbl_lines.append(f"{canonical_cid} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
            dst_lbl_path.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")

        # Record metadata
        metadata_by_split[proposed_split].append({
            "image": filename,
            "source_id": "construction_ppe",
            "group_id": group_id
        })

    # 5. Write metadata CSVs
    for split in ["train", "val", "test"]:
        meta_file = OUTPUT_DIR / "metadata" / f"{split}.csv"
        with open(meta_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "source_id", "group_id"])
            writer.writeheader()
            writer.writerows(metadata_by_split[split])
    print(f"Written metadata CSVs: train={len(metadata_by_split['train'])}, val={len(metadata_by_split['val'])}, test={len(metadata_by_split['test'])}")

    # 6. Copy contact sheets from scratch to output
    scratch_sheets = Path("scratch/remediated_visual_qa/contact_sheets")
    if scratch_sheets.exists():
        for sheet_p in scratch_sheets.glob("*.jpg"):
            shutil.copy2(sheet_p, OUTPUT_DIR / "contact_sheets" / sheet_p.name)
    print(f"Copied contact sheets to {OUTPUT_DIR / 'contact_sheets'}")

    # 7. Generate concise human QA handoff queue
    qa_queue_rows = []
    # Build lookup from relative_image_path to remed_rows
    remed_by_img = defaultdict(list)
    for r in remed_rows:
        remed_by_img[r["relative_image_path"]].append(r)

    regroup_by_fn = {r["filename"]: r for r in regroup_rows}

    for rel_path, r_list in sorted(remed_by_img.items()):
        fn = Path(rel_path).name
        reg_info = regroup_by_fn[fn]
        prop_split = reg_info["proposed_split"]
        grp = reg_info["group_id"]
        
        issue_types = "; ".join(r["issue_type"] for r in r_list)
        classes = "; ".join(r["canonical_class"] for r in r_list)
        evidences = "; ".join(r["evidence"] for r in r_list)
        boxes = REMEDIATED_BOXES.get(rel_path, [])
        box_summary = f"{len(boxes)} boxes (" + ", ".join(
            f"{CANONICAL_CLASSES[c]}:{sum(1 for b in boxes if b[0] == c)}"
            for c in sorted(set(b[0] for b in boxes))
        ) + ")" if boxes else "0 boxes (negative)"

        qa_queue_rows.append({
            "filename": fn,
            "source_split": r_list[0]["source_split"],
            "proposed_split": prop_split,
            "group_id": grp,
            "issue_types": issue_types,
            "target_classes": classes,
            "evidence": evidences,
            "worker_remediated_boxes": box_summary,
            "worker_review_status": "WORKER_VISUAL_QA_VERIFIED",
            "qa_overlay_path": f"data/processed/construction_ppe_corrected/qa_overlays/{prop_split}/{Path(fn).stem}_qa.jpg",
            "human_qa_verdict": "PENDING_HUMAN_QA",
            "human_qa_notes": ""
        })

    queue_fields = [
        "filename", "source_split", "proposed_split", "group_id", "issue_types",
        "target_classes", "evidence", "worker_remediated_boxes", "worker_review_status",
        "qa_overlay_path", "human_qa_verdict", "human_qa_notes"
    ]

    # Save queue in processed output and docs artifacts
    out_queue_processed = OUTPUT_DIR / "remediation_qa_queue.csv"
    out_queue_docs = DOCS_ARTIFACTS / "remediation_qa_queue.csv"
    for q_path in [out_queue_processed, out_queue_docs]:
        with open(q_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=queue_fields)
            writer.writeheader()
            writer.writerows(qa_queue_rows)
    print(f"Generated human QA handoff queue ({len(qa_queue_rows)} rows) at {out_queue_docs}")

    # 8. Write dataset_manifest.json
    manifest_data = {
        "dataset_name": "construction_ppe_corrected",
        "version": "1.0.0",
        "license": "AGPL-3.0 (academic / educational prototype)",
        "source_raw_path": str(RAW_ROOT),
        "total_images": len(regroup_rows),
        "splits": dict(split_counts),
        "class_mapping": {
            "0": "person",
            "1": "helmet",
            "2": "vest",
            "3": "fall (0 in PPE)",
            "4": "fire (0 in PPE)",
            "5": "smoke (0 in PPE)"
        },
        "instance_counts": {
            "person (0)": class_counts[0],
            "helmet (1)": class_counts[1],
            "vest (2)": class_counts[2],
            "fall (3)": 0,
            "fire (4)": 0,
            "smoke (5)": 0
        },
        "remediation_summary": {
            "total_remediated_images": remediated_count,
            "total_unchanged_images": unchanged_count,
            "worker_review_status": "WORKER_VISUAL_QA_VERIFIED",
            "human_qa_status": "PENDING_INDEPENDENT_HUMAN_QA"
        }
    }
    (OUTPUT_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved dataset manifest at {OUTPUT_DIR / 'dataset_manifest.json'}")
    print(json.dumps(manifest_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    build_dataset()
