"""Systematic diagnostic evaluation of all 326 Phase 2 candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


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


def main():
    crop_files = sorted(CROPS_DIR.glob("m_item_*.jpg"))
    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        all_cands = {r["candidate_id"]: r for r in csv.DictReader(f)}

    items = []
    for cf in crop_files:
        parts = cf.stem.split("_")
        rank = int(parts[2])
        cid = f"{parts[3]}_{parts[4]}"
        cname = parts[5]
        cand = all_cands[cid]
        items.append((rank, cid, cname, cand, cf))

    print(f"Total items: {len(items)}")

    # Load existing labels for all images
    img_labels = {}
    for rank, cid, cname, cand, cf in items:
        split = cand["split"]
        img_name = cand["image"]
        k = (split, img_name)
        if k not in img_labels:
            lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
            img_p = REMEDIATED_DIR / "images" / split / img_name
            with Image.open(img_p) as im:
                w_img, h_img = im.size
            boxes = []
            if lbl_p.exists():
                for line in lbl_p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        p = line.split()
                        c = int(p[0])
                        xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                        gx1 = (xc - w/2) * w_img
                        gy1 = (yc - h/2) * h_img
                        gx2 = (xc + w/2) * w_img
                        gy2 = (yc + h/2) * h_img
                        boxes.append((c, (gx1, gy1, gx2, gy2), (xc, yc, w, h)))
            img_labels[k] = (w_img, h_img, boxes)

    # Analyze helmets
    helmets_data = [it for it in items if it[2] == "helmet"]
    persons_data = [it for it in items if it[2] == "person"]

    print(f"Helmets count: {len(helmets_data)}")
    print(f"Persons count: {len(persons_data)}")

    # For helmets: check overlap with person boxes
    helmet_on_person = []
    helmet_isolated = []
    for rank, cid, cname, cand, cf in helmets_data:
        split = cand["split"]
        img_name = cand["image"]
        w_img, h_img, boxes = img_labels[(split, img_name)]
        x1, y1, x2, y2 = float(cand["x1"]), float(cand["y1"]), float(cand["x2"]), float(cand["y2"])
        cbox = (x1, y1, x2, y2)

        # Check if inside or overlapping any person box
        person_overlaps = []
        for c, pbox, norm in boxes:
            if c == 0:  # person
                iou = compute_iou(cbox, pbox)
                # Also check intersection area / helmet area
                ix1 = max(cbox[0], pbox[0])
                iy1 = max(cbox[1], pbox[1])
                ix2 = min(cbox[2], pbox[2])
                iy2 = min(cbox[3], pbox[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                h_area = (x2 - x1) * (y2 - y1)
                coverage = inter / h_area if h_area > 0 else 0
                person_overlaps.append((coverage, iou, pbox))

        max_cov = max((p[0] for p in person_overlaps), default=0.0)
        if max_cov >= 0.30:
            helmet_on_person.append((rank, cid, cand, max_cov))
        else:
            helmet_isolated.append((rank, cid, cand, max_cov))

    print(f"\nHelmets located on/inside detected persons: {len(helmet_on_person)}")
    print(f"Helmets isolated / not on detected person: {len(helmet_isolated)}")
    for rank, cid, cand, cov in helmet_isolated:
        print(f"  Isolated Helmet #{rank:03d} ({cid}): conf={float(cand['confidence']):.3f} on {cand['split']}/{cand['image']} (person coverage={cov:.2f})")


if __name__ == "__main__":
    main()
