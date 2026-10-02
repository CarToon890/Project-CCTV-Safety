"""Detailed per-candidate audit script for 326 Phase 2 items."""

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

    print(f"Total Phase 2 items: {len(items)}")

    # For each candidate, analyze:
    # 1. Existing labels on the image in dfire_remediated
    # 2. Overlap with fire/smoke GT
    # 3. Overlap with other candidates
    # 4. Aspect ratio & normalized size
    # 5. Border touching

    audit_summary = []
    for rank, cid, cname, cand, cf in items:
        split = cand["split"]
        img_name = cand["image"]
        img_p = REMEDIATED_DIR / "images" / split / img_name
        with Image.open(img_p) as im:
            w_img, h_img = im.size

        x1 = float(cand["x1"])
        y1 = float(cand["y1"])
        x2 = float(cand["x2"])
        y2 = float(cand["y2"])
        bw = x2 - x1
        bh = y2 - y1
        conf = float(cand["confidence"])

        lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
        lbl_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts = line.split()
                    c = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gx1 = (xc - w/2) * w_img
                    gy1 = (yc - h/2) * h_img
                    gx2 = (xc + w/2) * w_img
                    gy2 = (yc + h/2) * h_img
                    lbl_boxes.append((c, (gx1, gy1, gx2, gy2)))

        max_smoke_iou = max((compute_iou((x1, y1, x2, y2), b) for c, b in lbl_boxes if c == 5), default=0.0)
        max_fire_iou = max((compute_iou((x1, y1, x2, y2), b) for c, b in lbl_boxes if c == 4), default=0.0)
        max_person_iou = max((compute_iou((x1, y1, x2, y2), b) for c, b in lbl_boxes if c == 0), default=0.0)

        aspect = bh / bw if bw > 0 else 0
        norm_w = bw / w_img
        norm_h = bh / h_img
        norm_xc = (x1 + bw/2) / w_img
        norm_yc = (y1 + bh/2) / h_img

        touches_edge = (x1 <= 2.0 or y1 <= 2.0 or x2 >= w_img - 2.0 or y2 >= h_img - 2.0)

        audit_summary.append({
            "rank": rank,
            "cid": cid,
            "class_name": cname,
            "split": split,
            "image": img_name,
            "conf": conf,
            "box_pix": (x1, y1, x2, y2),
            "box_norm": (round(norm_xc, 8), round(norm_yc, 8), round(norm_w, 8), round(norm_h, 8)),
            "aspect": aspect,
            "norm_w": norm_w,
            "norm_h": norm_h,
            "max_smoke_iou": max_smoke_iou,
            "max_fire_iou": max_fire_iou,
            "max_person_iou": max_person_iou,
            "touches_edge": touches_edge,
            "crop_path": str(cf),
        })

    # Save detailed features JSON for inspection
    with open(ROOT / "docs/audit_artifacts/dfire/phase2_features.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Features for all {len(audit_summary)} items computed and saved.")


if __name__ == "__main__":
    main()
