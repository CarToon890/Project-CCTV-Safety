"""Audit 126 MEDIUM helmet candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


def main():
    crop_files = sorted(CROPS_DIR.glob("m_item_*_helmet.jpg"))
    print(f"Total helmet crop files: {len(crop_files)}")

    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        cands = {r["candidate_id"]: r for r in csv.DictReader(f)}

    # Analyze features of each helmet candidate
    for cf in crop_files:
        parts = cf.stem.split("_")
        rank = int(parts[2])
        cid = f"{parts[3]}_{parts[4]}"
        cand = cands[cid]

        split = cand["split"]
        img_name = cand["image"]
        conf = float(cand["confidence"])
        x1, y1, x2, y2 = float(cand["x1"]), float(cand["y1"]), float(cand["x2"]), float(cand["y2"])
        bw = x2 - x1
        bh = y2 - y1

        img_p = REMEDIATED_DIR / "images" / split / img_name
        with Image.open(img_p) as im:
            w_img, h_img = im.size

        norm_w = bw / w_img
        norm_h = bh / h_img
        aspect = bh / bw if bw > 0 else 0

        # Check existing labels on that image
        lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
        lbl_classes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lbl_classes.append(int(line.split()[0]))

        has_person = 0 in lbl_classes
        has_fire = 4 in lbl_classes
        has_smoke = 5 in lbl_classes

        # Print brief line
        if rank <= 25 or rank >= 115 or norm_w > 0.15 or norm_h > 0.15:
            print(f"Helmet #{rank:03d} ({cid}): conf={conf:.3f} size=({bw:.0f}x{bh:.0f}, {norm_w*100:.1f}%w x {norm_h*100:.1f}%h) on {split}/{img_name} [person_in_img={has_person}]")


if __name__ == "__main__":
    main()
