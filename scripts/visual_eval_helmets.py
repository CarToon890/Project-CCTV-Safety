"""Classify visual characteristics of all 126 MEDIUM helmet candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"
FEATURES_JSON = ROOT / "docs/audit_artifacts/dfire/phase2_features.json"


def main():
    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        all_features = json.load(f)

    helmets = [it for it in all_features if it["class_name"] == "helmet"]
    print(f"Loaded {len(helmets)} helmet items.")

    # Check image sizes, crop aspect ratios, brightness, etc.
    # Group by image
    by_img = {}
    for h in helmets:
        by_img.setdefault((h["split"], h["image"]), []).append(h)

    print(f"Unique images across 126 helmets: {len(by_img)}")

    # Let's inspect each helmet item
    for h in helmets:
        rank = h["rank"]
        cid = h["cid"]
        split = h["split"]
        img = h["image"]
        conf = h["conf"]
        bw = h["box_pix"][2] - h["box_pix"][0]
        bh = h["box_pix"][3] - h["box_pix"][1]
        aspect = h["aspect"]

        # Print detailed row for inspection
        print(f"H#{rank:03d} {cid} conf={conf:.3f} box=({bw:.0f}x{bh:.0f}, asp={aspect:.2f}) on {split}/{img}")


if __name__ == "__main__":
    main()
