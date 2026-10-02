"""Batch visual analysis of helmet and person crops for Phase 2 QA."""

from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"
FEATURES_JSON = ROOT / "docs/audit_artifacts/dfire/phase2_features.json"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


def analyze_crop(crop_path: Path):
    with Image.open(crop_path) as im:
        im_rgb = im.convert("RGB")
        w, h = im.size
        arr = np.array(im_rgb)

    # Compute color stats: average R, G, B, saturation, brightness
    mean_r = np.mean(arr[:, :, 0])
    mean_g = np.mean(arr[:, :, 1])
    mean_b = np.mean(arr[:, :, 2])
    brightness = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

    # Yellow/white detection: high R and G, high brightness
    is_yellowish = (mean_r > 120 and mean_g > 110 and mean_b < 100)
    is_whitish = (mean_r > 150 and mean_g > 150 and mean_b > 150)
    is_dark = (brightness < 60)

    return {
        "w": int(w), "h": int(h),
        "brightness": float(brightness),
        "is_yellowish": bool(is_yellowish),
        "is_whitish": bool(is_whitish),
        "is_dark": bool(is_dark),
    }


def main():
    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        features = json.load(f)

    # Check helmets first
    helmets = [f for f in features if f["class_name"] == "helmet"]
    persons = [f for f in features if f["class_name"] == "person"]

    print(f"Loaded {len(helmets)} helmets and {len(persons)} persons.")

    helmet_results = []
    for h in helmets:
        cf = Path(h["crop_path"])
        stats = analyze_crop(cf)
        h["stats"] = stats
        helmet_results.append(h)

    person_results = []
    for p in persons:
        cf = Path(p["crop_path"])
        stats = analyze_crop(cf)
        p["stats"] = stats
        person_results.append(p)

    with open(ROOT / "docs/audit_artifacts/dfire/phase2_crop_stats.json", "w", encoding="utf-8") as f:
        json.dump({"helmets": helmet_results, "persons": person_results}, f, indent=2)

    print("Crop analysis saved.")


if __name__ == "__main__":
    main()
