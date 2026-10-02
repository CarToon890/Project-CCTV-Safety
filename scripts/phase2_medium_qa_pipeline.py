"""Phase 2 D-Fire MEDIUM-tier Sampling and Diagnostics Pipeline.

Performs:
1. Complete census ingestion of all 126 MEDIUM helmet candidates.
2. Deterministic stratified sampling of exactly 200 MEDIUM person candidates
   using fixed random seed (seed=42) covering:
   - Splits: train, val, test (proportional: 146 train, 23 val, 31 test = 200)
   - Confidence bands: upper [0.60, 0.70), mid [0.50, 0.60), lower [0.40, 0.50)
   - Feature strata: smoke_fire_overlap, border_touching, extreme_aspect_ratio,
     multi_candidate_image, standard_geometry
3. Generates high-resolution diagnostic crops for all 326 candidates under
   docs/audit_artifacts/dfire/qa_crops_medium/
4. Emits initial candidate list with stratum records.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
OVERLAYS_DIR = ROOT / "data/processed/dfire_missing_label_full_scan/overlays"
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"

SEED = 42
TARGET_PERSONS = 200
EXPECTED_HELMETS = 126

CROPS_DIR.mkdir(parents=True, exist_ok=True)


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
    print("=" * 80)
    print("PHASE 2: D-FIRE MEDIUM-TIER SAMPLING & DIAGNOSTICS")
    print("=" * 80)

    # 1. Ingest candidates
    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_med = [r for r in reader if r["tier"] == "MEDIUM"]

    helmets = [r for r in all_med if r["class_name"] == "helmet"]
    persons = [r for r in all_med if r["class_name"] == "person"]

    print(f"Total MEDIUM candidates: {len(all_med)}")
    print(f"  Helmets: {len(helmets)} (100% census target: {EXPECTED_HELMETS})")
    print(f"  Persons: {len(persons)} (stratified sample target: {TARGET_PERSONS})")

    if len(helmets) != EXPECTED_HELMETS:
        raise ValueError(f"Expected {EXPECTED_HELMETS} helmets, got {len(helmets)}")

    # 2. Analyze persons features for stratification
    img_counts = Counter((r["split"], r["image"]) for r in persons)
    img_dims = {}

    categorized_persons = []
    for r in persons:
        split = r["split"]
        img_name = r["image"]
        img_p = REMEDIATED_DIR / "images" / split / img_name
        if img_p not in img_dims:
            with Image.open(img_p) as im:
                img_dims[img_p] = im.size
        w_img, h_img = img_dims[img_p]

        x1 = float(r["x1"])
        y1 = float(r["y1"])
        x2 = float(r["x2"])
        y2 = float(r["y2"])
        bw = x2 - x1
        bh = y2 - y1
        aspect = bh / bw if bw > 0 else 0
        conf = float(r["confidence"])

        conf_band = "upper_0.60_0.70" if conf >= 0.60 else ("mid_0.50_0.60" if conf >= 0.50 else "lower_0.40_0.50")

        # Check overlap with existing fire/smoke GT
        lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
        max_gt_iou = 0.0
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts = line.split()
                    cid = int(parts[0])
                    if cid in (4, 5):  # fire or smoke
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        gx1 = (xc - w/2) * w_img
                        gy1 = (yc - h/2) * h_img
                        gx2 = (xc + w/2) * w_img
                        gy2 = (yc + h/2) * h_img
                        iou = compute_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                        if iou > max_gt_iou:
                            max_gt_iou = iou

        is_overlap = max_gt_iou >= 0.20
        is_border = (x1 <= 2.0 or y1 <= 2.0 or x2 >= w_img - 2.0 or y2 >= h_img - 2.0)
        is_extreme_aspect = (aspect < 0.8 or aspect > 4.5)
        is_multi = img_counts[(split, img_name)] > 1

        if is_overlap:
            feature_stratum = "smoke_fire_overlap"
        elif is_extreme_aspect:
            feature_stratum = "extreme_aspect_ratio"
        elif is_border:
            feature_stratum = "border_touching"
        elif is_multi:
            feature_stratum = "multi_candidate_image"
        else:
            feature_stratum = "standard_geometry"

        categorized_persons.append({
            "raw_record": r,
            "candidate_id": r["candidate_id"],
            "split": split,
            "conf_band": conf_band,
            "feature_stratum": feature_stratum,
            "confidence": conf,
            "aspect": aspect,
            "max_gt_iou": max_gt_iou,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w_img": w_img, "h_img": h_img,
        })

    # 3. Stratified Sampling
    # Split quotas exactly proportional to total population:
    # train: 1346 / 1840 = 73.15% -> 146
    # val:   207 / 1840 = 11.25%  -> 23
    # test:  287 / 1840 = 15.60%  -> 31
    # Total = 146 + 23 + 31 = 200
    rng = random.Random(SEED)

    # Group by (split, feature_stratum, conf_band)
    strata_bins = defaultdict(list)
    for p in categorized_persons:
        strata_bins[(p["split"], p["feature_stratum"], p["conf_band"])].append(p)

    # Sort each bin deterministically
    for k in strata_bins:
        strata_bins[k].sort(key=lambda x: (x["candidate_id"]))
        rng.shuffle(strata_bins[k])

    # Allocate target quotas per split
    split_targets = {"train": 146, "val": 23, "test": 31}
    selected_persons = []

    for split, target_n in split_targets.items():
        split_candidates = [p for p in categorized_persons if p["split"] == split]
        # Feature allocation priority: ensure coverage of all features
        # 1. smoke_fire_overlap: allocate heavy sample
        # 2. extreme_aspect_ratio
        # 3. border_touching
        # 4. multi_candidate_image
        # 5. standard_geometry
        feature_order = [
            "smoke_fire_overlap",
            "extreme_aspect_ratio",
            "border_touching",
            "multi_candidate_image",
            "standard_geometry",
        ]
        split_selected = []
        # First pass: pick proportionally across (feature, conf_band)
        split_bins = {k: v[:] for k, v in strata_bins.items() if k[0] == split}
        
        # Round robin across feature bins to guarantee all strata representation
        while len(split_selected) < target_n:
            added_in_round = False
            for feat in feature_order:
                for cband in ("upper_0.60_0.70", "mid_0.50_0.60", "lower_0.40_0.50"):
                    bin_key = (split, feat, cband)
                    if strata_bins[bin_key]:
                        item = strata_bins[bin_key].pop(0)
                        split_selected.append(item)
                        added_in_round = True
                        if len(split_selected) == target_n:
                            break
                if len(split_selected) == target_n:
                    break
            if not added_in_round:
                break
        selected_persons.extend(split_selected)

    print(f"\nStratified person selection complete: {len(selected_persons)} persons sampled.")
    sel_splits = Counter(p["split"] for p in selected_persons)
    print(f"  Splits: {dict(sel_splits)}")
    sel_feats = Counter(p["feature_stratum"] for p in selected_persons)
    print(f"  Feature strata: {dict(sel_feats)}")
    sel_confs = Counter(p["conf_band"] for p in selected_persons)
    print(f"  Confidence bands: {dict(sel_confs)}")

    # 4. Prepare all 326 items for QA Queue
    all_326_items = []
    rank = 1

    # First: All 126 helmets
    for h in sorted(helmets, key=lambda x: float(x["confidence"]), reverse=True):
        split = h["split"]
        img_name = h["image"]
        img_p = REMEDIATED_DIR / "images" / split / img_name
        if img_p not in img_dims:
            with Image.open(img_p) as im:
                img_dims[img_p] = im.size
        w_img, h_img = img_dims[img_p]

        x1 = float(h["x1"])
        y1 = float(h["y1"])
        x2 = float(h["x2"])
        y2 = float(h["y2"])
        bw = x2 - x1
        bh = y2 - y1

        conf = float(h["confidence"])
        conf_band = "upper_0.60_0.70" if conf >= 0.60 else ("mid_0.50_0.60" if conf >= 0.50 else "lower_0.40_0.50")
        stratum = f"helmet_census_{split}_{conf_band}"

        all_326_items.append({
            "queue_rank": rank,
            "candidate_id": h["candidate_id"],
            "tier": "MEDIUM",
            "split": split,
            "image": img_name,
            "class_name": "helmet",
            "canonical_class": 1,
            "confidence": f"{conf:.6f}",
            "selection_stratum": stratum,
            "raw_record": h,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w_img": w_img, "h_img": h_img,
        })
        rank += 1

    # Second: Exactly 200 sampled persons
    for p in sorted(selected_persons, key=lambda x: x["confidence"], reverse=True):
        stratum = f"person_{p['split']}_{p['conf_band']}_{p['feature_stratum']}"
        all_326_items.append({
            "queue_rank": rank,
            "candidate_id": p["candidate_id"],
            "tier": "MEDIUM",
            "split": p["split"],
            "image": p["raw_record"]["image"],
            "class_name": "person",
            "canonical_class": 0,
            "confidence": f"{p['confidence']:.6f}",
            "selection_stratum": stratum,
            "raw_record": p["raw_record"],
            "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
            "w_img": p["w_img"], "h_img": p["h_img"],
        })
        rank += 1

    print(f"\nTotal QA items assembled: {len(all_326_items)} (126 helmets + 200 persons)")

    # 5. Extract Diagnostic Visual Crops
    print(f"\nExtracting diagnostic visual crops to {CROPS_DIR}...")
    for item in all_326_items:
        rank_i = item["queue_rank"]
        cid = item["candidate_id"]
        split = item["split"]
        img_name = item["image"]
        cname = item["class_name"]
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        w_img, h_img = item["w_img"], item["h_img"]

        bw = x2 - x1
        bh = y2 - y1
        margin_x = max(16.0, bw * 0.35)
        margin_y = max(16.0, bh * 0.35)

        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w_img, int(x2 + margin_x))
        cy2 = min(h_img, int(y2 + margin_y))

        # Check if overlay exists
        overlay_p = OVERLAYS_DIR / f"MEDIUM_{split}_{img_name}"
        src_p = overlay_p if overlay_p.exists() else (REMEDIATED_DIR / "images" / split / img_name)

        crop_fname = f"m_item_{rank_i:03d}_{cid}_{cname}.jpg"
        crop_out = CROPS_DIR / crop_fname
        if not crop_out.exists():
            with Image.open(src_p) as src_im:
                c_im = src_im.crop((cx1, cy1, cx2, cy2))
                c_im.save(crop_out)

    print(f"  All 326 diagnostic crops extracted.")

    # Save sampling metadata and interim queue
    sampling_meta = {
        "seed": SEED,
        "total_medium_candidates": len(all_med),
        "helmets_total": len(helmets),
        "helmets_evaluated": len(helmets),
        "persons_total": len(persons),
        "persons_sampled": len(selected_persons),
        "split_distribution": dict(sel_splits),
        "feature_strata_distribution": dict(sel_feats),
        "confidence_bands_distribution": dict(sel_confs),
    }
    with open(ROOT / "docs/audit_artifacts/dfire/dfire_phase2_sampling_spec.json", "w", encoding="utf-8") as f:
        json.dump(sampling_meta, f, indent=2)

    print("Phase 2 sampling and crop preparation complete.")


if __name__ == "__main__":
    main()
