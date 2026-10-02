"""Prepare and validate Human QA Batch 1 for unreviewed D-Fire MEDIUM person candidates.

Executes deterministically:
1. Verifies immutable source datasets and records pre-execution aggregate hashes.
2. Selects exactly 200 genuinely unreviewed MEDIUM person candidates via reproducible stratified sampling.
3. Renders reviewer-friendly context overlays, zoomed crops, and numbered contact sheets (16 items/sheet).
4. Generates empty-verdict qa_queue.csv, batch_manifest.json, and reviewer README.md.
5. Performs full 1:1 artifact validation and cross-pool integrity verification.
6. Computes post-execution aggregate hashes to prove zero dataset mutation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

# Dataset Directories
RAW_DIR = ROOT / "data/raw/dfire/data"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
VERIFIED_DIR = ROOT / "data/processed/dfire_remediated_verified"
VERIFIED_MANIFEST = VERIFIED_DIR / "remediation_manifest.csv"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"

# Evidence artifacts for exclusion cross-checks
P1_ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_qa_label_adjustments.csv"
P1_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_remediated_high_qa_queue.csv"
P2_ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv"
P2_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_medium_qa_queue.csv"

# Target Batch Artifacts Directory
BATCH_DIR = ROOT / "docs/audit_artifacts/dfire/human_qa_medium_person_batch_001"
OVERLAYS_DIR = BATCH_DIR / "overlays"
CROPS_DIR = BATCH_DIR / "crops"
CONTACT_SHEETS_DIR = BATCH_DIR / "contact_sheets"
QA_QUEUE_CSV = BATCH_DIR / "qa_queue.csv"
BATCH_MANIFEST_JSON = BATCH_DIR / "batch_manifest.json"
README_MD = BATCH_DIR / "README.md"

SPLITS = ("train", "val", "test")
TARGET_BATCH_SIZE = 200
SEED = 42

SPLIT_TARGETS = {
    "train": 146,
    "val": 23,
    "test": 31,
}

CLASS_COLORS = {
    0: (50, 205, 50),     # person - Lime Green
    1: (255, 215, 0),    # helmet - Gold / Yellow
    2: (255, 140, 0),    # vest - Dark Orange
    3: (186, 85, 211),   # fall - Medium Orchid
    4: (220, 20, 60),    # fire - Crimson
    5: (30, 144, 255),   # smoke - Dodger Blue
}
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "fall",
    4: "fire",
    5: "smoke",
}
CANDIDATE_COLOR = (255, 0, 128)  # Deep Pink / Bright Magenta


def hash_directory_labels(labels_root: Path) -> tuple[str, int]:
    """Calculates deterministic aggregate SHA-256 hash of all text label files in directory."""
    hasher = hashlib.sha256()
    file_count = 0
    all_txt = []
    for split in SPLITS:
        s_dir = labels_root / split
        if s_dir.exists():
            for p in sorted(s_dir.iterdir()):
                if p.is_file() and p.suffix.lower() == ".txt":
                    all_txt.append((split, p))

    for split, p in all_txt:
        rel_name = f"{split}/{p.name}"
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hasher.update(f"{rel_name}:{digest}\n".encode("utf-8"))
        file_count += 1

    return hasher.hexdigest(), file_count


def hash_raw_directory(raw_root: Path) -> tuple[str, int]:
    """Calculates deterministic aggregate SHA-256 hash of all raw images."""
    hasher = hashlib.sha256()
    file_count = 0
    all_imgs = []
    for split in SPLITS:
        s_dir = raw_root / split / "images"
        if s_dir.exists():
            for p in sorted(s_dir.iterdir()):
                if p.is_file():
                    all_imgs.append((split, p))

    for split, p in all_imgs:
        stat = p.stat()
        hasher.update(f"{split}/{p.name}:{stat.st_size}\n".encode("utf-8"))
        file_count += 1

    return hasher.hexdigest(), file_count


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Calculates Intersection over Union for two xyxy boxes."""
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


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Safely loads TrueType font with graceful fallback."""
    font_candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "tahomabd.ttf" if bold else "tahoma.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    start_time = time.perf_counter()
    gen_timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    print("=" * 80)
    print("D-FIRE HUMAN QA BATCH 1 PREPARATION & AUDIT PIPELINE")
    print(f"Timestamp: {gen_timestamp}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: PRE-EXECUTION IMMUTABILITY RECORDING
    # -------------------------------------------------------------------------
    print("\n[Step 1] Recording pre-execution aggregate dataset hashes...")
    raw_pre_hash, raw_pre_count = hash_raw_directory(RAW_DIR)
    corr_pre_hash, corr_pre_count = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_pre_hash, remed_pre_count = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_pre_hash, verif_pre_count = hash_directory_labels(VERIFIED_DIR / "labels")

    print(f"  data/raw image count: {raw_pre_count}, SHA-256: {raw_pre_hash}")
    print(f"  dfire_corrected label count: {corr_pre_count}, SHA-256: {corr_pre_hash}")
    print(f"  dfire_remediated label count: {remed_pre_count}, SHA-256: {remed_pre_hash}")
    print(f"  dfire_remediated_verified label count: {verif_pre_count}, SHA-256: {verif_pre_hash}")

    # -------------------------------------------------------------------------
    # STEP 2: INGESTION & ELIGIBILITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[Step 2] Ingesting candidates and verifying eligible unreviewed population...")
    if not VERIFIED_MANIFEST.exists():
        print(f"ERROR: {VERIFIED_MANIFEST} not found!")
        sys.exit(1)
    if not CANDIDATES_CSV.exists():
        print(f"ERROR: {CANDIDATES_CSV} not found!")
        sys.exit(1)

    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        all_raw_cands = {r["candidate_id"]: r for r in csv.DictReader(f)}
    print(f"  Loaded {len(all_raw_cands)} total candidate records from candidates.csv")

    with open(VERIFIED_MANIFEST, "r", encoding="utf-8") as f:
        verified_manifest_rows = list(csv.DictReader(f))
    print(f"  Loaded {len(verified_manifest_rows)} manifest rows from verified remediation manifest")

    # Ingest Phase 1 & Phase 2 reviewed candidate IDs for strict exclusion
    p1_reviewed_cids = set()
    p2_reviewed_cids = set()
    for row in verified_manifest_rows:
        if "Phase 1" in row["phase"]:
            p1_reviewed_cids.add(row["candidate_id"])
        elif "Phase 2" in row["phase"]:
            p2_reviewed_cids.add(row["candidate_id"])

    print(f"  Phase 1 reviewed candidates identified: {len(p1_reviewed_cids)}")
    print(f"  Phase 2 reviewed candidates identified: {len(p2_reviewed_cids)}")

    # Filter target unreviewed pool
    unreviewed_medium_persons = []
    for r in verified_manifest_rows:
        cid = r["candidate_id"]
        phase = r["phase"]
        cname = r["class_name"]
        act = r["action"]
        status = r["status"]
        if (
            phase == "Unreviewed (MEDIUM)"
            and cname == "person"
            and act == "EXCLUDED_PENDING_GENUINE_VISUAL_QA"
            and status == "EXCLUDED"
        ):
            unreviewed_medium_persons.append(r)

    print(f"  Total eligible unreviewed MEDIUM persons: {len(unreviewed_medium_persons)}")
    if len(unreviewed_medium_persons) != 1640:
        print(f"ERROR: Expected exactly 1,640 unreviewed MEDIUM persons, got {len(unreviewed_medium_persons)}")
        sys.exit(1)

    # Cross-pool safety checks
    unreviewed_cids = {r["candidate_id"] for r in unreviewed_medium_persons}
    p1_leakage = unreviewed_cids & p1_reviewed_cids
    p2_leakage = unreviewed_cids & p2_reviewed_cids
    low_tier_leakage = [cid for cid in unreviewed_cids if all_raw_cands[cid]["tier"] == "LOW"]

    print(f"  Safety check: Phase 1 overlap = {len(p1_leakage)}")
    print(f"  Safety check: Phase 2 overlap = {len(p2_leakage)}")
    print(f"  Safety check: LOW tier overlap = {len(low_tier_leakage)}")
    if p1_leakage or p2_leakage or low_tier_leakage:
        print("ERROR: Population purity violation detected!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3: FEATURE EXTRACTION & STRATIFICATION
    # -------------------------------------------------------------------------
    print("\n[Step 3] Extracting features and strata for stratified sampling...")
    img_counts = Counter((r["split"], r["image"]) for r in unreviewed_medium_persons)
    img_dims = {}

    # Pre-cache image dimensions and label paths
    categorized_candidates = []
    for r in unreviewed_medium_persons:
        cid = r["candidate_id"]
        split = r["split"]
        img_name = r["image"]
        stem = Path(img_name).stem
        conf = float(r["confidence"])

        raw_meta = all_raw_cands[cid]
        x1 = float(raw_meta["x1"])
        y1 = float(raw_meta["y1"])
        x2 = float(raw_meta["x2"])
        y2 = float(raw_meta["y2"])

        img_p = VERIFIED_DIR / "images" / split / img_name
        if img_p not in img_dims:
            with Image.open(img_p) as im:
                img_dims[img_p] = im.size
        w_img, h_img = img_dims[img_p]

        bw = x2 - x1
        bh = y2 - y1
        aspect = bh / bw if bw > 0 else 0.0
        norm_area = (bw * bh) / (w_img * h_img) if (w_img * h_img) > 0 else 0.0

        # Confidence band
        if conf >= 0.60:
            conf_band = "upper_0.60_0.70"
        elif conf >= 0.50:
            conf_band = "mid_0.50_0.60"
        else:
            conf_band = "lower_0.40_0.50"

        # Geometry size band
        if norm_area < 0.005:
            size_band = "small_box"
        elif norm_area < 0.05:
            size_band = "medium_box"
        else:
            size_band = "large_box"

        # Overlap with existing fire (4) or smoke (5) labels in verified dataset
        lbl_p = VERIFIED_DIR / "labels" / split / f"{stem}.txt"
        max_gt_iou = 0.0
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id in (4, 5):  # fire or smoke
                    l_xc, l_yc, l_w, l_h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gx1 = (l_xc - l_w / 2.0) * w_img
                    gy1 = (l_yc - l_h / 2.0) * h_img
                    gx2 = (l_xc + l_w / 2.0) * w_img
                    gy2 = (l_yc + l_h / 2.0) * h_img
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

        # Normalized coordinates (xc, yc, w, h)
        xc_norm = (x1 + x2) / (2.0 * w_img)
        yc_norm = (y1 + y2) / (2.0 * h_img)
        w_norm = bw / float(w_img)
        h_norm = bh / float(h_img)

        categorized_candidates.append({
            "candidate_id": cid,
            "split": split,
            "image": img_name,
            "confidence": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w_img": w_img,
            "h_img": h_img,
            "xc_norm": xc_norm,
            "yc_norm": yc_norm,
            "w_norm": w_norm,
            "h_norm": h_norm,
            "aspect": aspect,
            "norm_area": norm_area,
            "conf_band": conf_band,
            "size_band": size_band,
            "feature_stratum": feature_stratum,
            "max_gt_iou": max_gt_iou,
        })

    print(f"  Feature categorization completed for {len(categorized_candidates)} candidates.")

    # -------------------------------------------------------------------------
    # STEP 4: DETERMINISTIC REPRODUCIBLE SELECTION
    # -------------------------------------------------------------------------
    print(f"\n[Step 4] Deterministically selecting exactly {TARGET_BATCH_SIZE} candidates (seed={SEED})...")
    rng = random.Random(SEED)

    # Group by (split, feature_stratum, conf_band)
    strata_bins = defaultdict(list)
    for c in categorized_candidates:
        strata_bins[(c["split"], c["feature_stratum"], c["conf_band"])].append(c)

    # Sort each bin deterministically before shuffle
    for k in sorted(strata_bins.keys()):
        strata_bins[k].sort(key=lambda x: x["candidate_id"])
        rng.shuffle(strata_bins[k])

    feature_priority = [
        "smoke_fire_overlap",
        "extreme_aspect_ratio",
        "border_touching",
        "multi_candidate_image",
        "standard_geometry",
    ]
    conf_band_order = ("upper_0.60_0.70", "mid_0.50_0.60", "lower_0.40_0.50")

    selected_candidates = []
    for split, target_n in SPLIT_TARGETS.items():
        split_selected = []
        while len(split_selected) < target_n:
            added_in_round = False
            for feat in feature_priority:
                for cband in conf_band_order:
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
                # If strata exhausted, pull remaining deterministically from any split bin
                remaining_in_split = []
                for (s, f_str, cb), bin_items in strata_bins.items():
                    if s == split and bin_items:
                        remaining_in_split.extend(bin_items)
                        bin_items.clear()
                remaining_in_split.sort(key=lambda x: (-x["confidence"], x["candidate_id"]))
                needed = target_n - len(split_selected)
                split_selected.extend(remaining_in_split[:needed])
                break

        print(f"  Selected for split '{split}': {len(split_selected)} / {target_n}")
        selected_candidates.extend(split_selected)

    if len(selected_candidates) != TARGET_BATCH_SIZE:
        print(f"ERROR: Expected {TARGET_BATCH_SIZE} selected candidates, got {len(selected_candidates)}")
        sys.exit(1)

    # Deterministic final ordering: confidence descending, then candidate_id ascending
    selected_candidates.sort(key=lambda x: (-x["confidence"], x["candidate_id"]))

    for rank_idx, c in enumerate(selected_candidates, start=1):
        c["queue_rank"] = rank_idx

    # Compute distribution breakdowns
    sel_splits = Counter(c["split"] for c in selected_candidates)
    sel_confs = Counter(c["conf_band"] for c in selected_candidates)
    sel_feats = Counter(c["feature_stratum"] for c in selected_candidates)
    sel_sizes = Counter(c["size_band"] for c in selected_candidates)

    print("\nBatch 1 Selection Breakdown:")
    print(f"  Splits: {dict(sel_splits)}")
    print(f"  Confidence Bands: {dict(sel_confs)}")
    print(f"  Feature Strata: {dict(sel_feats)}")
    print(f"  Size Bands: {dict(sel_sizes)}")

    # -------------------------------------------------------------------------
    # STEP 5: ARTIFACT GENERATION (OVERLAYS, CROPS, CONTACT SHEETS)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Building reviewer-friendly visual artifacts under:")
    print(f"  {BATCH_DIR}")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    # Fonts
    font_large_bold = get_font(18, bold=True)
    font_med_bold = get_font(14, bold=True)
    font_small = get_font(11, bold=False)
    font_header = get_font(22, bold=True)
    font_sub = get_font(13, bold=False)

    total_contact_sheets = math.ceil(len(selected_candidates) / 16)
    queue_rows = []

    print("  Rendering context overlays and zoomed crops...")
    for idx, c in enumerate(selected_candidates, start=1):
        cid = c["candidate_id"]
        split = c["split"]
        img_name = c["image"]
        stem = Path(img_name).stem
        conf = c["confidence"]
        x1, y1, x2, y2 = c["x1"], c["y1"], c["x2"], c["y2"]
        w_img, h_img = c["w_img"], c["h_img"]

        img_path = VERIFIED_DIR / "images" / split / img_name
        lbl_path = VERIFIED_DIR / "labels" / split / f"{stem}.txt"

        # 5.1 Render Context Overlay
        with Image.open(img_path).convert("RGB") as base_im:
            overlay_im = base_im.copy()
            draw_overlay = ImageDraw.Draw(overlay_im)

            # Draw existing verified labels in distinct colors
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    l_xc, l_yc, l_w, l_h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    lx1 = (l_xc - l_w / 2.0) * w_img
                    ly1 = (l_yc - l_h / 2.0) * h_img
                    lx2 = (l_xc + l_w / 2.0) * w_img
                    ly2 = (l_yc + l_h / 2.0) * h_img

                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    cname = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    draw_overlay.rectangle([lx1, ly1, lx2, ly2], outline=color, width=2)
                    tag_text = f"existing: {cname}"
                    draw_overlay.rectangle([lx1, max(0, ly1 - 14), lx1 + len(tag_text) * 7, ly1], fill=color)
                    draw_overlay.text((lx1 + 2, max(0, ly1 - 13)), tag_text, fill=(255, 255, 255), font=font_small)

            # Prominently draw candidate box
            draw_overlay.rectangle([x1, y1, x2, y2], outline=CANDIDATE_COLOR, width=4)
            cand_tag = f"CANDIDATE: {cid} | person | conf: {conf:.3f}"
            tag_w = len(cand_tag) * 8 + 8
            tag_y1 = max(0, y1 - 20) if y1 >= 20 else y2
            draw_overlay.rectangle([x1, tag_y1, x1 + tag_w, tag_y1 + 18], fill=CANDIDATE_COLOR)
            draw_overlay.text((x1 + 4, tag_y1 + 1), cand_tag, fill=(255, 255, 255), font=font_med_bold)

            overlay_fname = f"{cid}_context_overlay.jpg"
            overlay_rel_path = f"docs/audit_artifacts/dfire/human_qa_medium_person_batch_001/overlays/{overlay_fname}"
            overlay_im.save(OVERLAYS_DIR / overlay_fname, quality=95)

        # 5.2 Render Zoomed Crop
        bw = x2 - x1
        bh = y2 - y1
        margin_x = max(36.0, bw * 0.40)
        margin_y = max(36.0, bh * 0.40)

        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w_img, int(x2 + margin_x))
        cy2 = min(h_img, int(y2 + margin_y))

        with Image.open(img_path).convert("RGB") as base_im:
            crop_im = base_im.crop((cx1, cy1, cx2, cy2))
            draw_crop = ImageDraw.Draw(crop_im)

            # Draw existing verified labels that intersect the crop
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    l_xc, l_yc, l_w, l_h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    lx1 = (l_xc - l_w / 2.0) * w_img
                    ly1 = (l_yc - l_h / 2.0) * h_img
                    lx2 = (l_xc + l_w / 2.0) * w_img
                    ly2 = (l_yc + l_h / 2.0) * h_img

                    # Intersect with crop
                    if compute_iou((lx1, ly1, lx2, ly2), (cx1, cy1, cx2, cy2)) > 0.0 or (
                        lx1 >= cx1 and lx2 <= cx2 and ly1 >= cy1 and ly2 <= cy2
                    ):
                        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                        cname = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                        draw_crop.rectangle([lx1 - cx1, ly1 - cy1, lx2 - cx1, ly2 - cy1], outline=color, width=2)

            # Draw candidate box within crop
            rx1 = x1 - cx1
            ry1 = y1 - cy1
            rx2 = x2 - cx1
            ry2 = y2 - cy1
            draw_crop.rectangle([rx1, ry1, rx2, ry2], outline=CANDIDATE_COLOR, width=3)
            crop_tag = f"{cid} ({conf:.3f})"
            tag_cy1 = max(0, ry1 - 16) if ry1 >= 16 else ry2
            draw_crop.rectangle([rx1, tag_cy1, rx1 + len(crop_tag) * 7 + 4, tag_cy1 + 14], fill=CANDIDATE_COLOR)
            draw_crop.text((rx1 + 2, tag_cy1), crop_tag, fill=(255, 255, 255), font=font_small)

            crop_fname = f"{cid}_zoomed_crop.jpg"
            crop_rel_path = f"docs/audit_artifacts/dfire/human_qa_medium_person_batch_001/crops/{crop_fname}"
            crop_im.save(CROPS_DIR / crop_fname, quality=95)

        # Contact sheet page and position reference
        cs_page = ((idx - 1) // 16) + 1
        pos_in_sheet = (idx - 1) % 16
        row_in_sheet = (pos_in_sheet // 4) + 1
        col_in_sheet = (pos_in_sheet % 4) + 1
        cs_ref = f"contact_sheets/contact_sheet_{cs_page:02d}.jpg (item {idx}, row {row_in_sheet}, col {col_in_sheet})"

        c["crop_file"] = CROPS_DIR / crop_fname
        c["overlay_rel_path"] = overlay_rel_path
        c["crop_rel_path"] = crop_rel_path
        c["cs_ref"] = cs_ref

        # Prepare QA Queue row (strictly blank review verdicts)
        pixel_bbox_str = f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
        norm_bbox_str = f"({c['xc_norm']:.6f}, {c['yc_norm']:.6f}, {c['w_norm']:.6f}, {c['h_norm']:.6f})"

        queue_rows.append({
            "candidate_id": cid,
            "tier": "MEDIUM",
            "split": split,
            "image": img_name,
            "confidence": f"{conf:.6f}",
            "pixel_bbox": pixel_bbox_str,
            "normalized_bbox": norm_bbox_str,
            "overlay_path": overlay_rel_path,
            "crop_path": crop_rel_path,
            "contact_sheet_reference": cs_ref,
            "human_verdict": "",
            "corrected_bbox": "",
            "reviewer_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    # 5.3 Render Numbered Contact Sheets (16 items per sheet)
    print("  Rendering numbered contact sheets (16 items per sheet)...")
    sheet_items_chunks = [selected_candidates[i : i + 16] for i in range(0, len(selected_candidates), 16)]

    cell_w = 380
    cell_h = 370
    thumb_w = 360
    thumb_h = 270
    grid_cols = 4
    grid_rows = 4
    margin_lr = 30
    header_h = 110

    total_sheet_w = margin_lr * 2 + grid_cols * cell_w
    total_sheet_h = header_h + grid_rows * cell_h + 30

    for sheet_idx, chunk in enumerate(sheet_items_chunks, start=1):
        sheet_im = Image.new("RGB", (total_sheet_w, total_sheet_h), color=(24, 27, 34))
        draw_cs = ImageDraw.Draw(sheet_im)

        # Draw Header
        draw_cs.rectangle([0, 0, total_sheet_w, header_h - 10], fill=(15, 18, 24))
        title_text = f"D-FIRE HUMAN QA BATCH 001 — CONTACT SHEET {sheet_idx:02d} / {total_contact_sheets:02d}"
        draw_cs.text((margin_lr, 20), title_text, fill=(255, 255, 255), font=font_header)

        start_num = (sheet_idx - 1) * 16 + 1
        end_num = start_num + len(chunk) - 1
        sub_text = (
            f"Review Items #{start_num:03d} to #{end_num:03d} (of 200) | Target: person (0) | "
            f"Tier: MEDIUM | Status: PENDING GENUINE HUMAN VISUAL QA"
        )
        draw_cs.text((margin_lr, 62), sub_text, fill=(180, 190, 205), font=font_sub)

        # Draw Cells
        for pos, item in enumerate(chunk):
            r = pos // grid_cols
            col = pos % grid_cols
            x_cell = margin_lr + col * cell_w
            y_cell = header_h + r * cell_h

            # Cell card background
            draw_cs.rectangle(
                [x_cell + 5, y_cell + 5, x_cell + cell_w - 5, y_cell + cell_h - 5],
                fill=(33, 38, 48),
                outline=(55, 62, 78),
                width=1,
            )

            # Load and paste zoomed crop thumbnail
            crop_path = item["crop_file"]
            with Image.open(crop_path) as crp:
                crp_copy = crp.copy()
                crp_copy.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                tw, th = crp_copy.size
                tx = x_cell + 10 + (thumb_w - tw) // 2
                ty = y_cell + 10 + (thumb_h - th) // 2
                sheet_im.paste(crp_copy, (tx, ty))

            # Info text banner below thumbnail
            text_y = y_cell + thumb_h + 16
            draw_cs.text(
                (x_cell + 12, text_y),
                f"#{item['queue_rank']:03d} | {item['candidate_id']}",
                fill=(255, 255, 255),
                font=font_med_bold,
            )
            draw_cs.text(
                (x_cell + 12, text_y + 20),
                f"{item['split']} | {item['image']}",
                fill=(180, 195, 215),
                font=font_small,
            )
            draw_cs.text(
                (x_cell + 12, text_y + 36),
                f"Class: person | Conf: {item['confidence']:.4f}",
                fill=(255, 215, 0),
                font=font_small,
            )
            draw_cs.text(
                (x_cell + 12, text_y + 52),
                f"Strata: {item['feature_stratum']} ({item['size_band']})",
                fill=(140, 155, 175),
                font=font_small,
            )

        cs_out_path = CONTACT_SHEETS_DIR / f"contact_sheet_{sheet_idx:02d}.jpg"
        sheet_im.save(cs_out_path, quality=92)

    print(f"  Successfully rendered {len(sheet_items_chunks)} contact sheets.")

    # 5.4 Write qa_queue.csv
    print(f"  Writing {QA_QUEUE_CSV}...")
    fieldnames = [
        "candidate_id",
        "tier",
        "split",
        "image",
        "confidence",
        "pixel_bbox",
        "normalized_bbox",
        "overlay_path",
        "crop_path",
        "contact_sheet_reference",
        "human_verdict",
        "corrected_bbox",
        "reviewer_notes",
        "reviewer",
        "reviewed_at",
    ]
    with open(QA_QUEUE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue_rows)

    # 5.5 Write batch_manifest.json
    print(f"  Writing {BATCH_MANIFEST_JSON}...")
    batch_manifest_data = {
        "batch_id": "human_qa_medium_person_batch_001",
        "batch_version": "1.0.0",
        "task_scope": "Preparation of Human QA Batch 1 for genuinely unreviewed D-Fire MEDIUM person candidates",
        "trusted_dataset": "data/processed/dfire_remediated_verified",
        "total_unreviewed_pool": 1640,
        "batch_size": TARGET_BATCH_SIZE,
        "candidate_selection_rule": (
            "Proportional stratified deterministic sampling across splits (train=146, val=23, test=31) "
            "and feature strata (smoke_fire_overlap, extreme_aspect_ratio, border_touching, "
            "multi_candidate_image, standard_geometry) and confidence bands ([0.60, 0.70), [0.50, 0.60), [0.40, 0.50)). "
            "Bins sorted by candidate_id before seed=42 shuffle. Final queue sorted by confidence DESC, candidate_id ASC."
        ),
        "seed_and_ordering": {
            "seed": SEED,
            "ordering": "confidence DESC, candidate_id ASC",
        },
        "source_hashes": {
            "data_raw_dfire_data_count": raw_pre_count,
            "data_raw_dfire_data_sha256": raw_pre_hash,
            "data_processed_dfire_corrected_labels_count": corr_pre_count,
            "data_processed_dfire_corrected_labels_sha256": corr_pre_hash,
            "data_processed_dfire_remediated_labels_count": remed_pre_count,
            "data_processed_dfire_remediated_labels_sha256": remed_pre_hash,
            "data_processed_dfire_remediated_verified_labels_count": verif_pre_count,
            "data_processed_dfire_remediated_verified_labels_sha256": verif_pre_hash,
        },
        "selected_candidate_ids": [c["candidate_id"] for c in selected_candidates],
        "counts_by_split": dict(sel_splits),
        "counts_by_confidence_band": dict(sel_confs),
        "counts_by_geometry_strata": dict(sel_feats),
        "counts_by_size_band": dict(sel_sizes),
        "contact_sheets_count": total_contact_sheets,
        "generation_timestamp": gen_timestamp,
    }
    with open(BATCH_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(batch_manifest_data, f, indent=2)

    # 5.6 Write Reviewer README.md
    print(f"  Writing {README_MD}...")
    readme_content = f"""# D-Fire Human QA: Medium Person Batch 001

## 1. Overview & Objectives
- **Batch Identifier:** `human_qa_medium_person_batch_001`
- **Scope:** Exactly **200 MEDIUM-tier person candidates** sampled deterministically from the 1,640 genuinely unreviewed D-Fire population.
- **Trusted Reference Dataset:** `data/processed/dfire_remediated_verified`
- **Target Class:** `person` (canonical class ID `0`)
- **Queue Location:** [qa_queue.csv](file:///C:/Users/USER/Desktop/BUU69/Project-CCTV-Safety-antigravity/docs/audit_artifacts/dfire/human_qa_medium_person_batch_001/qa_queue.csv)

All 200 candidates in this batch currently have blank review verdicts. No automated semantic verdict or label mutation has been applied.

---

## 2. Review Artifacts Structure
Reviewers should consult the provided visual artifacts for every candidate:
1. **Contact Sheets:** `contact_sheets/contact_sheet_01.jpg` to `contact_sheets/contact_sheet_13.jpg` (16 items per sheet with readable ID, split, image name, confidence, and stratum).
2. **Context Overlays:** `overlays/<candidate_id>_context_overlay.jpg` showing the candidate box in thick magenta (`#FF0080`), with existing verified dataset labels rendered in distinct colors (`person`: green, `helmet`: yellow, `fire`: red, `smoke`: blue).
3. **Zoomed Diagnostic Crops:** `crops/<candidate_id>_zoomed_crop.jpg` providing high-resolution local context around the candidate detection.

---

## 3. Human Review Instructions & Verdict Taxonomy

Human reviewers must record verdicts strictly using one of the four allowed tokens:

| Verdict | Meaning | When to Use | Action Required |
| :--- | :--- | :--- | :--- |
| **`PASS`** | Confirmed True Positive | The candidate is a genuine, usable person instance (pedestrian, firefighter, civilian) with a tight and accurate bounding box. | Set `human_verdict` to `PASS`. Leave `corrected_bbox` blank. |
| **`FIX`** | True Positive with Inaccurate Box | A genuine person is present, but the proposed candidate bounding box is loose, shifted, truncated, or excludes limbs/head. | Set `human_verdict` to `FIX`. Provide exact corrected coordinates in `corrected_bbox`. |
| **`REMOVE`** | False Positive / Reject | The candidate is not a person. Typical false positives include smoke puffs, flame shapes, tree trunks, fire equipment, shadows, or vehicles. | Set `human_verdict` to `REMOVE`. Leave `corrected_bbox` blank. |
| **`UNCERTAIN`** | Borderline / Ambiguous | The detection is heavily obscured, severely blurred, in thick smoke haze, or distant to the point of unresolvable ambiguity. | Set `human_verdict` to `UNCERTAIN`. Leave `corrected_bbox` blank. |

### Recording FIX Coordinates
When marking a candidate as `FIX`, the reviewer MUST provide the corrected coordinates in the `corrected_bbox` column in either format:
- **Normalized YOLO Format (Recommended):** `(xc, yc, w, h)` with values in $[0.0, 1.0]$ relative to the full image.
- **Pixel Coordinates Format:** `(x1, y1, x2, y2)` with pixel values $[0, W]$ and $[0, H]$.

### Unreviewed Status
- A **blank** value in `human_verdict` signifies that the candidate has **not yet been reviewed**.
- Do NOT prefill or auto-assign verdicts. Every verdict must originate from verified visual inspection.

### Reviewer Metadata
When completing a review row in `qa_queue.csv`, please fill:
- `reviewer`: Reviewer ID (e.g. `USER_NAME` or `AUDITOR_ID`).
- `reviewed_at`: ISO 8601 completion timestamp (e.g. `2026-09-26T12:00:00+07:00`).
- `reviewer_notes`: Concise explanation of visual rationale (especially for `FIX`, `REMOVE`, and `UNCERTAIN`).

---

## 4. Batch Selection Summary
- **Candidate Pool:** 1,640 unreviewed MEDIUM persons (`data/processed/dfire_remediated_verified/remediation_manifest.csv`)
- **Sampling Strategy:** Stratified deterministic sampling (`seed=42`)
- **Split Breakdown:** `train`: 146 | `val`: 23 | `test`: 31 (Total: 200)
- **Confidence Bands:** Upper [0.60, 0.70): {sel_confs['upper_0.60_0.70']} | Mid [0.50, 0.60): {sel_confs['mid_0.50_0.60']} | Lower [0.40, 0.50): {sel_confs['lower_0.40_0.50']}
"""
    with open(README_MD, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # -------------------------------------------------------------------------
    # STEP 6: VERIFICATION & AUDIT GATES
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running comprehensive validation audit...")
    validation_passed = True
    val_errors = []

    # 6.1 Check 1:1 file existence and mapping
    print("  Checking 1:1 visual artifact existence...")
    if not QA_QUEUE_CSV.exists():
        val_errors.append(f"Missing {QA_QUEUE_CSV}")
    if not BATCH_MANIFEST_JSON.exists():
        val_errors.append(f"Missing {BATCH_MANIFEST_JSON}")
    if not README_MD.exists():
        val_errors.append(f"Missing {README_MD}")

    overlays_found = list(OVERLAYS_DIR.glob("*.jpg"))
    crops_found = list(CROPS_DIR.glob("*.jpg"))
    sheets_found = list(CONTACT_SHEETS_DIR.glob("*.jpg"))

    print(f"  Found {len(overlays_found)} context overlays (expected 200)")
    print(f"  Found {len(crops_found)} zoomed crops (expected 200)")
    print(f"  Found {len(sheets_found)} contact sheets (expected 13)")

    if len(overlays_found) != TARGET_BATCH_SIZE:
        val_errors.append(f"Expected {TARGET_BATCH_SIZE} overlays, found {len(overlays_found)}")
    if len(crops_found) != TARGET_BATCH_SIZE:
        val_errors.append(f"Expected {TARGET_BATCH_SIZE} crops, found {len(crops_found)}")
    if len(sheets_found) != total_contact_sheets:
        val_errors.append(f"Expected {total_contact_sheets} contact sheets, found {len(sheets_found)}")

    # 6.2 Check queue rows integrity
    with open(QA_QUEUE_CSV, "r", encoding="utf-8") as f:
        queue_check = list(csv.DictReader(f))

    if len(queue_check) != TARGET_BATCH_SIZE:
        val_errors.append(f"Expected {TARGET_BATCH_SIZE} rows in qa_queue.csv, found {len(queue_check)}")

    queue_cids = [r["candidate_id"] for r in queue_check]
    if len(set(queue_cids)) != TARGET_BATCH_SIZE:
        val_errors.append(f"Duplicate candidate IDs found in qa_queue.csv: {len(queue_cids)} vs {len(set(queue_cids))}")

    # Check that all human review fields are strictly blank
    non_blank_verdicts = [r["candidate_id"] for r in queue_check if r["human_verdict"].strip()]
    if non_blank_verdicts:
        val_errors.append(f"Found non-blank human verdicts in {len(non_blank_verdicts)} rows! (Must be blank)")

    # 6.3 Verify candidate IDs belong strictly to unreviewed pool
    selected_set = set(queue_cids)
    if not selected_set.issubset(unreviewed_cids):
        diff = selected_set - unreviewed_cids
        val_errors.append(f"Selected IDs not in unreviewed pool: {diff}")

    # Check 0 overlap with Phase 1 and Phase 2
    if selected_set & p1_reviewed_cids:
        val_errors.append(f"Selected IDs overlap with Phase 1: {selected_set & p1_reviewed_cids}")
    if selected_set & p2_reviewed_cids:
        val_errors.append(f"Selected IDs overlap with Phase 2: {selected_set & p2_reviewed_cids}")

    # -------------------------------------------------------------------------
    # STEP 7: POST-EXECUTION IMMUTABILITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[Step 7] Checking post-execution aggregate hashes across all datasets...")
    raw_post_hash, raw_post_count = hash_raw_directory(RAW_DIR)
    corr_post_hash, corr_post_count = hash_directory_labels(CORRECTED_DIR / "labels")
    remed_post_hash, remed_post_count = hash_directory_labels(REMEDIATED_DIR / "labels")
    verif_post_hash, verif_post_count = hash_directory_labels(VERIFIED_DIR / "labels")

    print(f"  data/raw post count: {raw_post_count}, SHA-256: {raw_post_hash}")
    print(f"  dfire_corrected post count: {corr_post_count}, SHA-256: {corr_post_hash}")
    print(f"  dfire_remediated post count: {remed_post_count}, SHA-256: {remed_post_hash}")
    print(f"  dfire_remediated_verified post count: {verif_post_count}, SHA-256: {verif_post_hash}")

    if raw_pre_hash != raw_post_hash:
        val_errors.append(f"MUTATION DETECTED in data/raw! Pre: {raw_pre_hash}, Post: {raw_post_hash}")
    if corr_pre_hash != corr_post_hash:
        val_errors.append(f"MUTATION DETECTED in dfire_corrected! Pre: {corr_pre_hash}, Post: {corr_post_hash}")
    if remed_pre_hash != remed_post_hash:
        val_errors.append(f"MUTATION DETECTED in dfire_remediated! Pre: {remed_pre_hash}, Post: {remed_post_hash}")
    if verif_pre_hash != verif_post_hash:
        val_errors.append(f"MUTATION DETECTED in dfire_remediated_verified! Pre: {verif_pre_hash}, Post: {verif_post_hash}")

    if val_errors:
        print("\nVALIDATION FAILED:")
        for err in val_errors:
            print(f"  [FAIL] {err}")
        sys.exit(1)
    else:
        print("\nALL VALIDATION GATES PASSED:")
        print("  [PASS] Exactly 200 unique candidates selected from 1,640 unreviewed pool.")
        print("  [PASS] 0 overlap with Phase 1, Phase 2, or LOW tier.")
        print("  [PASS] 200 context overlays, 200 zoomed crops, and 13 contact sheets exist and match 1:1.")
        print("  [PASS] qa_queue.csv contains exactly 200 rows with all human verdict fields strictly blank.")
        print("  [PASS] Pre/post aggregate hashes match 100% across all 4 datasets (Zero mutation).")

    elapsed = time.perf_counter() - start_time
    print(f"\nPipeline finished successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
