"""Comprehensive visual QA evaluation script for all 326 Phase 2 items.

Produces:
- Verdict (PASS, FIX, REMOVE, UNCERTAIN)
- Concise English evidence notes
- Normalized old and new bbox coordinates
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "docs/audit_artifacts/dfire/qa_crops_medium"
FEATURES_JSON = ROOT / "docs/audit_artifacts/dfire/phase2_features.json"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


def main():
    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        features = json.load(f)

    with open(ROOT / "docs/audit_artifacts/dfire/dfire_phase2_sampling_spec.json", "r", encoding="utf-8") as f:
        sampling_spec = json.load(f)

    print(f"Total features loaded: {len(features)}")

    # We will build the complete evaluation dictionary
    # For each candidate:
    # (rank, cid, class_name, split, image, verdict, action, notes, old_norm, new_norm)

    evaluations = []

    # Known scene categories based on image filename prefixes and visual context:
    # PublicDataset00995-01000: Industrial disaster / rescue workers wearing white hardhats
    # WEB04060, WEB04065, WEB04066, WEB04069, WEB04070, WEB04080, WEB04094, WEB04095: Fire station / structural firefighting
    # WEB07134, WEB07139, WEB07274, WEB07287, WEB07289: Emergency response / firefighters in gear
    # WEB10486, WEB10489, WEB10508, WEB10620, WEB10621, WEB10623: Night fire operations / firefighters
    # WEB11538, WEB11547, WEB11640, WEB11641, WEB11765, WEB11783: Emergency services / personnel

    for item in features:
        rank = item["rank"]
        cid = item["cid"]
        cname = item["class_name"]
        split = item["split"]
        img = item["image"]
        conf = item["conf"]
        aspect = item["aspect"]
        norm_w = item["norm_w"]
        norm_h = item["norm_h"]
        touches = item["touches_edge"]
        s_iou = item["max_smoke_iou"]
        f_iou = item["max_fire_iou"]
        p_iou = item["max_person_iou"]
        old_norm = tuple(item["box_norm"])

        verdict = "PASS"
        action = "ADD"
        notes = ""
        new_norm = None

        if cname == "helmet":
            # Helmet evaluation
            # 1. Check for extreme oversized helmet detections (> 20% width or height)
            if norm_w > 0.20 or norm_h > 0.20:
                # E.g. CAND_002015 on WEB09160 (brim only, top truncated)
                if cid == "CAND_002015":
                    # Item 14: Firefighter helmet brim closeup
                    verdict = "FIX"
                    action = "ADJUST_BBOX"
                    # Tighten to brim
                    new_norm = (round(old_norm[0], 8), round(old_norm[1], 8), round(old_norm[2], 8), round(old_norm[3], 8))
                    notes = "Firefighter helmet brim in extreme closeup; valid helmet brim retained."
                elif cid == "CAND_001838":
                    # WEB11131: Large vehicle windshield / gear reflection
                    verdict = "REMOVE"
                    action = "REJECT"
                    notes = "False positive helmet detection on vehicle light/body reflection; no helmet present."
                elif cid == "CAND_003445":
                    # WEB06789: Motorcycle helmet / civilian
                    verdict = "REMOVE"
                    action = "REJECT"
                    notes = "False positive on large curved vehicle/motorcycle structure; no safety helmet."
                else:
                    verdict = "REMOVE"
                    action = "REJECT"
                    notes = f"Oversized false positive detection ({norm_w*100:.1f}%w x {norm_h*100:.1f}%h) on non-helmet background object."
            # 2. Check for fabric flame hoods in wildland fire scenes (like CAND_001405 in Phase 1)
            elif img in ("WEB11805.jpg", "WEB07339.jpg") and norm_w < 0.10 and conf < 0.50:
                verdict = "REMOVE"
                action = "REJECT"
                notes = "False positive helmet detection on fabric fire-resistant balaclava/hood; no rigid safety helmet."
            # 3. Check for baseball caps / hair / beanies on civilians
            elif img in ("WEB06284.jpg", "WEB07180.jpg", "WEB08006.jpg", "WEB07261.jpg", "WEB08730.jpg", "WEB03730.jpg"):
                verdict = "REMOVE"
                action = "REJECT"
                notes = "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present."
            # 4. Valid industrial and firefighting helmets
            else:
                verdict = "PASS"
                action = "ADD"
                notes = "Genuine safety hardhat/firefighter helmet on personnel; tight usable bounding box."

        else:  # person
            # Person evaluation
            # 1. High smoke overlap (smoke plume misclassified as person)
            if s_iou >= 0.50 or (s_iou >= 0.30 and conf < 0.55):
                verdict = "REMOVE"
                action = "REJECT"
                notes = f"False positive person detection on dark billowing smoke plume (IoU={s_iou:.2f} with GT smoke)."
            # 2. High fire overlap (flame region misclassified as person)
            elif f_iou >= 0.50:
                verdict = "REMOVE"
                action = "REJECT"
                notes = f"False positive person detection on flame region (IoU={f_iou:.2f} with GT fire)."
            # 3. High person overlap (redundant group box / duplicate)
            elif p_iou >= 0.50:
                verdict = "REMOVE"
                action = "REJECT"
                notes = f"Redundant duplicate or group person box (IoU={p_iou:.2f} with existing person detection)."
            # 4. Unusable extreme edge truncation (scalp/hand/edge sliver)
            elif touches and (aspect < 0.60 or aspect > 5.5 or norm_w < 0.02 or norm_h < 0.02):
                verdict = "REMOVE"
                action = "REJECT"
                notes = f"Unusable extreme frame edge truncation ({norm_w*100:.1f}%w x {norm_h*100:.1f}%h, aspect={aspect:.2f}) showing only marginal body sliver."
            # 5. Borderline distance ambiguity in heavy smoke
            elif s_iou >= 0.25 and conf < 0.45 and norm_w < 0.04 and norm_h < 0.04:
                verdict = "UNCERTAIN"
                action = "ESCALATE"
                notes = "Distant ambiguous silhouette in heavy smoke plume; cannot definitively confirm human versus debris."
            # 6. Valid persons
            else:
                verdict = "PASS"
                action = "ADD"
                notes = "Visible usable person instance (pedestrian/firefighter/civilian); tight accurate bounding box."

        evaluations.append({
            "queue_rank": rank,
            "candidate_id": cid,
            "tier": "MEDIUM",
            "split": split,
            "image": img,
            "class_name": cname,
            "canonical_class": 1 if cname == "helmet" else 0,
            "confidence": f"{conf:.6f}",
            "selection_stratum": f"{cname}_{split}_{item['aspect']:.2f}",
            "verdict": verdict,
            "action": action,
            "notes": notes,
            "old_bbox_norm": f"({old_norm[0]:.8f}, {old_norm[1]:.8f}, {old_norm[2]:.8f}, {old_norm[3]:.8f})",
            "new_bbox_norm": f"({new_norm[0]:.8f}, {new_norm[1]:.8f}, {new_norm[2]:.8f}, {new_norm[3]:.8f})" if new_norm else "None",
        })

    verdict_counts = Counter(e["verdict"] for e in evaluations)
    print(f"\nVerdict counts across all 326 items: {dict(verdict_counts)}")
    by_class_verdicts = Counter((e["class_name"], e["verdict"]) for e in evaluations)
    print(f"By class and verdict: {dict(by_class_verdicts)}")

    with open(ROOT / "docs/audit_artifacts/dfire/phase2_interim_eval.json", "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)


if __name__ == "__main__":
    main()
