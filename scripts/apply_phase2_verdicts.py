"""Apply Phase 2 Medium Visual QA verdicts and updates to dfire_remediated.

Scope:
- 326 candidates total:
  - 126 helmets (100% census)
  - 200 persons (stratified sample)
- Verdicts:
  - 269 PASS (108 helmets, 161 persons) -> ADD
  - 1 FIX (1 helmet: CAND_002015) -> ADJUST_BBOX
  - 54 REMOVE (15 helmets, 39 persons) -> REJECT (not added)
  - 2 UNCERTAIN (2 helmets: CAND_003159, CAND_003490) -> ESCALATE (not added, routed to owner queue)

Outputs / Updates:
1. docs/audit_artifacts/dfire/dfire_medium_qa_queue.csv (all 326 items)
2. docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv (all 326 items)
3. docs/audit_artifacts/dfire/dfire_phase2_owner_queue.csv (2 UNCERTAIN items)
4. data/processed/dfire_remediated/labels/{train,val,test}/*.txt (append 270 labels)
5. data/processed/dfire_remediated/remediation_manifest.csv (append 326 records)
6. data/processed/dfire_remediated/metadata/{train,val,test}.csv (update box counts)
7. data/processed/dfire_remediated/dataset_manifest.json (record Phase 2 completion)
8. docs/audit_artifacts/dfire/dfire_medium_visual_qa_report.md (concise report)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
FEATURES_JSON = ROOT / "docs/audit_artifacts/dfire/phase2_features.json"
SAMPLING_SPEC_JSON = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_sampling_spec.json"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"

QA_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_medium_qa_queue.csv"
ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv"
OWNER_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_owner_queue.csv"
REPORT_MD = ROOT / "docs/audit_artifacts/dfire/dfire_medium_visual_qa_report.md"

REVIEWER = "ANTIGRAVITY_VISUAL_QA"
REVIEWED_AT = "2026-09-26T03:00:00+07:00"

# Explicit sets of REMOVE, UNCERTAIN, and FIX candidates
REMOVE_HELMET_REASONS = {
    "CAND_001571": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_001838": "False positive helmet detection on vehicle light/body reflection; no helmet present.",
    "CAND_001856": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_002303": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_002720": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_002840": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_002923": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_003081": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_003102": "False positive helmet detection on fabric fire-resistant balaclava/hood; no rigid safety helmet.",
    "CAND_003127": "False positive helmet detection on fabric fire-resistant balaclava/hood; no rigid safety helmet.",
    "CAND_003139": "False positive helmet detection on fabric fire-resistant balaclava/hood; no rigid safety helmet.",
    "CAND_003289": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_003310": "False positive helmet detection on civilian baseball cap/hair; no safety hardhat/helmet present.",
    "CAND_003397": "Oversized false positive detection on non-helmet background structure.",
    "CAND_003445": "False positive detection on large curved motorcycle vehicle structure; no safety helmet.",
}

UNCERTAIN_HELMET_REASONS = {
    "CAND_003159": "Heavily blurred distant head in smoke haze (18x14 px); ambiguous whether white safety helmet or civilian cap.",
    "CAND_003490": "Heavily pixelated head in smoke haze (16x13 px); ambiguous whether safety headgear or bare head/fabric cap.",
}

FIX_HELMET_DATA = {
    "CAND_002015": {
        "notes": "Firefighter helmet brim in extreme closeup; tightened top edge coordinate to image boundary y=0.0.",
        "new_box_norm": (0.78203076, 0.08885496, 0.36562303, 0.17770992)
    }
}

REMOVE_PERSON_REASONS = {
    # Extreme edge truncations
    "CAND_001611": "Unusable extreme frame edge truncation (aspect=7.88) showing only marginal body sliver.",
    "CAND_001733": "Unusable extreme frame edge truncation (aspect=0.55) showing only marginal body sliver.",
    "CAND_001810": "Unusable extreme frame edge truncation (aspect=6.06) showing only marginal body sliver.",
    "CAND_002131": "Unusable extreme frame edge truncation (aspect=5.94) showing only marginal body sliver.",
    "CAND_002249": "Unusable extreme frame edge truncation (aspect=0.21) showing only marginal body sliver.",
    "CAND_002541": "Unusable extreme frame edge truncation (aspect=0.59) showing only marginal body sliver.",
    "CAND_002582": "Unusable extreme frame edge truncation (aspect=0.53) showing only marginal body sliver.",
    "CAND_002603": "Unusable extreme frame edge truncation (aspect=9.33) showing only marginal body sliver.",
    "CAND_002783": "Unusable extreme frame edge truncation (aspect=5.57) showing only marginal body sliver.",
    "CAND_002929": "Unusable extreme frame edge truncation (aspect=2.00, w=0.019) showing marginal crowd edge sliver.",
    "CAND_002944": "Unusable extreme frame edge truncation (aspect=5.69) showing only marginal body sliver.",
    "CAND_002982": "False positive person detection on dark billowing smoke plume (IoU=0.82 with GT smoke).",
    "CAND_003292": "Unusable extreme frame edge truncation (aspect=0.51) showing only marginal body sliver.",
    "CAND_003331": "Unusable extreme frame edge truncation (aspect=10.22) showing only marginal arm/back sliver.",
    "CAND_003499": "Unusable extreme frame edge truncation (aspect=0.33) showing only marginal body sliver.",
    # Smoke plume false positives
    "CAND_002362": "False positive person detection on dark billowing smoke plume on boat (IoU=0.94 with GT smoke).",
    "CAND_002670": "False positive person detection on dark billowing smoke plume (IoU=0.71 with GT smoke).",
    "CAND_002731": "False positive person detection on dark billowing smoke plume (IoU=0.84 with GT smoke).",
    "CAND_002871": "False positive person detection on dark billowing smoke plume (IoU=0.49 with GT smoke).",
    "CAND_003005": "False positive person detection on dark billowing smoke plume (IoU=0.50 with GT smoke).",
    "CAND_003037": "False positive person detection on dark billowing smoke plume (IoU=0.86 with GT smoke).",
    "CAND_003095": "False positive person detection on dark billowing smoke plume (IoU=0.64 with GT smoke).",
    "CAND_003106": "False positive person detection on dark billowing smoke plume (IoU=0.50 with GT smoke).",
    "CAND_003425": "False positive person detection on dark billowing smoke plume (IoU=0.31 with GT smoke).",
    # Flame false positives
    "CAND_002058": "False positive person detection on flame region (IoU=0.74 with GT fire).",
    "CAND_002827": "False positive person detection on flame region (IoU=0.84 with GT fire).",
    "CAND_003363": "False positive person detection on torch/burner flame region (IoU=0.70 with GT fire).",
    "CAND_003451": "False positive person detection on flame region (IoU=0.60 with GT fire).",
    "CAND_003481": "False positive person detection on flame region (IoU=0.79 with GT fire).",
    "CAND_003533": "False positive person detection on flame region (IoU=0.79 with GT fire).",
    # Visually verified additional false positives / duplicates / non-human objects
    "CAND_001572": "False positive person detection on stone sculpture/statue; non-human monument object.",
    "CAND_002325": "False positive person detection on volcano eruption smoke plume; landscape scene.",
    "CAND_002445": "Misaligned hybrid box covering mountain smoke and cutting off person head/body.",
    "CAND_002655": "Redundant multi-person group box spanning across two partial individuals.",
    "CAND_003085": "Unusable extreme frame edge truncation (aspect=0.33) showing only face/scalp sliver at boundary.",
    "CAND_003216": "False positive person detection on ground rock/rubble debris in snow scene.",
    "CAND_003335": "Unusable vertical edge strip (aspect=4.09) showing disconnected hands with red nail polish.",
    "CAND_003477": "Massive scene-level false detection covering entire street scene, house, and smoke.",
    "CAND_003488": "False positive person detection on distant horizon tree silhouette at sunset.",
}


def main():
    print("=" * 80)
    print("PHASE 2 D-FIRE MEDIUM-TIER QA AND REMEDIATION APPLICATION")
    print("=" * 80)

    with open(FEATURES_JSON, "r", encoding="utf-8") as f:
        features = json.load(f)

    with open(SAMPLING_SPEC_JSON, "r", encoding="utf-8") as f:
        sampling_spec = json.load(f)

    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        cands_raw = {r["candidate_id"]: r for r in csv.DictReader(f)}

    print(f"Loaded {len(features)} feature records.")

    # 1. Classify all 326 items
    qa_records = []
    adjustments_records = []
    owner_queue_records = []
    labels_to_add = []  # (split, image, class_id, norm_box, cid)

    pass_count = 0
    fix_count = 0
    remove_count = 0
    uncertain_count = 0

    for it in features:
        rank = it["rank"]
        cid = it["cid"]
        cname = it["class_name"]
        split = it["split"]
        img = it["image"]
        conf = it["conf"]
        ccls = 1 if cname == "helmet" else 0
        old_norm = tuple(it["box_norm"])
        cand_meta = cands_raw[cid]
        stratum = cand_meta.get("stratum", f"{cname}_{split}_{it['aspect']:.2f}")

        new_norm = None
        if cname == "helmet":
            if cid in REMOVE_HELMET_REASONS:
                verdict = "REMOVE"
                action = "REJECT"
                notes = REMOVE_HELMET_REASONS[cid]
                remove_count += 1
            elif cid in UNCERTAIN_HELMET_REASONS:
                verdict = "UNCERTAIN"
                action = "ESCALATE"
                notes = UNCERTAIN_HELMET_REASONS[cid]
                uncertain_count += 1
            elif cid in FIX_HELMET_DATA:
                verdict = "FIX"
                action = "ADJUST_BBOX"
                notes = FIX_HELMET_DATA[cid]["notes"]
                new_norm = FIX_HELMET_DATA[cid]["new_box_norm"]
                fix_count += 1
                labels_to_add.append((split, img, ccls, new_norm, cid))
            else:
                verdict = "PASS"
                action = "ADD"
                notes = "Genuine safety hardhat/firefighter helmet on personnel; tight usable bounding box."
                pass_count += 1
                labels_to_add.append((split, img, ccls, old_norm, cid))
        else:  # person
            if cid in REMOVE_PERSON_REASONS:
                verdict = "REMOVE"
                action = "REJECT"
                notes = REMOVE_PERSON_REASONS[cid]
                remove_count += 1
            else:
                verdict = "PASS"
                action = "ADD"
                notes = "Visible usable person instance (pedestrian/firefighter/civilian); tight accurate bounding box."
                pass_count += 1
                labels_to_add.append((split, img, ccls, old_norm, cid))

        old_norm_str = f"({old_norm[0]:.8f}, {old_norm[1]:.8f}, {old_norm[2]:.8f}, {old_norm[3]:.8f})"
        new_norm_str = f"({new_norm[0]:.8f}, {new_norm[1]:.8f}, {new_norm[2]:.8f}, {new_norm[3]:.8f})" if new_norm else "None"

        qa_records.append({
            "queue_rank": rank,
            "candidate_id": cid,
            "tier": "MEDIUM",
            "split": split,
            "image": img,
            "class_name": cname,
            "canonical_class": ccls,
            "confidence": f"{conf:.6f}",
            "x1": f"{it['box_pix'][0]:.2f}",
            "y1": f"{it['box_pix'][1]:.2f}",
            "x2": f"{it['box_pix'][2]:.2f}",
            "y2": f"{it['box_pix'][3]:.2f}",
            "aspect_ratio": f"{it['aspect']:.4f}",
            "selection_stratum": stratum,
            "verdict": verdict,
            "notes": notes,
            "reviewer": REVIEWER,
            "reviewed_at": REVIEWED_AT,
            "old_bbox_norm": old_norm_str,
            "new_bbox_norm": new_norm_str,
        })

        adjustments_records.append({
            "queue_rank": rank,
            "candidate_id": cid,
            "action": action,
            "canonical_class": ccls,
            "class_name": cname,
            "split": split,
            "image": img,
            "old_bbox_norm": old_norm_str,
            "new_bbox_norm": new_norm_str,
            "reason": notes,
        })

        if verdict == "UNCERTAIN":
            owner_queue_records.append({
                "queue_rank": rank,
                "candidate_id": cid,
                "tier": "MEDIUM",
                "split": split,
                "image": img,
                "class_name": cname,
                "canonical_class": ccls,
                "confidence": f"{conf:.6f}",
                "old_bbox_norm": old_norm_str,
                "ambiguity_reason": notes,
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            })

    print(f"\nVerdict Summary across 326 items:")
    print(f"  PASS: {pass_count} (Helmets: 108, Persons: 161)")
    print(f"  FIX: {fix_count} (Helmets: 1, Persons: 0)")
    print(f"  REMOVE: {remove_count} (Helmets: 15, Persons: 39)")
    print(f"  UNCERTAIN: {uncertain_count} (Helmets: 2, Persons: 0)")
    print(f"  Total Labels to Add: {len(labels_to_add)} (269 PASS + 1 FIX)")

    # 2. Write QA Queue CSV
    with open(QA_QUEUE_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(qa_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_records)
    print(f"\nSaved QA queue: {QA_QUEUE_CSV} ({len(qa_records)} rows)")

    # 3. Write Adjustments CSV
    with open(ADJUSTMENTS_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(adjustments_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(adjustments_records)
    print(f"Saved label adjustments: {ADJUSTMENTS_CSV} ({len(adjustments_records)} rows)")

    # 4. Write Owner Queue CSV
    with open(OWNER_QUEUE_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(owner_queue_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(owner_queue_records)
    print(f"Saved minimal owner queue: {OWNER_QUEUE_CSV} ({len(owner_queue_records)} rows)")

    # 5. Apply Labels to data/processed/dfire_remediated
    print("\n[Applying label additions to dfire_remediated]")
    by_file_additions = defaultdict(list)
    for split, img, ccls, box, cid in labels_to_add:
        by_file_additions[(split, img)].append((ccls, box, cid))

    files_modified = 0
    boxes_added_count = 0
    for (split, img), additions in by_file_additions.items():
        stem = Path(img).stem
        lbl_p = REMEDIATED_DIR / "labels" / split / f"{stem}.txt"
        existing_content = lbl_p.read_text(encoding="utf-8").strip()
        lines = existing_content.splitlines() if existing_content else []

        for ccls, box, cid in additions:
            line = f"{ccls} {box[0]:.8f} {box[1]:.8f} {box[2]:.8f} {box[3]:.8f}"
            lines.append(line)
            boxes_added_count += 1

        lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files_modified += 1

    print(f"Modified {files_modified} label files, appended {boxes_added_count} candidate bboxes.")

    # 6. Update metadata files
    print("\n[Updating metadata files]")
    for split in ("train", "val", "test"):
        meta_p = REMEDIATED_DIR / "metadata" / f"{split}.csv"
        with open(meta_p, "r", encoding="utf-8") as f:
            m_rows = list(csv.DictReader(f))

        new_m_rows = []
        for r in m_rows:
            img = r["image"]
            stem = Path(img).stem
            lbl_p = REMEDIATED_DIR / "labels" / split / f"{stem}.txt"
            lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]

            classes = [int(l.split()[0]) for l in lines]
            person_c = sum(1 for c in classes if c == 0)
            helmet_c = sum(1 for c in classes if c == 1)
            fire_c = sum(1 for c in classes if c == 4)
            smoke_c = sum(1 for c in classes if c == 5)
            tot_b = len(classes)

            r["person_boxes"] = person_c
            r["helmet_boxes"] = helmet_c
            r["fire_boxes"] = fire_c
            r["smoke_boxes"] = smoke_c
            r["total_boxes"] = tot_b
            new_m_rows.append(r)

        with open(meta_p, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "image", "source_split", "group_id", "category",
                "person_boxes", "helmet_boxes", "fire_boxes", "smoke_boxes", "total_boxes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_m_rows)
    print("Updated metadata counts for all splits.")

    # 7. Synchronize remediation_manifest.csv (Append Phase 2 rows without erasing Phase 1 history)
    print("\n[Synchronizing remediation_manifest.csv]")
    manifest_p = REMEDIATED_DIR / "remediation_manifest.csv"
    with open(manifest_p, "r", encoding="utf-8") as f:
        existing_manifest = list(csv.DictReader(f))

    print(f"Existing Phase 1 manifest records: {len(existing_manifest)}")

    phase2_manifest_records = []
    for it, adj in zip(features, adjustments_records):
        cid = it["cid"]
        action = adj["action"]
        split = it["split"]
        img = it["image"]
        stem = Path(img).stem
        cname = it["class_name"]
        conf = it["conf"]
        old_norm_str = adj["old_bbox_norm"]
        new_norm_str = adj["new_bbox_norm"]
        norm_to_record = new_norm_str if new_norm_str != "None" else old_norm_str
        orig_pix = f"({it['box_pix'][0]:.2f}, {it['box_pix'][1]:.2f}, {it['box_pix'][2]:.2f}, {it['box_pix'][3]:.2f})"

        if action in ("ADD", "FIX"):
            status = "ADDED"
            reason = f"PHASE_2_ACCEPTED_{action}"
            res_path = f"labels/{split}/{stem}.txt"
        elif action == "REJECT":
            status = "SKIPPED"
            reason = f"PHASE_2_REJECTED: {adj['reason']}"
            res_path = "NONE"
        else:  # ESCALATE
            status = "SKIPPED"
            reason = f"PHASE_2_UNCERTAIN_ESCALATED: {adj['reason']}"
            res_path = "NONE"

        phase2_manifest_records.append({
            "candidate_id": cid,
            "status": status,
            "reason": reason,
            "confidence": f"{conf:.6f}",
            "source_split": split,
            "source_image": img,
            "original_bbox": orig_pix,
            "normalized_bbox": norm_to_record,
            "resulting_label_path": res_path,
        })

    full_manifest = existing_manifest + phase2_manifest_records
    with open(manifest_p, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidate_id", "status", "reason", "confidence",
            "source_split", "source_image", "original_bbox", "normalized_bbox", "resulting_label_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_manifest)

    print(f"Total manifest records written: {len(full_manifest)} (1569 Phase 1 + 326 Phase 2)")

    # 8. Update dataset_manifest.json
    print("\n[Updating dataset_manifest.json]")
    dataset_manifest_p = REMEDIATED_DIR / "dataset_manifest.json"
    with open(dataset_manifest_p, "r", encoding="utf-8") as f:
        ds_man = json.load(f)

    ds_man["version"] = "2.1.0-medium-remediated"
    ds_man["description"] = (
        "D-Fire dataset with HIGH-tier remediation (1,562 net accepted) plus Phase 2 MEDIUM-tier "
        "visual QA remediation (100% census of 126 helmets, stratified 200 persons; 270 added, 54 removed, 2 uncertain)."
    )

    # Count final classes across all labels
    final_classes_per_split = {"train": Counter(), "val": Counter(), "test": Counter()}
    for s in ("train", "val", "test"):
        lbl_dir = REMEDIATED_DIR / "labels" / s
        for p in lbl_dir.glob("*.txt"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cls_id = int(line.split()[0])
                    final_classes_per_split[s][str(cls_id)] += 1

    total_canonical_classes = Counter()
    for s in ("train", "val", "test"):
        for cls_id, cnt in final_classes_per_split[s].items():
            total_canonical_classes[cls_id] += cnt

    ds_man["canonical_classes_after"] = dict(total_canonical_classes)
    ds_man["canonical_classes_per_split_after"] = {
        s: dict(final_classes_per_split[s]) for s in ("train", "val", "test")
    }

    ds_man["phase_2_medium_qa"] = {
        "helmets_census_reviewed": 126,
        "helmets_pass": 108,
        "helmets_fix": 1,
        "helmets_remove": 15,
        "helmets_uncertain": 2,
        "persons_stratified_reviewed": 200,
        "persons_pass": 161,
        "persons_fix": 0,
        "persons_remove": 39,
        "persons_uncertain": 0,
        "total_phase2_reviewed": 326,
        "net_added_to_dataset": 270,
        "rejected_false_positives": 54,
        "escalated_to_owner": 2,
        "sampling_seed": 42,
        "reviewer": REVIEWER,
        "completed_at": REVIEWED_AT,
        "owner_queue_file": "docs/audit_artifacts/dfire/dfire_phase2_owner_queue.csv",
        "qa_queue_file": "docs/audit_artifacts/dfire/dfire_medium_qa_queue.csv",
        "label_adjustments_file": "docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv"
    }

    with open(dataset_manifest_p, "w", encoding="utf-8") as f:
        json.dump(ds_man, f, indent=2)

    print("Updated dataset_manifest.json.")
    print(f"Final Class Counts: {dict(total_canonical_classes)}")
    print(f"Total Bounding Boxes: {sum(total_canonical_classes.values())}")


if __name__ == "__main__":
    main()
