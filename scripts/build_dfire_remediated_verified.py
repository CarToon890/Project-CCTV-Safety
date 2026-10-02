"""Build trusted Phase-2 verified dataset (data/processed/dfire_remediated_verified).

Rebuilds deterministically from immutable data/processed/dfire_corrected:
1. Reuses images via safe NTFS hardlinks from dfire_corrected.
2. Ingests base labels from dfire_corrected (16 person, 0 helmet, 14,683 fire, 11,854 smoke).
3. Applies trusted Phase 1 visual-QA outcomes (1,554 person + 8 helmet = 1,562 additions, 1 FIX, 7 removals).
4. Applies trusted Phase 2 visual-QA outcomes (161 person + 109 helmet = 270 additions, 1 FIX, 54 removals, 2 uncertain excluded).
5. Excludes 100% of Phase 3 candidates (all 2,547 heuristic additions excluded).
6. Produces exact reconciled counts:
   - person: 1,731
   - helmet: 117
   - vest: 0
   - fall: 0
   - fire: 14,683
   - smoke: 11,854
   - total: 28,385
7. Generates comprehensive trusted remediation_manifest.csv for all 5,579 candidates.
8. Writes updated metadata/{train,val,test}.csv, data.yaml, and dataset_manifest.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/dfire/data"
CORRECTED_DIR = ROOT / "data/processed/dfire_corrected"
TARGET_DIR = ROOT / "data/processed/dfire_remediated_verified"
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"

# Evidence artifacts
P1_REMEDIATED_MANIFEST = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
P1_ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_qa_label_adjustments.csv"
P1_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_remediated_high_qa_queue.csv"

P2_ADJUSTMENTS_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv"
P2_QUEUE_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_medium_qa_queue.csv"
P2_OWNER_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_owner_queue.csv"
P2_FEATURES_JSON = ROOT / "docs/audit_artifacts/dfire/phase2_features.json"

SPLITS = ("train", "val", "test")
EXPECTED_TOTAL_IMAGES = 21527
KNOWN_PRE_HASH = "ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da"

EXPECTED_COUNTS = {
    0: 1731,   # person
    1: 117,    # helmet
    2: 0,      # vest
    3: 0,      # fall
    4: 14683,  # fire
    5: 11854,  # smoke
}
EXPECTED_TOTAL_BOXES = 28385


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


def link_or_copy(src: Path, dst: Path):
    """Creates a hardlink to avoid duplicating images; falls back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def parse_bbox_tuple(box_str: str) -> tuple[float, float, float, float] | None:
    """Parses '(xc, yc, w, h)' into a float tuple."""
    if not box_str or box_str.strip().lower() in ("none", "unchanged", ""):
        return None
    cleaned = box_str.strip().strip("()").replace(",", " ")
    parts = [float(p) for p in cleaned.split() if p]
    if len(parts) == 4:
        return (parts[0], parts[1], parts[2], parts[3])
    return None


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


def main():
    t0 = time.perf_counter()
    print("=" * 80)
    print("BUILDING TRUSTED PHASE-2 VERIFIED D-FIRE DATASET")
    print("Target:", TARGET_DIR)
    print("=" * 80)

    # 1. Source Immutability Verifications
    print("\n[Step 1] Verifying immutable sources...")
    if not CORRECTED_DIR.exists():
        print("BLOCKED: source directory data/processed/dfire_corrected missing!")
        sys.exit(1)
    if not RAW_DIR.exists():
        print("BLOCKED: raw directory data/raw/dfire/data missing!")
        sys.exit(1)

    corr_hash, corr_count = hash_directory_labels(CORRECTED_DIR / "labels")
    print(f"  dfire_corrected label count: {corr_count}, SHA-256: {corr_hash}")
    if corr_hash != KNOWN_PRE_HASH:
        print(f"BLOCKED: dfire_corrected label hash mismatch! Expected {KNOWN_PRE_HASH}, got {corr_hash}")
        sys.exit(1)
    if corr_count != EXPECTED_TOTAL_IMAGES:
        print(f"BLOCKED: dfire_corrected expected {EXPECTED_TOTAL_IMAGES} label files, got {corr_count}")
        sys.exit(1)

    raw_images_count = sum(len(list((RAW_DIR / s / "images").iterdir())) for s in SPLITS if (RAW_DIR / s / "images").exists())
    print(f"  data/raw image count: {raw_images_count}")
    if raw_images_count != EXPECTED_TOTAL_IMAGES:
        print(f"BLOCKED: data/raw expected {EXPECTED_TOTAL_IMAGES} images, got {raw_images_count}")
        sys.exit(1)

    # 2. Ingest Evidence Sources
    print("\n[Step 2] Ingesting candidate scan and Phase 1/Phase 2 evidence artifacts...")
    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        all_candidates = list(csv.DictReader(f))
    print(f"  Total candidates in candidates.csv: {len(all_candidates)}")
    cands_map = {c["candidate_id"]: c for c in all_candidates}

    # Load Phase 1 evidence
    with open(P1_REMEDIATED_MANIFEST, "r", encoding="utf-8") as f:
        p1_manifest_rows = list(csv.DictReader(f))[:1569]
    with open(P1_ADJUSTMENTS_CSV, "r", encoding="utf-8") as f:
        p1_adj_rows = list(csv.DictReader(f))
    p1_adj_map = {r["candidate_id"]: r for r in p1_adj_rows}

    # Load Phase 2 evidence
    with open(P2_ADJUSTMENTS_CSV, "r", encoding="utf-8") as f:
        p2_adj_rows = list(csv.DictReader(f))
    p2_adj_map = {r["candidate_id"]: r for r in p2_adj_rows}

    # 3. Determine Candidate Resolutions and Labels to Apply
    print("\n[Step 3] Resolving Phase 1, Phase 2, and Unreviewed populations...")
    labels_to_add_by_file = defaultdict(list)
    remediation_manifest_rows = []

    p1_added_count = Counter()
    p2_added_count = Counter()
    p1_rejected_count = Counter()
    p2_rejected_count = Counter()
    p2_escalated_count = Counter()
    unreviewed_count = Counter()

    processed_cids = set()

    # Process Phase 1 candidates (1,569)
    for p1_row in p1_manifest_rows:
        cid = p1_row["candidate_id"]
        processed_cids.add(cid)
        cand_meta = cands_map[cid]
        cname = cand_meta["class_name"]
        ccls = int(cand_meta["canonical_class"])
        split = cand_meta["split"]
        img = cand_meta["image"]
        stem = Path(img).stem
        conf = float(cand_meta["confidence"])
        orig_box_str = f"({float(cand_meta['x1']):.2f}, {float(cand_meta['y1']):.2f}, {float(cand_meta['x2']):.2f}, {float(cand_meta['y2']):.2f})"

        # Check if item was in the 30-item follow-up QA queue
        if cid in p1_adj_map:
            adj = p1_adj_map[cid]
            action = adj["action"]  # PASS, FIX, REMOVE
            reason = adj["reason_and_evidence"]
            verdict = action

            if action == "REMOVE":
                status = "SKIPPED"
                final_action = "REJECT"
                final_bbox_str = "None"
                res_path = "NONE"
                p1_rejected_count[cname] += 1
            elif action == "FIX":
                status = "FIXED"
                final_action = "ADJUST_BBOX"
                final_bbox_str = adj["new_bbox_norm"]
                res_path = f"labels/{split}/{stem}.txt"
                p1_added_count[cname] += 1
                new_box_tuple = parse_bbox_tuple(final_bbox_str)
                labels_to_add_by_file[(split, stem)].append((ccls, new_box_tuple, cid, "Phase 1 (FIX)"))
            else:  # PASS
                status = "ADDED"
                final_action = "ADD"
                final_bbox_str = p1_row["normalized_bbox"]
                res_path = f"labels/{split}/{stem}.txt"
                p1_added_count[cname] += 1
                norm_tuple = parse_bbox_tuple(final_bbox_str)
                labels_to_add_by_file[(split, stem)].append((ccls, norm_tuple, cid, "Phase 1 (PASS)"))

            evidence_src = "docs/audit_artifacts/dfire/dfire_qa_label_adjustments.csv"
        else:
            # Baseline accepted HIGH candidate from Phase 1
            status = "ADDED"
            verdict = "PASS"
            final_action = "ADD"
            reason = "High-confidence detection verified under Phase 1 baseline acceptance."
            final_bbox_str = p1_row["normalized_bbox"]
            res_path = f"labels/{split}/{stem}.txt"
            p1_added_count[cname] += 1
            norm_tuple = parse_bbox_tuple(final_bbox_str)
            labels_to_add_by_file[(split, stem)].append((ccls, norm_tuple, cid, "Phase 1 (BASELINE)"))
            evidence_src = "data/processed/dfire_remediated/remediation_manifest.csv"

        remediation_manifest_rows.append({
            "candidate_id": cid,
            "phase": "Phase 1 (HIGH)",
            "class_name": cname,
            "canonical_class": ccls,
            "split": split,
            "image": img,
            "confidence": f"{conf:.6f}",
            "visual_verdict": verdict,
            "evidence_source_artifact": evidence_src,
            "action": final_action,
            "status": status,
            "original_bbox": orig_box_str,
            "final_bbox": final_bbox_str,
            "resulting_label_path": res_path,
            "notes": reason,
        })

    # Process Phase 2 candidates (326)
    for adj in p2_adj_rows:
        cid = adj["candidate_id"]
        processed_cids.add(cid)
        cand_meta = cands_map[cid]
        cname = cand_meta["class_name"]
        ccls = int(cand_meta["canonical_class"])
        split = cand_meta["split"]
        img = cand_meta["image"]
        stem = Path(img).stem
        conf = float(cand_meta["confidence"])
        orig_box_str = f"({float(cand_meta['x1']):.2f}, {float(cand_meta['y1']):.2f}, {float(cand_meta['x2']):.2f}, {float(cand_meta['y2']):.2f})"

        act = adj["action"]  # ADD, ADJUST_BBOX, REJECT, ESCALATE
        reason = adj["reason"]

        if act == "ADD":
            status = "ADDED"
            verdict = "PASS"
            final_action = "ADD"
            final_bbox_str = adj["old_bbox_norm"]
            res_path = f"labels/{split}/{stem}.txt"
            p2_added_count[cname] += 1
            norm_tuple = parse_bbox_tuple(final_bbox_str)
            labels_to_add_by_file[(split, stem)].append((ccls, norm_tuple, cid, "Phase 2 (PASS)"))
        elif act == "ADJUST_BBOX":
            status = "FIXED"
            verdict = "FIX"
            final_action = "ADJUST_BBOX"
            final_bbox_str = adj["new_bbox_norm"]
            res_path = f"labels/{split}/{stem}.txt"
            p2_added_count[cname] += 1
            new_box_tuple = parse_bbox_tuple(final_bbox_str)
            labels_to_add_by_file[(split, stem)].append((ccls, new_box_tuple, cid, "Phase 2 (FIX)"))
        elif act == "REJECT":
            status = "SKIPPED"
            verdict = "REMOVE"
            final_action = "REJECT"
            final_bbox_str = "None"
            res_path = "NONE"
            p2_rejected_count[cname] += 1
        elif act == "ESCALATE":
            status = "SKIPPED"
            verdict = "UNCERTAIN"
            final_action = "ESCALATE"
            final_bbox_str = "None"
            res_path = "NONE"
            p2_escalated_count[cname] += 1
        else:
            raise ValueError(f"Unexpected Phase 2 action: {act}")

        remediation_manifest_rows.append({
            "candidate_id": cid,
            "phase": "Phase 2 (MEDIUM)",
            "class_name": cname,
            "canonical_class": ccls,
            "split": split,
            "image": img,
            "confidence": f"{conf:.6f}",
            "visual_verdict": verdict,
            "evidence_source_artifact": "docs/audit_artifacts/dfire/dfire_phase2_label_adjustments.csv",
            "action": final_action,
            "status": status,
            "original_bbox": orig_box_str,
            "final_bbox": final_bbox_str,
            "resulting_label_path": res_path,
            "notes": reason,
        })

    # Process remaining candidates (all Phase 3 / unreviewed) -> 100% EXCLUDED
    remaining_cands = [c for c in all_candidates if c["candidate_id"] not in processed_cids]
    print(f"  Phase 1 processed: {len(p1_manifest_rows)}")
    print(f"  Phase 2 processed: {len(p2_adj_rows)}")
    print(f"  Unreviewed remaining: {len(remaining_cands)}")

    for cand_meta in remaining_cands:
        cid = cand_meta["candidate_id"]
        cname = cand_meta["class_name"]
        ccls = int(cand_meta["canonical_class"])
        split = cand_meta["split"]
        img = cand_meta["image"]
        conf = float(cand_meta["confidence"])
        orig_box_str = f"({float(cand_meta['x1']):.2f}, {float(cand_meta['y1']):.2f}, {float(cand_meta['x2']):.2f}, {float(cand_meta['y2']):.2f})"
        unreviewed_count[cname] += 1

        remediation_manifest_rows.append({
            "candidate_id": cid,
            "phase": f"Unreviewed ({cand_meta['tier']})",
            "class_name": cname,
            "canonical_class": ccls,
            "split": split,
            "image": img,
            "confidence": f"{conf:.6f}",
            "visual_verdict": "UNREVIEWED",
            "evidence_source_artifact": "NONE (Heuristics rejected; genuine visual QA pending)",
            "action": "EXCLUDED_PENDING_GENUINE_VISUAL_QA",
            "status": "EXCLUDED",
            "original_bbox": orig_box_str,
            "final_bbox": "None",
            "resulting_label_path": "NONE",
            "notes": "Candidate excluded from trusted Phase-2 verified dataset; pending future genuine visual QA.",
        })

    print("\nSummary of candidate resolutions:")
    print(f"  Phase 1 added: {dict(p1_added_count)} (Total: {sum(p1_added_count.values())})")
    print(f"  Phase 1 rejected: {dict(p1_rejected_count)} (Total: {sum(p1_rejected_count.values())})")
    print(f"  Phase 2 added: {dict(p2_added_count)} (Total: {sum(p2_added_count.values())})")
    print(f"  Phase 2 rejected: {dict(p2_rejected_count)} (Total: {sum(p2_rejected_count.values())})")
    print(f"  Phase 2 escalated (uncertain): {dict(p2_escalated_count)} (Total: {sum(p2_escalated_count.values())})")
    print(f"  Unreviewed excluded: {dict(unreviewed_count)} (Total: {sum(unreviewed_count.values())})")
    print(f"  Total candidate manifest rows: {len(remediation_manifest_rows)}")

    # 4. Create directory structure
    print(f"\n[Step 4] Creating directory structure under {TARGET_DIR}...")
    for s in SPLITS:
        (TARGET_DIR / "images" / s).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / s).mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "metadata").mkdir(parents=True, exist_ok=True)

    # 5. Safe NTFS Hardlinking for Images
    print("\n[Step 5] Creating safe NTFS hardlinks for images...")
    hardlink_count = 0
    copy_count = 0
    for s in SPLITS:
        src_dir = CORRECTED_DIR / "images" / s
        dst_dir = TARGET_DIR / "images" / s
        for img_p in src_dir.iterdir():
            if img_p.is_file():
                dst_p = dst_dir / img_p.name
                if not dst_p.exists():
                    try:
                        os.link(img_p, dst_p)
                        hardlink_count += 1
                    except Exception:
                        shutil.copy2(img_p, dst_p)
                        copy_count += 1
                else:
                    hardlink_count += 1

    print(f"  Images linked/copied: {hardlink_count} hardlinks, {copy_count} copies (Total: {hardlink_count + copy_count})")
    if (hardlink_count + copy_count) != EXPECTED_TOTAL_IMAGES:
        print(f"BLOCKED: Expected {EXPECTED_TOTAL_IMAGES} images, got {hardlink_count + copy_count}")
        sys.exit(1)

    # 6. Rebuild Label Files (Independent Copies from dfire_corrected + Trusted Phase 1 & 2 additions)
    print("\n[Step 6] Rebuilding independent label files...")
    rebuilt_label_count = 0
    verified_class_counts = Counter()
    verified_split_counts = {s: Counter() for s in SPLITS}

    for s in SPLITS:
        corr_lbl_dir = CORRECTED_DIR / "labels" / s
        dst_lbl_dir = TARGET_DIR / "labels" / s

        for corr_lbl in corr_lbl_dir.glob("*.txt"):
            stem = corr_lbl.stem
            dst_lbl = dst_lbl_dir / corr_lbl.name

            # Read base lines from dfire_corrected
            base_lines = [l.strip() for l in corr_lbl.read_text(encoding="utf-8").splitlines() if l.strip()]

            # Add additions for this file
            file_additions = labels_to_add_by_file.get((s, stem), [])
            addition_lines = []
            for ccls, norm_box, cid, src_info in file_additions:
                line_str = f"{ccls} {norm_box[0]:.8f} {norm_box[1]:.8f} {norm_box[2]:.8f} {norm_box[3]:.8f}"
                addition_lines.append(line_str)

            all_lines = base_lines + addition_lines
            for line in all_lines:
                parts = line.split()
                c = int(parts[0])
                verified_class_counts[c] += 1
                verified_split_counts[s][c] += 1

            # Write independent label file
            content = "\n".join(all_lines) + ("\n" if all_lines else "")
            dst_lbl.write_text(content, encoding="utf-8")
            rebuilt_label_count += 1

    print(f"  Rebuilt label files: {rebuilt_label_count}")
    print(f"  Verified class counts: {dict(verified_class_counts)}")
    total_rebuilt_boxes = sum(verified_class_counts.values())
    print(f"  Total bounding boxes: {total_rebuilt_boxes}")
    for s in SPLITS:
        print(f"    Split {s}: {dict(verified_split_counts[s])} (Total: {sum(verified_split_counts[s].values())})")

    # 7. Exact Reconciled Counts Assertion
    print("\n[Step 7] Checking exact count reconciliation...")
    mismatch = False
    for c, exp_c in EXPECTED_COUNTS.items():
        act_c = verified_class_counts.get(c, 0)
        if act_c != exp_c:
            print(f"  MISMATCH on class {c}: expected {exp_c}, found {act_c}!")
            mismatch = True
        else:
            print(f"  Class {c}: {act_c} == {exp_c} (MATCH)")

    if total_rebuilt_boxes != EXPECTED_TOTAL_BOXES:
        print(f"  MISMATCH on total boxes: expected {EXPECTED_TOTAL_BOXES}, found {total_rebuilt_boxes}!")
        mismatch = True
    else:
        print(f"  Total boxes: {total_rebuilt_boxes} == {EXPECTED_TOTAL_BOXES} (MATCH)")

    if mismatch:
        print("\nBLOCKED: Final counts do not match expected Phase-2 counts! Stopping as required.")
        sys.exit(1)

    # 8. Write Metadata CSVs
    print("\n[Step 8] Writing metadata CSVs...")
    for s in SPLITS:
        corr_meta_p = CORRECTED_DIR / "metadata" / f"{s}.csv"
        dst_meta_p = TARGET_DIR / "metadata" / f"{s}.csv"

        with open(corr_meta_p, "r", encoding="utf-8") as f:
            corr_meta_rows = list(csv.DictReader(f))

        dst_rows = []
        for r in corr_meta_rows:
            img = r["image"]
            stem = Path(img).stem
            lbl_p = TARGET_DIR / "labels" / s / f"{stem}.txt"
            lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]

            classes = [int(l.split()[0]) for l in lines]
            person_c = sum(1 for c in classes if c == 0)
            helmet_c = sum(1 for c in classes if c == 1)
            vest_c = sum(1 for c in classes if c == 2)
            fall_c = sum(1 for c in classes if c == 3)
            fire_c = sum(1 for c in classes if c == 4)
            smoke_c = sum(1 for c in classes if c == 5)
            tot_b = len(classes)

            dst_rows.append({
                "image": img,
                "source_split": r.get("source_split", s),
                "group_id": r["group_id"],
                "category": r["category"],
                "person_boxes": person_c,
                "helmet_boxes": helmet_c,
                "fire_boxes": fire_c,
                "smoke_boxes": smoke_c,
                "total_boxes": tot_b,
            })

        with open(dst_meta_p, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "image", "source_split", "group_id", "category",
                "person_boxes", "helmet_boxes", "fire_boxes", "smoke_boxes", "total_boxes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dst_rows)

    print("  Saved metadata for train, val, test.")

    # 9. Write Remediation Manifest CSV
    print("\n[Step 9] Writing remediation manifest CSV...")
    manifest_dst = TARGET_DIR / "remediation_manifest.csv"
    manifest_fields = [
        "candidate_id", "phase", "class_name", "canonical_class", "split", "image",
        "confidence", "visual_verdict", "evidence_source_artifact", "action", "status",
        "original_bbox", "final_bbox", "resulting_label_path", "notes"
    ]
    with open(manifest_dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(remediation_manifest_rows)
    print(f"  Saved remediation manifest to {manifest_dst} ({len(remediation_manifest_rows)} rows).")

    # 10. Write data.yaml and dataset_manifest.json
    print("\n[Step 10] Writing data.yaml and dataset_manifest.json...")
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: person
  1: helmet
  2: vest
  3: fall
  4: fire
  5: smoke
"""
    (TARGET_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    dataset_manifest = {
        "dataset_name": "dfire_remediated_verified",
        "version": "2.1.0-verified-phase2",
        "date_rebuilt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": (
            "Trusted Phase-2 verified snapshot of D-Fire dataset with actual visual-QA verified labels only "
            "(Phase 1 HIGH + Phase 2 MEDIUM census/sample). All Phase 3 heuristic additions completely excluded."
        ),
        "immutable_base": "data/processed/dfire_corrected",
        "base_labels_sha256": KNOWN_PRE_HASH,
        "total_images": EXPECTED_TOTAL_IMAGES,
        "total_boxes": EXPECTED_TOTAL_BOXES,
        "canonical_classes": dict(verified_class_counts),
        "canonical_classes_per_split": {s: dict(verified_split_counts[s]) for s in SPLITS},
        "phase1_verified_candidates": {
            "census_population": 1569,
            "pass_retained": p1_added_count["person"] + p1_added_count["helmet"] - 1,
            "fix_adjusted": 1,
            "remove_rejected": sum(p1_rejected_count.values()),
            "total_applied": sum(p1_added_count.values()),
        },
        "phase2_verified_candidates": {
            "reviewed_population": 326,
            "pass_added": 269,
            "fix_adjusted": 1,
            "remove_rejected": sum(p2_rejected_count.values()),
            "uncertain_escalated": sum(p2_escalated_count.values()),
            "total_applied": sum(p2_added_count.values()),
        },
        "phase3_candidates_status": {
            "status": "EXCLUDED_PENDING_GENUINE_VISUAL_QA",
            "unreviewed_population": len(remaining_cands),
            "applied_boxes": 0,
            "notes": "100% of Phase 3 candidates excluded. Awaiting future genuine visual QA.",
        },
    }

    with open(TARGET_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\nRebuild complete in {elapsed:.2f} seconds.")
    print("SUCCESS: Trusted Phase-2 dataset created at:", TARGET_DIR)


if __name__ == "__main__":
    main()
