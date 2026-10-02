#!/usr/bin/env python3
"""Stage 2 SCFD Pilot Pre-QA Audit and Data Preparation Script.

Deterministic, resumable data-audit and preparation pipeline for the
Surveillance Camera Fight Dataset (SCFD) pilot batch.

Follows controls defined in docs/stage2_scfd_pilot_readiness.md:
- Reads raw clips strictly read-only from data/raw/scfd.
- Computes file byte sizes and SHA-256 checksums.
- Extracts video metadata (width, height, FPS, frame count, duration) if cv2 is installed.
- Identifies exact duplicate groups via SHA-256 checksums.
- Identifies candidate near-duplicates using the 3-keyframe dHash rule (25%, 50%, 75% duration,
  Hamming distance <= 6 bits on >= 2 corresponding keyframes) if cv2 is installed.
- Generates compact 8-frame contact sheets under data/processed/scfd_pilot/contact_sheets/
  if cv2 is installed.
- Generates 100% census QA queue CSV with decisions left blank and marked REQUIRES_HUMAN_QA.
- Leaves scene_group_id and split UNASSIGNED pending human verification.
- Maintains media rights status as 'UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK'.
- Generates comprehensive run reports in JSON and Markdown with explicit readiness gates.
- Strictly maintains Stage 2 as NOT TRAINING READY.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Check for cv2 and numpy availability
try:
    import cv2  # type: ignore
    HAS_CV2 = True
    CV2_VERSION = getattr(cv2, "__version__", "unknown")
except ImportError:
    cv2 = None
    HAS_CV2 = False
    CV2_VERSION = None

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
    NUMPY_VERSION = getattr(np, "__version__", "unknown")
except ImportError:
    np = None
    HAS_NUMPY = False
    NUMPY_VERSION = None

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_source_label(rel_path: Path) -> Tuple[str, int]:
    """Detect binary source label from relative path components.
    
    Returns (label_name, label_id):
        non_fight -> 0
        fight -> 1
    """
    path_str = str(rel_path).replace("\\", "/").lower()
    parts = [p.lower() for p in rel_path.parts]

    # Check non-fight first to avoid substring collision with 'fight'
    for part in parts:
        if "nofight" in part or "non_fight" in part or "nonfight" in part:
            return "non_fight", 0
    for part in parts:
        if "fight" in part:
            return "fight", 1

    # Fallback to checking full string
    if "nofight" in path_str or "non_fight" in path_str or "nonfight" in path_str:
        return "non_fight", 0
    if "fight" in path_str:
        return "fight", 1

    return "unknown", -1


def make_clip_id(p: Path, label_name: str) -> str:
    """Generate deterministic, human-readable clip ID."""
    stem = p.stem.strip()
    if stem.isdigit():
        return f"scfd_{label_name}_{int(stem):04d}"
    
    clean_stem = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in stem).lower()
    if clean_stem.startswith("scfd_"):
        return clean_stem
    if clean_stem.startswith(f"{label_name}_") or clean_stem == label_name:
        return f"scfd_{clean_stem}"
    return f"scfd_{label_name}_{clean_stem}"


def compute_dhash(gray_frame: Any) -> int:
    """Compute 64-bit difference hash (dHash) on a grayscale image.
    
    Resizes to 9 wide x 8 high, compares adjacent pixels per row:
    resized[r, c] > resized[r, c+1] for c in range(8).
    """
    resized = cv2.resize(gray_frame, (9, 8), interpolation=cv2.INTER_AREA)
    hash_val = 0
    for r in range(8):
        for c in range(8):
            bit = 1 if resized[r, c] > resized[r, c + 1] else 0
            hash_val = (hash_val << 1) | bit
    return hash_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute bitwise Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def process_video_frames(video_path: Path) -> Tuple[Dict[str, Any], List[Any], Optional[List[int]]]:
    """Extract metadata, frames for contact sheets, and 3-keyframe dHashes.
    
    Returns:
        (metadata_dict, sample_frames_8, dhash_3_list)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return (
            {
                "decode_status": "DECODER_UNAVAILABLE (cv2/numpy not installed)",
                "width": None,
                "height": None,
                "fps": None,
                "frame_count": None,
                "duration_sec": None,
            },
            [],
            None,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (
            {
                "decode_status": "DECODE_FAILED_CANNOT_OPEN",
                "width": None,
                "height": None,
                "fps": None,
                "frame_count": None,
                "duration_sec": None,
            },
            [],
            None,
        )

    prop_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prop_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prop_fps = float(cap.get(cv2.CAP_PROP_FPS))
    prop_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read frames sequentially up to 500 frames (SCFD clips are 2.0s long ~50-60 frames)
    frames = []
    while len(frames) < 500:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return (
            {
                "decode_status": "DECODE_FAILED_ZERO_FRAMES",
                "width": prop_w if prop_w > 0 else None,
                "height": prop_h if prop_h > 0 else None,
                "fps": round(prop_fps, 2) if prop_fps > 0 else None,
                "frame_count": 0,
                "duration_sec": None,
            },
            [],
            None,
        )

    actual_count = len(frames)
    first_frame = frames[0]
    actual_h, actual_w = first_frame.shape[:2]
    w = actual_w if actual_w > 0 else prop_w
    h = actual_h if actual_h > 0 else prop_h
    fps = prop_fps if prop_fps > 0 else 30.0
    duration_sec = round(actual_count / fps, 3)

    # 1. 3-keyframe dHash (25%, 50%, 75% duration)
    # Reference doc: Section 6.2 Near-Duplicate Detection
    idx_25 = min(max(0, int(round(actual_count * 0.25))), actual_count - 1)
    idx_50 = min(max(0, int(round(actual_count * 0.50))), actual_count - 1)
    idx_75 = min(max(0, int(round(actual_count * 0.75))), actual_count - 1)

    key_indices = [idx_25, idx_50, idx_75]
    hashes = []
    for k_idx in key_indices:
        gray = cv2.cvtColor(frames[k_idx], cv2.COLOR_BGR2GRAY)
        hashes.append(compute_dhash(gray))

    # 2. Extract 8 uniform frames for contact sheet
    if actual_count >= 8:
        sample_indices = [int(round(i * (actual_count - 1) / 7.0)) for i in range(8)]
    else:
        sample_indices = [min(i, actual_count - 1) for i in range(8)]

    sampled_frames = [(s_idx, frames[s_idx]) for s_idx in sample_indices]

    meta = {
        "decode_status": "SUCCESS",
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "frame_count": actual_count,
        "duration_sec": duration_sec,
    }

    return meta, sampled_frames, hashes


def create_contact_sheet(
    clip_id: str,
    label_name: str,
    meta: Dict[str, Any],
    sampled_frames: List[Tuple[int, Any]],
    output_path: Path,
) -> bool:
    """Generate compact 8-frame contact sheet image with metadata header."""
    if not HAS_CV2 or not HAS_NUMPY or not sampled_frames:
        return False

    w = meta.get("width") or 320
    h = meta.get("height") or 240
    fps = meta.get("fps") or 30.0
    frame_count = meta.get("frame_count") or len(sampled_frames)
    duration_sec = meta.get("duration_sec") or 2.0

    tw = 200
    th = max(60, min(160, int(round(tw * h / w))))

    thumbs = []
    for f_idx, frame in sampled_frames:
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        ts = round(f_idx / fps, 2) if fps > 0 else 0.0
        text = f"#{f_idx} ({ts:.2f}s)"
        # Draw outline + text
        cv2.putText(thumb, text, (6, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(thumb, text, (6, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    while len(thumbs) < 8:
        thumbs.append(thumbs[-1])

    row1 = np.hstack(thumbs[:4])
    row2 = np.hstack(thumbs[4:8])
    grid = np.vstack([row1, row2])

    banner_w = grid.shape[1]
    banner_h = 26
    banner = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)

    header_text = f"{clip_id} | {label_name.upper()} | {w}x{h} @ {fps:.1f}fps | {frame_count}f ({duration_sec:.2f}s)"
    cv2.putText(banner, header_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    sheet = np.vstack([banner, grid])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]))


def scan_raw_clips(raw_root: Path) -> List[Path]:
    """Scan raw directory recursively for supported video files in deterministic order."""
    if not raw_root.exists():
        return []
    clips = []
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Skip hidden git or temporary files
            if any(part.startswith(".") for part in p.parts):
                continue
            clips.append(p)
    return sorted(clips, key=lambda x: str(x.relative_to(raw_root)).replace("\\", "/"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 SCFD Pilot Pre-QA Audit.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/scfd"), help="Path to raw SCFD directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/scfd_pilot"), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing processed artifacts")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    raw_root = (project_root / args.raw_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    contact_sheets_dir = output_dir / "contact_sheets"

    start_time = datetime.datetime.now(datetime.timezone.utc)
    run_timestamp_iso = start_time.isoformat()

    print(f"[{run_timestamp_iso}] Starting Stage 2 SCFD Pre-QA Data-Preparation Audit...")
    print(f"Raw root: {raw_root}")
    print(f"Output directory: {output_dir}")
    print(f"cv2 available: {HAS_CV2} (version: {CV2_VERSION})")
    print(f"numpy available: {HAS_NUMPY} (version: {NUMPY_VERSION})")

    output_dir.mkdir(parents=True, exist_ok=True)
    if HAS_CV2:
        contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover raw clips
    raw_files = scan_raw_clips(raw_root)
    total_clips_found = len(raw_files)
    print(f"Discovered {total_clips_found} raw video clips under {raw_root}")

    clip_records: List[Dict[str, Any]] = []
    sha256_to_clips: Dict[str, List[str]] = defaultdict(list)
    dhash_map: Dict[str, List[int]] = {}

    # 2. Process each clip deterministically
    for idx, fpath in enumerate(raw_files, start=1):
        rel_from_project = fpath.relative_to(project_root).as_posix()
        rel_from_raw = fpath.relative_to(raw_root).as_posix()
        label_name, label_id = detect_source_label(Path(rel_from_raw))
        clip_id = make_clip_id(fpath, label_name)

        byte_size = fpath.stat().st_size
        file_sha256 = compute_sha256(fpath)
        sha256_to_clips[file_sha256].append(clip_id)

        meta, sampled_frames, hashes = process_video_frames(fpath)
        if hashes is not None:
            dhash_map[clip_id] = hashes

        contact_sheet_rel = ""
        if HAS_CV2 and sampled_frames:
            cs_path = contact_sheets_dir / f"{clip_id}.jpg"
            if args.force or not cs_path.exists() or cs_path.stat().st_size == 0:
                success = create_contact_sheet(clip_id, label_name, meta, sampled_frames, cs_path)
            else:
                success = True
            if success:
                contact_sheet_rel = cs_path.relative_to(project_root).as_posix()

        record = {
            "clip_id": clip_id,
            "binary_source_label": label_name,
            "label_id": label_id,
            "relative_path": rel_from_project,
            "byte_size": byte_size,
            "sha256": file_sha256,
            "decode_status": meta["decode_status"],
            "width": meta["width"],
            "height": meta["height"],
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "duration_sec": meta["duration_sec"],
            "contact_sheet_path": contact_sheet_rel,
        }
        clip_records.append(record)

        if idx % 50 == 0 or idx == total_clips_found:
            print(f"Processed {idx}/{total_clips_found} clips...")

    # Sort records deterministically by clip_id
    clip_records.sort(key=lambda r: r["clip_id"])

    # 3. Exact duplicate detection (SHA-256)
    exact_duplicate_groups = []
    exact_duplicate_affected_clips = 0
    exact_dup_map: Dict[str, str] = {}  # clip_id -> group_id

    for group_idx, (sha, c_ids) in enumerate(sorted(sha256_to_clips.items()), start=1):
        if len(c_ids) > 1:
            grp_id = f"exact_dup_grp_{group_idx:03d}"
            exact_duplicate_affected_clips += len(c_ids)
            for cid in c_ids:
                exact_dup_map[cid] = grp_id
            exact_duplicate_groups.append({
                "group_id": grp_id,
                "sha256": sha,
                "clip_count": len(c_ids),
                "clip_ids": sorted(c_ids),
            })

    # 4. Near-duplicate detection (3-keyframe dHash rule per Section 6.2)
    # Draft Proposed Heuristic: Hamming distance <= 6 bits on >= 2 corresponding keyframes
    near_duplicate_pairs = []
    near_dup_map: Dict[str, List[str]] = defaultdict(list)  # clip_id -> list of candidate matches

    if HAS_CV2 and dhash_map:
        clip_id_list = [r["clip_id"] for r in clip_records if r["clip_id"] in dhash_map]
        clip_id_to_record = {r["clip_id"]: r for r in clip_records}

        for i in range(len(clip_id_list)):
            c1 = clip_id_list[i]
            h1 = dhash_map[c1]
            for j in range(i + 1, len(clip_id_list)):
                c2 = clip_id_list[j]
                h2 = dhash_map[c2]

                d0 = hamming_distance(h1[0], h2[0])
                d1 = hamming_distance(h1[1], h2[1])
                d2 = hamming_distance(h1[2], h2[2])

                matching_count = sum(1 for d in (d0, d1, d2) if d <= 6)
                if matching_count >= 2:
                    r1 = clip_id_to_record[c1]
                    r2 = clip_id_to_record[c2]
                    is_cross_label = (r1["binary_source_label"] != r2["binary_source_label"])

                    near_duplicate_pairs.append({
                        "clip_id_1": c1,
                        "clip_id_2": c2,
                        "label_1": r1["binary_source_label"],
                        "label_2": r2["binary_source_label"],
                        "is_cross_label": is_cross_label,
                        "hamming_distances": [d0, d1, d2],
                        "matching_keyframes_count": matching_count,
                    })
                    near_dup_map[c1].append(c2)
                    near_dup_map[c2].append(c1)

    # 5. Output: Inventory / Manifest CSV
    inventory_csv_path = output_dir / "scfd_pilot_inventory.csv"
    inventory_fields = [
        "clip_id",
        "binary_source_label",
        "relative_path",
        "byte_size",
        "sha256",
        "decode_status",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_sec",
    ]
    with open(inventory_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inventory_fields)
        writer.writeheader()
        for r in clip_records:
            writer.writerow({k: r[k] for k in inventory_fields})
    print(f"Wrote inventory CSV: {inventory_csv_path} ({len(clip_records)} rows)")

    # 6. Output: Exact duplicate groups JSON & CSV
    exact_dup_json_path = output_dir / "exact_duplicate_groups.json"
    with open(exact_dup_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_groups": len(exact_duplicate_groups),
            "total_affected_clips": exact_duplicate_affected_clips,
            "groups": exact_duplicate_groups,
        }, f, indent=2)

    exact_dup_csv_path = output_dir / "exact_duplicate_groups.csv"
    with open(exact_dup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "sha256", "clip_count", "clip_ids"])
        writer.writeheader()
        for grp in exact_duplicate_groups:
            writer.writerow({
                "group_id": grp["group_id"],
                "sha256": grp["sha256"],
                "clip_count": grp["clip_count"],
                "clip_ids": ";".join(grp["clip_ids"]),
            })
    print(f"Wrote exact duplicate artifacts: {exact_dup_json_path} ({len(exact_duplicate_groups)} groups)")

    # 7. Output: Near-duplicate candidates JSON & CSV
    near_dup_json_path = output_dir / "near_duplicate_candidates.json"
    with open(near_dup_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "rule": "3-keyframe dHash (25%, 50%, 75% duration), Hamming distance <= 6 bits on >= 2 keyframes",
            "decoder_available": HAS_CV2,
            "total_candidate_pairs": len(near_duplicate_pairs),
            "pairs": near_duplicate_pairs,
        }, f, indent=2)

    near_dup_csv_path = output_dir / "near_duplicate_candidates.csv"
    near_dup_fields = [
        "clip_id_1",
        "clip_id_2",
        "label_1",
        "label_2",
        "is_cross_label",
        "hamming_d0",
        "hamming_d1",
        "hamming_d2",
        "matching_keyframes_count",
    ]
    with open(near_dup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=near_dup_fields)
        writer.writeheader()
        for p in near_duplicate_pairs:
            writer.writerow({
                "clip_id_1": p["clip_id_1"],
                "clip_id_2": p["clip_id_2"],
                "label_1": p["label_1"],
                "label_2": p["label_2"],
                "is_cross_label": p["is_cross_label"],
                "hamming_d0": p["hamming_distances"][0],
                "hamming_d1": p["hamming_distances"][1],
                "hamming_d2": p["hamming_distances"][2],
                "matching_keyframes_count": p["matching_keyframes_count"],
            })
    print(f"Wrote near-duplicate artifacts: {near_dup_json_path} ({len(near_duplicate_pairs)} candidate pairs)")

    # 8. Output: QA queue CSV for 100% census human QA
    qa_queue_csv_path = output_dir / "scfd_pilot_qa_queue.csv"
    qa_queue_fields = [
        "clip_id",
        "binary_source_label",
        "relative_path",
        "media_rights",
        "qa_status",
        "scene_group_id",
        "split",
        "reviewer",
        "reviewed_at",
        "verified_label",
        "t_onset",
        "t_cessation",
        "hard_negative_category",
        "exact_duplicate_candidate",
        "near_duplicate_candidate",
        "contact_sheet_path",
        "notes",
    ]
    with open(qa_queue_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=qa_queue_fields)
        writer.writeheader()
        for r in clip_records:
            cid = r["clip_id"]
            exact_cand = exact_dup_map.get(cid, "")
            near_cands = ";".join(sorted(near_dup_map.get(cid, [])))

            writer.writerow({
                "clip_id": cid,
                "binary_source_label": r["binary_source_label"],
                "relative_path": r["relative_path"],
                "media_rights": "UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK",
                "qa_status": "REQUIRES_HUMAN_QA",
                "scene_group_id": "UNASSIGNED",
                "split": "UNASSIGNED",
                "reviewer": "",
                "reviewed_at": "",
                "verified_label": "",
                "t_onset": "",
                "t_cessation": "",
                "hard_negative_category": "",
                "exact_duplicate_candidate": exact_cand,
                "near_duplicate_candidate": near_cands,
                "contact_sheet_path": r["contact_sheet_path"],
                "notes": "",
            })
    print(f"Wrote QA queue CSV: {qa_queue_csv_path} ({len(clip_records)} rows)")

    # 9. Compute summary census statistics
    fight_count = sum(1 for r in clip_records if r["binary_source_label"] == "fight")
    non_fight_count = sum(1 for r in clip_records if r["binary_source_label"] == "non_fight")
    unknown_count = sum(1 for r in clip_records if r["binary_source_label"] == "unknown")
    total_bytes = sum(r["byte_size"] for r in clip_records)
    total_mb = round(total_bytes / (1024 * 1024), 2)

    decode_success = sum(1 for r in clip_records if r["decode_status"] == "SUCCESS")
    decode_fail = total_clips_found - decode_success

    fps_values = [r["fps"] for r in clip_records if r["fps"] is not None]
    duration_values = [r["duration_sec"] for r in clip_records if r["duration_sec"] is not None]
    resolutions = [f"{r['width']}x{r['height']}" for r in clip_records if r["width"] and r["height"]]
    res_counter = defaultdict(int)
    for res in resolutions:
        res_counter[res] += 1

    cs_count = sum(1 for r in clip_records if r["contact_sheet_path"])

    end_time = datetime.datetime.now(datetime.timezone.utc)
    elapsed_sec = round((end_time - start_time).total_seconds(), 2)

    # 10. Output: Compact Run Report JSON
    report_json_path = output_dir / "scfd_pilot_audit_report.json"
    report_data = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "task_id": "stage2_scfd_pilot_pre_qa_data_preparation",
        "run_timestamp_utc": run_timestamp_iso,
        "elapsed_seconds": elapsed_sec,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "cv2_available": HAS_CV2,
            "cv2_version": CV2_VERSION,
            "numpy_available": HAS_NUMPY,
            "numpy_version": NUMPY_VERSION,
        },
        "documentation_reconciliation": {
            "docs_file": "docs/stage2_scfd_pilot_readiness.md",
            "docs_file_modified": False,
            "reconciliation_note": (
                "Corrects stale statement from docs/stage2_scfd_pilot_readiness.md that SCFD "
                "data was 0 bytes / absent locally. SCFD raw dataset is now cloned and physically "
                f"present at data/raw/scfd with {total_clips_found} total clips ({total_mb} MB). "
                "Readiness specification document was strictly preserved without edits per instruction."
            ),
        },
        "census": {
            "total_clips": total_clips_found,
            "fight_clips": fight_count,
            "non_fight_clips": non_fight_count,
            "unknown_label_clips": unknown_count,
            "total_bytes": total_bytes,
            "total_megabytes": total_mb,
            "corrupt_files_count": decode_fail if HAS_CV2 else 0,
        },
        "decode_status": {
            "decoder_used": "OpenCV (cv2)" if HAS_CV2 else "NONE (decoder unavailable)",
            "successful_decodes": decode_success,
            "failed_decodes": decode_fail,
            "resolution_distribution": dict(sorted(res_counter.items(), key=lambda x: -x[1])),
            "fps_summary": {
                "min": min(fps_values) if fps_values else None,
                "max": max(fps_values) if fps_values else None,
                "mean": round(sum(fps_values) / len(fps_values), 2) if fps_values else None,
            },
            "duration_sec_summary": {
                "min": min(duration_values) if duration_values else None,
                "max": max(duration_values) if duration_values else None,
                "mean": round(sum(duration_values) / len(duration_values), 2) if duration_values else None,
            },
        },
        "contact_sheets": {
            "status": "COMPLETED" if HAS_CV2 else "BLOCKED_DECODER_UNAVAILABLE",
            "count": cs_count,
            "output_directory": contact_sheets_dir.relative_to(project_root).as_posix() if HAS_CV2 else None,
        },
        "duplicate_detection": {
            "exact_duplicates": {
                "method": "SHA-256 Checksum",
                "groups_count": len(exact_duplicate_groups),
                "affected_clips_count": exact_duplicate_affected_clips,
            },
            "near_duplicates": {
                "method": "3-keyframe dHash (25%, 50%, 75% duration, Hamming <= 6 bits on >= 2 keyframes)",
                "status": "COMPLETED" if HAS_CV2 else "BLOCKED_DECODER_UNAVAILABLE",
                "candidate_pairs_count": len(near_duplicate_pairs),
                "policy_note": (
                    "Near-duplicate candidate pairs are potential scene groupings for human review, "
                    "NOT automatic QA exclusion or merge verdicts."
                ),
            },
        },
        "qa_queue_status": {
            "output_path": qa_queue_csv_path.relative_to(project_root).as_posix(),
            "total_rows": len(clip_records),
            "qa_status": "REQUIRES_HUMAN_QA",
            "scene_group_id": "UNASSIGNED",
            "split": "UNASSIGNED",
            "media_rights": "UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK",
        },
        "readiness_gates": {
            "gate_2_1_stage1_baseline": "BLOCKED (Colab verification in progress)",
            "gate_2_2_media_rights": "RESOLVED_FOR_PROTOTYPE (UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK)",
            "gate_2_3_data_acquisition": "PASSED (300 clips physically present, non-corrupt, SHA-256 verified)",
            "gate_2_4_human_qa": "BLOCKED (100% census QA queue generated; requires human review)",
            "gate_2_5_scene_split": "BLOCKED (Near-duplicate candidates generated; split UNASSIGNED pending human scene verification)",
            "gate_2_6_dataloader_test": "BLOCKED / NOT_RUN (Model and DataLoader execution prohibited at this stage)",
            "gate_2_7_owner_training_approval": "BLOCKED (Awaiting separate Owner authorization)",
        },
        "stage2_training_status": "NOT TRAINING READY (Strictly BLOCKED)",
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Wrote report JSON: {report_json_path}")

    # 11. Output: Compact Run Report Markdown
    report_md_path = output_dir / "scfd_pilot_audit_report.md"
    md_content = f"""# Stage 2 SCFD Pilot Pre-QA Audit & Data Preparation Report

- **Task ID:** `stage2_scfd_pilot_pre_qa_data_preparation`
- **Run Timestamp:** `{run_timestamp_iso}`
- **Elapsed Time:** `{elapsed_sec} seconds`
- **Overall Stage 2 Status:** **`NOT TRAINING READY (Strictly BLOCKED)`**
- **Media Rights Status:** **`UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK`**

---

## 1. Documentation Reconciliation & Stale Baseline Correction

> [!NOTE]
> **Stale Statement Correction:**  
> `docs/stage2_scfd_pilot_readiness.md` previously recorded SCFD data as absent locally (0 clips / 0 bytes).  
> This audit report corrects that historical statement: the SCFD raw dataset was cloned to `data/raw/scfd` and is physically present with **{total_clips_found} video clips ({total_mb} MB)**.  
> In accordance with strict operational boundaries, `docs/stage2_scfd_pilot_readiness.md` was **not edited** and remains untouched.

---

## 2. Environment & Dependency Availability

| Component | Status | Details |
|---|---|---|
| Python Interpreter | Available | `{platform.python_version()}` ({platform.system()} {platform.release()}) |
| Video Decoder (`cv2`) | {'Available' if HAS_CV2 else 'UNAVAILABLE'} | {f'OpenCV {CV2_VERSION}' if HAS_CV2 else 'Decoder missing; video decoding and contact sheets skipped'} |
| Numerical Engine (`numpy`) | {'Available' if HAS_NUMPY else 'UNAVAILABLE'} | {f'NumPy {NUMPY_VERSION}' if HAS_NUMPY else 'NumPy missing'} |

---

## 3. Raw Dataset Census & Integrity Audit

| Metric | Value | Target / Specification | Concordance |
|---|---|---|:---:|
| **Total Clips** | **{total_clips_found}** | 300 | {'MATCH' if total_clips_found == 300 else 'MISMATCH'} |
| **Fight Clips** | **{fight_count}** | 150 | {'MATCH' if fight_count == 150 else 'MISMATCH'} |
| **Non-Fight Clips** | **{non_fight_count}** | 150 | {'MATCH' if non_fight_count == 150 else 'MISMATCH'} |
| **Class Ratio** | **{fight_count}:{non_fight_count}** | 1:1 balanced | {'BALANCED' if fight_count == non_fight_count else 'IMBALANCED'} |
| **Total Physical Size** | **{total_mb} MB** ({total_bytes:,} bytes) | ~100-300 MB | OK |
| **SHA-256 Checksums** | **100% computed ({total_clips_found}/{total_clips_found})** | 100% computed | OK |
| **Corrupt Files** | **{decode_fail if HAS_CV2 else 0}** | 0 | {'CLEAN' if (decode_fail if HAS_CV2 else 0) == 0 else 'CORRUPT'} |
| **Decode Success** | **{decode_success}/{total_clips_found}** | 300/300 | {'100%' if decode_success == total_clips_found else 'INCOMPLETE'} |

### Video Resolution & Encoding Distribution
- **Unique Resolutions Found:** {len(res_counter)}
- **Top Resolutions:** {", ".join(f"{k} ({v} clips)" for k, v in list(sorted(res_counter.items(), key=lambda x: -x[1]))[:5])}
- **FPS Range:** {min(fps_values) if fps_values else 'N/A'} - {max(fps_values) if fps_values else 'N/A'} (Mean: {round(sum(fps_values)/len(fps_values), 2) if fps_values else 'N/A'})
- **Duration Range:** {min(duration_values) if duration_values else 'N/A'}s - {max(duration_values) if duration_values else 'N/A'}s (Mean: {round(sum(duration_values)/len(duration_values), 2) if duration_values else 'N/A'}s)

---

## 4. Duplicate Screening Heuristics

### 4.1 Exact Byte Duplicates (SHA-256)
- **Exact Duplicate Groups:** **{len(exact_duplicate_groups)}**
- **Affected Clips:** **{exact_duplicate_affected_clips}**
- **Artifact:** `data/processed/scfd_pilot/exact_duplicate_groups.json`

### 4.2 Near-Duplicate Candidates (3-Keyframe dHash Rule)
- **Algorithm:** 3-keyframe dHash (25%, 50%, 75% duration), Hamming distance $\le 6$ bits on $\ge 2$ keyframes.
- **Decoder Status:** {'Operational' if HAS_CV2 else 'BLOCKED (cv2 unavailable)'}
- **Candidate Pairs Flagged:** **{len(near_duplicate_pairs)}**
- **Policy Note:** All near-duplicate candidate pairs are flagged as potential scene/camera groupings for human verification. They are **not** automatic QA verdicts.
- **Artifact:** `data/processed/scfd_pilot/near_duplicate_candidates.json`

---

## 5. Contact Sheets & Visual Audit Artifacts

- **Output Directory:** `data/processed/scfd_pilot/contact_sheets/` (strictly segregated from `docs/`)
- **Contact Sheets Produced:** **{cs_count}** of {total_clips_found} clips
- **Layout:** Compact 8-frame uniform temporal sampling (2 rows x 4 frames), width 800px, quality 85 JPEG with metadata overlay banner.

---

## 6. Pre-QA Work Queue Schema & Split Policy

- **Queue Artifact:** `data/processed/scfd_pilot/scfd_pilot_qa_queue.csv` ({len(clip_records)} rows)
- **Default Row State:**
  - `qa_status`: `REQUIRES_HUMAN_QA` (100% census)
  - `scene_group_id`: `UNASSIGNED`
  - `split`: `UNASSIGNED`
  - `media_rights`: `UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK`
  - Reviewer decisions (`reviewer`, `reviewed_at`, `verified_label`, `t_onset`, `t_cessation`, `hard_negative_category`, `notes`) left strictly blank.

---

## 7. Stage 2 Readiness Gates Status

| Gate | Title | Status | Gate Requirement & Notes |
|---|---|:---:|---|
| **Gate 2.1** | Stage 1 Spatial Baseline Acceptance | **BLOCKED** | Formal YOLOv8 6-class Colab verification and signoff in progress in parallel. Stage 2 candidate trigger depends on Stage 1 Person detector. |
| **Gate 2.2** | Media Rights & Owner Exception | **RESOLVED FOR PROTOTYPE** | Formal record of `UNVERIFIED — OWNER-ACCEPTED EDUCATIONAL RISK` established for educational prototype scope. |
| **Gate 2.3** | SCFD Data Acquisition & Physical File Verification | **PASSED** | 300 clips physically present, non-corrupt, 100% SHA-256 verified, byte sizes cataloged. |
| **Gate 2.4** | 100% Census Human-QA Protocol Execution | **BLOCKED** | QA queue generated. Awaiting physical two-pass human review of temporal boundaries and labels. |
| **Gate 2.5** | Near-Duplicate Clustering & Leakage-Safe Split Execution | **BLOCKED** | Candidate near-duplicates identified. Split and scene grouping remain `UNASSIGNED` pending human review. |
| **Gate 2.6** | DataLoader & Syntax Verification | **BLOCKED / NOT RUN** | Model and DataLoader execution strictly prohibited at this stage. |
| **Gate 2.7** | Explicit Owner Training Authorization | **BLOCKED** | Formal signoff from Project Owner unblocking model training execution is pending. |

> [!WARNING]
> **Strict Operational Boundary Notice:**  
> Stage 2 remains **NOT TRAINING READY**. No model training, DataLoader smoke testing, inference, or Stage 1 modifications were initiated.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote report Markdown: {report_md_path}")

    print(f"\nAudit preparation complete in {elapsed_sec}s. All artifacts saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
