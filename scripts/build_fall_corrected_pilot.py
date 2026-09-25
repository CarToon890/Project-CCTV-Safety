#!/usr/bin/env python
"""Build corrected Fall Detection pilot dataset.

Parses 8 CVAT XML annotations, maps state tracks to Stage 1 canonical classes (0:person, 3:fall),
enforces conservative tail truncation, applies deterministic temporal decimation informed by dHash,
splits by actor/session groups, generates YOLO dataset structure + data.yaml, renders QA overlays/contact sheets,
and generates audit CSV artifacts.

Standard library, OpenCV, and Pillow only.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

CANONICAL_CLASSES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "fall",
    4: "fire",
    5: "smoke",
}

PILOT_CLIPS = [
    "FD0035", "FD0044", "FD0049", "FD0024",
    "FD0027", "FD0051", "FD0052", "FD0054",
]


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash)."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | (pixels[start + col] > pixels[start + col + 1])
    return value


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def parse_cvat_xml(xml_path: Path) -> dict[str, Any]:
    """Parse CVAT 1.1 XML annotation file for tracks and bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta = root.find("meta")
    job = meta.find("job") if meta is not None else None
    xml_size = int(job.findtext("size")) if job is not None and job.findtext("size") else None
    orig_size = meta.find("original_size") if meta is not None else None
    ow = int(orig_size.findtext("width")) if orig_size is not None else 1920
    oh = int(orig_size.findtext("height")) if orig_size is not None else 1080

    tracks = root.findall("track")
    boxes_by_frame: dict[int, list[dict[str, Any]]] = {}
    keyframes: set[int] = set()
    state_boundaries: set[int] = set()
    track_records: list[dict[str, Any]] = []

    for t in tracks:
        tid = t.get("id")
        label = t.get("label", "").strip()
        boxes = t.findall("box")
        inside_boxes = [b for b in boxes if b.get("outside") == "0"]
        if not inside_boxes:
            continue

        frames = [int(b.get("frame")) for b in inside_boxes]
        start_f, end_f = min(frames), max(frames)
        state_boundaries.add(start_f)
        state_boundaries.add(end_f)

        for b in inside_boxes:
            f_idx = int(b.get("frame"))
            is_key = (b.get("keyframe") == "1")
            if is_key:
                keyframes.add(f_idx)

            xtl = float(b.get("xtl"))
            ytl = float(b.get("ytl"))
            xbr = float(b.get("xbr"))
            ybr = float(b.get("ybr"))

            # Clamp coordinates to image boundaries
            xtl = max(0.0, min(float(ow), xtl))
            ytl = max(0.0, min(float(oh), ytl))
            xbr = max(0.0, min(float(ow), xbr))
            ybr = max(0.0, min(float(oh), ybr))

            box_dict = {
                "track_id": tid,
                "label": label,
                "frame": f_idx,
                "keyframe": is_key,
                "xtl": xtl,
                "ytl": ytl,
                "xbr": xbr,
                "ybr": ybr,
                "width": max(0.0, xbr - xtl),
                "height": max(0.0, ybr - ytl),
            }

            # Deduplicate multiple boxes on same frame if identical track/label
            existing = boxes_by_frame.get(f_idx, [])
            if not any(e["label"].lower() == label.lower() and abs(e["xtl"] - xtl) < 5.0 for e in existing):
                boxes_by_frame.setdefault(f_idx, []).append(box_dict)

        track_records.append({
            "track_id": tid,
            "label": label,
            "start_frame": start_f,
            "end_frame": end_f,
            "total_inside": len(inside_boxes),
        })

    annotated_frames = sorted(boxes_by_frame.keys())
    last_annotated_frame = annotated_frames[-1] if annotated_frames else -1

    return {
        "xml_size": xml_size,
        "width": ow,
        "height": oh,
        "track_records": track_records,
        "boxes_by_frame": boxes_by_frame,
        "keyframes": keyframes,
        "state_boundaries": state_boundaries,
        "annotated_frames": annotated_frames,
        "last_annotated_frame": last_annotated_frame,
    }


def select_decimated_frames(
    clip_info: dict[str, Any],
    xml_data: dict[str, Any],
    video_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply deterministic decimation informed by dHash and track semantics.

    Selection rules:
    1. Retain all state boundaries (start/end of standing, falling, fallen).
    2. Retain all CVAT keyframes (keyframe=1).
    3. Dense active falling motion (stride = 3 frames).
    4. Candidate pre-fall normal and fallen rest frames (stride = 10 frames).
    5. Evaluate consecutive candidate dHash: prune if Hamming distance <= 3,
       unless the frame is a protected keyframe or state boundary.
    6. Strictly truncate tail frames (frame > last_annotated_frame).
    """
    total_video_frames = clip_info["total_frames"]
    boxes_by_frame = xml_data["boxes_by_frame"]
    keyframes = xml_data["keyframes"]
    state_boundaries = xml_data["state_boundaries"]
    annotated_frames = xml_data["annotated_frames"]
    last_annotated = xml_data["last_annotated_frame"]

    # Candidate selection
    candidates: set[int] = set()
    candidates.update(keyframes)
    candidates.update(state_boundaries)

    for f_idx in annotated_frames:
        boxes = boxes_by_frame.get(f_idx, [])
        if not boxes:
            continue
        lbl = boxes[0]["label"].lower()
        if "falling" in lbl:
            if f_idx % 3 == 0:
                candidates.add(f_idx)
        else:
            if f_idx % 10 == 0:
                candidates.add(f_idx)

    sorted_candidates = sorted(candidates)

    cap = cv2.VideoCapture(str(video_path))
    retained_records: list[dict[str, Any]] = []
    last_hash: int | None = None
    dhash_dropped_count = 0

    for f_idx in sorted_candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = compute_dhash(pil_img)

        is_critical = (f_idx in keyframes) or (f_idx in state_boundaries)

        if last_hash is None or is_critical:
            retained = True
            last_hash = curr_hash
        else:
            dist = hamming_distance(last_hash, curr_hash)
            if dist <= 3:
                retained = False
                dhash_dropped_count += 1
            else:
                retained = True
                last_hash = curr_hash

        if retained:
            boxes = boxes_by_frame[f_idx]
            retained_records.append({
                "frame_idx": f_idx,
                "frame_bgr": frame,
                "dhash": curr_hash,
                "dhash_hex": hex(curr_hash),
                "is_keyframe": f_idx in keyframes,
                "is_state_boundary": f_idx in state_boundaries,
                "boxes": boxes,
            })

    cap.release()

    tail_dropped = max(0, total_video_frames - (last_annotated + 1))
    annotated_count = len(annotated_frames)
    total_dropped = annotated_count - len(retained_records)

    stats = {
        "total_video_frames": total_video_frames,
        "annotated_frames": annotated_count,
        "last_annotated_frame": last_annotated,
        "tail_dropped": tail_dropped,
        "candidate_count": len(sorted_candidates),
        "dhash_dropped": dhash_dropped_count,
        "retained_count": len(retained_records),
        "total_dropped": total_dropped,
    }

    return retained_records, stats


def convert_boxes_to_yolo(
    boxes: list[dict[str, Any]],
    img_w: int = 1920,
    img_h: int = 1080,
) -> list[tuple[int, float, float, float, float]]:
    """Convert CVAT boxes to Stage 1 YOLO label entries.

    Mapping rules:
    - pre-fall normal states ('standing', 'Sleeping') -> class 0 person only
    - active falling and fallen-on-floor states ('falling', 'Falling', 'fallen', 'Fallen')
      -> two co-occurring boxes using the same verified coordinates: class 0 person and class 3 fall
    - never create helmet (1), vest (2), fire (4), smoke (5) boxes
    """
    yolo_entries = []

    for b in boxes:
        lbl = b["label"].strip().lower()
        xtl, ytl, xbr, ybr = b["xtl"], b["ytl"], b["xbr"], b["ybr"]

        # Clamp and normalize
        w_px = max(1.0, xbr - xtl)
        h_px = max(1.0, ybr - ytl)
        cx_px = xtl + (w_px / 2.0)
        cy_px = ytl + (h_px / 2.0)

        cx = round(max(0.0, min(1.0, cx_px / img_w)), 6)
        cy = round(max(0.0, min(1.0, cy_px / img_h)), 6)
        w = round(max(0.0001, min(1.0, w_px / img_w)), 6)
        h = round(max(0.0001, min(1.0, h_px / img_h)), 6)

        if "standing" in lbl or "sleeping" in lbl:
            # Pre-fall normal state -> class 0 person only
            yolo_entries.append((0, cx, cy, w, h))
        elif "falling" in lbl or "fallen" in lbl:
            # Falling or Fallen -> two co-occurring boxes with identical coordinates
            yolo_entries.append((0, cx, cy, w, h))
            yolo_entries.append((3, cx, cy, w, h))
        else:
            raise ValueError(f"Unrecognized CVAT label: {b['label']}")

    return yolo_entries


def draw_qa_overlay(
    frame_bgr: Any,
    clip_id: str,
    frame_idx: int,
    total_frames: int,
    fps: float,
    split: str,
    actor_group: str,
    yolo_entries: list[tuple[int, float, float, float, float]],
    raw_state_label: str,
    img_w: int = 1920,
    img_h: int = 1080,
) -> Image.Image:
    """Render high-clarity visual QA overlay with bounding boxes and state metadata."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    time_sec = frame_idx / fps if fps > 0 else 0.0

    # Banner header
    banner_height = 54
    draw.rectangle([(0, 0), (img_w, banner_height)], fill=(20, 24, 32, 230))

    header_text = (
        f"PILOT QA | {clip_id} ({split}) | Frame {frame_idx:04d}/{total_frames:04d} ({time_sec:.2f}s) "
        f"| CVAT: {raw_state_label} | Group: {actor_group}"
    )
    draw.text((18, 16), header_text, fill=(255, 255, 255))

    # Box rendering
    # Color scheme:
    # Class 0 person only (pre-fall): Cyan
    # Class 0 + Class 3 (falling): Orange / Amber
    # Class 0 + Class 3 (fallen): Magenta / Crimson
    has_fall = any(c == 3 for c, *_ in yolo_entries)
    is_fallen = "fallen" in raw_state_label.lower()

    if not has_fall:
        box_color = (0, 210, 255)
        badge_text = "0:person [NORMAL]"
    elif is_fallen:
        box_color = (255, 45, 95)
        badge_text = "0:person + 3:fall [FALLEN REST]"
    else:
        box_color = (255, 170, 0)
        badge_text = "0:person + 3:fall [ACTIVE FALL]"

    # Draw boxes
    for cls_id, cx, cy, w, h in yolo_entries:
        if cls_id == 3 and not is_fallen:
            continue  # don't overdraw identical box twice

        w_px = w * img_w
        h_px = h * img_h
        xtl = (cx * img_w) - (w_px / 2.0)
        ytl = (cy * img_h) - (h_px / 2.0)
        xbr = xtl + w_px
        ybr = ytl + h_px

        draw.rectangle([(xtl, ytl), (xbr, ybr)], outline=box_color, width=4)

        # Label badge
        badge_w = len(badge_text) * 9 + 18
        draw.rectangle([(xtl, max(banner_height, ytl - 26)), (xtl + badge_w, max(banner_height + 26, ytl))], fill=box_color)
        draw.text((xtl + 8, max(banner_height + 4, ytl - 22)), badge_text, fill=(0, 0, 0))

    return img


def create_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    title: str,
    cols: int = 5,
    thumb_w: int = 360,
    thumb_h: int = 202,
) -> None:
    """Combine sampled frames into a contact sheet grid."""
    if not frame_paths:
        return

    # Select up to 25 evenly spaced frames if many
    max_thumbs = 25
    if len(frame_paths) > max_thumbs:
        step = len(frame_paths) / max_thumbs
        selected = [frame_paths[int(i * step)] for i in range(max_thumbs)]
    else:
        selected = frame_paths

    num_thumbs = len(selected)
    rows = math.ceil(num_thumbs / cols)

    margin = 12
    header_h = 60
    sheet_w = cols * thumb_w + (cols + 1) * margin
    sheet_h = rows * thumb_h + (rows + 1) * margin + header_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(26, 30, 38))
    draw = ImageDraw.Draw(sheet)

    draw.rectangle([(0, 0), (sheet_w, header_h)], fill=(16, 18, 24))
    draw.text((margin, 20), title, fill=(245, 245, 245))

    for idx, fpath in enumerate(selected):
        r = idx // cols
        c = idx % cols
        x = margin + c * (thumb_w + margin)
        y = header_h + margin + r * (thumb_h + margin)

        with Image.open(fpath) as img:
            thumb = img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            sheet.paste(thumb, (x, y))

        fname = fpath.stem.replace("_overlay", "")
        draw.rectangle([(x, y + thumb_h - 22), (x + thumb_w, y + thumb_h)], fill=(0, 0, 0, 190))
        draw.text((x + 6, y + thumb_h - 18), fname, fill=(230, 230, 230))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def build_corrected_pilot(
    raw_dir: Path,
    output_dir: Path,
    grouping_csv: Path,
    docs_dir: Path,
) -> dict[str, Any]:
    """Execute complete corrected pilot build."""
    print("=" * 80)
    print("STAGE 1 PRE-TRAINING PREPARATION: FALL CORRECTED PILOT BUILD")
    print("=" * 80)

    # 1. Load actor grouping
    print(f"Loading actor grouping from {grouping_csv}...")
    grouping_by_clip = {}
    with open(grouping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            grouping_by_clip[r["clip_id"]] = r

    # Verify grouping
    for cid in PILOT_CLIPS:
        if cid not in grouping_by_clip:
            raise ValueError(f"Clip {cid} missing from {grouping_csv}")

    # Check for group leakage
    groups_in_splits: dict[str, set[str]] = {}
    for cid in PILOT_CLIPS:
        g = grouping_by_clip[cid]["actor_group"]
        s = grouping_by_clip[cid]["proposed_split"]
        groups_in_splits.setdefault(g, set()).add(s)

    for g, splits in groups_in_splits.items():
        if len(splits) > 1:
            raise ValueError(f"FATAL: Actor group '{g}' crosses splits: {splits}")
    print("Actor grouping verified: 0 cross-split leakage among pilot clips.")

    # 2. Load manifest
    manifest_csv = raw_dir / "annotations" / "cvat" / "annotation_manifest.csv"
    manifest_by_clip: dict[str, dict[str, Any]] = {}
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for cid in PILOT_CLIPS:
                if cid in grouping_by_clip and grouping_by_clip[cid]["relative_path"] == r["mapped_video_rel_path"]:
                    manifest_by_clip[cid] = r

    # Prepare output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    overlays_dir = output_dir / "qa_overlays"
    sheets_dir = output_dir / "qa_contact_sheets"

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        (overlays_dir / split).mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    pilot_manifest_rows: list[dict[str, Any]] = []
    tail_exclusion_rows: list[dict[str, Any]] = []
    decimation_stats_rows: list[dict[str, Any]] = []

    total_retained_pairs = 0
    instances_by_class: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    split_pair_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_group_sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    overlays_by_clip: dict[str, list[Path]] = {}
    transitions_by_clip: dict[str, list[Path]] = {}

    print(f"\nProcessing 8 CVAT-annotated clips...")

    for cid in PILOT_CLIPS:
        grp_info = grouping_by_clip[cid]
        man_info = manifest_by_clip[cid]
        split = grp_info["proposed_split"]
        actor_grp = grp_info["actor_group"]
        split_group_sets[split].add(actor_grp)

        vpath = raw_dir / grp_info["relative_path"]
        xml_path = raw_dir / man_info["annotation_rel_path"]

        cap = cv2.VideoCapture(str(vpath))
        total_vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        clip_info = {
            "clip_id": cid,
            "total_frames": total_vframes,
            "fps": fps,
            "class_label": grp_info["class_label"],
            "split": split,
            "actor_group": actor_grp,
        }

        # Parse XML
        xml_data = parse_cvat_xml(xml_path)
        last_annot = xml_data["last_annotated_frame"]
        annot_frames = xml_data["annotated_frames"]
        boxes_by_frame = xml_data["boxes_by_frame"]

        # Calculate internal gaps and tail ranges
        all_possible_in_range = set(range(annot_frames[0], last_annot + 1)) if annot_frames else set()
        internal_gaps = sorted(all_possible_in_range - set(annot_frames))
        tail_start = last_annot + 1
        tail_end = total_vframes - 1
        tail_count = max(0, total_vframes - tail_start)

        tail_exclusion_rows.append({
            "clip_id": cid,
            "video_rel_path": grp_info["relative_path"],
            "total_frames": total_vframes,
            "fps": round(fps, 2),
            "first_annotated_frame": annot_frames[0] if annot_frames else 0,
            "last_annotated_frame": last_annot,
            "tail_excluded_start": tail_start if tail_count > 0 else "N/A",
            "tail_excluded_end": tail_end if tail_count > 0 else "N/A",
            "tail_excluded_count": tail_count,
            "internal_gaps_count": len(internal_gaps),
            "internal_gaps_ranges": str(internal_gaps[:10]) + ("..." if len(internal_gaps) > 10 else ""),
            "total_unannotated_excluded": tail_count + len(internal_gaps),
        })

        # Decimation
        retained_records, stats = select_decimated_frames(clip_info, xml_data, vpath)
        total_retained_pairs += len(retained_records)
        split_pair_counts[split] += len(retained_records)

        pre_fall_count = 0
        falling_count = 0
        fallen_count = 0

        clip_overlays = []
        clip_transitions = []

        print(f"  [{cid}] ({split:<5}) frames: {total_vframes:<4} | annot: {len(annot_frames):<4} | tail drop: {tail_count:<3} | retained: {len(retained_records):<3}")

        # Export images, labels, and QA overlays
        for rec in retained_records:
            f_idx = rec["frame_idx"]
            frame_bgr = rec["frame_bgr"]
            boxes = rec["boxes"]
            raw_state = boxes[0]["label"]

            raw_lbl = raw_state.lower()
            if "standing" in raw_lbl or "sleeping" in raw_lbl:
                pre_fall_count += 1
                state_cat = "pre_fall"
            elif "falling" in raw_lbl:
                falling_count += 1
                state_cat = "falling"
            elif "fallen" in raw_lbl:
                fallen_count += 1
                state_cat = "fallen"
            else:
                state_cat = "unknown"

            # Convert to YOLO entries
            yolo_entries = convert_boxes_to_yolo(boxes, img_w=1920, img_h=1080)

            # Class count update
            for cls_id, *_ in yolo_entries:
                instances_by_class[cls_id] += 1

            # Filenames
            stem = f"{cid}_frame_{f_idx:04d}"
            img_filename = f"{stem}.jpg"
            lbl_filename = f"{stem}.txt"
            overlay_filename = f"{stem}_overlay.jpg"

            img_path = images_dir / split / img_filename
            lbl_path = labels_dir / split / lbl_filename
            overlay_path = overlays_dir / split / overlay_filename

            # Save JPEG image
            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save YOLO text label
            lbl_lines = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_id, cx, cy, w, h in yolo_entries]
            lbl_path.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

            # Render overlay
            overlay_img = draw_qa_overlay(
                frame_bgr=frame_bgr,
                clip_id=cid,
                frame_idx=f_idx,
                total_frames=total_vframes,
                fps=fps,
                split=split,
                actor_group=actor_grp,
                yolo_entries=yolo_entries,
                raw_state_label=raw_state,
                img_w=1920,
                img_h=1080,
            )
            overlay_img.save(overlay_path, quality=85)
            clip_overlays.append(overlay_path)

            if rec["is_state_boundary"] or "falling" in raw_lbl:
                clip_transitions.append(overlay_path)

            # Record manifest row
            class_ids_str = ",".join(str(c) for c, *_ in yolo_entries)
            pilot_manifest_rows.append({
                "pilot_id": stem,
                "clip_id": cid,
                "frame_idx": f_idx,
                "time_sec": round(f_idx / fps, 3) if fps > 0 else 0.0,
                "split": split,
                "actor_group": actor_grp,
                "state_category": state_cat,
                "cvat_state_label": raw_state,
                "image_rel_path": f"images/{split}/{img_filename}",
                "label_rel_path": f"labels/{split}/{lbl_filename}",
                "box_count": len(yolo_entries),
                "class_ids": class_ids_str,
                "dhash_hex": rec["dhash_hex"],
                "is_keyframe": rec["is_keyframe"],
                "is_state_boundary": rec["is_state_boundary"],
            })

        overlays_by_clip[cid] = clip_overlays
        transitions_by_clip[cid] = clip_transitions

        # Create contact sheets
        cs_path = sheets_dir / f"{cid}_contact_sheet.png"
        cs_title = f"{cid} ({split}) — {actor_grp} — {len(clip_overlays)} Retained Pilot Frames"
        create_contact_sheet(clip_overlays, cs_path, cs_title)

        if clip_transitions:
            ts_path = sheets_dir / f"{cid}_transition_sheet.png"
            ts_title = f"{cid} ({split}) — Key State Transitions (Onset/Impact) — {len(clip_transitions)} Frames"
            create_contact_sheet(clip_transitions, ts_path, ts_title)

        decimation_stats_rows.append({
            "clip_id": cid,
            "class_label": grp_info["class_label"],
            "split": split,
            "actor_group": actor_grp,
            "total_frames": total_vframes,
            "annotated_frames": len(annot_frames),
            "tail_dropped": tail_count,
            "internal_gap_dropped": len(internal_gaps),
            "candidate_frames": stats["candidate_count"],
            "dhash_dropped": stats["dhash_dropped"],
            "retained_frames": len(retained_records),
            "pre_fall_retained": pre_fall_count,
            "falling_retained": falling_count,
            "fallen_retained": fallen_count,
            "person_instances": pre_fall_count + (falling_count + fallen_count),
            "fall_instances": falling_count + fallen_count,
        })

    # 3. Generate data.yaml
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = [
        f"path: ../data/processed/fall_corrected_pilot",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        f"",
        f"nc: 6",
        f"names:",
    ]
    for cid in range(6):
        data_yaml_content.append(f"  {cid}: {CANONICAL_CLASSES[cid]}")
    data_yaml_content.append("")
    data_yaml_path.write_text("\n".join(data_yaml_content), encoding="utf-8")
    print(f"\nGenerated {data_yaml_path}")

    # 4. Write CSV audit artifacts to docs/audit_artifacts/fall/
    # A. fall_pilot_manifest.csv
    pilot_manifest_path = docs_dir / "fall_pilot_manifest.csv"
    with open(pilot_manifest_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "pilot_id", "clip_id", "frame_idx", "time_sec", "split", "actor_group",
            "state_category", "cvat_state_label", "image_rel_path", "label_rel_path",
            "box_count", "class_ids", "dhash_hex", "is_keyframe", "is_state_boundary"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pilot_manifest_rows)
    print(f"Wrote {pilot_manifest_path} ({len(pilot_manifest_rows)} rows)")

    # B. fall_tail_exclusion_log.csv
    tail_log_path = docs_dir / "fall_tail_exclusion_log.csv"
    with open(tail_log_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "clip_id", "video_rel_path", "total_frames", "fps", "first_annotated_frame",
            "last_annotated_frame", "tail_excluded_start", "tail_excluded_end",
            "tail_excluded_count", "internal_gaps_count", "internal_gaps_ranges",
            "total_unannotated_excluded"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tail_exclusion_rows)
    print(f"Wrote {tail_log_path} ({len(tail_exclusion_rows)} rows)")

    # C. fall_decimation_stats.csv
    decimation_csv_path = docs_dir / "fall_decimation_stats.csv"
    with open(decimation_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "clip_id", "class_label", "split", "actor_group", "total_frames",
            "annotated_frames", "tail_dropped", "internal_gap_dropped",
            "candidate_frames", "dhash_dropped", "retained_frames",
            "pre_fall_retained", "falling_retained", "fallen_retained",
            "person_instances", "fall_instances"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decimation_stats_rows)
    print(f"Wrote {decimation_csv_path} ({len(decimation_stats_rows)} rows)")

    summary = {
        "total_retained_pairs": total_retained_pairs,
        "split_pair_counts": split_pair_counts,
        "instances_by_class": instances_by_class,
        "split_group_sets": {k: list(v) for k, v in split_group_sets.items()},
    }

    print("\n" + "=" * 80)
    print(f"PILOT BUILD COMPLETE:")
    print(f"  Total Image-Label Pairs: {total_retained_pairs}")
    print(f"  Split Counts: train={split_pair_counts['train']}, val={split_pair_counts['val']}, test={split_pair_counts['test']}")
    print(f"  Class Instances: 0:person={instances_by_class[0]}, 3:fall={instances_by_class[3]} (all others=0)")
    print("=" * 80)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corrected Fall Detection pilot dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/fall_detection_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/fall_corrected_pilot"))
    parser.add_argument("--grouping-csv", type=Path, default=Path("docs/audit_artifacts/fall/fall_actor_grouping.csv"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/audit_artifacts/fall"))
    args = parser.parse_args()

    build_corrected_pilot(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        grouping_csv=args.grouping_csv,
        docs_dir=args.docs_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
