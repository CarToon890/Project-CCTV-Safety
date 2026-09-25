#!/usr/bin/env python
"""Test deterministic decimation rules on the 8 CVAT-annotated clips."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import cv2
from PIL import Image

RAW_DIR = Path("data/raw/fall_detection_dataset")
MANIFEST_PATH = RAW_DIR / "annotations" / "cvat" / "annotation_manifest.csv"
GROUPING_PATH = Path("docs/audit_artifacts/fall/fall_actor_grouping.csv")

def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
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

def parse_clip_annotations(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = root.findall("track")
    
    boxes_by_frame = {}
    keyframes = set()
    state_boundaries = set()
    track_details = []
    
    for t in tracks:
        tid = t.get("id")
        label = t.get("label")
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
            if b.get("keyframe") == "1":
                keyframes.add(f_idx)
            boxes_by_frame.setdefault(f_idx, []).append({
                "track_id": tid,
                "label": label,
                "xtl": float(b.get("xtl")),
                "ytl": float(b.get("ytl")),
                "xbr": float(b.get("xbr")),
                "ybr": float(b.get("ybr")),
                "keyframe": b.get("keyframe") == "1",
            })
        
        track_details.append({
            "track_id": tid,
            "label": label,
            "start_frame": start_f,
            "end_frame": end_f,
            "count": len(inside_boxes)
        })

    return {
        "boxes_by_frame": boxes_by_frame,
        "keyframes": keyframes,
        "state_boundaries": state_boundaries,
        "track_details": track_details,
    }

def test_decimation():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
        
    print(f"{'Clip':<10} | {'Total':<6} | {'Annot':<6} | {'Tail Drop':<10} | {'Pre-Fall':<8} | {'Falling':<8} | {'Fallen':<8} | {'Retained':<8} | {'Dropped':<8}")
    print("-" * 90)
    
    grand_total_frames = 0
    grand_annot_frames = 0
    grand_tail_dropped = 0
    grand_retained = 0
    grand_dropped = 0
    
    for r in manifest_rows:
        xml_path = RAW_DIR / r["annotation_rel_path"]
        vpath = RAW_DIR / r["mapped_video_rel_path"]
        
        cap = cv2.VideoCapture(str(vpath))
        v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        data = parse_clip_annotations(xml_path)
        boxes_by_frame = data["boxes_by_frame"]
        keyframes = data["keyframes"]
        state_boundaries = data["state_boundaries"]
        
        annot_frames = sorted(boxes_by_frame.keys())
        last_annot = annot_frames[-1] if annot_frames else -1
        tail_dropped = v_total - (last_annot + 1)
        
        # Rule:
        # 1. Must be in annot_frames (<= last_annot and has box)
        # 2. Candidate frames:
        #    - All keyframes
        #    - All state boundaries (start/end of standing, falling, fallen)
        #    - During falling: stride = 3 (dense capture of transition)
        #    - During standing/sleeping: stride = 10 (~0.17-0.20s)
        #    - During fallen: stride = 10 (~0.17-0.20s)
        
        candidates = set()
        candidates.update(keyframes)
        candidates.update(state_boundaries)
        
        for f_idx in annot_frames:
            boxes = boxes_by_frame[f_idx]
            lbl = boxes[0]["label"].lower()
            if "falling" in lbl:
                if f_idx % 3 == 0:
                    candidates.add(f_idx)
            else:
                if f_idx % 10 == 0:
                    candidates.add(f_idx)
                    
        sorted_candidates = sorted(candidates)
        
        # Check dHash on candidates to prune consecutive near-duplicates (hamming <= 3)
        # BUT protect keyframes and state_boundaries from being dropped!
        cap = cv2.VideoCapture(str(vpath))
        retained = []
        last_hash = None
        
        for f_idx in sorted_candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            curr_hash = compute_dhash(pil_img)
            
            is_critical = (f_idx in keyframes) or (f_idx in state_boundaries)
            
            if last_hash is None or is_critical:
                retained.append(f_idx)
                last_hash = curr_hash
            else:
                dist = hamming_distance(last_hash, curr_hash)
                if dist <= 3:
                    # Near-duplicate, drop
                    pass
                else:
                    retained.append(f_idx)
                    last_hash = curr_hash
        cap.release()
        
        dropped = len(annot_frames) - len(retained)
        
        # Count by state in retained
        pre_cnt = sum(1 for f in retained if "standing" in boxes_by_frame[f][0]["label"].lower() or "sleeping" in boxes_by_frame[f][0]["label"].lower())
        falling_cnt = sum(1 for f in retained if "falling" in boxes_by_frame[f][0]["label"].lower())
        fallen_cnt = sum(1 for f in retained if "fallen" in boxes_by_frame[f][0]["label"].lower())
        
        clip_name = vpath.stem
        print(f"{clip_name:<10} | {v_total:<6} | {len(annot_frames):<6} | {tail_dropped:<10} | {pre_cnt:<8} | {falling_cnt:<8} | {fallen_cnt:<8} | {len(retained):<8} | {dropped:<8}")
        
        grand_total_frames += v_total
        grand_annot_frames += len(annot_frames)
        grand_tail_dropped += tail_dropped
        grand_retained += len(retained)
        grand_dropped += dropped
        
    print("-" * 90)
    print(f"{'TOTAL':<10} | {grand_total_frames:<6} | {grand_annot_frames:<6} | {grand_tail_dropped:<10} | {'':<8} | {'':<8} | {'':<8} | {grand_retained:<8} | {grand_dropped:<8}")

if __name__ == "__main__":
    test_decimation()
