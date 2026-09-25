#!/usr/bin/env python
"""Inspect the 8 CVAT XML files and corresponding MP4 video clips."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

RAW_DIR = Path("data/raw/fall_detection_dataset")
MANIFEST_PATH = RAW_DIR / "annotations" / "cvat" / "annotation_manifest.csv"

def inspect():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} entries from manifest.")
    for r in rows:
        xml_path = RAW_DIR / r["annotation_rel_path"]
        video_path = RAW_DIR / r["mapped_video_rel_path"]
        
        cap = cv2.VideoCapture(str(video_path))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        tree = ET.parse(xml_path)
        root = tree.getroot()
        meta = root.find("meta")
        job = meta.find("job") if meta is not None else None
        xml_size = int(job.findtext("size")) if job is not None and job.findtext("size") else None

        tracks = root.findall("track")
        print(f"\n--- {r['annotation_file']} -> {video_path.name} ---")
        print(f"Video frames: {video_frames}, FPS: {fps:.2f}, XML size: {xml_size}, Tracks: {len(tracks)}")

        track_info = []
        all_frames_with_boxes = set()
        keyframes = set()
        loose_boxes = []

        for t in tracks:
            tid = t.get("id")
            label = t.get("label")
            boxes = t.findall("box")
            inside_boxes = [b for b in boxes if b.get("outside") == "0"]
            if not inside_boxes:
                print(f"  Track id={tid} label={label}: EMPTY (0 inside boxes)")
                continue

            frames = [int(b.get("frame")) for b in inside_boxes]
            min_f, max_f = min(frames), max(frames)
            for b in inside_boxes:
                f_idx = int(b.get("frame"))
                all_frames_with_boxes.add(f_idx)
                if b.get("keyframe") == "1":
                    keyframes.add(f_idx)
                
                xtl = float(b.get("xtl"))
                ytl = float(b.get("ytl"))
                xbr = float(b.get("xbr"))
                ybr = float(b.get("ybr"))
                w = xbr - xtl
                h = ybr - ytl
                if w > 1000 or w <= 0 or h <= 0 or xtl < 0 or ytl < 0 or xbr > 1920 or ybr > 1080:
                    loose_boxes.append((f_idx, label, xtl, ytl, xbr, ybr, w, h))

            track_info.append((tid, label, min_f, max_f, len(inside_boxes)))
            print(f"  Track id={tid} label='{label}': frame {min_f}..{max_f} ({len(inside_boxes)} frames)")

        sorted_frames = sorted(all_frames_with_boxes)
        min_annotated = sorted_frames[0] if sorted_frames else 0
        max_annotated = sorted_frames[-1] if sorted_frames else 0
        missing_in_range = [f for f in range(min_annotated, max_annotated + 1) if f not in all_frames_with_boxes]
        tail_excluded = list(range(max_annotated + 1, video_frames))

        print(f"  Overall annotated range: {min_annotated}..{max_annotated} ({len(sorted_frames)} frames)")
        print(f"  Keyframes count: {len(keyframes)}")
        print(f"  Missing frames in range [{min_annotated}..{max_annotated}]: {len(missing_in_range)} frames {missing_in_range[:10]}{'...' if len(missing_in_range)>10 else ''}")
        print(f"  Tail frames beyond {max_annotated}: {len(tail_excluded)} frames (range {max_annotated+1}..{video_frames-1})")
        if loose_boxes:
            print(f"  Abnormal / loose boxes: {len(loose_boxes)} instances. First 3: {loose_boxes[:3]}")

if __name__ == "__main__":
    inspect()
