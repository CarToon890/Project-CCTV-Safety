#!/usr/bin/env python
"""Analyze visual content and actors of sitting and ADL clips."""

import csv
import cv2
from pathlib import Path
from PIL import Image

CLIPS_CSV = Path("data/raw/fall_detection_dataset/metadata/clips_index.csv")
GROUPING_CSV = Path("docs/audit_artifacts/fall/fall_actor_grouping.csv")
RAW_ROOT = Path("data/raw/fall_detection_dataset")
THUMB_DIR = Path("data/processed/fall_annotation_campaign/thumbnails")

def analyze():
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(GROUPING_CSV, "r", encoding="utf-8") as f:
        grouping = {r["clip_id"]: r for r in csv.DictReader(f)}

    with open(CLIPS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    sitting_clips = [r for r in rows if r["class_label"] == "sitting_to_fall"]
    adl_clips = [r for r in rows if r["class_label"] == "adl_no_fall"]
    
    print("=== SITTING TO FALL CLIPS ===")
    for r in sitting_clips:
        cid = r["clip_id"]
        rel = r["relative_path"]
        grp_info = grouping.get(cid, {})
        grp = grp_info.get("actor_group", "unknown")
        split = grp_info.get("proposed_split", "unknown")
        vpath = RAW_ROOT / rel
        cap = cv2.VideoCapture(str(vpath))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # sample at 10%, 50%, 80%
        for pct, name in [(0.1, "start"), (0.5, "mid"), (0.8, "end")]:
            fidx = int(nframes * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if ret:
                thumb = cv2.resize(frame, (320, int(320 * h / w)))
                cv2.imwrite(str(THUMB_DIR / f"{cid}_{name}.jpg"), thumb)
        cap.release()
        print(f"{cid}: group={grp}, split={split}, frames={nframes}, dim={w}x{h}")

    print("\n=== ADL NO FALL CLIPS ===")
    for r in adl_clips:
        cid = r["clip_id"]
        rel = r["relative_path"]
        grp_info = grouping.get(cid, {})
        grp = grp_info.get("actor_group", "unknown")
        split = grp_info.get("proposed_split", "unknown")
        vpath = RAW_ROOT / rel
        cap = cv2.VideoCapture(str(vpath))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for pct, name in [(0.1, "start"), (0.5, "mid"), (0.8, "end")]:
            fidx = int(nframes * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if ret:
                thumb = cv2.resize(frame, (320, int(320 * h / w)))
                cv2.imwrite(str(THUMB_DIR / f"{cid}_{name}.jpg"), thumb)
        cap.release()
        print(f"{cid}: group={grp}, split={split}, frames={nframes}, dim={w}x{h}")

if __name__ == "__main__":
    analyze()
