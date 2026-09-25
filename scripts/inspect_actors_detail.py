#!/usr/bin/env python
"""Inspect detailed actor and scene visual characteristics for sitting and ADL clips."""

import csv
import cv2
import numpy as np
from pathlib import Path

CLIPS_CSV = Path("data/raw/fall_detection_dataset/metadata/clips_index.csv")
RAW_ROOT = Path("data/raw/fall_detection_dataset")

def analyze_details():
    with open(CLIPS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    target_cids = [
        # ADL
        "FD0001", "FD0002", "FD0003", "FD0004", "FD0005", "FD0006",
        # Sitting
        "FD0007", "FD0008", "FD0009", "FD0010", "FD0011", "FD0012",
        "FD0013", "FD0014", "FD0015", "FD0016", "FD0017", "FD0018",
        "FD0019", "FD0020", "FD0021", "FD0022"
    ]
    
    for r in rows:
        cid = r["clip_id"]
        if cid not in target_cids:
            continue
        rel = r["relative_path"]
        vpath = RAW_ROOT / rel
        cap = cv2.VideoCapture(str(vpath))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # sample at 25% (sitting/action start) and 75% (fallen/completion)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(nframes * 0.25))
        ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(nframes * 0.75))
        ret2, f2 = cap.read()
        cap.release()
        
        # average color / brightness in center to get an idea of scene
        mean_bgr1 = np.mean(f1, axis=(0,1)) if ret1 else [0,0,0]
        mean_bgr2 = np.mean(f2, axis=(0,1)) if ret2 else [0,0,0]
        
        print(f"{cid} ({r['class_label']}): {rel} | {w}x{h} | {nframes}f | {fps:.1f}fps | mean_bgr1={mean_bgr1.astype(int)} | mean_bgr2={mean_bgr2.astype(int)}")

if __name__ == "__main__":
    analyze_details()
