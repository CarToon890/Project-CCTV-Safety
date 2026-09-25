#!/usr/bin/env python
"""Analyze exact frame action phases for ADL and sitting clips."""

import cv2
import numpy as np
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS = {
    "FD0001": "data/adl_no_fall/video_20260216_151903.mp4",
    "FD0002": "data/adl_no_fall/video_20260216_152523.mp4",
    "FD0003": "data/adl_no_fall/video_20260216_152939.mp4",
    "FD0004": "data/adl_no_fall/video_20260216_154339.mp4",
    "FD0005": "data/adl_no_fall/video_20260223_151242.mp4",
    "FD0006": "data/adl_no_fall/video_20260223_151533.mp4",
    "FD0007": "data/sitting_to_fall/video_20260216_151241.mp4",
    "FD0010": "data/sitting_to_fall/video_20260216_151650.mp4",
    "FD0014": "data/sitting_to_fall/video_20260223_151341.mp4",
    "FD0018": "data/sitting_to_fall/video_20260223_151902.mp4",
    "FD0020": "data/sitting_to_fall/video_20260223_153406.mp4",
}

# For sitting clips, detect where motion occurs (frame differencing)
def find_motion_burst(video_path):
    cap = cv2.VideoCapture(str(RAW_ROOT / video_path))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    diffs = []
    
    for i in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            diffs.append((i, diff))
        prev_gray = gray
    cap.release()
    
    # Sort by diff
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_diffs = diffs[:10]
    top_diffs.sort(key=lambda x: x[0])
    return nframes, top_diffs

for cid, rel in CLIPS.items():
    nframes, top_diffs = find_motion_burst(rel)
    burst_frames = [f for f, d in top_diffs]
    min_b = min(burst_frames) if burst_frames else 0
    max_b = max(burst_frames) if burst_frames else 0
    print(f"{cid} ({nframes} frames): motion peak range ~ frames {min_b}..{max_b}")
