#!/usr/bin/env python
"""Determine exact transition intervals for the 4 sitting clips."""

import cv2
import numpy as np
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS = [
    ("FD0007", "data/sitting_to_fall/video_20260216_151241.mp4", 250, 370),
    ("FD0010", "data/sitting_to_fall/video_20260216_151650.mp4", 170, 270),
    ("FD0014", "data/sitting_to_fall/video_20260223_151341.mp4", 270, 360),
    ("FD0020", "data/sitting_to_fall/video_20260223_153406.mp4", 250, 340),
]

for cid, rel, start_f, end_f in CLIPS:
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    prev_center_y = None
    y_velocities = []
    
    for fidx in range(start_f, end_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            break
        # calculate center of mass of person / motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # diff from previous
        if prev_center_y is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            if M["m00"] > 100:
                cy = M["m01"] / M["m00"]
                y_velocities.append((fidx, cy, M["m00"]))
        prev_gray = gray
        prev_center_y = True
    cap.release()
    
    # print significant motion frames
    y_velocities.sort(key=lambda x: x[2], reverse=True)
    top10 = sorted(y_velocities[:10], key=lambda x: x[0])
    frames = [f for f, cy, m in top10]
    print(f"{cid} motion peak window: frames {min(frames)} to {max(frames)}")
