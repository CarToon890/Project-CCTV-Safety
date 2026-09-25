#!/usr/bin/env python
"""Determine exact frame state boundaries for the 10 campaign clips."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS = [
    # 6 ADL
    ("FD0001", "data/adl_no_fall/video_20260216_151903.mp4"),
    ("FD0002", "data/adl_no_fall/video_20260216_152523.mp4"),
    ("FD0003", "data/adl_no_fall/video_20260216_152939.mp4"),
    ("FD0004", "data/adl_no_fall/video_20260216_154339.mp4"),
    ("FD0005", "data/adl_no_fall/video_20260223_151242.mp4"),
    ("FD0006", "data/adl_no_fall/video_20260223_151533.mp4"),
    # 4 Sitting-to-fall
    ("FD0007", "data/sitting_to_fall/video_20260216_151241.mp4"),
    ("FD0010", "data/sitting_to_fall/video_20260216_151650.mp4"),
    ("FD0014", "data/sitting_to_fall/video_20260223_151341.mp4"),
    ("FD0020", "data/sitting_to_fall/video_20260223_153406.mp4"),
]

def scan_clip(cid, rel):
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"=== {cid} ({rel}) | {w}x{h} | {nframes} frames | {fps:.2f} fps ===")

for cid, rel in CLIPS:
    scan_clip(cid, rel)
