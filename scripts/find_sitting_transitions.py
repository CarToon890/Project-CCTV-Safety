#!/usr/bin/env python
"""Find exact transition, impact, and fallen frame indices for the 4 sitting clips."""

import cv2
import numpy as np
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS = [
    ("FD0007", "data/sitting_to_fall/video_20260216_151241.mp4"),
    ("FD0010", "data/sitting_to_fall/video_20260216_151650.mp4"),
    ("FD0014", "data/sitting_to_fall/video_20260223_151341.mp4"),
    ("FD0020", "data/sitting_to_fall/video_20260223_153406.mp4"),
]

out_dir = Path("data/processed/fall_annotation_campaign/transitions_inspect")
out_dir.mkdir(parents=True, exist_ok=True)

for cid, rel in CLIPS:
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Save frames around the motion burst in steps of 5
    # Let's inspect frames across the clip to find exact boundaries
    print(f"\n--- {cid} ({nframes} frames) ---")
    for fidx in range(0, nframes, max(1, nframes // 30)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            thumb = cv2.resize(frame, (320, int(320 * h / w)))
            cv2.putText(thumb, f"{cid} f{fidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / f"{cid}_f{fidx:04d}.jpg"), thumb)
    cap.release()

print("Transitions inspection frames saved.")
