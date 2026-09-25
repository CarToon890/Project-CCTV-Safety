#!/usr/bin/env python
"""Identify actor clothing / appearance in sitting clips."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS = [
    "FD0007", "FD0008", "FD0009", "FD0010", "FD0011", "FD0012", "FD0013",
    "FD0014", "FD0015", "FD0016", "FD0017", "FD0018", "FD0019",
    "FD0020", "FD0021", "FD0022"
]

PATHS = {
    "FD0007": "data/sitting_to_fall/video_20260216_151241.mp4",
    "FD0008": "data/sitting_to_fall/video_20260216_151407.mp4",
    "FD0009": "data/sitting_to_fall/video_20260216_151454.mp4",
    "FD0010": "data/sitting_to_fall/video_20260216_151650.mp4",
    "FD0011": "data/sitting_to_fall/video_20260216_151931.mp4",
    "FD0012": "data/sitting_to_fall/video_20260216_152158.mp4",
    "FD0013": "data/sitting_to_fall/video_20260216_154403.mp4",
    "FD0014": "data/sitting_to_fall/video_20260223_151341.mp4",
    "FD0015": "data/sitting_to_fall/video_20260223_151402.mp4",
    "FD0016": "data/sitting_to_fall/video_20260223_151433.mp4",
    "FD0017": "data/sitting_to_fall/video_20260223_151550.mp4",
    "FD0018": "data/sitting_to_fall/video_20260223_151902.mp4",
    "FD0019": "data/sitting_to_fall/video_20260223_151930.mp4",
    "FD0020": "data/sitting_to_fall/video_20260223_153406.mp4",
    "FD0021": "data/sitting_to_fall/video_20260223_153438.mp4",
    "FD0022": "data/sitting_to_fall/video_20260223_153506.mp4",
}

for cid in CLIPS:
    vpath = RAW_ROOT / PATHS[cid]
    cap = cv2.VideoCapture(str(vpath))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Grab frame at 20%
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(nframes * 0.2))
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Save a clear image for visual check
        out_p = Path(f"data/processed/fall_annotation_campaign/thumbnails/{cid}_inspect.jpg")
        cv2.imwrite(str(out_p), frame)
        print(f"Saved {cid} frame at {out_p}")
