#!/usr/bin/env python
"""Inspect exact action states for the 10 selected campaign clips."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

CLIPS_10 = [
    # 6 ADL
    {"clip_id": "FD0001", "rel": "data/adl_no_fall/video_20260216_151903.mp4", "label": "adl_no_fall"},
    {"clip_id": "FD0002", "rel": "data/adl_no_fall/video_20260216_152523.mp4", "label": "adl_no_fall"},
    {"clip_id": "FD0003", "rel": "data/adl_no_fall/video_20260216_152939.mp4", "label": "adl_no_fall"},
    {"clip_id": "FD0004", "rel": "data/adl_no_fall/video_20260216_154339.mp4", "label": "adl_no_fall"},
    {"clip_id": "FD0005", "rel": "data/adl_no_fall/video_20260223_151242.mp4", "label": "adl_no_fall"},
    {"clip_id": "FD0006", "rel": "data/adl_no_fall/video_20260223_151533.mp4", "label": "adl_no_fall"},
    # 4 Sitting
    {"clip_id": "FD0007", "rel": "data/sitting_to_fall/video_20260216_151241.mp4", "label": "sitting_to_fall"},
    {"clip_id": "FD0010", "rel": "data/sitting_to_fall/video_20260216_151650.mp4", "label": "sitting_to_fall"},
    {"clip_id": "FD0014", "rel": "data/sitting_to_fall/video_20260223_151341.mp4", "label": "sitting_to_fall"},
    {"clip_id": "FD0020", "rel": "data/sitting_to_fall/video_20260223_153406.mp4", "label": "sitting_to_fall"},
]

out_base = Path("data/processed/fall_annotation_campaign/inspections")
out_base.mkdir(parents=True, exist_ok=True)

for item in CLIPS_10:
    cid = item["clip_id"]
    rel = item["rel"]
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    clip_dir = out_base / cid
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    # Save frames every ~10 frames or specific milestones
    step = max(1, nframes // 20)
    for fidx in range(0, nframes, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            thumb = cv2.resize(frame, (320, int(320 * h / w)))
            cv2.putText(thumb, f"{cid} f{fidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(clip_dir / f"f{fidx:04d}.jpg"), thumb)
    cap.release()
    print(f"Processed {cid}: {nframes} frames, {w}x{h}, {fps:.2f} fps")
