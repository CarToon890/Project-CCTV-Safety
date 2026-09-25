#!/usr/bin/env python
"""Inspect actions and state transitions in ADL and sitting clips."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

# Let's inspect ADL clips: FD0001..FD0006
adl_clips = [
    ("FD0001", "data/adl_no_fall/video_20260216_151903.mp4"),
    ("FD0002", "data/adl_no_fall/video_20260216_152523.mp4"),
    ("FD0003", "data/adl_no_fall/video_20260216_152939.mp4"),
    ("FD0004", "data/adl_no_fall/video_20260216_154339.mp4"),
    ("FD0005", "data/adl_no_fall/video_20260223_151242.mp4"),
    ("FD0006", "data/adl_no_fall/video_20260223_151533.mp4"),
]

# Candidate sitting clips
sit_clips = [
    ("FD0007", "data/sitting_to_fall/video_20260216_151241.mp4"),
    ("FD0010", "data/sitting_to_fall/video_20260216_151650.mp4"),
    ("FD0014", "data/sitting_to_fall/video_20260223_151341.mp4"),
    ("FD0018", "data/sitting_to_fall/video_20260223_151902.mp4"),
    ("FD0020", "data/sitting_to_fall/video_20260223_153406.mp4"),
]

out_dir = Path("data/processed/fall_annotation_campaign/action_inspect")
out_dir.mkdir(parents=True, exist_ok=True)

def sample_clip(cid, rel, n_samples=10):
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = nframes // n_samples
    for i in range(n_samples):
        fidx = min(i * step, nframes - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            # resize for inspection
            h, w = frame.shape[:2]
            thumb = cv2.resize(frame, (320, int(320 * h / w)))
            cv2.putText(thumb, f"{cid} f{fidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{cid}_f{fidx:04d}.jpg"), thumb)
    cap.release()

print("Sampling ADL clips...")
for cid, rel in adl_clips:
    sample_clip(cid, rel, 8)

print("Sampling candidate sitting clips...")
for cid, rel in sit_clips:
    sample_clip(cid, rel, 10)

print("Done sampling.")
