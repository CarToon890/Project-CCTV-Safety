#!/usr/bin/env python
"""Compare actors in candidate sitting clips against pilot clips."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

# Let's extract a frame from FD0049 (Actor C)
cap = cv2.VideoCapture(str(RAW_ROOT / "data/standing_to_fall/video_20260216_155034.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, f = cap.read()
cap.release()
if ret:
    cv2.imwrite("data/processed/fall_annotation_campaign/thumbnails/FD0049_actorC.jpg", f)
    print("Saved FD0049_actorC.jpg")
