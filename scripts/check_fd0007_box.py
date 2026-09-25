#!/usr/bin/env python
import cv2
from pathlib import Path

raw_p = Path("data/raw/fall_detection_dataset/data/sitting_to_fall/video_20260216_151241.mp4")
cap = cv2.VideoCapture(str(raw_p))
cap.set(cv2.CAP_PROP_POS_FRAMES, 350)
ret, f = cap.read()
cap.release()

# ymin=640, xmin=390, ymax=950, xmax=1430
vis = f.copy()
# Canonical dual-box: class 0 person AND class 3 fall sharing coordinates
cv2.rectangle(vis, (390, 640), (1430, 950), (255, 255, 0), 3) # class 0
cv2.rectangle(vis, (392, 642), (1428, 948), (0, 0, 255), 2)   # class 3
cv2.putText(vis, "0: person | 3: fall", (390, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.imwrite("data/processed/fall_annotation_campaign/fd0007_f350_check.jpg", vis)
print("Saved f350 check")
