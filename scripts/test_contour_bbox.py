#!/usr/bin/env python
"""Test contour-based bbox extraction on static background frames."""

import cv2
import numpy as np
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

def test_bbox(video_rel, fidx):
    cap = cv2.VideoCapture(str(RAW_ROOT / video_rel))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Compute median background from a few frames
    frames = []
    for step in range(0, nframes, max(1, nframes // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        r, f = cap.read()
        if r:
            frames.append(f)
    bg = np.median(frames, axis=0).astype(np.uint8)
    
    # Grab target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    r, frame = cap.read()
    cap.release()
    if not r:
        return None
    
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    person_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000: # significant person area
            x, y, bw, bh = cv2.boundingRect(c)
            # YOLO normalized: xc, yc, norm_w, norm_h
            person_boxes.append((x, y, bw, bh, (x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h))
    
    # Draw on frame
    vis = frame.copy()
    for x, y, bw, bh, xc, yc, nw, nh in person_boxes:
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.putText(vis, f"person {nw:.2f}x{nh:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    out_p = Path("data/processed/fall_annotation_campaign/test_box.jpg")
    cv2.imwrite(str(out_p), vis)
    print(f"Detected {len(person_boxes)} boxes on frame {fidx}. Saved to {out_p}")
    return person_boxes

test_bbox("data/sitting_to_fall/video_20260216_151241.mp4", 100)
