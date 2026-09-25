#!/usr/bin/env python
"""Inspect FD0027 tracks in detail."""
import xml.etree.ElementTree as ET
from pathlib import Path

xml_path = Path("data/raw/fall_detection_dataset/annotations/cvat/video_20260223_150939.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

for t in root.findall("track"):
    tid = t.get("id")
    label = t.get("label")
    boxes = t.findall("box")
    inside = [b for b in boxes if b.get("outside") == "0"]
    frames = [int(b.get("frame")) for b in inside]
    print(f"Track {tid} ({label}): {len(inside)} inside boxes, frames: {min(frames) if frames else None}..{max(frames) if frames else None}")
    if tid in ("1", "4"):
        for b in inside:
            if int(b.get("frame")) == 181:
                print(f"  Frame 181 in track {tid}: xtl={b.get('xtl')}, ytl={b.get('ytl')}, xbr={b.get('xbr')}, ybr={b.get('ybr')}")
