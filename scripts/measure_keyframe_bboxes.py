#!/usr/bin/env python
"""Inspect and verify bounding box coordinates for campaign keyframes."""

import cv2
import json
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

# Clip metadata
CLIPS_META = {
    "FD0001": ("data/adl_no_fall/video_20260216_151903.mp4", 1920, 1080),
    "FD0002": ("data/adl_no_fall/video_20260216_152523.mp4", 1920, 1080),
    "FD0003": ("data/adl_no_fall/video_20260216_152939.mp4", 1080, 1920),
    "FD0004": ("data/adl_no_fall/video_20260216_154339.mp4", 1920, 1080),
    "FD0005": ("data/adl_no_fall/video_20260223_151242.mp4", 1920, 1080),
    "FD0006": ("data/adl_no_fall/video_20260223_151533.mp4", 1920, 1080),
    "FD0007": ("data/sitting_to_fall/video_20260216_151241.mp4", 1920, 1080),
    "FD0010": ("data/sitting_to_fall/video_20260216_151650.mp4", 1920, 1080),
    "FD0014": ("data/sitting_to_fall/video_20260223_151341.mp4", 1920, 1080),
    "FD0020": ("data/sitting_to_fall/video_20260223_153406.mp4", 1080, 1920),
}

# Candidate manual annotations: [ymin, xmin, ymax, xmax] in pixel coordinates
# Each box will be overlaid, saved, and visually reviewed.
CANDIDATE_BOXES = {
    # FD0001 (Actor A, dark top, classroom)
    ("FD0001", 100): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [280, 520, 950, 980]}]  # person sitting in chair
    },
    ("FD0001", 260): {
        "action": "adl_standing_transition",
        "boxes": [{"cls": 0, "rect": [250, 520, 950, 980]}]  # rising from chair
    },
    ("FD0001", 300): {
        "action": "adl_walking",
        "boxes": [{"cls": 0, "rect": [210, 580, 950, 1050]}]  # standing/walking
    },
    # FD0002 (Actor pink top, classroom)
    ("FD0002", 20): {
        "action": "adl_standing",
        "boxes": [{"cls": 0, "rect": [95, 550, 960, 915]}]  # standing upright
    },
    ("FD0002", 60): {
        "action": "adl_bending",
        "boxes": [{"cls": 0, "rect": [360, 560, 960, 980]}]  # bending forward
    },
    # FD0003 (2 persons, classroom, portrait 1080x1920)
    ("FD0003", 30): {
        "action": "adl_standing_multi",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]}, # male foreground
            {"cls": 0, "rect": [430, 725, 1260, 965]}, # female background
        ]
    },
    # FD0004 (Actor black tee, classroom)
    ("FD0004", 50): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [230, 360, 830, 810]}]
    },
    ("FD0004", 125): {
        "action": "adl_leaning",
        "boxes": [{"cls": 0, "rect": [250, 360, 830, 810]}]
    },
    # FD0005 (Actor E, studio bed)
    ("FD0005", 50): {
        "action": "adl_lying",
        "boxes": [{"cls": 0, "rect": [380, 380, 640, 1750]}] # lying horizontal
    },
    ("FD0005", 220): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}] # sitting upright
    },
    # FD0006 (Plaid shirt, studio bed)
    ("FD0006", 40): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}] # sitting on bed edge
    },
    ("FD0006", 110): {
        "action": "adl_bending",
        "boxes": [{"cls": 0, "rect": [260, 560, 800, 930]}] # deep bend
    },
    # FD0007 (Actor B, classroom chair sitting-to-fall)
    ("FD0007", 100): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [290, 720, 960, 1050]}]
    },
    ("FD0007", 295): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [450, 670, 960, 1140]},
            {"cls": 3, "rect": [450, 670, 960, 1140]},
        ]
    },
    ("FD0007", 325): {
        "action": "fall_impact",
        "boxes": [
            {"cls": 0, "rect": [680, 680, 965, 1380]},
            {"cls": 3, "rect": [680, 680, 965, 1380]},
        ]
    },
    ("FD0007", 350): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [700, 680, 965, 1420]},
            {"cls": 3, "rect": [700, 680, 965, 1420]},
        ]
    },
    # FD0010 (Actor C, classroom chair sitting-to-fall)
    ("FD0010", 100): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}]
    },
    ("FD0010", 215): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [280, 390, 940, 960]},
            {"cls": 3, "rect": [280, 390, 940, 960]},
        ]
    },
    ("FD0010", 230): {
        "action": "fall_impact",
        "boxes": [
            {"cls": 0, "rect": [580, 400, 940, 1260]},
            {"cls": 3, "rect": [580, 400, 940, 1260]},
        ]
    },
    ("FD0010", 260): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ]
    },
    # FD0014 (Plaid shirt, studio bed sitting-to-fall)
    ("FD0014", 150): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [10, 680, 775, 1020]}]
    },
    ("FD0014", 310): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [320, 710, 800, 1140]},
            {"cls": 3, "rect": [320, 710, 800, 1140]},
        ]
    },
    ("FD0014", 325): {
        "action": "fall_impact",
        "boxes": [
            {"cls": 0, "rect": [620, 730, 840, 1420]},
            {"cls": 3, "rect": [620, 730, 840, 1420]},
        ]
    },
    ("FD0014", 345): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [640, 730, 840, 1450]},
            {"cls": 3, "rect": [640, 730, 840, 1450]},
        ]
    },
    # FD0020 (Actor E, studio bed sitting-to-fall, portrait 1080x1920)
    ("FD0020", 150): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}]
    },
    ("FD0020", 288): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [650, 470, 1120, 1060]},
            {"cls": 3, "rect": [650, 470, 1120, 1060]},
        ]
    },
    ("FD0020", 297): {
        "action": "fall_impact",
        "boxes": [
            {"cls": 0, "rect": [950, 520, 1370, 1060]},
            {"cls": 3, "rect": [950, 520, 1370, 1060]},
        ]
    },
    ("FD0020", 340): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [1080, 520, 1500, 1080]},
            {"cls": 3, "rect": [1080, 520, 1500, 1080]},
        ]
    },
}

out_qa = Path("data/processed/fall_annotation_campaign/qa_measurements")
out_qa.mkdir(parents=True, exist_ok=True)

def render_measurement(cid, fidx, info):
    rel, w, h = CLIPS_META[cid]
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    
    vis = frame.copy()
    action = info["action"]
    boxes = info["boxes"]
    
    # Class colors: 0 (person) = cyan (255, 255, 0), 3 (fall) = magenta (255, 0, 255)
    colors = {0: (255, 255, 0), 3: (255, 0, 255)}
    
    for b in boxes:
        cls_id = b["cls"]
        ymin, xmin, ymax, xmax = b["rect"]
        color = colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 4)
        label_text = f"cls {cls_id} ({'person' if cls_id==0 else 'fall'})"
        cv2.putText(vis, label_text, (xmin, max(ymin - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
    cv2.putText(vis, f"{cid} f{fidx:04d} [{action}]", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    out_path = out_qa / f"{cid}_f{fidx:04d}_meas.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Rendered {out_path}")

for (cid, fidx), info in CANDIDATE_BOXES.items():
    render_measurement(cid, fidx, info)

print("Measurements rendered.")
