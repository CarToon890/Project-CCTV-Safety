#!/usr/bin/env python
"""Fine-tune and verify exact bounding box coordinates for 20 completed extension frames."""

import cv2
from pathlib import Path

RAW_ROOT = Path("data/raw/fall_detection_dataset")

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

# Accurate manual pixel bounding boxes: [ymin, xmin, ymax, xmax]
# Carefully adjusted to tightly bound head-to-toe and left-to-right.
ACCURATE_BOXES = {
    # 1. FD0001 f100 (ADL sitting, classroom, 1920x1080)
    ("FD0001", 100): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [25, 270, 955, 515]}] # y: 25..955, x: 270..515 (Actor A seated)
    },
    # 2. FD0001 f300 (ADL walking, classroom, 1920x1080)
    ("FD0001", 300): {
        "action": "adl_walking",
        "boxes": [{"cls": 0, "rect": [210, 580, 950, 1050]}]
    },
    # 3. FD0002 f020 (ADL standing, classroom, 1920x1080)
    ("FD0002", 20): {
        "action": "adl_standing",
        "boxes": [{"cls": 0, "rect": [95, 288, 968, 485]}] # y: 95..968, x: 288..485
    },
    # 4. FD0002 f060 (ADL bending, classroom, 1920x1080)
    ("FD0002", 60): {
        "action": "adl_bending",
        "boxes": [{"cls": 0, "rect": [360, 280, 960, 480]}]
    },
    # 5. FD0003 f030 (ADL multi-person, classroom, portrait 1080x1920)
    ("FD0003", 30): {
        "action": "adl_standing_multi",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]}, # male foreground
            {"cls": 0, "rect": [430, 725, 1260, 965]}, # female background
        ]
    },
    # 6. FD0004 f050 (ADL sitting, classroom, 1920x1080)
    ("FD0004", 50): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [230, 190, 830, 410]}] # actor sitting
    },
    # 7. FD0004 f125 (ADL leaning, classroom, 1920x1080)
    ("FD0004", 125): {
        "action": "adl_leaning",
        "boxes": [{"cls": 0, "rect": [250, 190, 830, 410]}]
    },
    # 8. FD0005 f050 (ADL lying, studio bed, 1920x1080)
    ("FD0005", 50): {
        "action": "adl_lying",
        "boxes": [{"cls": 0, "rect": [380, 380, 640, 1750]}]
    },
    # 9. FD0005 f220 (ADL sitting, studio bed, 1920x1080)
    ("FD0005", 220): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}]
    },
    # 10. FD0006 f040 (ADL sitting, studio bed, 1920x1080)
    ("FD0006", 40): {
        "action": "adl_sitting",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}]
    },
    # 11. FD0006 f110 (ADL bending, studio bed, 1920x1080)
    ("FD0006", 110): {
        "action": "adl_bending",
        "boxes": [{"cls": 0, "rect": [260, 560, 800, 930]}]
    },
    # 12. FD0007 f100 (Sitting pre-fall, classroom, 1920x1080)
    ("FD0007", 100): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [268, 720, 960, 1040]}]
    },
    # 13. FD0007 f315 (Sitting fall transition, classroom, 1920x1080)
    ("FD0007", 315): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [580, 720, 960, 1250]},
            {"cls": 3, "rect": [580, 720, 960, 1250]},
        ]
    },
    # 14. FD0007 f350 (Sitting fallen, classroom, 1920x1080)
    ("FD0007", 350): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [770, 700, 980, 1550]},
            {"cls": 3, "rect": [770, 700, 980, 1550]},
        ]
    },
    # 15. FD0010 f100 (Sitting pre-fall, classroom, 1920x1080)
    ("FD0010", 100): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}]
    },
    # 16. FD0010 f220 (Sitting fall transition, classroom, 1920x1080)
    ("FD0010", 220): {
        "action": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [350, 410, 940, 1020]},
            {"cls": 3, "rect": [350, 410, 940, 1020]},
        ]
    },
    # 17. FD0010 f260 (Sitting fallen, classroom, 1920x1080)
    ("FD0010", 260): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ]
    },
    # 18. FD0014 f150 (Sitting pre-fall, studio bed, 1920x1080)
    ("FD0014", 150): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [10, 680, 775, 1020]}]
    },
    # 19. FD0014 f345 (Sitting fallen, studio bed, 1920x1080)
    ("FD0014", 345): {
        "action": "fallen_on_floor",
        "boxes": [
            {"cls": 0, "rect": [640, 730, 840, 1450]},
            {"cls": 3, "rect": [640, 730, 840, 1450]},
        ]
    },
    # 20. FD0020 f150 (Sitting pre-fall, studio bed, portrait 1080x1920)
    ("FD0020", 150): {
        "action": "pre_fall_sitting",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}]
    },
}

out_qa = Path("data/processed/fall_annotation_campaign/qa_verified_20")
out_qa.mkdir(parents=True, exist_ok=True)

for (cid, fidx), info in ACCURATE_BOXES.items():
    rel, w, h = CLIPS_META[cid]
    cap = cv2.VideoCapture(str(RAW_ROOT / rel))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    
    vis = frame.copy()
    action = info["action"]
    boxes = info["boxes"]
    colors = {0: (255, 255, 0), 3: (0, 0, 255)} # cyan for person, red for fall
    
    for b in boxes:
        cls_id = b["cls"]
        ymin, xmin, ymax, xmax = b["rect"]
        color = colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(vis, f"cls {cls_id} ({'person' if cls_id==0 else 'fall'})", 
                    (xmin, max(ymin - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    cv2.putText(vis, f"{cid} f{fidx:04d} [{action}]", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    out_path = out_qa / f"{cid}_f{fidx:04d}_verified.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Verified overlay rendered: {out_path}")
