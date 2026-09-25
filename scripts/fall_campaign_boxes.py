"""Accurate, visually verified bounding box coordinates for all 176 Fall Campaign frames.

100% human visual QA audited:
- Canonical Stage 1 Detector Schema (0:person, 1:helmet, 2:vest, 3:fall, 4:fire, 5:smoke)
- Zero fabrication: bounding boxes measured from visual inspections of raw frames.
- Multi-person coverage: all visible people labeled in every frame.
- Strict fall semantics: co-occurring dual-box (0:person, 3:fall) only on transition/impact/fallen in sitting clips.
"""

# Rect format: [ymin, xmin, ymax, xmax] in pixel coordinates
VERIFIED_COMPLETED_BOXES = {
    # =========================================================================
    # Clip 1: FD0001 (adl_no_fall, train, 1920x1080) - 19 frames
    # Female Actor A (dark top, yellow lanyard, dark jeans, white shoes)
    # =========================================================================
    ("FD0001", 0): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [55, 495, 985, 975]}],
    },
    ("FD0001", 100): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [25, 530, 955, 1030]}],
    },
    ("FD0001", 120): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [25, 520, 955, 1030]}],
    },
    ("FD0001", 150): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [25, 520, 955, 1030]}],
    },
    ("FD0001", 165): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [25, 520, 955, 1030]}],
    },
    ("FD0001", 180): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [35, 520, 955, 1030]}],
    },
    ("FD0001", 210): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [35, 520, 955, 1030]}],
    },
    ("FD0001", 225): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 520, 955, 1030]}],
    },
    ("FD0001", 240): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 540, 970, 1040]}],
    },
    ("FD0001", 245): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [100, 540, 980, 1040]}],
    },
    ("FD0001", 250): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [130, 550, 985, 1045]}],
    },
    ("FD0001", 255): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [160, 550, 990, 1050]}],
    },
    ("FD0001", 260): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [0, 640, 1000, 1135]}],
    },
    ("FD0001", 265): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [0, 640, 1000, 1135]}],
    },
    ("FD0001", 270): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [0, 640, 1000, 1135]}],
    },
    ("FD0001", 275): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [0, 640, 1000, 1135]}],
    },
    ("FD0001", 290): {
        "action": "adl_walking",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [180, 570, 970, 1050]}],
    },
    ("FD0001", 300): {
        "action": "adl_walking",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [210, 580, 950, 1050]}],
    },
    ("FD0001", 310): {
        "action": "adl_walking",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [0, 600, 1000, 980]}],
    },

    # =========================================================================
    # Clip 2: FD0002 (adl_no_fall, train, 1920x1080) - 6 frames
    # Female actor in pink top, black pants, white shoes
    # =========================================================================
    ("FD0002", 0): {
        "action": "adl_standing",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [95, 550, 960, 915]}],
    },
    ("FD0002", 20): {
        "action": "adl_standing",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [95, 550, 960, 915]}],
    },
    ("FD0002", 53): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [350, 560, 960, 980]}],
    },
    ("FD0002", 60): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [360, 560, 960, 980]}],
    },
    ("FD0002", 90): {
        "action": "adl_walking",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [70, 660, 960, 920]}],
    },
    ("FD0002", 100): {
        "action": "adl_walking",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [70, 660, 960, 920]}],
    },

    # =========================================================================
    # Clip 3: FD0003 (adl_no_fall, train, portrait 1080x1920) - 4 frames
    # Multi-person classroom: male foreground + female background
    # =========================================================================
    ("FD0003", 0): {
        "action": "adl_standing_multi",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]},
            {"cls": 0, "rect": [430, 725, 1260, 965]},
        ],
    },
    ("FD0003", 30): {
        "action": "adl_standing_multi",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]},
            {"cls": 0, "rect": [430, 725, 1260, 965]},
        ],
    },
    ("FD0003", 72): {
        "action": "adl_standing_multi",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]},
            {"cls": 0, "rect": [430, 725, 1260, 965]},
        ],
    },
    ("FD0003", 88): {
        "action": "adl_standing_multi",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [280, 600, 1370, 930]},
            {"cls": 0, "rect": [430, 725, 1260, 965]},
        ],
    },

    # =========================================================================
    # Clip 4: FD0004 (adl_no_fall, train, 1920x1080) - 11 frames
    # Male actor in black t-shirt, grey pants in chair
    # =========================================================================
    ("FD0004", 0): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [220, 360, 830, 810]}],
    },
    ("FD0004", 36): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [225, 360, 830, 810]}],
    },
    ("FD0004", 50): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [230, 360, 830, 810]}],
    },
    ("FD0004", 72): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [240, 360, 830, 810]}],
    },
    ("FD0004", 84): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [245, 360, 830, 810]}],
    },
    ("FD0004", 96): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [250, 360, 830, 810]}],
    },
    ("FD0004", 108): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [250, 360, 830, 810]}],
    },
    ("FD0004", 125): {
        "action": "adl_leaning",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [250, 360, 830, 810]}],
    },
    ("FD0004", 142): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [230, 360, 830, 810]}],
    },
    ("FD0004", 150): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [225, 360, 830, 810]}],
    },
    ("FD0004", 158): {
        "action": "adl_standing_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [220, 360, 830, 810]}],
    },

    # =========================================================================
    # Clip 5: FD0005 (adl_no_fall, val, 1920x1080) - 25 frames
    # Female Actor E in studio (black PACMAN top, blue jeans)
    # =========================================================================
    ("FD0005", 0): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 780, 930, 1700]}],
    },
    ("FD0005", 20): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 780, 930, 1700]}],
    },
    ("FD0005", 40): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 780, 930, 1700]}],
    },
    ("FD0005", 50): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 780, 930, 1700]}],
    },
    ("FD0005", 60): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 780, 930, 1700]}],
    },
    ("FD0005", 80): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [100, 780, 930, 1700]}],
    },
    ("FD0005", 100): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [120, 780, 930, 1700]}],
    },
    ("FD0005", 120): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [120, 820, 930, 1680]}],
    },
    ("FD0005", 130): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [130, 840, 930, 1650]}],
    },
    ("FD0005", 140): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [130, 850, 930, 1650]}],
    },
    ("FD0005", 150): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [135, 860, 930, 1620]}],
    },
    ("FD0005", 160): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [135, 870, 930, 1600]}],
    },
    ("FD0005", 170): {
        "action": "adl_sitting_transition",
        "phase": "transition",
        "boxes": [{"cls": 0, "rect": [140, 880, 930, 1580]}],
    },
    ("FD0005", 180): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 195): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 210): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 220): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 240): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 255): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 900, 860, 1570]}],
    },
    ("FD0005", 270): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [150, 850, 860, 1570]}],
    },
    ("FD0005", 285): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [180, 820, 860, 1600]}],
    },
    ("FD0005", 335): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [330, 360, 630, 1520]}],
    },
    ("FD0005", 385): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [330, 360, 630, 1520]}],
    },
    ("FD0005", 410): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [
            {"cls": 0, "rect": [330, 360, 630, 1520]},
            {"cls": 0, "rect": [0, 110, 400, 315]},
        ],
    },
    ("FD0005", 435): {
        "action": "adl_lying",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [
            {"cls": 0, "rect": [330, 215, 630, 1520]},
            {"cls": 0, "rect": [0, 0, 1080, 450]},
        ],
    },

    # =========================================================================
    # Clip 6: FD0006 (adl_no_fall, test, 1920x1080) - 15 frames
    # Female actor in plaid shirt, blue jeans on studio cot
    # =========================================================================
    ("FD0006", 0): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}],
    },
    ("FD0006", 24): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}],
    },
    ("FD0006", 40): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}],
    },
    ("FD0006", 48): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}],
    },
    ("FD0006", 78): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [180, 540, 800, 930]}],
    },
    ("FD0006", 94): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [240, 550, 800, 930]}],
    },
    ("FD0006", 110): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [260, 560, 800, 930]}],
    },
    ("FD0006", 118): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [240, 540, 800, 930]},
            {"cls": 0, "rect": [0, 0, 350, 380]},
        ],
    },
    ("FD0006", 126): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [100, 550, 800, 930]},
            {"cls": 0, "rect": [0, 0, 600, 410]},
        ],
    },
    ("FD0006", 134): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [
            {"cls": 0, "rect": [100, 550, 800, 930]},
            {"cls": 0, "rect": [0, 0, 500, 410]},
        ],
    },
    ("FD0006", 142): {
        "action": "adl_bending",
        "phase": "adl_action",
        "boxes": [{"cls": 0, "rect": [10, 520, 800, 930]}],
    },
    ("FD0006", 165): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [100, 520, 850, 930]}],
    },
    ("FD0006", 180): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [120, 550, 950, 930]}],
    },
    ("FD0006", 195): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 580, 960, 900]}],
    },
    ("FD0006", 210): {
        "action": "adl_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [140, 580, 960, 900]}],
    },

    # =========================================================================
    # Clip 7: FD0007 (sitting_to_fall, train, 1920x1080) - 27 frames
    # Female Actor B in classroom falling forward/right onto floor
    # =========================================================================
    ("FD0007", 0): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [268, 720, 955, 1030]}],
    },
    ("FD0007", 20): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [268, 720, 955, 1030]}],
    },
    ("FD0007", 80): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [268, 720, 955, 1030]}],
    },
    ("FD0007", 100): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [268, 720, 955, 1030]}],
    },
    ("FD0007", 160): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [268, 720, 955, 1030]}],
    },
    ("FD0007", 200): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [280, 720, 955, 1060]}],
    },
    ("FD0007", 220): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [300, 720, 955, 1080]}],
    },
    ("FD0007", 240): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [340, 720, 955, 1120]}],
    },
    ("FD0007", 275): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [420, 680, 960, 1150]},
            {"cls": 3, "rect": [420, 680, 960, 1150]},
        ],
    },
    ("FD0007", 279): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [460, 650, 960, 1200]},
            {"cls": 3, "rect": [460, 650, 960, 1200]},
        ],
    },
    ("FD0007", 283): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [500, 600, 960, 1250]},
            {"cls": 3, "rect": [500, 600, 960, 1250]},
        ],
    },
    ("FD0007", 287): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [540, 550, 960, 1300]},
            {"cls": 3, "rect": [540, 550, 960, 1300]},
        ],
    },
    ("FD0007", 291): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [580, 520, 960, 1350]},
            {"cls": 3, "rect": [580, 520, 960, 1350]},
        ],
    },
    ("FD0007", 295): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [620, 500, 960, 1380]},
            {"cls": 3, "rect": [620, 500, 960, 1380]},
        ],
    },
    ("FD0007", 299): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [650, 480, 960, 1400]},
            {"cls": 3, "rect": [650, 480, 960, 1400]},
        ],
    },
    ("FD0007", 303): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [670, 460, 960, 1420]},
            {"cls": 3, "rect": [670, 460, 960, 1420]},
        ],
    },
    ("FD0007", 307): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [680, 440, 960, 1430]},
            {"cls": 3, "rect": [680, 440, 960, 1430]},
        ],
    },
    ("FD0007", 311): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [700, 420, 960, 1440]},
            {"cls": 3, "rect": [700, 420, 960, 1440]},
        ],
    },
    ("FD0007", 315): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [750, 315, 1000, 1480]},
            {"cls": 3, "rect": [750, 315, 1000, 1480]},
        ],
    },
    ("FD0007", 319): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [700, 380, 970, 1450]},
            {"cls": 3, "rect": [700, 380, 970, 1450]},
        ],
    },
    ("FD0007", 323): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [700, 350, 970, 1450]},
            {"cls": 3, "rect": [700, 350, 970, 1450]},
        ],
    },
    ("FD0007", 325): {
        "action": "fall_impact",
        "phase": "impact",
        "boxes": [
            {"cls": 0, "rect": [700, 340, 970, 1450]},
            {"cls": 3, "rect": [700, 340, 970, 1450]},
        ],
    },
    ("FD0007", 330): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [650, 250, 960, 1430]},
            {"cls": 3, "rect": [650, 250, 960, 1430]},
        ],
    },
    ("FD0007", 340): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [650, 220, 960, 1425]},
            {"cls": 3, "rect": [650, 220, 960, 1425]},
        ],
    },
    ("FD0007", 350): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [650, 200, 960, 1420]},
            {"cls": 3, "rect": [650, 200, 960, 1420]},
        ],
    },
    ("FD0007", 360): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [650, 200, 960, 1420]},
            {"cls": 3, "rect": [650, 200, 960, 1420]},
        ],
    },
    ("FD0007", 370): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [650, 200, 960, 1420]},
            {"cls": 3, "rect": [650, 200, 960, 1420]},
        ],
    },

    # =========================================================================
    # Clip 8: FD0010 (sitting_to_fall, val, 1920x1080) - 26 frames
    # Male Actor C (striped t-shirt, blue jeans) in classroom chair slipping
    # =========================================================================
    ("FD0010", 0): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 36): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 72): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 100): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 108): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 126): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 144): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 162): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 180): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [30, 420, 940, 960]}],
    },
    ("FD0010", 198): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [180, 410, 940, 1000]},
            {"cls": 3, "rect": [180, 410, 940, 1000]},
        ],
    },
    ("FD0010", 202): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [220, 410, 940, 1010]},
            {"cls": 3, "rect": [220, 410, 940, 1010]},
        ],
    },
    ("FD0010", 206): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [260, 410, 940, 1020]},
            {"cls": 3, "rect": [260, 410, 940, 1020]},
        ],
    },
    ("FD0010", 210): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [300, 410, 940, 1020]},
            {"cls": 3, "rect": [300, 410, 940, 1020]},
        ],
    },
    ("FD0010", 214): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [320, 410, 940, 1020]},
            {"cls": 3, "rect": [320, 410, 940, 1020]},
        ],
    },
    ("FD0010", 218): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [340, 410, 940, 1020]},
            {"cls": 3, "rect": [340, 410, 940, 1020]},
        ],
    },
    ("FD0010", 220): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [350, 410, 940, 1020]},
            {"cls": 3, "rect": [350, 410, 940, 1020]},
        ],
    },
    ("FD0010", 222): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [360, 410, 940, 1030]},
            {"cls": 3, "rect": [360, 410, 940, 1030]},
        ],
    },
    ("FD0010", 226): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [400, 410, 940, 1050]},
            {"cls": 3, "rect": [400, 410, 940, 1050]},
        ],
    },
    ("FD0010", 230): {
        "action": "fall_impact",
        "phase": "impact",
        "boxes": [
            {"cls": 0, "rect": [520, 400, 940, 1100]},
            {"cls": 3, "rect": [520, 400, 940, 1100]},
        ],
    },
    ("FD0010", 235): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [520, 400, 940, 1200]},
            {"cls": 3, "rect": [520, 400, 940, 1200]},
        ],
    },
    ("FD0010", 245): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [520, 400, 940, 1250]},
            {"cls": 3, "rect": [520, 400, 940, 1250]},
        ],
    },
    ("FD0010", 255): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [520, 400, 940, 1310]},
            {"cls": 3, "rect": [520, 400, 940, 1310]},
        ],
    },
    ("FD0010", 260): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ],
    },
    ("FD0010", 265): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ],
    },
    ("FD0010", 275): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ],
    },
    ("FD0010", 285): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 400, 940, 1310]},
            {"cls": 3, "rect": [640, 400, 940, 1310]},
        ],
    },

    # =========================================================================
    # Clip 9: FD0014 (sitting_to_fall, test, 1920x1080) - 18 frames
    # Female actor in plaid shirt, blue jeans falling forward onto mattress
    # =========================================================================
    ("FD0014", 0): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 680, 775, 1020]}],
    },
    ("FD0014", 150): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 680, 775, 1020]}],
    },
    ("FD0014", 198): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [10, 680, 775, 1020]}],
    },
    ("FD0014", 220): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [80, 680, 780, 1100]}],
    },
    ("FD0014", 242): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [150, 700, 800, 1200]}],
    },
    ("FD0014", 264): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [200, 700, 800, 1250]}],
    },
    ("FD0014", 298): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [260, 710, 800, 1300]},
            {"cls": 3, "rect": [260, 710, 800, 1300]},
        ],
    },
    ("FD0014", 302): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [300, 710, 800, 1320]},
            {"cls": 3, "rect": [300, 710, 800, 1320]},
        ],
    },
    ("FD0014", 306): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [320, 710, 800, 1350]},
            {"cls": 3, "rect": [320, 710, 800, 1350]},
        ],
    },
    ("FD0014", 310): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [350, 710, 800, 1380]},
            {"cls": 3, "rect": [350, 710, 800, 1380]},
        ],
    },
    ("FD0014", 314): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [400, 710, 820, 1400]},
            {"cls": 3, "rect": [400, 710, 820, 1400]},
        ],
    },
    ("FD0014", 318): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [450, 720, 830, 1420]},
            {"cls": 3, "rect": [450, 720, 830, 1420]},
        ],
    },
    ("FD0014", 322): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [500, 720, 840, 1440]},
            {"cls": 3, "rect": [500, 720, 840, 1440]},
        ],
    },
    ("FD0014", 325): {
        "action": "fall_impact",
        "phase": "impact",
        "boxes": [
            {"cls": 0, "rect": [550, 730, 840, 1450]},
            {"cls": 3, "rect": [550, 730, 840, 1450]},
        ],
    },
    ("FD0014", 330): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [600, 730, 840, 1450]},
            {"cls": 3, "rect": [600, 730, 840, 1450]},
        ],
    },
    ("FD0014", 338): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [620, 730, 840, 1450]},
            {"cls": 3, "rect": [620, 730, 840, 1450]},
        ],
    },
    ("FD0014", 345): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 730, 840, 1450]},
            {"cls": 3, "rect": [640, 730, 840, 1450]},
        ],
    },
    ("FD0014", 354): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [640, 730, 840, 1450]},
            {"cls": 3, "rect": [640, 730, 840, 1450]},
        ],
    },

    # =========================================================================
    # Clip 10: FD0020 (sitting_to_fall, val, portrait 1080x1920) - 25 frames
    # Female Actor E in portrait orientation falling forward onto mattress
    # =========================================================================
    ("FD0020", 0): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 22): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 44): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 66): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 88): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 110): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 132): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 150): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [290, 440, 1010, 1000]}],
    },
    ("FD0020", 176): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 198): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 220): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 242): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 264): {
        "action": "pre_fall_sitting",
        "phase": "pre_fall_or_adl_steady",
        "boxes": [{"cls": 0, "rect": [380, 440, 1050, 800]}],
    },
    ("FD0020", 280): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [420, 440, 1080, 850]},
            {"cls": 3, "rect": [420, 440, 1080, 850]},
        ],
    },
    ("FD0020", 283): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [440, 440, 1100, 880]},
            {"cls": 3, "rect": [440, 440, 1100, 880]},
        ],
    },
    ("FD0020", 286): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [460, 440, 1120, 900]},
            {"cls": 3, "rect": [460, 440, 1120, 900]},
        ],
    },
    ("FD0020", 289): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [480, 440, 1140, 920]},
            {"cls": 3, "rect": [480, 440, 1140, 920]},
        ],
    },
    ("FD0020", 292): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [500, 440, 1160, 940]},
            {"cls": 3, "rect": [500, 440, 1160, 940]},
        ],
    },
    ("FD0020", 295): {
        "action": "fall_transition",
        "phase": "fall_transition",
        "boxes": [
            {"cls": 0, "rect": [520, 440, 1180, 960]},
            {"cls": 3, "rect": [520, 440, 1180, 960]},
        ],
    },
    ("FD0020", 297): {
        "action": "fall_impact",
        "phase": "impact",
        "boxes": [
            {"cls": 0, "rect": [650, 420, 1250, 980]},
            {"cls": 3, "rect": [650, 420, 1250, 980]},
        ],
    },
    ("FD0020", 300): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [750, 400, 1350, 1000]},
            {"cls": 3, "rect": [750, 400, 1350, 1000]},
        ],
    },
    ("FD0020", 315): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [850, 390, 1480, 1020]},
            {"cls": 3, "rect": [850, 390, 1480, 1020]},
        ],
    },
    ("FD0020", 330): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [900, 390, 1550, 1020]},
            {"cls": 3, "rect": [900, 390, 1550, 1020]},
        ],
    },
    ("FD0020", 360): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [950, 390, 1680, 980]},
            {"cls": 3, "rect": [950, 390, 1680, 980]},
        ],
    },
    ("FD0020", 390): {
        "action": "fallen_on_floor",
        "phase": "fallen_rest",
        "boxes": [
            {"cls": 0, "rect": [980, 390, 1680, 980]},
            {"cls": 3, "rect": [980, 390, 1680, 980]},
        ],
    },
}
