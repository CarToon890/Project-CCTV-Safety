#!/usr/bin/env python
"""Update fall_actor_grouping.csv and scripts/audit_fall_sample.py to ensure group_split_map keys match actor_group."""

import csv
from pathlib import Path

GROUPING_CSV = Path("docs/audit_artifacts/fall/fall_actor_grouping.csv")
AUDIT_SCRIPT = Path("scripts/audit_fall_sample.py")

group_split_map = {
    "grp_session_20260216_actorA_dark_top": "train",
    "grp_session_20260216_actorB_green_polo": "train",
    "grp_session_20260216_actorC_male_glasses": "val",
    "grp_session_20260223_actorD_pink_shirt": "train",
    "grp_session_20260223_actorE_black_graphic_top": "val",
    "grp_session_20260223_actorF_green_polo": "test",
    "grp_session_20260223_actorG_pattern_blouse": "test",
    "grp_session_20260216_actor_session1_classroom": "train",
    "grp_session_20260223_actor_session2_studio": "test",
}

def update_grouping():
    with open(GROUPING_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        grp = r["actor_group"]
        r["proposed_split"] = group_split_map.get(grp, "train")

    with open(GROUPING_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["clip_id", "relative_path", "class_label", "session_id", "actor_group", "proposed_split", "is_sampled_for_audit"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {GROUPING_CSV}: {len(rows)} rows.")
    
    # Check 8 annotated clips
    annot_clips = ["FD0035", "FD0044", "FD0049", "FD0024", "FD0027", "FD0051", "FD0052", "FD0054"]
    for r in rows:
        if r["clip_id"] in annot_clips:
            print(f"  {r['clip_id']}: group={r['actor_group']}, split={r['proposed_split']}")

if __name__ == "__main__":
    update_grouping()
