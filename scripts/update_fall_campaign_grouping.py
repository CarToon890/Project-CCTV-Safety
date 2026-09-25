#!/usr/bin/env python
"""Update fall_actor_grouping.csv to isolate actor groups and prevent split leakage.

Refines groupings for campaign clips based on visual identity:
- FD0005, FD0020, FD0022 -> grp_session_20260223_actorE_black_graphic_top (val)
- FD0010 -> grp_session_20260216_actorC_male_glasses (val)
- FD0015 -> grp_session_20260223_actorD_pink_shirt (train)
- FD0018, FD0019 -> grp_session_20260223_actorF_green_polo (test)
- Sets is_sampled_for_audit = True for all 10 campaign clips (FD0001-FD0007, FD0010, FD0014, FD0020)
  while preserving True for the 8 pilot clips.
"""

import csv
from pathlib import Path

GROUPING_CSV = Path("docs/audit_artifacts/fall/fall_actor_grouping.csv")

REVISIONS = {
    # clip_id: (new_actor_group, new_proposed_split)
    "FD0005": ("grp_session_20260223_actorE_black_graphic_top", "val"),
    "FD0010": ("grp_session_20260216_actorC_male_glasses", "val"),
    "FD0015": ("grp_session_20260223_actorD_pink_shirt", "train"),
    "FD0018": ("grp_session_20260223_actorF_green_polo", "test"),
    "FD0019": ("grp_session_20260223_actorF_green_polo", "test"),
    "FD0020": ("grp_session_20260223_actorE_black_graphic_top", "val"),
    "FD0022": ("grp_session_20260223_actorE_black_graphic_top", "val"),
    "FD0002": ("grp_session_20260216_actor_session1_classroom", "train"),
    "FD0003": ("grp_session_20260216_actor_session1_classroom", "train"),
    "FD0004": ("grp_session_20260216_actor_session1_classroom", "train"),
}

CAMPAIGN_10_CLIPS = {
    "FD0001", "FD0002", "FD0003", "FD0004", "FD0005", "FD0006",
    "FD0007", "FD0010", "FD0014", "FD0020"
}

PILOT_8_CLIPS = {
    "FD0035", "FD0044", "FD0049", "FD0024", "FD0027", "FD0051", "FD0052", "FD0054"
}

def update_grouping():
    with open(GROUPING_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        cid = r["clip_id"]
        if cid in REVISIONS:
            new_grp, new_split = REVISIONS[cid]
            r["actor_group"] = new_grp
            r["proposed_split"] = new_split

        if cid in CAMPAIGN_10_CLIPS or cid in PILOT_8_CLIPS:
            r["is_sampled_for_audit"] = "True"

    with open(GROUPING_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["clip_id", "relative_path", "class_label", "session_id", "actor_group", "proposed_split", "is_sampled_for_audit"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {GROUPING_CSV} successfully.")
    
    # Audit group to split consistency across all rows
    group_splits = {}
    leakage = False
    for r in rows:
        grp = r["actor_group"]
        sp = r["proposed_split"]
        if grp not in group_splits:
            group_splits[grp] = set()
        group_splits[grp].add(sp)
    
    print("\n--- Group to Split Consistency Audit ---")
    for grp, splits in sorted(group_splits.items()):
        print(f"  {grp} -> {splits}")
        if len(splits) > 1:
            print(f"  [ERROR] Leakage detected in {grp}: spans multiple splits {splits}")
            leakage = True
            
    if not leakage:
        print("\n[SUCCESS] Zero actor group leakage across splits!")

if __name__ == "__main__":
    update_grouping()
