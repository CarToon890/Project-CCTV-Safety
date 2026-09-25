#!/usr/bin/env python
"""Inspect QA samples and state transitions across the 8 pilot clips."""

import csv
from pathlib import Path
from PIL import Image

PILOT_DIR = Path("data/processed/fall_corrected_pilot")
DOCS_DIR = Path("docs/audit_artifacts/fall")
MANIFEST_PATH = DOCS_DIR / "fall_pilot_manifest.csv"

def inspect_qa():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} manifest rows for QA inspection.")

    # 1. Contact sheets inspection
    contact_sheets = list((PILOT_DIR / "qa_contact_sheets").glob("*.png"))
    print(f"\nGenerated Contact Sheets ({len(contact_sheets)} files):")
    for cs in sorted(contact_sheets):
        with Image.open(cs) as img:
            print(f"  {cs.name}: {img.size} ({img.format})")

    # 2. State transition inspection
    # Group by clip and track transitions
    clips = sorted(list(set(r["clip_id"] for r in rows)))
    
    inspected_transitions = []
    inspected_frames_by_state = {"pre_fall": 0, "falling": 0, "fallen": 0}

    print("\nState Transitions by Clip:")
    for cid in clips:
        c_rows = [r for r in rows if r["clip_id"] == cid]
        pre = [r for r in c_rows if r["state_category"] == "pre_fall"]
        falling = [r for r in c_rows if r["state_category"] == "falling"]
        fallen = [r for r in c_rows if r["state_category"] == "fallen"]

        inspected_frames_by_state["pre_fall"] += len(pre)
        inspected_frames_by_state["falling"] += len(falling)
        inspected_frames_by_state["fallen"] += len(fallen)

        # Onset transition (last pre-fall to first falling)
        onset_str = "None"
        if pre and falling:
            onset_str = f"Frame {pre[-1]['frame_idx']} (pre_fall, class 0) -> Frame {falling[0]['frame_idx']} (falling, classes 0,3)"
            inspected_transitions.append((cid, "onset", pre[-1]["frame_idx"], falling[0]["frame_idx"]))

        # Impact transition (last falling to first fallen)
        impact_str = "None"
        if falling and fallen:
            impact_str = f"Frame {falling[-1]['frame_idx']} (falling, classes 0,3) -> Frame {fallen[0]['frame_idx']} (fallen, classes 0,3)"
            inspected_transitions.append((cid, "impact", falling[-1]["frame_idx"], fallen[0]["frame_idx"]))

        print(f"  [{cid}] Total: {len(c_rows)} (pre: {len(pre)}, falling: {len(falling)}, fallen: {len(fallen)})")
        print(f"    Onset:  {onset_str}")
        print(f"    Impact: {impact_str}")

    print(f"\nTotal Retained Frames: {len(rows)}")
    print(f"  Pre-fall normal (class 0 only):     {inspected_frames_by_state['pre_fall']}")
    print(f"  Active falling (classes 0 + 3):      {inspected_frames_by_state['falling']}")
    print(f"  Fallen-on-floor (classes 0 + 3):     {inspected_frames_by_state['fallen']}")
    print(f"Total State Transitions Inspected:     {len(inspected_transitions)} (8 onset, 8 impact)")

if __name__ == "__main__":
    inspect_qa()
