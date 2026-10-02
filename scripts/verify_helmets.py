"""Verify all 126 helmets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "docs/audit_artifacts/dfire/phase2_features.json", "r", encoding="utf-8") as f:
    feats = json.load(f)

helmets = [it for it in feats if it["class_name"] == "helmet"]
print(f"Total helmets: {len(helmets)}")

remove_cids = {
    "CAND_001571", "CAND_001838", "CAND_001856", "CAND_002303", "CAND_002720",
    "CAND_002840", "CAND_002923", "CAND_003081", "CAND_003102", "CAND_003127",
    "CAND_003139", "CAND_003289", "CAND_003310", "CAND_003397", "CAND_003445"
}
uncertain_cids = {"CAND_003159", "CAND_003490"}
fix_cids = {"CAND_002015"}

pass_helmets = []
for h in helmets:
    cid = h["cid"]
    if cid in remove_cids:
        print(f"REMOVE Helmet #{h['rank']:03d} {cid} on {h['split']}/{h['image']}")
    elif cid in uncertain_cids:
        print(f"UNCERTAIN Helmet #{h['rank']:03d} {cid} on {h['split']}/{h['image']}")
    elif cid in fix_cids:
        print(f"FIX Helmet #{h['rank']:03d} {cid} on {h['split']}/{h['image']}")
    else:
        pass_helmets.append(h)

print(f"\nCounts: REMOVE={len(remove_cids)}, UNCERTAIN={len(uncertain_cids)}, FIX={len(fix_cids)}, PASS={len(pass_helmets)}")
print(f"Total = {len(remove_cids) + len(uncertain_cids) + len(fix_cids) + len(pass_helmets)}")
