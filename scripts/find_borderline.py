"""Find borderline ambiguous candidates among Phase 2 items."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "docs/audit_artifacts/dfire/phase2_interim_eval.json", "r", encoding="utf-8") as f:
    evals = json.load(f)

print("Checking borderline candidates:")
for e in evals:
    conf = float(e["confidence"])
    cname = e["class_name"]
    cid = e["candidate_id"]
    split = e["split"]
    img = e["image"]
    # Look for very small candidates with conf < 0.45 in smoke
    if conf < 0.43:
        print(f"Rank #{e['queue_rank']:03d} {cid} ({cname} {conf:.3f}) on {split}/{img}: {e['verdict']} - {e['notes'][:60]}")
