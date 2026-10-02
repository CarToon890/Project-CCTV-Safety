"""Find some PASS person items to inspect."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "docs/audit_artifacts/dfire/phase2_interim_eval.json", "r", encoding="utf-8") as f:
    evals = json.load(f)

pass_persons = [e for e in evals if e["class_name"] == "person" and e["verdict"] == "PASS"]
for p in pass_persons[::20]:
    print(f"Rank #{p['queue_rank']:03d} {p['candidate_id']} {p['split']}/{p['image']} conf={p['confidence']}")
