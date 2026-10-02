"""Find all tiny person candidates to visually inspect."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "docs/audit_artifacts/dfire/phase2_features.json", "r", encoding="utf-8") as f:
    feats = json.load(f)

tiny_persons = []
for item in feats:
    if item["class_name"] == "person":
        w = item["norm_w"]
        h = item["norm_h"]
        area = w * h
        if area < 0.003 or (item["conf"] < 0.43 and area < 0.008):
            tiny_persons.append(item)

print(f"Total tiny person candidates: {len(tiny_persons)}")
for it in tiny_persons:
    print(f"Rank #{it['rank']:03d} {it['cid']} ({it['conf']:.3f}) area={it['norm_w']*it['norm_h']:.5f} ({it['norm_w']:.3f}x{it['norm_h']:.3f}) on {it['split']}/{it['image']}")
