"""Verify Phase 1 candidates from manifest and candidates.csv."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cands_csv = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
manifest_csv = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"

with open(cands_csv, "r", encoding="utf-8") as f:
    cands_map = {r["candidate_id"]: r for r in csv.DictReader(f)}

with open(manifest_csv, "r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))

p1_manifest = manifest_rows[:1569]
p1_added = [r for r in p1_manifest if r["status"] in ("ADDED", "FIXED_IN_VISUAL_QA")]
p1_removed = [r for r in p1_manifest if r["status"] == "REMOVED_IN_VISUAL_QA"]

print(f"P1 total manifest: {len(p1_manifest)}")
print(f"P1 added/fixed: {len(p1_added)}")
print(f"P1 removed: {len(p1_removed)}")

added_classes = Counter(cands_map[r["candidate_id"]]["class_name"] for r in p1_added)
removed_classes = Counter(cands_map[r["candidate_id"]]["class_name"] for r in p1_removed)

print("P1 added classes:", dict(added_classes))
print("P1 removed classes:", dict(removed_classes))
