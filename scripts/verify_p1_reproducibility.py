"""Verify reproducibility of Phase 1 candidate bboxes between manifest and raw scan."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest_csv = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
p1_adj_csv = ROOT / "docs/audit_artifacts/dfire/dfire_qa_label_adjustments.csv"

with open(manifest_csv, "r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))[:1569]

with open(p1_adj_csv, "r", encoding="utf-8") as f:
    adj_rows = list(csv.DictReader(f))

adj_by_cid = {r["candidate_id"]: r for r in adj_rows}

print(f"Loaded {len(manifest_rows)} Phase 1 manifest rows.")
print(f"Loaded {len(adj_rows)} Phase 1 adjustment rows.")

# Check statuses in manifest
status_counts = {}
for r in manifest_rows:
    s = r["status"]
    status_counts[s] = status_counts.get(s, 0) + 1
print("Manifest statuses:", status_counts)

# Check CAND_000122 in manifest
for r in manifest_rows:
    if r["candidate_id"] == "CAND_000122":
        print("CAND_000122 in manifest:", r)
        break

# Check REMOVED in manifest
removed_cids = [r["candidate_id"] for r in manifest_rows if r["status"] == "REMOVED_IN_VISUAL_QA"]
print(f"Removed cids ({len(removed_cids)}):", removed_cids)
