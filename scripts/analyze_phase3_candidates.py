import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
PHASE2_OWNER_CSV = ROOT / "docs/audit_artifacts/dfire/dfire_phase2_owner_queue.csv"

with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
    all_cands = list(csv.DictReader(f))

with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
    existing_manifest = list(csv.DictReader(f))

existing_cids = {r["candidate_id"] for r in existing_manifest}

with open(PHASE2_OWNER_CSV, "r", encoding="utf-8") as f:
    p2_owner = list(csv.DictReader(f))

print(f"Total candidates in candidates.csv: {len(all_cands)}")
print(f"Existing rows in remediation_manifest: {len(existing_manifest)}")
print(f"Unique candidate_ids in manifest: {len(existing_cids)}")
print(f"Phase 2 owner queue rows: {[r['candidate_id'] for r in p2_owner]}")

remaining_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]
print(f"Remaining candidates to process: {len(remaining_cands)}")

rem_tier_class = Counter((c["tier"], c["class_name"]) for c in remaining_cands)
print("Remaining breakdown by tier and class:")
for (tier, cname), cnt in sorted(rem_tier_class.items()):
    print(f"  {tier} - {cname}: {cnt}")

rem_splits = Counter(c["split"] for c in remaining_cands)
print(f"Remaining by split: {dict(rem_splits)}")
