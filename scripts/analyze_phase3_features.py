import csv
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
OVERLAYS_DIR = ROOT / "data/processed/dfire_missing_label_full_scan/overlays"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"

with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
    all_cands = list(csv.DictReader(f))

with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
    existing_cids = {r["candidate_id"] for r in csv.DictReader(f)}

phase3_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]
print(f"Total Phase 3 candidates: {len(phase3_cands)}")

# Check overlays availability
has_overlay_count = 0
for c in phase3_cands:
    tier = c["tier"]
    split = c["split"]
    img = c["image"]
    overlay_p = OVERLAYS_DIR / f"{tier}_{split}_{img}"
    if overlay_p.exists():
        has_overlay_count += 1

print(f"Overlays existing for Phase 3 candidates: {has_overlay_count} / {len(phase3_cands)}")

# Unique images
unique_imgs = {(c["split"], c["image"]) for c in phase3_cands}
print(f"Unique images across Phase 3 candidates: {len(unique_imgs)}")

# Check class breakdown
print("Class breakdown:")
for (tier, cname), cnt in Counter((c["tier"], c["class_name"]) for c in phase3_cands).items():
    print(f"  {tier} {cname}: {cnt}")

# Check confidence distribution
print("Confidence distribution:")
for tier in ["MEDIUM", "LOW"]:
    confs = [float(c["confidence"]) for c in phase3_cands if c["tier"] == tier]
    print(f"  {tier}: min={min(confs):.4f}, median={sorted(confs)[len(confs)//2]:.4f}, max={max(confs):.4f}")

# Check existing boxes in those images
total_existing_boxes = 0
for split, img in list(unique_imgs)[:10]:
    lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img).stem}.txt"
    if lbl_p.exists():
        lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_existing_boxes += len(lines)
print(f"Sample of 10 images existing label check passed.")
