import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
sys.path.insert(0, str(ROOT))
from scripts.test_phase3_evaluator import evaluate_candidate

with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
    all_cands = list(csv.DictReader(f))

with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
    existing_cids = {r["candidate_id"] for r in csv.DictReader(f)}

p3_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]

batch_size = 250
num_batches = (len(p3_cands) + batch_size - 1) // batch_size

# Pre-cache image sizes and existing labels
img_cache = {}
for c in p3_cands:
    key = (c["split"], c["image"])
    if key not in img_cache:
        img_p = REMEDIATED_DIR / "images" / c["split"] / c["image"]
        with Image.open(img_p) as im:
            w_img, h_img = im.size
        lbl_p = REMEDIATED_DIR / "labels" / c["split"] / f"{Path(c['image']).stem}.txt"
        lbl_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts = line.split()
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    lbl_boxes.append((cls_id, ((xc - w/2)*w_img, (yc - h/2)*h_img, (xc + w/2)*w_img, (yc + h/2)*h_img)))
        img_cache[key] = (w_img, h_img, lbl_boxes)

total_verdicts = Counter()
tier_class_verdicts = defaultdict(Counter)

for b_idx in range(num_batches):
    b_start = b_idx * batch_size
    b_end = min(len(p3_cands), (b_idx + 1) * batch_size)
    batch = p3_cands[b_start:b_end]
    b_verdicts = Counter()
    for c in batch:
        w_img, h_img, existing_labels = img_cache[(c["split"], c["image"])]
        v, act, new_box, notes = evaluate_candidate(c, w_img, h_img, existing_labels, {})
        b_verdicts[v] += 1
        total_verdicts[v] += 1
        tier_class_verdicts[(c["tier"], c["class_name"])][v] += 1
    print(f"Batch {b_idx + 1:02d} ({b_start:4d}..{b_end:4d}, n={len(batch):3d}): {dict(b_verdicts)}")

print(f"\nTotal Verdicts: {dict(total_verdicts)}")
print("\nBy Tier and Class:")
for (tier, cname), v_cnt in sorted(tier_class_verdicts.items()):
    print(f"  {tier} {cname}: {dict(v_cnt)}")
