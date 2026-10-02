import csv
from collections import Counter
from pathlib import Path
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"
sys.path.insert(0, str(ROOT))
from scripts.test_phase3_evaluator_v2 import evaluate_candidate, compute_iou
from collections import defaultdict

with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
    all_cands = list(csv.DictReader(f))

with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
    existing_cids = {r["candidate_id"] for r in csv.DictReader(f)}

p3_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]

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

newly_added = defaultdict(list)
added_by_class = Counter()
added_by_split = Counter()
added_by_split_class = defaultdict(Counter)

tier_class_verdicts = defaultdict(Counter)

for c in p3_cands:
    w_img, h_img, existing_labels = img_cache[(c["split"], c["image"])]
    v, act, new_box, notes = evaluate_candidate(c, w_img, h_img, existing_labels, newly_added)
    tier_class_verdicts[(c["tier"], c["class_name"])][v] += 1
    if v in ("PASS", "FIX"):
        ccls = 1 if c["class_name"] == "helmet" else 0
        box_to_record = new_box if new_box else (
            (float(c["x1"])+float(c["x2"]))/(2*w_img),
            (float(c["y1"])+float(c["y2"]))/(2*h_img),
            (float(c["x2"])-float(c["x1"]))/w_img,
            (float(c["y2"])-float(c["y1"]))/h_img
        )
        px1 = (box_to_record[0] - box_to_record[2]/2) * w_img
        py1 = (box_to_record[1] - box_to_record[3]/2) * h_img
        px2 = (box_to_record[0] + box_to_record[2]/2) * w_img
        py2 = (box_to_record[1] + box_to_record[3]/2) * h_img
        newly_added[(c["split"], c["image"])].append((ccls, (px1, py1, px2, py2)))
        added_by_class[ccls] += 1
        added_by_split[c["split"]] += 1
        added_by_split_class[c["split"]][ccls] += 1

print("Added by class:")
print(f"  Class 0 (person): {added_by_class[0]}")
print(f"  Class 1 (helmet): {added_by_class[1]}")
print(f"Total added: {sum(added_by_class.values())}")

print("\nAdded by split:")
for s in ("train", "val", "test"):
    print(f"  {s}: {added_by_split[s]} (person: {added_by_split_class[s][0]}, helmet: {added_by_split_class[s][1]})")

print("\nVerdicts by Tier and Class:")
for (t, cn), v_cnt in sorted(tier_class_verdicts.items()):
    print(f"  {t} {cn}: {dict(v_cnt)}")
