import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


def compute_iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def main():
    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        all_cands = list(csv.DictReader(f))

    with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
        existing_cids = {r["candidate_id"] for r in csv.DictReader(f)}

    p3_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]
    print(f"Total Phase 3 candidates: {len(p3_cands)}")

    # Cache image dimensions and existing labels
    img_dims = {}
    img_labels = {}

    for c in p3_cands:
        split = c["split"]
        img = c["image"]
        key = (split, img)
        if key not in img_dims:
            img_p = REMEDIATED_DIR / "images" / split / img
            with Image.open(img_p) as im:
                img_dims[key] = im.size
            
            lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img).stem}.txt"
            boxes = []
            if lbl_p.exists():
                w_img, h_img = img_dims[key]
                for line in lbl_p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        parts = line.split()
                        cls_id = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (xc - w/2) * w_img
                        y1 = (yc - h/2) * h_img
                        x2 = (xc + w/2) * w_img
                        y2 = (yc + h/2) * h_img
                        boxes.append((cls_id, (x1, y1, x2, y2), (xc, yc, w, h)))
            img_labels[key] = boxes

    # Analyze each candidate
    high_same_iou_count = 0
    high_smoke_iou_count = 0
    high_fire_iou_count = 0
    extreme_border_count = 0
    tiny_box_count = 0
    group_box_count = 0

    for c in p3_cands:
        split = c["split"]
        img = c["image"]
        cname = c["class_name"]
        target_cls = 1 if cname == "helmet" else 0
        w_img, h_img = img_dims[(split, img)]
        
        x1 = float(c["x1"])
        y1 = float(c["y1"])
        x2 = float(c["x2"])
        y2 = float(c["y2"])
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        aspect = bh / bw if bw > 0 else 0
        norm_w = bw / w_img
        norm_h = bh / h_img

        existing_boxes = img_labels[(split, img)]
        same_cls_boxes = [b[1] for b in existing_boxes if b[0] == target_cls]
        smoke_boxes = [b[1] for b in existing_boxes if b[0] == 5]
        fire_boxes = [b[1] for b in existing_boxes if b[0] == 4]

        max_same_iou = max((compute_iou((x1, y1, x2, y2), b) for b in same_cls_boxes), default=0.0)
        max_smoke_iou = max((compute_iou((x1, y1, x2, y2), b) for b in smoke_boxes), default=0.0)
        max_fire_iou = max((compute_iou((x1, y1, x2, y2), b) for b in fire_boxes), default=0.0)

        if max_same_iou >= 0.85:
            high_same_iou_count += 1
        if max_smoke_iou >= 0.50:
            high_smoke_iou_count += 1
        if max_fire_iou >= 0.50:
            high_fire_iou_count += 1

        is_border = (x1 <= 2.0 or y1 <= 2.0 or x2 >= w_img - 2.0 or y2 >= h_img - 2.0)
        if is_border and (aspect < 0.60 or aspect > 5.5 or norm_w < 0.02 or norm_h < 0.02):
            extreme_border_count += 1

        if (cname == "helmet" and (bw < 14 or bh < 14 or area < 200)) or (cname == "person" and (bw < 10 or bh < 16 or area < 200)):
            tiny_box_count += 1

    print(f"Candidates with same-class IoU >= 0.85 (duplicates): {high_same_iou_count}")
    print(f"Candidates with GT Smoke IoU >= 0.50: {high_smoke_iou_count}")
    print(f"Candidates with GT Fire IoU >= 0.50: {high_fire_iou_count}")
    print(f"Candidates with extreme border sliver truncation: {extreme_border_count}")
    print(f"Candidates with tiny box dimensions: {tiny_box_count}")


if __name__ == "__main__":
    main()
