import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data/processed/dfire_missing_label_full_scan/candidates.csv"
REMEDIATION_MANIFEST_CSV = ROOT / "data/processed/dfire_remediated/remediation_manifest.csv"
REMEDIATED_DIR = ROOT / "data/processed/dfire_remediated"


def main():
    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        all_cands = list(csv.DictReader(f))

    with open(REMEDIATION_MANIFEST_CSV, "r", encoding="utf-8") as f:
        existing_cids = {r["candidate_id"] for r in csv.DictReader(f)}

    p3_cands = [c for c in all_cands if c["candidate_id"] not in existing_cids]

    # Helmets in LOW tier
    low_helmets = [c for c in p3_cands if c["class_name"] == "helmet"]
    print(f"=== LOW Helmets ({len(low_helmets)}) ===")
    
    # Check what images have existing persons or helmets
    helmet_img_contexts = []
    for h in low_helmets:
        split = h["split"]
        img = h["image"]
        lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img).stem}.txt"
        existing_classes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing_classes.append(int(line.split()[0]))
        has_person = 0 in existing_classes
        has_helmet = 1 in existing_classes
        has_fire = 4 in existing_classes
        has_smoke = 5 in existing_classes
        helmet_img_contexts.append((has_person, has_helmet, has_fire, has_smoke))

    print("LOW Helmets image context breakdown:")
    print(f"  Images with existing person: {sum(1 for c in helmet_img_contexts if c[0])} / {len(low_helmets)}")
    print(f"  Images with existing helmet: {sum(1 for c in helmet_img_contexts if c[1])} / {len(low_helmets)}")
    print(f"  Images with NO person: {sum(1 for c in helmet_img_contexts if not c[0])} / {len(low_helmets)}")

    # Low persons
    low_persons = [c for c in p3_cands if c["class_name"] == "person" and c["tier"] == "LOW"]
    print(f"\n=== LOW Persons ({len(low_persons)}) ===")
    low_p_contexts = []
    for p in low_persons:
        split = p["split"]
        img = p["image"]
        lbl_p = REMEDIATED_DIR / "labels" / split / f"{Path(img).stem}.txt"
        existing_classes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing_classes.append(int(line.split()[0]))
        low_p_contexts.append((0 in existing_classes, 4 in existing_classes, 5 in existing_classes))

    print(f"  Images with existing person: {sum(1 for c in low_p_contexts if c[0])} / {len(low_persons)}")
    print(f"  Images with NO existing person: {sum(1 for c in low_p_contexts if not c[0])} / {len(low_persons)}")


if __name__ == "__main__":
    main()
