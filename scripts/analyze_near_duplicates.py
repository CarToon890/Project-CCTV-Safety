import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

sys.path.insert(0, ".")
from cctv_safety.dataset import perceptual_hash

ROOT = Path("data/processed/stage1_unified")
IMG_DIR = ROOT / "images"

print("=" * 80)
print("STAGE 1 UNIFIED DATASET: NEAR-DUPLICATE INTEGRITY ANALYSIS")
print("=" * 80)

# Collect all images by split
images_by_split = {}
all_images = []
for split in ("train", "val", "test"):
    s_dir = IMG_DIR / split
    s_imgs = sorted([p for p in s_dir.glob("*") if p.is_file()])
    images_by_split[split] = s_imgs
    all_images.extend((split, p) for p in s_imgs)
    print(f"Split {split:<5}: {len(s_imgs)} images")

print(f"Total images: {len(all_images)}")

# Perceptual hash helper
def compute_hash(item):
    split, path = item
    try:
        h = perceptual_hash(path)
        return split, path, h
    except Exception as e:
        return split, path, None

t0 = time.time()
print(f"Computing 64-bit perceptual hashes (multi-threaded)...")
with ThreadPoolExecutor(max_workers=8) as ex:
    hashed_records = list(ex.map(compute_hash, all_images))
t1 = time.time()
print(f"Hashed {len(hashed_records)} images in {t1 - t0:.2f}s ({(t1 - t0)/len(hashed_records)*1000:.2f} ms/img)")

# Check cross-split and intra-split near duplicates (Hamming distance <= 5)
print("\nAnalyzing pairwise Hamming distances (threshold <= 5)...")
t2 = time.time()
intra_matches = defaultdict(int)
cross_matches = defaultdict(list)
total_near_dups = 0

# Convert to list for fast indexing
records = [(s, p.name, h) for s, p, h in hashed_records if h is not None]
n = len(records)

# Check cross-split matches specifically
# We organize hashes by split for ultra-fast cross-split checking
by_split = {"train": [], "val": [], "test": []}
for s, name, h in records:
    by_split[s].append((name, h))

cross_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
cross_near_dups = Counter()

for s1, s2 in cross_pairs:
    list1 = by_split[s1]
    list2 = by_split[s2]
    print(f"Checking {s1} ({len(list1)}) vs {s2} ({len(list2)})...")
    for name1, h1 in list1:
        for name2, h2 in list2:
            dist = (h1 ^ h2).bit_count()
            if dist <= 5:
                cross_near_dups[(s1, s2)] += 1
                if len(cross_matches[f"{s1}_{s2}"]) < 20:
                    cross_matches[f"{s1}_{s2}"].append((name1, name2, dist))

t3 = time.time()
print(f"Cross-split comparison completed in {t3 - t2:.2f}s")
print("\n--- CROSS-SPLIT NEAR-DUPLICATE RESULTS ---")
for s1, s2 in cross_pairs:
    cnt = cross_near_dups[(s1, s2)]
    print(f"  {s1} vs {s2} near duplicates (dist <= 5): {cnt}")
    if cnt > 0:
        print(f"    Sample matches (up to 5):")
        for m in cross_matches[f"{s1}_{s2}"][:5]:
            print(f"      {m[0]} <-> {m[1]} (dist={m[2]})")

# Summary report
summary = {
    "total_images_analyzed": len(all_images),
    "hash_algorithm": "perceptual_hash_64bit",
    "threshold_max_distance": 5,
    "cross_split_near_duplicates": {
        f"{s1}_vs_{s2}": cross_near_dups[(s1, s2)] for s1, s2 in cross_pairs
    },
    "cross_split_near_duplicate_samples": {
        k: [{"left": m[0], "right": m[1], "distance": m[2]} for m in v[:10]]
        for k, v in cross_matches.items()
    }
}

out_path = Path("reports/stage1_near_duplicate_analysis.json")
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nWrote near-duplicate analysis to {out_path}")
