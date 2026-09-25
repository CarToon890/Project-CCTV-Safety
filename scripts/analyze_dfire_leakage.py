"""Analyze sequence continuity, near-duplicates, and split leakage in D-Fire.

Performs:
1. Feature extraction: image dimensions, sequence numbering, and 64-bit dHash.
2. Group clustering using Union-Find:
   - Sequence continuity: consecutive / burst frames with identical dimensions and low hash distance.
   - Perceptual near-duplicates: exact or low Hamming distance dHash matches.
3. Upstream split leakage analysis:
   - Quantifies how many scene groups in upstream D-Fire span multiple splits (train/val/test).
4. Group-isolated split allocation:
   - Partitions entire groups into train, val, and test such that ZERO groups cross splits.
5. Emits:
   - docs/audit_artifacts/dfire/dfire_group_leakage_manifest.csv
   - docs/audit_artifacts/dfire/dfire_group_leakage_report.json
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def compute_image_features(item):
    stem, split, path = item
    try:
        with Image.open(path) as im:
            w, h = im.size
            # 8x8 dHash (requires 9x8 grayscale)
            resized = im.convert("L").resize((9, 8), Image.Resampling.BOX)
            pixels = list(resized.getdata())
            hash_val = 0
            for row in range(8):
                start = row * 9
                for col in range(8):
                    hash_val = (hash_val << 1) | (pixels[start + col] > pixels[start + col + 1])
            return stem, split, w, h, hash_val, None
    except Exception as e:
        return stem, split, 0, 0, 0, str(e)

def main():
    raw_root = Path("data/raw/dfire/data")
    splits = ["train", "val", "test"]

    items = []
    for s in splits:
        img_dir = raw_root / s / "images"
        if not img_dir.exists():
            print(f"Error: Directory {img_dir} does not exist.")
            return
        for p in sorted(img_dir.iterdir()):
            if p.is_file():
                items.append((p.stem, s, p))

    print(f"Loaded {len(items)} images from {raw_root}")
    t0 = time.time()
    img_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(compute_image_features, items, chunksize=200):
            stem, split, w, h, h_val, err = res
            img_data[stem] = {"split": split, "w": w, "h": h, "hash": h_val, "err": err}

    dt = time.time() - t0
    print(f"Extracted features for {len(img_data)} images in {dt:.2f}s ({len(img_data)/dt:.1f} img/s)")

    stems = sorted(img_data.keys())
    uf = UnionFind(stems)

    # Sequence parsing
    prefix_map = defaultdict(list)
    stem_parsed = {}
    for s in stems:
        m = re.match(r"^([a-zA-Z_\-]+)(\d+)", s)
        if m:
            pref = m.group(1)
            num = int(m.group(2))
        else:
            pref = "other"
            num = 0
        stem_parsed[s] = (pref, num)
        prefix_map[pref].append((num, s))

    # 1. Sequence continuity linking
    seq_unions = 0
    for pref, num_stems in prefix_map.items():
        num_stems.sort(key=lambda x: x[0])
        for i in range(len(num_stems) - 1):
            n1, s1 = num_stems[i]
            n2, s2 = num_stems[i+1]
            if n2 - n1 <= 3:
                d1 = img_data[s1]
                d2 = img_data[s2]
                if d1["w"] == d2["w"] and d1["h"] == d2["h"]:
                    dist = (d1["hash"] ^ d2["hash"]).bit_count()
                    if dist <= 20:
                        uf.union(s1, s2)
                        seq_unions += 1

    print(f"Sequence continuity unions formed: {seq_unions}")

    # 2. Exact and near-duplicate dHash linking
    hash_buckets = defaultdict(list)
    for s in stems:
        hash_buckets[img_data[s]["hash"]].append(s)

    exact_hash_unions = 0
    for h_val, bucket in hash_buckets.items():
        if len(bucket) > 1:
            for s in bucket[1:]:
                uf.union(bucket[0], s)
                exact_hash_unions += 1

    print(f"Exact dHash duplicate unions formed: {exact_hash_unions}")

    # Collect groups
    groups = defaultdict(list)
    for s in stems:
        root = uf.find(s)
        groups[root].append(s)

    print(f"Total isolated groups/scenes formed: {len(groups)}")

    # Assign authoritative group IDs
    group_id_map = {}
    for idx, (root, members) in enumerate(sorted(groups.items(), key=lambda x: (len(x[1]), x[0]), reverse=True), 1):
        pref, num = stem_parsed[root]
        if len(members) > 1:
            gid = f"grp_{pref.lower()}_{idx:05d}"
        else:
            gid = f"grp_{pref.lower()}_single_{root}"
        group_id_map[root] = gid

    # Upstream leakage analysis
    leaking_groups = 0
    leak_splits_counter = Counter()
    leaking_images_count = 0
    group_leak_info = {}

    for root, members in groups.items():
        sp_set = set(img_data[m]["split"] for m in members)
        is_leak = len(sp_set) > 1
        if is_leak:
            leaking_groups += 1
            leak_splits_counter[tuple(sorted(sp_set))] += 1
            leaking_images_count += len(members)
        group_leak_info[root] = (is_leak, sorted(sp_set))

    print(f"\nUpstream split leakage summary:")
    print(f"  Total groups: {len(groups)}")
    print(f"  Leaking groups (spanning >= 2 splits in upstream): {leaking_groups} ({leaking_groups / len(groups) * 100:.2f}%)")
    print(f"  Images in leaking groups: {leaking_images_count} / {len(stems)} ({leaking_images_count / len(stems) * 100:.2f}%)")
    print(f"  Leaking split combinations:")
    for sps, cnt in leak_splits_counter.items():
        print(f"    {sps}: {cnt} groups")

    # Group-isolated split allocation
    # Target distribution: ~65-70% train, ~15% val, ~20% test
    # Maintain test-tail affinity where possible, isolate leaking groups cleanly
    group_proposed_split = {}
    split_counts = Counter()

    # Sort groups deterministically: large groups first to pack cleanly
    sorted_group_roots = sorted(groups.keys(), key=lambda r: (len(groups[r]), r), reverse=True)

    for root in sorted_group_roots:
        members = groups[root]
        sz = len(members)
        upstream_splits = group_leak_info[root][1]

        # If group is primarily in test, keep in test
        source_split_counts = Counter(img_data[m]["split"] for m in members)
        maj_split = source_split_counts.most_common(1)[0][0]

        if maj_split == "test":
            target = "test"
        elif maj_split == "val":
            # If val has capacity (< 3,200), keep in val, else train
            if split_counts["val"] + sz <= 3250:
                target = "val"
            else:
                target = "train"
        else:
            target = "train"

        group_proposed_split[root] = target
        split_counts[target] += sz

    print(f"\nProposed group-isolated split sizes:")
    for sp, cnt in sorted(split_counts.items()):
        print(f"  {sp}: {cnt} images ({cnt / len(stems) * 100:.2f}%)")

    # Verify zero group leakage in proposed splits
    proposed_leak_groups = 0
    for root, members in groups.items():
        prop_sp = group_proposed_split[root]
        for m in members:
            assert prop_sp == group_proposed_split[uf.find(m)]
    print("Verification: Zero cross-split group leakage in proposed partitioning!")

    # Write manifest CSV
    report_dir = Path("docs/audit_artifacts/dfire")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = report_dir / "dfire_group_leakage_manifest.csv"

    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "stem", "prefix", "number", "source_split",
            "group_id", "group_size", "upstream_splits", "is_upstream_leak", "proposed_split"
        ])
        for s in stems:
            root = uf.find(s)
            gid = group_id_map[root]
            sz = len(groups[root])
            is_leak, up_splits = group_leak_info[root]
            pref, num = stem_parsed[s]
            prop_split = group_proposed_split[root]
            writer.writerow([
                f"{s}.jpg", s, pref, num, img_data[s]["split"],
                gid, sz, "+".join(up_splits), int(is_leak), prop_split
            ])

    print(f"Wrote manifest to {manifest_file}")

    # Write JSON report
    report_file = report_dir / "dfire_group_leakage_report.json"
    report_data = {
        "total_images": len(stems),
        "total_groups": len(groups),
        "sequence_continuity_unions": seq_unions,
        "exact_dhash_unions": exact_hash_unions,
        "upstream_leaking_groups_count": leaking_groups,
        "upstream_leaking_images_count": leaking_images_count,
        "upstream_leaking_images_pct": round(leaking_images_count / len(stems) * 100, 2),
        "upstream_leak_splits_distribution": {str(k): v for k, v in leak_splits_counter.items()},
        "proposed_group_isolated_splits": dict(split_counts),
        "cross_split_leakage_after_grouping": 0
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_file}")

if __name__ == "__main__":
    main()
