"""Prepare SCFD Stage 2 Pilot Reproducible Group-Aware Splits and Packaging.

Ensures:
1. All 7 owner-signed cross-label clusters remain atomic within their assigned split.
2. All 84 near-duplicate review pairs (connected components) remain atomic.
3. Target ratios (70% train, 15% val, 15% test) and binary class balance (1:1) are achieved.
4. Materializes portable package under data/processed/scfd_pilot/ without altering data/raw/.
"""

import csv
import json
import os
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCFD_PROCESSED = REPO_ROOT / "data" / "processed" / "scfd_pilot"
RAW_SCFD = REPO_ROOT / "data" / "raw" / "scfd"

OWNER_CLUSTERS = [
    ("Cluster_A", ["scfd_fight_fi046", "scfd_fight_fi047", "scfd_fight_fi049", "scfd_fight_fi050", "scfd_non_fight_nofi023", "scfd_non_fight_nofi024"]),
    ("Cluster_B", ["scfd_fight_fi065", "scfd_fight_fi066", "scfd_fight_fi067", "scfd_non_fight_nofi097", "scfd_non_fight_nofi098"]),
    ("Cluster_C", ["scfd_fight_fi073", "scfd_fight_fi074", "scfd_non_fight_nofi147"]),
    ("Cluster_D", ["scfd_fight_fi059", "scfd_non_fight_nofi148"]),
    ("Cluster_E", ["scfd_fight_fi107", "scfd_non_fight_nofi027"]),
    ("Cluster_F", ["scfd_fight_fi116", "scfd_non_fight_nofi051"]),
    ("Cluster_G", ["scfd_fight_fi122", "scfd_fight_fi123", "scfd_non_fight_nofi055", "scfd_non_fight_nofi056"]),
]

OWNER_CLIPS_NON_FIGHT = {
    "scfd_non_fight_nofi011",
    "scfd_non_fight_nofi052",
    "scfd_non_fight_nofi095",
    "scfd_non_fight_nofi115",
    "scfd_non_fight_nofi116",
}

def load_data():
    # 1. Inventory
    inventory = {}
    with open(SCFD_PROCESSED / "scfd_pilot_inventory.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inventory[row["clip_id"]] = row

    # 2. Worker visual review
    worker_reviews = {}
    with open(SCFD_PROCESSED / "scfd_worker_visual_review.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            worker_reviews[row["clip_id"]] = row

    # 3. Near-dup pairs
    near_dup_pairs = []
    with open(SCFD_PROCESSED / "scfd_worker_near_duplicate_review.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            near_dup_pairs.append(row)

    return inventory, worker_reviews, near_dup_pairs

def build_connected_components(all_clips, near_dup_pairs):
    adj = defaultdict(set)
    for c in all_clips:
        adj[c] = set()

    # Add owner cluster edges
    for cname, members in OWNER_CLUSTERS:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                adj[members[i]].add(members[j])
                adj[members[j]].add(members[i])

    # Add near-dup edges
    for row in near_dup_pairs:
        if row["plausibly_same_scene_or_group"] == "True":
            c1, c2 = row["clip_id_1"], row["clip_id_2"]
            adj[c1].add(c2)
            adj[c2].add(c1)

    visited = set()
    components = []
    for c in sorted(all_clips):
        if c not in visited:
            comp = []
            queue = [c]
            visited.add(c)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in sorted(adj[curr]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(comp))

    return components

def find_optimal_split(components, inventory):
    # Form component objects
    comp_list = []
    for idx, comp in enumerate(components):
        f_count = sum(1 for c in comp if inventory[c]["binary_source_label"] == "fight")
        nf_count = sum(1 for c in comp if inventory[c]["binary_source_label"] == "non_fight")
        comp_list.append({
            "idx": idx,
            "clips": comp,
            "fight": f_count,
            "non_fight": nf_count,
            "size": len(comp)
        })

    multi = [c for c in comp_list if c["size"] > 1]
    single_f = [c for c in comp_list if c["size"] == 1 and c["fight"] == 1]
    single_nf = [c for c in comp_list if c["size"] == 1 and c["non_fight"] == 1]

    # Target counts:
    # Train: 210 clips (105 F, 105 NF)
    # Val: 45 clips (22 F, 23 NF)
    # Test: 45 clips (23 F, 22 NF)
    target_f = {"train": 105, "val": 22, "test": 23}
    target_nf = {"train": 105, "val": 23, "test": 22}

    # Deterministic search
    for seed in range(1000):
        rng = random.Random(seed)
        shuffled_multi = list(multi)
        rng.shuffle(shuffled_multi)

        splits = {"train": [], "val": [], "test": []}
        curr_f = {"train": 0, "val": 0, "test": 0}
        curr_nf = {"train": 0, "val": 0, "test": 0}

        possible = True
        for m in shuffled_multi:
            # Candidates that can accommodate this multi-component
            candidates = []
            for s in ("val", "test", "train"):
                if (curr_f[s] + m["fight"] <= target_f[s]) and (curr_nf[s] + m["non_fight"] <= target_nf[s]):
                    candidates.append(s)
            if not candidates:
                possible = False
                break
            chosen = rng.choice(candidates)
            splits[chosen].append(m)
            curr_f[chosen] += m["fight"]
            curr_nf[chosen] += m["non_fight"]

        if not possible:
            continue

        # Check singletons fill
        rem_f = {s: target_f[s] - curr_f[s] for s in ("train", "val", "test")}
        rem_nf = {s: target_nf[s] - curr_nf[s] for s in ("train", "val", "test")}

        if all(v >= 0 for v in rem_f.values()) and sum(rem_f.values()) == len(single_f) and \
           all(v >= 0 for v in rem_nf.values()) and sum(rem_nf.values()) == len(single_nf):
            
            # Found exact solution!
            print(f"Optimal split found with seed={seed}")
            
            # Assign singletons deterministically
            shuffled_sf = list(single_f)
            shuffled_snf = list(single_nf)
            rng.shuffle(shuffled_sf)
            rng.shuffle(shuffled_snf)

            idx_f = 0
            idx_nf = 0
            for s in ("train", "val", "test"):
                count_f = rem_f[s]
                count_nf = rem_nf[s]
                splits[s].extend(shuffled_sf[idx_f : idx_f + count_f])
                splits[s].extend(shuffled_snf[idx_nf : idx_nf + count_nf])
                idx_f += count_f
                idx_nf += count_nf

            return splits

    raise RuntimeError("Could not find exact split assignment!")

def main():
    print("Loading data...")
    inventory, worker_reviews, near_dup_pairs = load_data()
    all_clips = sorted(inventory.keys())

    print("Building connected components (scene groups)...")
    components = build_connected_components(all_clips, near_dup_pairs)
    print(f"Total scene groups (connected components): {len(components)}")

    # Assign stable scene_group_id
    # Sort multi-components first, then singletons
    components.sort(key=lambda c: (-len(c), c[0]))
    clip_to_group = {}
    group_to_clips = {}
    for idx, comp in enumerate(components):
        gid = f"scene_grp_{idx+1:03d}"
        group_to_clips[gid] = comp
        for c in comp:
            clip_to_group[c] = gid

    print("Optimizing train/val/test assignment...")
    splits = find_optimal_split(components, inventory)

    clip_to_split = {}
    for s_name, comp_items in splits.items():
        for comp in comp_items:
            for c in comp["clips"]:
                clip_to_split[c] = s_name

    # Verify counts
    split_counts = {s: Counter() for s in ("train", "val", "test")}
    for c, s in clip_to_split.items():
        lbl = inventory[c]["binary_source_label"]
        split_counts[s][lbl] += 1

    print("\n--- Split Verification ---")
    for s in ("train", "val", "test"):
        total = sum(split_counts[s].values())
        print(f"Split {s:>5}: total={total:3d} (fight={split_counts[s]['fight']:3d}, non_fight={split_counts[s]['non_fight']:3d})")

    # Verify zero leakage across groups
    groups_by_split = defaultdict(set)
    for c, s in clip_to_split.items():
        groups_by_split[s].add(clip_to_group[c])

    assert not (groups_by_split["train"] & groups_by_split["val"]), "Train/Val leakage!"
    assert not (groups_by_split["train"] & groups_by_split["test"]), "Train/Test leakage!"
    assert not (groups_by_split["val"] & groups_by_split["test"]), "Val/Test leakage!"
    print("PASS: Zero scene group leakage across train/val/test splits.")

    # Verify owner clusters intact
    for cname, members in OWNER_CLUSTERS:
        member_splits = {clip_to_split[m] for m in members}
        assert len(member_splits) == 1, f"Owner {cname} split across {member_splits}!"
        print(f"PASS: {cname} (size {len(members)}) intact in split '{list(member_splits)[0]}'")

    # Write manifests
    print("\nWriting manifests...")
    for s in ("train", "val", "test"):
        manifest_path = SCFD_PROCESSED / f"{s}_manifest.csv"
        rows = []
        for c in sorted(all_clips):
            if clip_to_split[c] == s:
                inv = inventory[c]
                rev = worker_reviews[c]
                rows.append({
                    "clip_id": c,
                    "split": s,
                    "scene_group_id": clip_to_group[c],
                    "binary_source_label": inv["binary_source_label"],
                    "label_id": 1 if inv["binary_source_label"] == "fight" else 0,
                    "relative_path": inv["relative_path"],
                    "byte_size": inv["byte_size"],
                    "sha256": inv["sha256"],
                    "duration_sec": inv["duration_sec"],
                    "fps": inv["fps"],
                    "frame_count": inv["frame_count"],
                    "width": inv["width"],
                    "height": inv["height"],
                    "hard_negative_category": rev["hard_negative_category"],
                    "qa_status": "OWNER_HUMAN_QA_APPROVED",
                    "owner_approved": True,
                })
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {manifest_path.name} ({len(rows)} rows)")

    # Canonical scfd_pilot_manifest.json
    manifest_json_path = SCFD_PROCESSED / "scfd_pilot_manifest.json"
    clips_json = []
    for c in sorted(all_clips):
        inv = inventory[c]
        rev = worker_reviews[c]
        s = clip_to_split[c]
        lbl_id = 1 if inv["binary_source_label"] == "fight" else 0
        lbl_name = inv["binary_source_label"]
        t_onset = float(rev["t_onset"]) if rev["t_onset"] != "N/A" else 0.0
        t_cessation = float(rev["t_cessation"]) if rev["t_cessation"] != "N/A" else float(inv["duration_sec"])

        clips_json.append({
            "clip_id": c,
            "relative_path": inv["relative_path"],
            "file_sha256": inv["sha256"],
            "split": s,
            "scene_group_id": clip_to_group[c],
            "label": lbl_id,
            "label_name": lbl_name,
            "duration_sec": float(inv["duration_sec"]),
            "total_frames": int(inv["frame_count"]),
            "fps": float(inv["fps"]),
            "resolution": [int(inv["width"]), int(inv["height"])],
            "temporal_event_range": [t_onset, t_cessation],
            "hard_negative_category": rev["hard_negative_category"] if lbl_name == "non_fight" else None,
            "qa_status": "OWNER_HUMAN_QA_APPROVED",
        })

    import hashlib
    # Compute content hash of clips
    clips_bytes = json.dumps(clips_json, sort_keys=True).encode("utf-8")
    manifest_sha = hashlib.sha256(clips_bytes).hexdigest()

    manifest_data = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "dataset_name": "scfd_pilot_temporal_fight",
        "dataset_version": "1.0.0-pilot",
        "rights_status": "UNVERIFIED_OWNER_ACCEPTED_RISK",
        "owner_risk_acceptance": True,
        "total_clips": 300,
        "class_mapping": {
            "0": "non_fight",
            "1": "fight"
        },
        "splits": {
            "train": 210,
            "val": 45,
            "test": 45
        },
        "input_sampling_spec": {
            "frames_per_clip": 16,
            "sampling_method": "uniform_temporal",
            "target_resolution": [224, 224],
            "clip_duration_seconds": 2.0
        },
        "manifest_sha256": manifest_sha,
        "clips": clips_json
    }

    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  Wrote {manifest_json_path.name}")

    # Update scfd_pilot_qa_queue.csv
    print("\nUpdating scfd_pilot_qa_queue.csv...")
    qa_queue_rows = []
    with open(SCFD_PROCESSED / "scfd_pilot_qa_queue.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["clip_id"]
            rev = worker_reviews[cid]
            row["scene_group_id"] = clip_to_group[cid]
            row["split"] = clip_to_split[cid]
            row["verified_label"] = rev["worker_verified_label"]
            row["t_onset"] = rev["t_onset"]
            row["t_cessation"] = rev["t_cessation"]
            row["hard_negative_category"] = rev["hard_negative_category"]

            if cid in OWNER_CLIPS_NON_FIGHT:
                row["qa_status"] = "OWNER_HUMAN_QA_APPROVED"
                row["reviewer"] = "project_owner"
                row["reviewed_at"] = "2026-09-27"
                row["notes"] = "Project owner human sign-off (2026-09-27): confirmed non-fight under clip-level physical-fight definition."
            else:
                row["qa_status"] = "OWNER_HUMAN_QA_APPROVED"
                row["reviewer"] = "project_owner"
                row["reviewed_at"] = "2026-09-27"
                row["notes"] = (
                    "Project owner human visual QA PASS (2026-09-27): owner visually inspected this clip and approved/pass "
                    "(genuine human QA, not just worker QA); verified label confirmed as recorded. Does not review raw "
                    f"rights/provenance or approve training. Worker evidence: {rev['visual_evidence']}"
                )

            qa_queue_rows.append(row)

    with open(SCFD_PROCESSED / "scfd_pilot_qa_queue.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qa_queue_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qa_queue_rows)
    print(f"  Updated scfd_pilot_qa_queue.csv ({len(qa_queue_rows)} rows)")

    # Materialize train/val/test folders
    print("\nMaterializing portable training package (train/val/test clip copies)...")
    copied_bytes = 0
    copied_count = 0
    for s in ("train", "val", "test"):
        split_dir = SCFD_PROCESSED / s
        split_dir.mkdir(parents=True, exist_ok=True)

    for cid in sorted(all_clips):
        s = clip_to_split[cid]
        src_path = REPO_ROOT / inventory[cid]["relative_path"]
        dst_path = SCFD_PROCESSED / s / f"{cid}.mp4"
        if not dst_path.exists() or dst_path.stat().st_size != int(inventory[cid]["byte_size"]):
            shutil.copy2(src_path, dst_path)
            copied_bytes += dst_path.stat().st_size
            copied_count += 1

    print(f"  Materialized {len(all_clips)} clips across train/val/test ({copied_bytes / (1024*1024):.2f} MB copied)")
    print("\nDone!")

if __name__ == "__main__":
    main()
