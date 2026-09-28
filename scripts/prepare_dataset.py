#!/usr/bin/env python
"""Combine approved, exhaustively-labelled YOLO sources without group leakage.

Provisional Stage 1 unified dataset preparation pipeline.
Enforces 6 canonical spatial classes, rejects 'fight', applies exclusions,
preserves true group IDs across splits, and strictly verifies all inputs before copying.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_safety.dataset import assign_group_splits, copy_prepared_sample, parse_yolo_label
from cctv_safety.schema import CLASS_TO_ID, DETECTOR_SCHEMA_VERSION, IMAGE_SUFFIXES


def resolve_image(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.exists():
        return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = root / f"{relative}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(relative)


def load_exclusions(exclusion_path: Path | None) -> set[str]:
    """Load filenames marked for exclusion from training."""
    if not exclusion_path or not exclusion_path.exists():
        return set()
    excluded = set()
    with exclusion_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decision = row.get("decision", "").strip().upper()
            if "EXCLUDE" in decision:
                fname = row.get("filename", "").strip()
                if fname:
                    excluded.add(fname)
                    excluded.add(Path(fname).name)
                    excluded.add(Path(fname).stem)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare unified Stage 1 dataset.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/datasets.local.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage1_unified"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Verify sources and metadata without copying files.")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("No sources defined in manifest")

    # Sample representation:
    # dict: {source, source_names, mapping, image, label, group_id, split, output_name}
    prepared_samples = []
    group_sizes = Counter()
    split_group_sets: dict[str, set[str]] = defaultdict(set)
    exclusions_applied: list[str] = []

    print("=" * 80)
    print("STAGE 1 UNIFIED DATASET PREPARATION")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: DISCOVER AND INVENTORY ALL SOURCES
    # -------------------------------------------------------------------------
    for source in sources:
        sid = source["id"]
        print(f"\n--- Ingesting source: {sid} ---")
        if not source.get("license_approved"):
            raise ValueError(f"{sid}: license is not approved")
        if not source.get("exhaustive_labels"):
            raise ValueError(f"{sid}: labels are not marked exhaustive")
        if source.get("annotation_format") != "yolo":
            raise ValueError(f"{sid}: only YOLO sources are currently supported")

        root = Path(source["local_path"])
        if not root.exists():
            raise FileNotFoundError(f"{sid}: local path does not exist: {root}")

        source_names = source["class_names"]
        mapping = source.get("class_mapping", {})

        # Load exclusions
        excl_file = Path(source["exclusions"]) if source.get("exclusions") else None
        excluded_names = load_exclusions(excl_file)
        if excluded_names:
            print(f"  Loaded {len(excluded_names)} exclusion tokens from {excl_file}")

        source_samples = []

        # Case A: Manifest file explicitly provided (e.g. Fall pilot manifest)
        if source.get("manifest"):
            man_path = Path(source["manifest"])
            if not man_path.exists():
                raise FileNotFoundError(f"{sid}: manifest file missing: {man_path}")
            with man_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_rel = row.get("image_rel_path") or row.get("image")
                    fname = Path(img_rel).name
                    if fname in excluded_names or Path(fname).stem in excluded_names:
                        exclusions_applied.append(f"{sid}::{fname}")
                        continue
                    img_p = root / img_rel
                    lbl_rel = row.get("label_rel_path") or row.get("label")
                    if lbl_rel:
                        lbl_p = root / lbl_rel
                    else:
                        lbl_p = root / "labels" / row.get("split", "train") / f"{Path(fname).stem}.txt"
                    split = row.get("split")
                    # Preserve true group ID: prefer actor_group, fallback to group_id or clip_id
                    raw_grp = row.get("actor_group") or row.get("group_id") or row.get("clip_id")
                    group_id = f"{sid}::{raw_grp}"
                    source_samples.append({
                        "source": source,
                        "source_names": source_names,
                        "mapping": mapping,
                        "image": img_p,
                        "label": lbl_p,
                        "group_id": group_id,
                        "split": split,
                        "output_name": f"{sid}__{fname}",
                    })

        # Case B: Split metadata directory (metadata/train.csv, metadata/val.csv, metadata/test.csv)
        elif (root / "metadata").is_dir() and any((root / "metadata" / f"{s}.csv").exists() for s in ("train", "val", "test")):
            for split in ("train", "val", "test"):
                s_meta = root / "metadata" / f"{split}.csv"
                if not s_meta.exists():
                    continue
                with s_meta.open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fname = row["image"]
                        if fname in excluded_names or Path(fname).stem in excluded_names:
                            exclusions_applied.append(f"{sid}::{fname}")
                            continue
                        img_p = root / "images" / split / fname
                        lbl_p = root / "labels" / split / f"{Path(fname).stem}.txt"
                        raw_grp = row["group_id"]
                        group_id = f"{sid}::{raw_grp}"
                        source_samples.append({
                            "source": source,
                            "source_names": source_names,
                            "mapping": mapping,
                            "image": img_p,
                            "label": lbl_p,
                            "group_id": group_id,
                            "split": split,
                            "output_name": f"{sid}__{fname}",
                        })

        # Case C: Single metadata.csv
        else:
            meta_file = root / source.get("metadata", "metadata.csv")
            if not meta_file.exists():
                raise FileNotFoundError(f"{sid}: metadata file missing: {meta_file}")
            with meta_file.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_img = row["image"]
                    if raw_img in excluded_names or Path(raw_img).stem in excluded_names:
                        exclusions_applied.append(f"{sid}::{raw_img}")
                        continue
                    img_p = resolve_image(root, raw_img)
                    if img_p.name in excluded_names or img_p.stem in excluded_names:
                        exclusions_applied.append(f"{sid}::{img_p.name}")
                        continue
                    lbl_p = root / row.get("label", f"labels/{img_p.stem}.txt")
                    split = row.get("split")
                    raw_grp = row.get("group_id", img_p.stem)
                    group_id = f"{sid}::{raw_grp}"
                    source_samples.append({
                        "source": source,
                        "source_names": source_names,
                        "mapping": mapping,
                        "image": img_p,
                        "label": lbl_p,
                        "group_id": group_id,
                        "split": split,
                        "output_name": f"{sid}__{img_p.name}",
                    })

        print(f"  Ingested {len(source_samples)} samples from {sid}")
        prepared_samples.extend(source_samples)

    # -------------------------------------------------------------------------
    # PHASE 2: SPLIT ASSIGNMENT & LEAKAGE VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- Verifying split assignments and group isolation ---")
    all_preassigned = all(sample["split"] in ("train", "val", "test") for sample in prepared_samples)

    if all_preassigned:
        print("  All samples have verified pre-assigned splits from source metadata.")
        for s in prepared_samples:
            split_group_sets[s["split"]].add(s["group_id"])
    else:
        print("  Assigning group splits deterministically using seed", args.seed)
        for s in prepared_samples:
            group_sizes[s["group_id"]] += 1
        split_by_group = assign_group_splits(group_sizes, seed=args.seed)
        for s in prepared_samples:
            s["split"] = split_by_group[s["group_id"]]
            split_group_sets[s["split"]].add(s["group_id"])

    # Cross-split leakage assertion
    leakage_errors = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_group_sets[left] & split_group_sets[right])
        if overlap:
            leakage_errors.append(f"Group leakage between {left} and {right}: {len(overlap)} groups (e.g. {overlap[:5]})")

    if leakage_errors:
        print("FATAL: Group leakage detected!")
        for err in leakage_errors:
            print(f"  - {err}")
        raise ValueError("Group leakage detected across splits. Aborting.")
    print("  PASS: Zero cross-split group leakage across all sources.")

    # -------------------------------------------------------------------------
    # PHASE 3: STRICT PRE-VERIFICATION OF ALL IMAGE-LABEL PAIRS & LABELS
    # -------------------------------------------------------------------------
    print(f"\n--- Pre-verifying all {len(prepared_samples)} input image-label pairs ---")
    pre_errors = []
    verified_class_counts = Counter()

    for idx, sample in enumerate(prepared_samples):
        img_p: Path = sample["image"]
        lbl_p: Path = sample["label"]
        sid = sample["source"]["id"]
        source_names = sample["source_names"]
        mapping = sample["mapping"]

        if not img_p.is_file():
            pre_errors.append(f"{sid}: Missing image file: {img_p}")
            continue
        if not lbl_p.is_file():
            pre_errors.append(f"{sid}: Missing label file for image: {img_p.name} (expected {lbl_p})")
            continue

        try:
            boxes = parse_yolo_label(lbl_p)
        except Exception as exc:
            pre_errors.append(f"{sid}: Malformed label {lbl_p}: {exc}")
            continue

        for cid, xc, yc, w, h in boxes:
            if cid < 0 or cid >= len(source_names):
                pre_errors.append(f"{sid}: Class ID {cid} out of range in {lbl_p}")
                continue
            src_name = source_names[cid]
            canon_name = mapping.get(src_name)
            if canon_name is None:
                continue
            if canon_name == "fight":
                pre_errors.append(
                    f"{sid}: fight is a temporal event and cannot be mapped into detector schema v2 ({lbl_p})"
                )
            if canon_name not in CLASS_TO_ID:
                pre_errors.append(f"{sid}: Unknown canonical class '{canon_name}' in {lbl_p}")
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                pre_errors.append(f"{sid}: Out of range box ({xc}, {yc}, {w}, {h}) in {lbl_p}")
            verified_class_counts[canon_name] += 1

    if pre_errors:
        print(f"FATAL: {len(pre_errors)} verification errors found before copying!")
        for err in pre_errors[:20]:
            print(f"  - {err}")
        if len(pre_errors) > 20:
            print(f"  ... and {len(pre_errors) - 20} more.")
        raise ValueError(f"Pre-verification failed with {len(pre_errors)} errors.")

    print(f"  PASS: All {len(prepared_samples)} image-label pairs and boxes verified cleanly.")
    print("  Verified class instance totals:")
    for cname in CLASS_TO_ID:
        print(f"    {CLASS_TO_ID[cname]}: {cname:<8} = {verified_class_counts[cname]}")

    if args.dry_run:
        print("\nDry-run complete. Exiting without modifying disk.")
        return 0

    # -------------------------------------------------------------------------
    # PHASE 4: PREPARE OUTPUT DIRECTORIES & COPY SAMPLES
    # -------------------------------------------------------------------------
    print(f"\n--- Generating unified dataset at {args.output} ---")
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    metadata_rows: dict[str, list[dict[str, str]]] = {name: [] for name in ("train", "val", "test")}
    written_instance_counts = Counter()

    for idx, sample in enumerate(prepared_samples):
        img_p = sample["image"]
        lbl_p = sample["label"]
        split = sample["split"]
        out_name = sample["output_name"]
        sid = sample["source"]["id"]
        source_names = sample["source_names"]
        mapping = sample["mapping"]
        grp_id = sample["group_id"]

        img_target, lbl_target = copy_prepared_sample(img_p, lbl_p, args.output, split, out_name)

        # Parse and remap boxes
        remapped_lines = []
        for src_id, xc, yc, w, h in parse_yolo_label(lbl_p):
            src_name = source_names[src_id]
            canon_name = mapping.get(src_name)
            if canon_name is None:
                continue
            if canon_name == "fight":
                raise ValueError(f"{sid}: fight encountered during write ({lbl_p})")
            canon_id = CLASS_TO_ID[canon_name]
            written_instance_counts[canon_name] += 1
            remapped_lines.append(f"{canon_id} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")

        lbl_target.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
        metadata_rows[split].append({
            "image": img_target.name,
            "source_id": sid,
            "group_id": grp_id,
        })

    # -------------------------------------------------------------------------
    # PHASE 5: WRITE METADATA, MANIFEST, AND DATA.YAML
    # -------------------------------------------------------------------------
    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in metadata_rows.items():
        with (metadata_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "source_id", "group_id"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {metadata_dir / f'{split}.csv'} ({len(rows)} rows)")

    # Data.yaml
    data_yaml_path = args.output / "data.yaml"
    data_yaml_content = [
        f"path: '{args.output.resolve().as_posix()}'",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        f"",
        f"nc: 6",
        f"names: ['person', 'helmet', 'vest', 'fall', 'fire', 'smoke']",
        f"",
    ]
    data_yaml_path.write_text("\n".join(data_yaml_content), encoding="utf-8")
    print(f"  Wrote {data_yaml_path}")

    # Dataset manifest
    summary = {
        "detector_schema_version": DETECTOR_SCHEMA_VERSION,
        "class_names": list(CLASS_TO_ID),
        "seed": args.seed,
        "source_manifest": str(args.manifest),
        "output_path": str(args.output),
        "provisional": True,
        "validation_only": True,
        "training_approved": False,
        "sources": [s["id"] for s in sources],
        "exclusions_applied": sorted(list(set(exclusions_applied))),
        "images_by_split": {k: len(v) for k, v in metadata_rows.items()},
        "total_images": sum(len(v) for v in metadata_rows.values()),
        "instances": dict(written_instance_counts),
        "total_instances": sum(written_instance_counts.values()),
        "warning": (
            "Provisional validation-only Stage 1 unified dataset. "
            "Public-dataset baseline; not validated on target CCTV cameras. "
            "Does NOT approve training."
        ),
    }
    manifest_out = args.output / "dataset_manifest.json"
    manifest_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Wrote {manifest_out}")

    print("\n" + "=" * 80)
    print("STAGE 1 UNIFIED DATASET GENERATION SUCCESSFUL")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
