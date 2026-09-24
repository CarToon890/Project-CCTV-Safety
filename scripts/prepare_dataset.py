#!/usr/bin/env python
"""Combine approved, exhaustively-labelled YOLO sources without group leakage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/datasets.local.yaml"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    samples = []
    group_sizes = Counter()

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for source in config.get("sources", []):
        if not source.get("license_approved"):
            raise ValueError(f"{source['id']}: license is not approved")
        if not source.get("exhaustive_labels"):
            raise ValueError(f"{source['id']}: labels are not marked exhaustive")
        if source.get("annotation_format") != "yolo":
            raise ValueError(f"{source['id']}: only YOLO sources are currently supported")
        root = Path(source["local_path"])
        source_names = source["class_names"]
        mapping = source.get("class_mapping", {})
        metadata = root / source.get("metadata", "metadata.csv")
        with metadata.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image = resolve_image(root, row["image"])
                label = root / row.get("label", f"labels/{image.stem}.txt")
                group_id = f"{source['id']}::{row['group_id']}"
                samples.append((source, source_names, mapping, image, label, group_id))
                group_sizes[group_id] += 1

    split_by_group = assign_group_splits(group_sizes, seed=args.seed)
    metadata_rows = {name: [] for name in ("train", "val", "test")}
    instance_counts = Counter()
    for source, source_names, mapping, image, label, group_id in samples:
        split = split_by_group[group_id]
        output_name = f"{source['id']}__{image.name}"
        image_target, label_target = copy_prepared_sample(image, label, args.output, split, output_name)
        remapped = []
        for source_id, x, y, width, height in parse_yolo_label(label):
            source_name = source_names[source_id]
            canonical_name = mapping.get(source_name)
            if canonical_name is None:
                continue
            if canonical_name == "fight":
                raise ValueError(
                    f"{source['id']}: fight is a temporal event and cannot be mapped "
                    "into detector schema v2"
                )
            if canonical_name not in CLASS_TO_ID:
                raise ValueError(f"{source['id']}: unknown canonical class {canonical_name}")
            class_id = CLASS_TO_ID[canonical_name]
            instance_counts[canonical_name] += 1
            remapped.append(f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
        label_target.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
        metadata_rows[split].append({"image": image_target.name, "source_id": source["id"], "group_id": group_id})

    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in metadata_rows.items():
        with (metadata_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "source_id", "group_id"])
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "detector_schema_version": DETECTOR_SCHEMA_VERSION,
        "class_names": list(CLASS_TO_ID),
        "seed": args.seed,
        "source_manifest": str(args.manifest),
        "images_by_split": {key: len(value) for key, value in metadata_rows.items()},
        "instances": dict(instance_counts),
        "warning": "Public-dataset baseline; not validated on target CCTV cameras.",
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
