"""Dataset preparation primitives with leakage and label-safety checks."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .schema import CLASS_NAMES, IMAGE_SUFFIXES


@dataclass
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


def iter_images(root: Path) -> Iterable[Path]:
    return (path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_number}: expected 5 values, got {len(parts)}")
        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
        boxes.append((class_id, *values))
    return boxes


def validate_dataset(root: Path) -> dict:
    issues: list[ValidationIssue] = []
    counts = Counter()
    images_total = 0
    groups_by_split: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            issues.append(ValidationIssue("error", "missing_directory", str(image_dir), "Image split is missing"))
            continue
        if not label_dir.is_dir():
            issues.append(ValidationIssue("error", "missing_directory", str(label_dir), "Label split is missing"))
            continue
        for image_path in iter_images(image_dir):
            images_total += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                issues.append(ValidationIssue("error", "missing_label", str(image_path), "No matching label file"))
                continue
            try:
                boxes = parse_yolo_label(label_path)
            except (ValueError, UnicodeError) as exc:
                issues.append(ValidationIssue("error", "invalid_label", str(label_path), str(exc)))
                continue
            for class_id, x, y, width, height in boxes:
                if class_id not in range(len(CLASS_NAMES)):
                    issues.append(ValidationIssue("error", "invalid_class", str(label_path), f"Class {class_id} is outside 0-{len(CLASS_NAMES)-1}"))
                else:
                    counts[CLASS_NAMES[class_id]] += 1
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    issues.append(ValidationIssue("error", "invalid_box", str(label_path), f"Out-of-range box: {x} {y} {width} {height}"))
                if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
                    issues.append(ValidationIssue("warning", "box_outside_image", str(label_path), "Box crosses an image boundary"))

        metadata_path = root / "metadata" / f"{split}.csv"
        if not metadata_path.exists():
            issues.append(ValidationIssue("error", "missing_metadata", str(metadata_path), "Metadata is required for leakage checks"))
        else:
            with metadata_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"image", "source_id", "group_id"}
                if not required.issubset(reader.fieldnames or set()):
                    issues.append(ValidationIssue("error", "invalid_metadata", str(metadata_path), f"Required columns: {sorted(required)}"))
                else:
                    for row in reader:
                        if not row["group_id"].strip():
                            issues.append(ValidationIssue("error", "missing_group", str(metadata_path), f"Missing group_id for {row['image']}"))
                        groups_by_split[split].add(row["group_id"].strip())

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(groups_by_split[left] & groups_by_split[right])
        if overlap:
            issues.append(ValidationIssue("error", "group_leakage", str(root / "metadata"), f"{left}/{right} share group_id values: {overlap[:10]}"))

    duplicates = find_duplicates(root / "images") if (root / "images").exists() else []
    for paths in duplicates:
        issues.append(ValidationIssue("error", "exact_duplicate", str(paths[0]), "Duplicate image: " + ", ".join(map(str, paths[1:]))))

    return {
        "schema": list(CLASS_NAMES),
        "images": images_total,
        "instances": dict(counts),
        "issues": [asdict(issue) for issue in issues],
        "valid": not any(issue.severity == "error" for issue in issues),
    }


def find_duplicates(root: Path) -> list[list[Path]]:
    by_digest: dict[str, list[Path]] = defaultdict(list)
    for path in iter_images(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest[digest].append(path)
    return [paths for paths in by_digest.values() if len(paths) > 1]


def perceptual_hash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((hash_size + 1, hash_size)).getdata())
    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | (pixels[start + column] > pixels[start + column + 1])
    return value


def find_near_duplicates(root: Path, max_distance: int = 5) -> list[tuple[Path, Path, int]]:
    hashed = [(path, perceptual_hash(path)) for path in iter_images(root)]
    matches = []
    for index, (left_path, left_hash) in enumerate(hashed):
        for right_path, right_hash in hashed[index + 1:]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_distance:
                matches.append((left_path, right_path, distance))
    return matches


def assign_group_splits(groups: dict[str, int], ratios=(0.7, 0.2, 0.1), seed: int = 42) -> dict[str, str]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values")
    names = ("train", "val", "test")
    normalized = [value / sum(ratios) for value in ratios]
    targets = [sum(groups.values()) * value for value in normalized]
    assigned = {name: 0 for name in names}
    result: dict[str, str] = {}
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    for group_id, size in items:
        split_index = min(range(3), key=lambda index: assigned[names[index]] / max(targets[index], 1e-9))
        split = names[split_index]
        result[group_id] = split
        assigned[split] += size
    return result


def copy_prepared_sample(image: Path, label: Path, output: Path, split: str, output_name: str | None = None) -> tuple[Path, Path]:
    name = output_name or image.name
    image_target = output / "images" / split / name
    label_target = output / "labels" / split / f"{Path(name).stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)
    shutil.copy2(label, label_target)
    return image_target, label_target


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

