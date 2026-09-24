#!/usr/bin/env python
"""Validate labels, metadata leakage, exact duplicates and near duplicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_safety.dataset import find_near_duplicates, validate_dataset, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_validation.json"))
    parser.add_argument("--near-duplicates", action="store_true")
    parser.add_argument("--max-distance", type=int, default=5)
    args = parser.parse_args()
    report = validate_dataset(args.dataset)
    if args.near_duplicates:
        report["near_duplicates"] = [
            {"left": str(left), "right": str(right), "distance": distance}
            for left, right, distance in find_near_duplicates(args.dataset / "images", args.max_distance)
        ]
    write_report(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

