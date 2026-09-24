#!/usr/bin/env python
"""Download only explicitly approved datasets from the source manifest."""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/datasets.local.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--execute", action="store_true", help="Perform downloads; default is a safe inventory preview")
    args = parser.parse_args()
    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    failures = 0
    for source in config.get("sources", []):
        source_id = source["id"]
        method = source.get("download", {}).get("type", "manual")
        value = source.get("download", {}).get("value", "")
        approved = bool(source.get("license_approved"))
        print(f"{source_id}: method={method} license={source.get('license', 'UNKNOWN')} approved={approved}")
        if not args.execute:
            continue
        if not approved:
            print("  SKIP: license_approved is false")
            failures += 1
            continue
        target = args.output / source_id
        target.mkdir(parents=True, exist_ok=True)
        if method == "manual":
            print(f"  MANUAL: obtain from {value} and place under {target}")
        elif method == "direct":
            destination = target / Path(value).name
            urllib.request.urlretrieve(value, destination)
            print(f"  downloaded {destination}")
        elif method == "git":
            if any(target.iterdir()):
                print(f"  SKIP: {target} is not empty")
            else:
                subprocess.run(["git", "clone", "--depth", "1", value, str(target)], check=True)
        elif method == "kaggle":
            subprocess.run(["kaggle", "datasets", "download", "-d", value, "-p", str(target), "--unzip"], check=True)
        else:
            print(f"  ERROR: unsupported download type {method}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

