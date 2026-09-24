import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cctv_safety.dataset import assign_group_splits, validate_dataset


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root: Path, leaked: bool = False) -> None:
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (10 if split == "train" else 20 if split == "val" else 30, 0, 0)).save(image_dir / f"{split}.jpg")
            (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        metadata = root / "metadata"
        metadata.mkdir()
        for split in ("train", "val", "test"):
            group = "shared" if leaked and split in ("train", "val") else f"group-{split}"
            with (metadata / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image", "source_id", "group_id"])
                writer.writeheader()
                writer.writerow({"image": f"{split}.jpg", "source_id": "sample", "group_id": group})

    def test_valid_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            report = validate_dataset(root)
            self.assertTrue(report["valid"], report["issues"])
            self.assertEqual(report["instances"]["person"], 3)

    def test_group_leakage_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root, leaked=True)
            report = validate_dataset(root)
            self.assertFalse(report["valid"])
            self.assertIn("group_leakage", {issue["code"] for issue in report["issues"]})

    def test_group_assignment_is_deterministic_and_exclusive(self):
        groups = {f"video-{index}": index + 1 for index in range(12)}
        first = assign_group_splits(groups, seed=42)
        second = assign_group_splits(groups, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(groups))
        self.assertTrue(set(first.values()).issubset({"train", "val", "test"}))


if __name__ == "__main__":
    unittest.main()

