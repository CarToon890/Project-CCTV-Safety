import unittest

from cctv_safety.ppe import Detection, assess_ppe


class PpeAssociationTests(unittest.TestCase):
    def test_fully_equipped_person(self):
        detections = [
            Detection("person", 0.9, (0, 0, 100, 200)),
            Detection("helmet", 0.8, (30, 5, 70, 45)),
            Detection("vest", 0.8, (20, 60, 80, 130)),
        ]
        self.assertEqual(assess_ppe(detections)[0]["alerts"], [])

    def test_missing_ppe(self):
        result = assess_ppe([Detection("person", 0.9, (0, 0, 100, 200))])[0]
        self.assertEqual(result["alerts"], ["no_helmet", "no_vest"])

    def test_ppe_is_not_assigned_to_wrong_person(self):
        detections = [
            Detection("person", 0.9, (0, 0, 100, 200)),
            Detection("person", 0.9, (200, 0, 300, 200)),
            Detection("helmet", 0.8, (230, 5, 270, 45)),
        ]
        results = assess_ppe(detections)
        self.assertIn("no_helmet", results[0]["alerts"])
        self.assertNotIn("no_helmet", results[1]["alerts"])


if __name__ == "__main__":
    unittest.main()

