"""Deterministic person-to-PPE association for baseline inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def _centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _inside_region(point: tuple[float, float], person: tuple[float, float, float, float], top: float, bottom: float) -> bool:
    x1, y1, x2, y2 = person
    px, py = point
    height = y2 - y1
    return x1 <= px <= x2 and y1 + height * top <= py <= y1 + height * bottom


def assess_ppe(
    detections: list[Detection],
    min_confidence: float = 0.25,
    head_region: tuple[float, float] = (0.0, 0.35),
    torso_region: tuple[float, float] = (0.2, 0.75),
) -> list[dict]:
    for name, region in (("head_region", head_region), ("torso_region", torso_region)):
        if len(region) != 2 or not 0 <= region[0] < region[1] <= 1:
            raise ValueError(f"{name} must be an increasing pair within 0..1")
    people = [item for item in detections if item.class_name == "person" and item.confidence >= min_confidence]
    helmets = [item for item in detections if item.class_name == "helmet" and item.confidence >= min_confidence]
    vests = [item for item in detections if item.class_name == "vest" and item.confidence >= min_confidence]
    results = []
    for index, person in enumerate(people):
        helmet_matches = [item for item in helmets if _inside_region(_centre(item.xyxy), person.xyxy, *head_region)]
        vest_matches = [item for item in vests if _inside_region(_centre(item.xyxy), person.xyxy, *torso_region)]
        results.append({
            "person_index": index,
            "person_confidence": person.confidence,
            "has_helmet": bool(helmet_matches),
            "has_vest": bool(vest_matches),
            "alerts": [name for name, present in (("no_helmet", helmet_matches), ("no_vest", vest_matches)) if not present],
            "method": "ppe-centre-in-person-region",
        })
    return results
