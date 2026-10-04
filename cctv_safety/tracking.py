"""Experimental IoU/centre-distance person tracking and fight-candidate trigger."""

from __future__ import annotations

from math import hypot

PROXIMITY_DIAGONALS = 0.16
MOTION_DIAGONALS_PER_S = 0.25
MAX_TRACK_GAP_S = 0.75


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def add_track_ids(frames: list[dict], width: int, height: int, max_track_gap_s: float = MAX_TRACK_GAP_S) -> None:
    """Attach greedy, frame-to-frame IDs and normalized centre speed to person detections."""
    diagonal = hypot(width, height) or 1.0
    tracks: dict[int, dict] = {}
    next_id = 1
    for frame in frames:
        time_s = float(frame["time_s"] or 0.0)
        detections = [d for d in frame["detections"] if d["class_name"] == "person"]
        available = {
            track_id: track for track_id, track in tracks.items()
            if 0 <= time_s - track["time_s"] <= max_track_gap_s
        }
        candidates = []
        for det_index, detection in enumerate(detections):
            box = detection["xyxy"]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            for track_id, track in available.items():
                dt = time_s - track["time_s"]
                if dt <= 0:
                    continue
                old = track["xyxy"]
                ox, oy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
                distance = hypot(cx - ox, cy - oy)
                overlap = _iou(box, old)
                if overlap >= 0.05 or distance <= max(20.0, diagonal * dt * 1.0):
                    candidates.append((1.0 - overlap + distance / diagonal, det_index, track_id, cx, cy, dt))

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        matches: dict[int, tuple[int, float, float, float]] = {}
        for _, det_index, track_id, cx, cy, dt in sorted(candidates):
            if det_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(det_index)
            matched_tracks.add(track_id)
            matches[det_index] = (track_id, cx, cy, dt)

        current = []
        for det_index, detection in enumerate(detections):
            box = detection["xyxy"]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if det_index in matches:
                track_id, _, _, dt = matches[det_index]
                track = tracks[track_id]
                speed = hypot(cx - track["cx"], cy - track["cy"]) / diagonal / dt
            else:
                track_id = next_id
                next_id += 1
                speed = 0.0
            tracks[track_id] = {"xyxy": box, "cx": cx, "cy": cy, "time_s": time_s}
            detection["track_id"] = track_id
            detection["motion_diagonals_per_s"] = round(speed, 4)
            current.append({"track_id": track_id, "cx": cx, "cy": cy, "speed": speed})
        frame["_tracking"] = current


def summarize_window(frames: list[dict], start_s: float, end_s: float, width: int, height: int,
                     proximity_diagonals: float = PROXIMITY_DIAGONALS,
                     motion_diagonals_per_s: float = MOTION_DIAGONALS_PER_S) -> dict:
    """Summarize proximity/motion evidence and the experimental candidate trigger."""
    diagonal = hypot(width, height) or 1.0
    selected = [f for f in frames if start_s <= (f["time_s"] or 0.0) < end_s]
    seen_tracks: set[int] = set()
    max_tracks = proximity_frames = motion_frames = 0
    reasons: set[str] = set()
    for frame in selected:
        people = frame.get("_tracking", [])
        seen_tracks.update(p["track_id"] for p in people)
        max_tracks = max(max_tracks, len(people))
        close = any(
            hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / diagonal <= proximity_diagonals
            for i, a in enumerate(people) for b in people[i + 1:]
        )
        moving_with_multiple_people = len(people) >= 2 and any(
            p["speed"] >= motion_diagonals_per_s for p in people
        )
        if close:
            proximity_frames += 1
            reasons.add("people_in_close_proximity")
        if moving_with_multiple_people:
            motion_frames += 1
            reasons.add("multi_person_motion")

    triggered = proximity_frames > 0 or motion_frames > 0
    return {
        "candidate_triggered": triggered,
        "trigger_reasons": sorted(reasons),
        "tracked_people": len(seen_tracks),
        "max_concurrent_tracks": max_tracks,
        "proximity_frames": proximity_frames,
        "motion_frames": motion_frames,
    }
