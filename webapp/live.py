"""Single-stream, in-memory live/replay worker for the local pilot."""
from __future__ import annotations

import base64
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from cctv_safety.privacy import YuNetFaceAnonymizer, anonymize
from cctv_safety.tracking import add_track_ids, summarize_window
from cctv_safety.stage2 import FRAMES
from webapp import config, media
from webapp.models import resolve_device


class LiveSession:
    def __init__(self, source_type, source, model, device, registry, settings, weights_dir,
                 device_requested="auto"):
        self.source_type = source_type
        self.source = source
        self.model = model
        self.device = resolve_device(device)
        self.device_requested = device_requested or "auto"
        self.registry = registry
        self.settings = settings
        self.weights_dir = weights_dir
        self.state = "starting"
        self.reason = None
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_face_scanned = 0
        self.source_fps = None
        self.started_at = time.time()
        self.started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._worker_failed = threading.Event()
        self._frames = queue.Queue(maxsize=12)
        self._events = queue.Queue(maxsize=64)
        self._metrics_lock = threading.Lock()
        self._timings = {name: deque(maxlen=120) for name in
                         ("decode", "face", "yolo", "x3d", "preview_encode", "frame_total")}
        self._last_health_at = 0.0
        self.on_finish = None
        self._capture_thread = threading.Thread(target=self._capture, daemon=True)
        self._worker_thread = threading.Thread(target=self._process, daemon=True)

    def start(self):
        self.state = "running"
        self._worker_thread.start()
        self._capture_thread.start()

    def stop(self):
        self._stop.set()
        if self.state == "running":
            self.state = "stopping"

    def status(self):
        with self._metrics_lock:
            timings = {
                name: {"last_ms": round(samples[-1], 1),
                       "mean_ms": round(sum(samples) / len(samples), 1),
                       "samples": len(samples)}
                for name, samples in self._timings.items() if samples
            }
        elapsed = max(0.001, time.monotonic() - self.started_monotonic)
        return {"state": self.state, "device_requested": self.device_requested,
                "device_used": self.device,
                "device_components": {"yolo": self.device, "x3d": self.device, "yunet": "cpu"},
                "frames_received": self.frames_received, "frames_processed": self.frames_processed,
                "frames_face_scanned": self.frames_face_scanned,
                "queue_depth": self._frames.qsize(), "reason": self.reason,
                "playback_mode": "processing_paced" if self.source_type == "file" else "realtime",
                "source_fps": self.source_fps,
                "processing_fps": round(self.frames_processed / elapsed, 2),
                "timings_ms": timings}

    def _record_timing(self, name, elapsed_ms):
        with self._metrics_lock:
            self._timings[name].append(max(0.0, float(elapsed_ms)))

    def publish(self, event):
        try:
            self._events.put_nowait(event)
        except queue.Full:
            # Preserve model/health events by evicting an old preview first.
            try:
                old = self._events.get_nowait()
                if old.get("type") != "preview":
                    self._events.put_nowait(old)
                    return
                self._events.put_nowait(event)
            except queue.Empty:
                pass
            except queue.Full:
                pass

    def _capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.state, self.reason = "error", "source_unavailable"
            self.publish({"type": "health", **self.status()})
            self._frames.put(None)
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = round(float(fps), 3) if fps and fps > 0 else None
        pace = 1.0 / fps if self.source_type == "file" and fps and fps > 0 else 0.0
        try:
            index = 0
            wall_start = time.monotonic()
            while not self._stop.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                self._record_timing("decode", (time.monotonic() - t0) * 1000)
                if not ok:
                    break
                self.frames_received += 1
                if frame is None or frame.size == 0 or frame.shape[0] * frame.shape[1] > config.MAX_VIDEO_PIXELS:
                    self.state, self.reason = "error", "frame_dimensions_unsupported"
                    self._stop.set()
                    self.publish({"type": "health", **self.status()})
                    break
                if self.source_type == "file" and fps and fps > 0:
                    source_time = index / fps
                else:
                    source_time = time.monotonic() - wall_start
                captured_at = time.monotonic()
                # Back-pressure the source; never drop a frame already returned by VideoCapture.
                frame_item = (frame, source_time, captured_at)
                try:
                    self._frames.put(frame_item, timeout=0.25)
                except queue.Full:
                    if self._worker_failed.is_set():
                        break
                    if self.source_type == "file":
                        # A replay file can be slowed by back-pressure without losing frames.
                        # Keep the frame already read and resume when the worker frees a slot.
                        self._frames.put(frame_item)
                        index += 1
                        continue
                    if not self._stop.is_set():
                        self.state, self.reason = "degraded", "processing_overload"
                        self._stop.set()
                    self.publish({"type": "health", **self.status()})
                    self._frames.put(frame_item)
                    break
                index += 1
                if pace:
                    time.sleep(max(0.0, pace - (time.monotonic() - t0)))
        finally:
            cap.release()
            # Drain every captured frame before terminating the inference worker.
            if self._worker_failed.is_set():
                while True:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        break
            else:
                self._frames.put(None)

    def _process(self):
        try:
            anonymizer = YuNetFaceAnonymizer(self.weights_dir, device="cpu")
            history = deque(maxlen=250)
            recent_entries = deque(maxlen=250)
            last_yolo = last_x3d = -1.0
            last_candidate_event = -1.0
            while True:
                item = self._frames.get()
                if item is None:
                    break
                frame_started = time.monotonic()
                frame, t, captured_at = item
                stage_started = time.monotonic()
                safe, _ = anonymize(anonymizer, frame)
                self._record_timing("face", (time.monotonic() - stage_started) * 1000)
                self.frames_face_scanned += 1
                self.frames_processed += 1
                # Retain only the X3D-size representation; keeping seconds of full-resolution
                # surveillance frames here would multiply memory use on 4K sources.
                history.append((t, media.letterbox_bgr(frame)))
                if t - last_yolo >= 0.1:
                    options = {k: self.settings["yolo"][k] for k in ("iou", "imgsz", "max_det")}
                    stage_started = time.monotonic()
                    detections = self.registry.stage1(self.model, [frame], self.settings["yolo"]["thresholds"], options, device=self.device)[0]
                    self._record_timing("yolo", (time.monotonic() - stage_started) * 1000)
                    entry = {"time_s": t, "detections": [{"class_name": d.class_name, "xyxy": list(d.xyxy), "confidence": float(d.confidence)} for d in detections]}
                    recent_entries.append(entry)
                    confidence_by_class = {}
                    for detection in detections:
                        current = confidence_by_class.setdefault(detection.class_name, {"count": 0, "max_confidence": 0.0})
                        current["count"] += 1
                        current["max_confidence"] = max(current["max_confidence"], float(detection.confidence))
                    self.publish({"type": "confidence", "model": self.model, "time_s": round(t, 3),
                                  "detection_count": len(detections), "classes": confidence_by_class})
                    add_track_ids(list(recent_entries), frame.shape[1], frame.shape[0], self.settings["tracking"]["max_track_gap_s"])
                    last_yolo = t
                    summary = summarize_window(list(recent_entries), max(0.0, t - 2), t + 0.001,
                        frame.shape[1], frame.shape[0], self.settings["tracking"]["proximity_diagonals"],
                        self.settings["tracking"]["motion_diagonals_per_s"])
                    if summary["candidate_triggered"] and t - last_candidate_event >= 1.0:
                        self.publish({"type": "candidate", "time_s": round(t, 3), "stage": "preliminary",
                                      "processing_latency_ms": round((time.monotonic() - captured_at) * 1000, 1),
                                      **summary})
                        last_candidate_event = t
                if len(history) >= FRAMES and t - last_x3d >= 0.5:
                    first_t = t - 2.0
                    clip_source = [item for ts, item in history if first_t <= ts <= t]
                    if len(clip_source) >= FRAMES:
                        import numpy as np
                        indices = np.linspace(0, len(clip_source) - 1, FRAMES).round().astype(int)
                        stage_started = time.monotonic()
                        probs = self.registry.stage2([[clip_source[int(i)] for i in indices]], device=self.device)[0]
                        self._record_timing("x3d", (time.monotonic() - stage_started) * 1000)
                        self.publish({"type": "x3d", "start_s": round(first_t, 3), "end_s": round(t, 3),
                                      "non_fight": round(probs[0], 4), "fight": round(probs[1], 4),
                                      "label": "fight" if probs[1] > self.settings["x3d"]["fight_threshold"] else "non_fight"})
                        last_x3d = t
                if self.frames_processed % 2 == 0:
                    stage_started = time.monotonic()
                    ok, encoded = cv2.imencode(".jpg", safe, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    self._record_timing("preview_encode", (time.monotonic() - stage_started) * 1000)
                    if ok:
                        self.publish({"type": "preview", "jpeg_b64": base64.b64encode(encoded).decode("ascii"), "time_s": round(t, 3),
                                      "processing_latency_ms": round((time.monotonic() - captured_at) * 1000, 1)})
                self._record_timing("frame_total", (time.monotonic() - frame_started) * 1000)
                now = time.monotonic()
                if now - self._last_health_at >= 2.0:
                    self.publish({"type": "health", **self.status()})
                    self._last_health_at = now
            if self.state not in {"degraded", "error"}:
                self.state = "stopped"
            self.publish({"type": "health", **self.status()})
        except Exception as exc:  # fail closed: no original frame is ever published
            self.state, self.reason = "error", type(exc).__name__
            self._worker_failed.set()
            self._stop.set()
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            self.publish({"type": "health", **self.status()})
        finally:
            if self.on_finish:
                self.on_finish()


def validate_source(source_type: str, source: str, repo_root: Path) -> str:
    if source_type == "rtsp":
        if not source.lower().startswith("rtsp://"):
            raise ValueError("RTSP source must use rtsp://")
        return source
    if source_type != "file":
        raise ValueError("source_type must be file or rtsp")
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    path = candidate.resolve()
    allowed = (repo_root / "data").resolve()
    if allowed not in path.parents or path.suffix.lower() not in config.VIDEO_EXTENSIONS or not path.is_file():
        raise ValueError("Replay video must be an existing supported file under the repository data/ directory")
    if path.stat().st_size > config.MAX_UPLOAD_BYTES:
        raise ValueError("Replay video exceeds the 100 MB pilot limit")
    return str(path)
