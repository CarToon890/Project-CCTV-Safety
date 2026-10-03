"""Upload validation, image/video decoding, frame sampling and Stage 2 windowing."""

from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cctv_safety.stage2 import FRAMES, SIZE, letterbox, sample_indices
from webapp import config


class ApiError(Exception):
    """An error that maps to the contract's ``{"error": {"code", "message"}}`` body."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def media_kind(filename: str | None) -> str | None:
    """``"image"`` / ``"video"`` by file extension (case-insensitive), else ``None``."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in config.IMAGE_EXTENSIONS:
        return "image"
    if suffix in config.VIDEO_EXTENSIONS:
        return "video"
    return None


def upload_size(fileobj) -> int:
    position = fileobj.tell()
    fileobj.seek(0, os.SEEK_END)
    size = fileobj.tell()
    fileobj.seek(position)
    return size


def save_to_temp(fileobj, suffix: str, max_bytes: int) -> Path:
    """Copy an upload into a closed temp file (OpenCV needs a path). Caller must delete it."""
    fileobj.seek(0)
    handle = tempfile.NamedTemporaryFile(prefix="cctv_upload_", suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        written = 0
        with handle:
            while chunk := fileobj.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise too_large(max_bytes)
                handle.write(chunk)
    except BaseException:
        remove_quietly(path)
        raise
    return path


def remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def too_large(max_bytes: int) -> ApiError:
    return ApiError(413, "file_too_large", f"The upload exceeds the size limit of {max_bytes / (1024 * 1024):g} MB ({max_bytes} bytes).")


def decode_failed(detail: str) -> ApiError:
    return ApiError(422, "decode_failed", f"The file could not be decoded: {detail}.")


def decode_image(path: Path) -> np.ndarray:
    """Decode an image to a BGR uint8 array (OpenCV first, Pillow fallback)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if frame is None:
        try:
            from PIL import Image

            with Image.open(path) as image:
                frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            frame = None
    if frame is None or frame.size == 0:
        raise decode_failed("not a readable image")
    return np.ascontiguousarray(frame)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def probe_video(path: Path) -> VideoInfo:
    """Read metadata and check the first frame decodes; raises ``decode_failed`` otherwise."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise decode_failed("OpenCV cannot open this video (codec/container unsupported)")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            raise decode_failed("the video reports fps <= 0")
        if count <= 0:
            raise decode_failed("the video has 0 frames")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise decode_failed("no decodable frames")
        height, width = frame.shape[:2]
        return VideoInfo(width=int(width), height=int(height), fps=fps, frame_count=count)
    finally:
        cap.release()


def iter_frames(path: Path, indices: list[int], transform=None):
    """Yield ``(index, frame)`` for the requested frame numbers, ascending, reading sequentially
    (exact, no seeking). Only the current frame is held in memory.

    ``transform`` (e.g. :func:`letterbox_bgr`) is applied to each yielded frame. If the stream
    ends before a requested index (container over-reports its length), the last decoded frame
    is yielded for the remaining indices.
    """
    transform = transform or (lambda frame: frame)
    wanted = sorted(set(indices))
    if not wanted:
        return
    cap = cv2.VideoCapture(str(path))
    pointer = 0
    last = None
    try:
        if not cap.isOpened():
            raise decode_failed("OpenCV cannot open this video")
        position = 0
        while pointer < len(wanted):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            last = frame
            if position == wanted[pointer]:
                pointer += 1
                yield position, transform(frame)
            position += 1
    finally:
        cap.release()
    if pointer < len(wanted):
        if last is None:
            raise decode_failed("no decodable frames")
        filler = transform(last)
        for index in wanted[pointer:]:
            yield index, filler


def iter_video_frames(path: Path):
    """Yield every actually decoded source frame in order without trusting metadata count."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise decode_failed("OpenCV cannot open this video")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1
        if index == 0:
            raise decode_failed("no decodable frames")
    finally:
        cap.release()


def read_frames(path: Path, indices: list[int], transform=None) -> dict[int, np.ndarray]:
    """All requested frames as ``{index: frame}`` (see :func:`iter_frames`)."""
    return dict(iter_frames(path, indices, transform))


def letterbox_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Stage 2 letterbox applied while still in BGR, to bound memory per sampled frame.

    The BGR->RGB swap is a channel permutation and INTER_AREA resize / zero padding act per
    channel, so ``cvtColor(letterbox(bgr))`` equals ``letterbox(cvtColor(bgr))`` byte for byte.
    The result is already SIZE x SIZE, so the letterbox inside
    :func:`cctv_safety.stage2.preprocess_frames` is an exact copy (cv2.resize with an
    unchanged size copies the input) and the tensor matches the notebook path.
    """
    return letterbox(frame_bgr, SIZE)


def stage1_indices(frame_count: int, max_frames: int) -> list[int]:
    n = min(max_frames, frame_count)
    values = np.linspace(0, frame_count - 1, n).round().astype(int)
    return sorted({int(value) for value in values})


def dense_indices(info: VideoInfo, sample_fps: float) -> list[int]:
    """Dense Stage 1 sampling: ``t_k = k / sample_fps`` while ``t_k < duration_s``; frame index
    ``min(int(round(k / sample_fps * fps)), frame_count - 1)``; de-duplicated, first kept."""
    indices: list[int] = []
    seen: set[int] = set()
    k = 0
    while k / sample_fps < info.duration_s:
        index = min(int(round(k / sample_fps * info.fps)), info.frame_count - 1)
        if index not in seen:
            seen.add(index)
            indices.append(index)
        k += 1
    return indices


def stage2_windows(duration_s: float) -> list[tuple[float, float]]:
    """Non-overlapping 2 s windows; a trailing partial window needs >= 1 s; D < 2 s gives one window."""
    width = config.WINDOW_S
    if duration_s < width:
        return [(0.0, duration_s)]
    windows = []
    k = 0
    while True:
        start = k * width
        if start >= duration_s:
            break
        end = min(start + width, duration_s)
        if end - start < config.MIN_PARTIAL_WINDOW_S:
            break
        windows.append((start, end))
        k += 1
    return windows


def window_frame_indices(start_s: float, end_s: float, info: VideoInfo) -> list[int]:
    first = int(round(start_s * info.fps))
    last = min(int(round(end_s * info.fps)), info.frame_count) - 1
    first = min(max(first, 0), info.frame_count - 1)
    last = max(last, first)
    return sample_indices(first, last, FRAMES)


def encode_jpeg_b64(frame_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
    if not ok:
        raise decode_failed("frame could not be re-encoded as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
