"""FastAPI app for the Upload & Analyze prototype (contract: docs/web_api_contract.md).

Run: ``.venv-cuda/Scripts/python -m uvicorn webapp.api:app --port 8000`` then open
http://localhost:8000/. Models load lazily on the first analyze request.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cctv_safety.detector import load_thresholds  # noqa: E402
from cctv_safety.ppe import assess_ppe  # noqa: E402
from cctv_safety.schema import CLASS_NAMES, CLASS_TO_ID, DETECTOR_SCHEMA_VERSION  # noqa: E402
from cctv_safety.stage2 import CLASS_NAMES as STAGE2_CLASS_NAMES, FRAMES  # noqa: E402
from webapp import config, media  # noqa: E402
from webapp.media import ApiError  # noqa: E402
from webapp.models import ModelLoadFailed, ModelRegistry, WeightsMissing  # noqa: E402

logger = logging.getLogger("webapp.api")

_HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def _r(value: float) -> float:
    return round(float(value), 4)


def _envelope(model: str | None, is_model_output: bool = True) -> dict:
    return {
        "is_model_output": is_model_output,
        "model": model,
        "schema_version": config.SCHEMA_VERSION,
        "disclaimer": config.DISCLAIMER,
    }


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _status_models(status: dict) -> tuple[dict, str]:
    models = status.get("models", status)
    return {name: models.get(name, "missing") for name in config.WEIGHT_FILES}, str(status.get("device", "cpu"))


def _source(width: int, height: int, info: media.VideoInfo | None) -> dict:
    return {
        "width": int(width),
        "height": int(height),
        "fps": _r(info.fps) if info else None,
        "frame_count": int(info.frame_count) if info else None,
        "duration_s": _r(info.duration_s) if info else None,
    }


def _check_upload(file: UploadFile | None) -> tuple[str | None, int]:
    if file is None or not isinstance(file, StarletteUploadFile):
        raise ApiError(400, "missing_file", "No file was uploaded; send it in the multipart field 'file'.")
    size = file.size if file.size is not None else media.upload_size(file.file)
    if size <= 0:
        raise ApiError(400, "missing_file", "The uploaded file is empty (0 bytes).")
    return media.media_kind(file.filename), size


def _check_kind_and_size(file: UploadFile, kind: str | None, size: int) -> None:
    if kind is None:
        allowed = ", ".join(config.IMAGE_EXTENSIONS + config.VIDEO_EXTENSIONS)
        raise ApiError(415, "unsupported_media_type",
                       f"Unsupported file type '{Path(file.filename or '').suffix}'. Allowed: {allowed}.")
    if size > config.MAX_UPLOAD_BYTES:
        raise media.too_large(config.MAX_UPLOAD_BYTES)


def _too_long(info: media.VideoInfo) -> ApiError:
    windows = len(media.stage2_windows(info.duration_s))
    return ApiError(422, "video_too_long",
                    f"The video is {info.duration_s:.1f} s long and would need {windows} windows of "
                    f"{config.WINDOW_S:g} s; the limit is {config.MAX_WINDOWS} windows (about 60 s).")


def create_app(registry=None, weights_dir: Path | None = None) -> FastAPI:
    weights_dir = Path(weights_dir) if weights_dir is not None else config.WEIGHTS_DIR
    if registry is None:
        registry = ModelRegistry(weights_dir, load_thresholds(config.THRESHOLDS_PATH))
    thresholds = getattr(registry, "thresholds", None) or load_thresholds(config.THRESHOLDS_PATH)
    thresholds = {name: float(thresholds[name]) for name in CLASS_NAMES}
    ppe_min_confidence = min(thresholds["person"], thresholds["helmet"], thresholds["vest"])

    app = FastAPI(title="CCTV Safety — Upload & Analyze (pilot)", version=config.SCHEMA_VERSION)
    app.state.registry = registry

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(WeightsMissing)
    async def _weights_missing(_: Request, exc: WeightsMissing):
        return _error(503, "weights_missing", str(exc))

    @app.exception_handler(ModelLoadFailed)
    async def _load_failed(_: Request, exc: ModelLoadFailed):
        return _error(503, "model_load_failed", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = _HTTP_ERROR_CODES.get(exc.status_code)
        if code is not None and request.url.path.startswith("/api/"):
            text = "No API endpoint at this path." if exc.status_code == 404 else (
                f"Method {request.method} is not allowed for this endpoint.")
            response = _error(exc.status_code, code, f"{text} ({request.method} {request.url.path})")
            if exc.headers:
                response.headers.update(exc.headers)
            return response
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _internal_error(request: Request, exc: Exception):
        # Full traceback goes to the server log only; the body never carries it.
        logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return _error(500, "internal_error", "Internal server error while processing the request; see the server log.")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = exc.errors()
        if any("file" in (err.get("loc") or ()) for err in errors):
            return _error(400, "missing_file", "No file was uploaded; send it in the multipart field 'file'.")
        detail = "; ".join(f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg')}" for err in errors)
        return _error(400, "invalid_parameter", f"Invalid request parameters: {detail}")

    def ensure_model(name: str) -> None:
        """weights_missing / model_load_failed come before decode_failed (contract section 4)."""
        models, _ = _status_models(registry.status())
        if models.get(name) == "missing":
            raise WeightsMissing(name, weights_dir / config.WEIGHT_FILES[name])
        load = getattr(registry, "load", None)
        if callable(load):
            load(name)

    @app.get("/api/health")
    def health():
        models, device = _status_models(registry.status())
        return {**_envelope(None, is_model_output=False), "status": "ok", "device": device, "models": models}

    @app.post("/api/stage1/analyze")
    def stage1_analyze(
        file: UploadFile | None = File(None),
        model: str | None = Form(None),
        max_frames: str | None = Form(None),
        mode: str | None = Form(None),
        sample_fps: str | None = Form(None),
    ):
        kind, size = _check_upload(file)
        if model not in config.STAGE1_MODELS:
            raise ApiError(400, "invalid_parameter",
                           f"'model' must be one of {list(config.STAGE1_MODELS)}; got {model!r}.")
        mode_used = mode.strip() if mode is not None and mode.strip() != "" else "frames"
        if mode_used not in config.STAGE1_MODES:
            raise ApiError(400, "invalid_parameter",
                           f"'mode' must be one of {list(config.STAGE1_MODES)}; got {mode!r}.")
        dense = mode_used == "dense"
        limit = config.DEFAULT_MAX_FRAMES
        fps_used = config.DEFAULT_SAMPLE_FPS
        if not dense and max_frames is not None and max_frames.strip() != "":
            try:
                limit = int(max_frames.strip())
            except ValueError:
                limit = -1
            if not 1 <= limit <= config.MAX_FRAMES_LIMIT:
                raise ApiError(400, "invalid_parameter",
                               f"'max_frames' must be an integer in 1..{config.MAX_FRAMES_LIMIT}; got {max_frames!r}.")
        if dense and sample_fps is not None and sample_fps.strip() != "":
            try:
                fps_used = float(sample_fps.strip())
            except ValueError:
                fps_used = float("nan")
            if not (math.isfinite(fps_used) and config.MIN_SAMPLE_FPS <= fps_used <= config.MAX_SAMPLE_FPS):
                raise ApiError(400, "invalid_parameter",
                               f"'sample_fps' must be a number in {config.MIN_SAMPLE_FPS:g}..{config.MAX_SAMPLE_FPS:g}; "
                               f"got {sample_fps!r}.")
        _check_kind_and_size(file, kind, size)
        if dense and kind != "video":
            raise ApiError(422, "video_required",
                           "Stage 1 dense mode needs a video file (mp4, avi, mov, mkv); an image was uploaded. "
                           "Use mode=frames for images.")
        ensure_model(model)

        frames: list[dict] = []
        info = None
        path = media.save_to_temp(file.file, Path(file.filename).suffix.lower(), config.MAX_UPLOAD_BYTES)
        try:
            if kind == "image":
                frame = media.decode_image(path)
                height, width = frame.shape[:2]
                (detections,) = registry.stage1(model, [frame])
                frames.append(_frame_entry(0, None, media.encode_jpeg_b64(frame), detections))
            elif not dense:
                info = media.probe_video(path)
                width, height = info.width, info.height
                indices = media.stage1_indices(info.frame_count, limit)
                decoded = media.read_frames(path, indices)
                per_frame = registry.stage1(model, [decoded[index] for index in indices])
                for index, detections in zip(indices, per_frame):
                    frames.append(_frame_entry(index, _r(index / info.fps),
                                               media.encode_jpeg_b64(decoded[index]), detections))
            else:
                info = media.probe_video(path)
                width, height = info.width, info.height
                if len(media.stage2_windows(info.duration_s)) > config.MAX_WINDOWS:
                    raise _too_long(info)
                # Detect while reading, in small batches: no full-res frame list, no JPEG encoding.
                batch: list = []
                for item in media.iter_frames(path, media.dense_indices(info, fps_used)):
                    batch.append(item)
                    if len(batch) >= config.DENSE_BATCH:
                        frames.extend(_detect_batch(model, batch, info))
                        batch = []
                if batch:
                    frames.extend(_detect_batch(model, batch, info))
        finally:
            media.remove_quietly(path)

        return {
            **_envelope(model),
            "detector_schema_version": DETECTOR_SCHEMA_VERSION,
            "class_names": list(CLASS_NAMES),
            "media_type": kind,
            "source": _source(width, height, info),
            "thresholds": {name: _r(value) for name, value in thresholds.items()},
            "ppe_min_confidence": _r(ppe_min_confidence),
            "mode": mode_used,
            "sample_fps": _r(fps_used) if dense else None,
            "frames": frames,
        }

    def _detect_batch(model: str, batch: list, info: media.VideoInfo) -> list[dict]:
        per_frame = registry.stage1(model, [frame for _, frame in batch])
        return [_frame_entry(index, _r(index / info.fps), None, detections)
                for (index, _), detections in zip(batch, per_frame)]

    def _frame_entry(index: int, time_s, jpeg_b64, detections) -> dict:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        ppe_rows = assess_ppe(ordered, ppe_min_confidence)
        for row in ppe_rows:
            row["person_confidence"] = _r(row["person_confidence"])
        return {
            "index": int(index),
            "time_s": time_s,
            "image_jpeg_b64": jpeg_b64,
            "detections": [
                {
                    "class_id": CLASS_TO_ID[item.class_name],
                    "class_name": item.class_name,
                    "confidence": _r(item.confidence),
                    "xyxy": [_r(value) for value in item.xyxy],
                }
                for item in ordered
            ],
            "ppe": ppe_rows,
        }

    @app.post("/api/stage2/analyze")
    def stage2_analyze(file: UploadFile | None = File(None)):
        kind, size = _check_upload(file)
        _check_kind_and_size(file, kind, size)
        if kind != "video":
            raise ApiError(422, "video_required",
                           "Stage 2 (X3D-S fight classifier) needs a video file (mp4, avi, mov, mkv); "
                           "an image was uploaded.")
        ensure_model(config.STAGE2_MODEL)

        path = media.save_to_temp(file.file, Path(file.filename).suffix.lower(), config.MAX_UPLOAD_BYTES)
        try:
            info = media.probe_video(path)
            spans = media.stage2_windows(info.duration_s)
            if len(spans) > config.MAX_WINDOWS:
                raise _too_long(info)
            clip_indices = [media.window_frame_indices(start, end, info) for start, end in spans]
            decoded = media.read_frames(path, [i for clip in clip_indices for i in clip], media.letterbox_bgr)
        finally:
            media.remove_quietly(path)

        clips = [[decoded[i] for i in clip] for clip in clip_indices]
        probs = registry.stage2(clips)
        windows = []
        for index, ((start, end), (p_non, p_fight)) in enumerate(zip(spans, probs)):
            windows.append({
                "index": index,
                "start_s": _r(start),
                "end_s": _r(end),
                "probs": {"non_fight": _r(p_non), "fight": _r(p_fight)},
                "label": STAGE2_CLASS_NAMES[1] if p_fight > p_non else STAGE2_CLASS_NAMES[0],
            })
        return {
            **_envelope(config.STAGE2_MODEL),
            "source": _source(info.width, info.height, info),
            "class_names": list(STAGE2_CLASS_NAMES),
            "frames_per_window": FRAMES,
            "window_s": config.WINDOW_S,
            "windows": windows,
            "summary": {
                "max_fight_prob": max(window["probs"]["fight"] for window in windows),
                "fight_windows": sum(1 for window in windows if window["label"] == "fight"),
                "total_windows": len(windows),
            },
        }

    api_methods: dict[str, set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/api/"):
            api_methods.setdefault(path, set()).update(getattr(route, "methods", None) or ())

    @app.api_route("/api/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                   include_in_schema=False)
    def api_fallback(rest: str):
        """Unknown /api/ paths and wrong methods get the contract error body (before the static mount)."""
        allowed = api_methods.get(f"/api/{rest}")
        if allowed:
            raise StarletteHTTPException(405, headers={"Allow": ", ".join(sorted(allowed))})
        raise StarletteHTTPException(404)

    if config.STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="mockup")
    return app


app = create_app()
