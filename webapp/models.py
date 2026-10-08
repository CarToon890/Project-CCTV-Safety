"""Lazy, cached, thread-locked model registry for the web prototype."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from cctv_safety.ppe import Detection
from webapp import config


class WeightsMissing(Exception):
    """The weights file for the requested model is absent."""

    def __init__(self, name: str, path: Path | None = None):
        """``WeightsMissing(name, path)`` builds the contract message; ``WeightsMissing(message)`` keeps it."""
        self.name = name
        self.path = Path(path) if path is not None else None
        if self.path is None:
            super().__init__(name)
            return
        super().__init__(
            f"Model weights for '{name}' were not found: {_display_path(self.path)}. "
            "Copy the .pt file into the weights directory (CCTV_WEIGHTS_DIR)."
        )


class ModelLoadFailed(Exception):
    """Weights are present but loading or the runtime contract failed."""


def _display_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def runtime_status() -> dict:
    """Report runtime capabilities without loading model weights."""
    import torch
    cuda = bool(torch.cuda.is_available())
    gpu = None
    if cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "total_vram_bytes": int(props.total_memory)}
        except Exception:  # noqa: BLE001 - health should remain available
            gpu = {"name": "NVIDIA CUDA device", "total_vram_bytes": None}
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        providers = []
    try:
        import cv2
        opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:  # noqa: BLE001
        opencv_cuda = False
    # YuNet is deliberately kept on CPU; CUDA selection applies to YOLO/X3D only.
    face_backend = "opencv-yunet-cpu"
    return {"cuda_available": cuda, "gpu": gpu, "onnx_providers": providers,
            "face_backend": face_backend, "full_pipeline_cuda_ready": False,
            "opencv_cuda_available": opencv_cuda,
            "cuda_execution_mode": "cuda_models_cpu_yunet" if cuda else "cpu_only"}


def resolve_device(requested: str | None) -> str:
    """Resolve auto/cpu/cuda explicitly; never silently fall back from cuda."""
    if requested is not None and not isinstance(requested, str):
        raise ValueError("device must be one of auto, cpu, cuda")
    requested = (requested or "auto").strip().lower()
    if requested == "cuda:0":
        requested = "cuda"
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, cuda")
    status = runtime_status()
    if requested == "cuda":
        # PyTorch runs YOLO/X3D on CUDA; Live and Upload keep YuNet on CPU.
        if not status["cuda_available"]:
            raise RuntimeError("PyTorch CUDA is unavailable; install a CUDA-enabled PyTorch build and verify the NVIDIA driver")
        return "cuda:0"
    if requested == "auto":
        # Select CUDA for model inference when PyTorch reports it; YuNet remains CPU-only.
        return "cuda:0" if status["cuda_available"] else "cpu"
    return "cpu"


class ModelRegistry:
    """Loads each model on first use only; ``status()`` never loads anything."""

    def __init__(self, weights_dir: Path, thresholds: dict):
        self.weights_dir = Path(weights_dir)
        self.thresholds = dict(thresholds)
        self.device = resolve_device("auto")
        self._models: dict[tuple[str, str], object] = {}
        self._load_lock = threading.Lock()
        self._infer_locks = {name: threading.Lock() for name in config.WEIGHT_FILES}

    def path(self, name: str) -> Path:
        return self.weights_dir / config.WEIGHT_FILES[name]

    def status(self) -> dict:
        result: dict = {}
        for name in config.WEIGHT_FILES:
            if any(model_name == name for model_name, _ in self._models):
                result[name] = "loaded"
            elif self.path(name).is_file():
                result[name] = "available"
            else:
                result[name] = "missing"
        result["device"] = self.device
        result["runtime"] = runtime_status()
        return result

    def load(self, name: str, device: str | None = None):
        """Return the cached model, loading it on first use."""
        if name not in config.WEIGHT_FILES:
            raise ValueError(f"Unknown model: {name}")
        device = resolve_device(device or self.device)
        key = (name, device)
        model = self._models.get(key)
        if model is not None:
            return model
        with self._load_lock:
            model = self._models.get(key)
            if model is not None:
                return model
            path = self.path(name)
            if not path.is_file():
                raise WeightsMissing(name, path)
            try:
                model = self._load(name, path, device)
            except Exception as exc:  # noqa: BLE001 - surfaced as 503 model_load_failed
                raise ModelLoadFailed(f"Could not load model '{name}' from {_display_path(path)}: {exc}") from exc
            self._models[key] = model
            return model

    def _load(self, name: str, path: Path, device: str):
        if name == config.STAGE2_MODEL:
            from cctv_safety.stage2 import load_x3d_s

            return load_x3d_s(path, device=device)
        from cctv_safety.detector import load_yolo, validate_runtime_contract

        model = load_yolo(path)
        validate_runtime_contract(model.names, self.thresholds)
        model.to(device)
        return model

    def stage1(self, name: str, frames_bgr: list[np.ndarray], thresholds: dict | None = None,
               yolo_options: dict | None = None, device: str | None = None) -> list[list[Detection]]:
        if name not in config.STAGE1_MODELS:
            raise ValueError(f"Unknown Stage 1 model: {name}")
        from cctv_safety.detector import detect

        model = self.load(name, device)
        active_thresholds = thresholds or self.thresholds
        options = yolo_options or {}
        with self._infer_locks[name]:
            return [detect(model, frame, active_thresholds, **options) for frame in frames_bgr]

    def stage2(self, clips: list[list[np.ndarray]], device: str | None = None) -> list[tuple[float, float]]:
        import torch

        from cctv_safety.stage2 import predict_probs, preprocess_frames

        model = self.load(config.STAGE2_MODEL, device)
        probs: list[tuple[float, float]] = []
        with self._infer_locks[config.STAGE2_MODEL]:
            for start in range(0, len(clips), config.STAGE2_BATCH):
                chunk = clips[start:start + config.STAGE2_BATCH]
                batch = torch.stack([preprocess_frames(clip) for clip in chunk])
                probs.extend(predict_probs(model, batch))
        return probs
