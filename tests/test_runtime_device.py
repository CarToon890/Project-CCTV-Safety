from __future__ import annotations

import pytest

from webapp import models
from webapp.live import validate_source


def test_auto_uses_cuda_for_models_while_yunet_can_stay_on_cpu(monkeypatch):
    monkeypatch.setattr(models, "runtime_status", lambda: {
        "cuda_available": True, "full_pipeline_cuda_ready": False,
        "opencv_cuda_available": False,
        "onnx_providers": ["CUDAExecutionProvider"], "face_backend": "opencv-yunet-cpu",
    })
    assert models.resolve_device("auto") == "cuda:0"
    assert models.resolve_device("cuda") == "cuda:0"


def test_explicit_cuda_fails_when_torch_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(models, "runtime_status", lambda: {
        "cuda_available": False, "full_pipeline_cuda_ready": False,
        "opencv_cuda_available": False,
        "onnx_providers": ["CPUExecutionProvider"], "face_backend": "opencv-yunet-cpu",
    })
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        models.resolve_device("cuda")
    assert models.resolve_device("auto") == "cpu"
    assert models.resolve_device("cpu") == "cpu"


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="auto, cpu, cuda"):
        models.resolve_device("metal")


def test_registry_keeps_device_specific_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "runtime_status", lambda: {
        "cuda_available": True, "full_pipeline_cuda_ready": True,
        "opencv_cuda_available": True,
        "onnx_providers": [], "face_backend": "opencv-yunet-cpu",
    })
    registry = models.ModelRegistry(tmp_path, {"person": 0.25})
    registry._load = lambda name, path, device: (name, device)
    (tmp_path / "YOLOv8n_best.pt").touch()
    assert registry.load("yolov8n", "cpu") == ("yolov8n", "cpu")
    assert registry.load("yolov8n", "cuda") == ("yolov8n", "cuda:0")
    assert len(registry._models) == 2


def test_live_status_reports_mixed_model_and_privacy_devices(monkeypatch):
    from webapp.live import LiveSession

    monkeypatch.setattr(models, "runtime_status", lambda: {
        "cuda_available": True, "full_pipeline_cuda_ready": False,
        "opencv_cuda_available": False, "onnx_providers": [],
        "face_backend": "opencv-yunet-cpu",
    })
    session = LiveSession("rtsp", "rtsp://camera/stream", "yolov8n", "cuda",
                          object(), {}, None, device_requested="cuda")
    status = session.status()
    assert status["device_requested"] == "cuda"
    assert status["device_used"] == "cuda:0"
    assert status["device_components"] == {
        "yolo": "cuda:0", "x3d": "cuda:0", "yunet": "cpu",
    }


def test_live_source_validation_limits_replay_to_data_tree(tmp_path):
    with pytest.raises(ValueError, match="RTSP"):
        validate_source("rtsp", "https://example.invalid/stream", tmp_path)
    with pytest.raises(ValueError, match="under the repository data"):
        validate_source("file", "outside.mp4", tmp_path)
