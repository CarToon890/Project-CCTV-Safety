"""HTTP contract tests for the Upload & Analyze API (docs/web_api_contract.md + tests/fixtures/api).

All tests except the final real-weights smoke test inject a deterministic stub registry, so they
run on any machine without weights or a GPU.
"""

from __future__ import annotations

import base64
import io
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cctv_safety.ppe import Detection, assess_ppe
from cctv_safety.schema import CLASS_NAMES, CLASS_TO_ID
from cctv_safety.stage2 import sample_indices
from conftest import (REPO_ROOT, WEIGHTS_DIR, X3D_WEIGHTS, YOLO_WEIGHTS, image_bytes, load_fixture,
                      require_weights, write_video)
from webapp.api import create_app
from webapp.models import WeightsMissing

STAGE1 = "/api/stage1/analyze"
STAGE2 = "/api/stage2/analyze"
DEFAULT_LIMIT = 100 * 1024 * 1024
THRESHOLDS_YAML = REPO_ROOT / "configs" / "thresholds.yaml"


# ---------------------------------------------------------------- stub registry
def stub_detections(frame: np.ndarray) -> list[Detection]:
    """Deterministic boxes scaled to the frame, all above the configured thresholds, sorted desc."""
    h, w = frame.shape[:2]
    return [
        Detection("person", 0.91, (0.10 * w, 0.10 * h, 0.40 * w, 0.90 * h)),
        Detection("helmet", 0.77, (0.18 * w, 0.10 * h, 0.32 * w, 0.22 * h)),
        Detection("person", 0.74, (0.60 * w, 0.10 * h, 0.85 * w, 0.90 * h)),
        Detection("vest", 0.66, (0.15 * w, 0.35 * h, 0.35 * w, 0.60 * h)),
        Detection("fire", 0.30, (0.05 * w, 0.70 * h, 0.20 * w, 0.95 * h)),
    ]


STUB_FIGHT = (0.0868, 0.5, 0.7915)  # cycles per window; 0.5 exercises the tie -> non_fight rule


class StubRegistry:
    def __init__(self, missing: bool = False):
        self.missing = missing
        self.stage1_calls: list[tuple[str, list[np.ndarray]]] = []
        self.stage2_calls: list[list[list[np.ndarray]]] = []

    def status(self) -> dict:
        state = "missing" if self.missing else "available"
        return {"yolov8n": state, "yolov8s": state, "x3d_s": state, "device": "cpu"}

    def stage1(self, name, frames_bgr):
        if self.missing:
            raise WeightsMissing(name, WEIGHTS_DIR / f"{name}.pt")
        self.stage1_calls.append((name, list(frames_bgr)))
        return [stub_detections(f) for f in frames_bgr]

    def stage2(self, clips):
        if self.missing:
            raise WeightsMissing("x3d_s", X3D_WEIGHTS)
        self.stage2_calls.append(list(clips))
        out = []
        for i, _ in enumerate(clips):
            fight = STUB_FIGHT[i % len(STUB_FIGHT)]
            out.append((1.0 - fight, fight))
        return out


@pytest.fixture
def stub():
    return StubRegistry()


@pytest.fixture
def client(stub):
    return TestClient(create_app(registry=stub))


@pytest.fixture
def missing_client():
    return TestClient(create_app(registry=StubRegistry(missing=True)))


# ---------------------------------------------------------------- helpers
def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a[k], b[k]) if k in b else a[k] for k in a}
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return a


def assert_structure(actual, fixture, path="$"):
    """Recursively check that ``actual`` has the same keys and JSON types as the fixture."""
    if fixture is None:
        assert actual is None, f"{path}: expected null, got {actual!r}"
    elif isinstance(fixture, bool):
        assert type(actual) is bool, f"{path}: expected bool, got {actual!r}"
    elif isinstance(fixture, int):
        assert type(actual) is int, f"{path}: expected int, got {actual!r}"
    elif isinstance(fixture, float):
        assert type(actual) is float, f"{path}: expected float, got {actual!r}"
    elif isinstance(fixture, str):
        assert isinstance(actual, str), f"{path}: expected string, got {actual!r}"
    elif isinstance(fixture, dict):
        assert isinstance(actual, dict), f"{path}: expected object, got {actual!r}"
        assert set(actual) == set(fixture), f"{path}: keys {sorted(actual)} != {sorted(fixture)}"
        for key in fixture:
            assert_structure(actual[key], fixture[key], f"{path}.{key}")
    elif isinstance(fixture, list):
        assert isinstance(actual, list), f"{path}: expected array, got {actual!r}"
        if fixture:
            template = reduce(_merge, fixture)
            for i, item in enumerate(actual):
                assert_structure(item, template, f"{path}[{i}]")
    else:  # pragma: no cover
        raise AssertionError(f"{path}: unsupported fixture type {type(fixture)}")


def post_file(client, url, name, content, data=None, ctype="application/octet-stream"):
    return client.post(url, files={"file": (name, content, ctype)}, data=data or {})


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("error_stage2_image.json"))
    assert body["error"]["code"] == code, body
    assert body["error"]["message"].strip()
    return body


def decode_b64_jpeg(value: str) -> Image.Image:
    assert not value.startswith("data:")
    raw = base64.b64decode(value, validate=True)
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI
    return Image.open(io.BytesIO(raw))


def expected_stage1_indices(frame_count, max_frames):
    n = min(max_frames, frame_count)
    return sorted({int(v) for v in np.linspace(0, frame_count - 1, n).round().astype(int)})


def gray_index(frame: np.ndarray) -> int:
    """Frame number encoded as grey level (centre pixel, so any centred letterbox is fine)."""
    h, w = frame.shape[:2]
    return int(round(float(frame[h // 2, w // 2].mean()) / 4))


def thresholds():
    import yaml

    return {k: float(v) for k, v in yaml.safe_load(THRESHOLDS_YAML.read_text(encoding="utf-8")).items()}


def check_envelope(body, model, is_model_output=True):
    fixture = load_fixture("health.json")
    assert body["is_model_output"] is is_model_output
    assert body["model"] == model
    assert body["schema_version"] == "1.0"
    assert body["disclaimer"] == fixture["disclaimer"]


# ---------------------------------------------------------------- health / static
def test_health_matches_fixture(client, stub):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert_structure(body, load_fixture("health.json"))
    check_envelope(body, None, is_model_output=False)
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["models"] == {"yolov8n": "available", "yolov8s": "available", "x3d_s": "available"}
    assert not stub.stage1_calls and not stub.stage2_calls


def test_health_reports_missing(missing_client):
    body = missing_client.get("/api/health").json()
    assert body["models"] == {"yolov8n": "missing", "yolov8s": "missing", "x3d_s": "missing"}


def test_root_serves_mockup_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.text == (REPO_ROOT / "mockup" / "index.html").read_text(encoding="utf-8")


def test_analyze_js_is_served(client):
    resp = client.get("/analyze.js")
    assert resp.status_code == 200
    assert resp.content == (REPO_ROOT / "mockup" / "analyze.js").read_bytes()


# ---------------------------------------------------------------- stage 1: image
def test_stage1_image_matches_fixture(client, stub):
    resp = post_file(client, STAGE1, "photo.jpg", image_bytes(160, 120), {"model": "yolov8n"}, "image/jpeg")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_image.json"))
    check_envelope(body, "yolov8n")
    assert body["detector_schema_version"] == 2
    assert body["class_names"] == list(CLASS_NAMES)
    assert body["media_type"] == "image"
    assert body["source"] == {"width": 160, "height": 120, "fps": None, "frame_count": None, "duration_s": None}
    th = thresholds()
    assert body["thresholds"] == pytest.approx(th)
    assert set(body["thresholds"]) == set(CLASS_NAMES)
    assert body["ppe_min_confidence"] == pytest.approx(min(th["person"], th["helmet"], th["vest"]))
    assert len(body["frames"]) == 1
    frame = body["frames"][0]
    assert frame["index"] == 0 and frame["time_s"] is None
    assert decode_b64_jpeg(frame["image_jpeg_b64"]).size == (160, 120)
    assert len(stub.stage1_calls) == 1 and stub.stage1_calls[0][0] == "yolov8n"


def test_stage1_image_frame_is_bgr_full_resolution(client, stub):
    resp = post_file(client, STAGE1, "red.png", image_bytes(160, 120, (255, 0, 0), "PNG"), {"model": "yolov8n"})
    assert resp.status_code == 200, resp.text
    (_, frames), = stub.stage1_calls
    assert len(frames) == 1
    f = frames[0]
    assert f.shape == (120, 160, 3) and f.dtype == np.uint8
    assert f[..., 2].min() > 240 and f[..., 0].max() < 15 and f[..., 1].max() < 15  # red in BGR order


def test_stage1_detections_and_ppe_rows(client, stub):
    resp = post_file(client, STAGE1, "photo.jpg", image_bytes(200, 100), {"model": "yolov8s"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "yolov8s" and stub.stage1_calls[0][0] == "yolov8s"
    frame = body["frames"][0]
    expected = stub_detections(np.zeros((100, 200, 3), np.uint8))
    assert len(frame["detections"]) == len(expected)
    confs = [d["confidence"] for d in frame["detections"]]
    assert confs == sorted(confs, reverse=True)
    for got, want in zip(frame["detections"], expected):
        assert got["class_name"] == want.class_name
        assert got["class_id"] == CLASS_TO_ID[want.class_name]
        assert got["confidence"] == pytest.approx(want.confidence, abs=1e-4)
        assert got["xyxy"] == pytest.approx(list(want.xyxy), abs=1e-3)
        assert got["confidence"] >= body["thresholds"][got["class_name"]]
    rows = assess_ppe(expected, body["ppe_min_confidence"])
    assert len(frame["ppe"]) == len(rows) == 2
    for got, want in zip(frame["ppe"], rows):
        assert set(got) == set(want)
        assert got["person_confidence"] == pytest.approx(want["person_confidence"], abs=1e-4)
        assert {k: v for k, v in got.items() if k != "person_confidence"} == \
               {k: v for k, v in want.items() if k != "person_confidence"}
    assert frame["ppe"][0]["alerts"] == [] and frame["ppe"][1]["alerts"] == ["no_helmet", "no_vest"]


@pytest.mark.parametrize("name,fmt", [("a.png", "PNG"), ("a.bmp", "BMP"), ("a.webp", "WEBP"),
                                      ("a.jpeg", "JPEG"), ("UPPER.JPG", "JPEG")])
def test_stage1_accepts_all_image_extensions(client, name, fmt):
    resp = post_file(client, STAGE1, name, image_bytes(64, 48, fmt=fmt), {"model": "yolov8n"}, "text/plain")
    assert resp.status_code == 200, resp.text
    assert resp.json()["media_type"] == "image"


def test_stage1_image_ignores_max_frames(client, stub):
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n", "max_frames": "5"})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["frames"]) == 1


# ---------------------------------------------------------------- stage 1: video
def test_stage1_video_matches_fixture(client, stub, tmp_path):
    path = write_video(tmp_path / "clip.mp4", n_frames=50, fps=25.0, width=160, height=120)
    resp = post_file(client, STAGE1, "clip.mp4", path.read_bytes(), {"model": "yolov8s", "max_frames": "4"}, "video/mp4")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_video.json"))
    check_envelope(body, "yolov8s")
    assert body["media_type"] == "video"
    src = body["source"]
    assert (src["width"], src["height"], src["frame_count"]) == (160, 120, 50)
    assert src["fps"] == pytest.approx(25.0) and src["duration_s"] == pytest.approx(2.0)
    indices = [f["index"] for f in body["frames"]]
    assert indices == expected_stage1_indices(50, 4) == [0, 16, 33, 49]
    for f in body["frames"]:
        assert f["time_s"] == pytest.approx(f["index"] / 25.0, abs=1e-4)
        assert decode_b64_jpeg(f["image_jpeg_b64"]).size == (160, 120)
        assert len(f["detections"]) == 5
    assert len(stub.stage1_calls[0][1]) == 4


def test_stage1_video_samples_the_right_frames(client, stub, tmp_path):
    path = write_video(tmp_path / "ids.avi", n_frames=30, fps=10.0)
    resp = post_file(client, STAGE1, "ids.avi", path.read_bytes(), {"model": "yolov8n", "max_frames": "5"})
    assert resp.status_code == 200, resp.text
    want = expected_stage1_indices(30, 5)
    assert [f["index"] for f in resp.json()["frames"]] == want
    (_, frames), = stub.stage1_calls
    assert [gray_index(f) for f in frames] == want


@pytest.mark.parametrize("n_frames,max_frames,expected_len", [
    (40, None, 16),   # default max_frames = 16
    (6, "60", 6),     # fewer frames than requested -> every frame
    (40, "1", 1),     # single frame -> frame 0
    (80, "60", 60),   # upper bound
])
def test_stage1_video_max_frames(client, stub, tmp_path, n_frames, max_frames, expected_len):
    path = write_video(tmp_path / "v.mp4", n_frames=n_frames, fps=20.0)
    data = {"model": "yolov8n"}
    if max_frames is not None:
        data["max_frames"] = max_frames
    resp = post_file(client, STAGE1, "v.mp4", path.read_bytes(), data)
    assert resp.status_code == 200, resp.text
    frames = resp.json()["frames"]
    assert len(frames) == expected_len
    mf = 16 if max_frames is None else int(max_frames)
    assert [f["index"] for f in frames] == expected_stage1_indices(n_frames, mf)
    assert frames[0]["index"] == 0
    if expected_len > 1:
        assert frames[-1]["index"] == n_frames - 1


# ---------------------------------------------------------------- stage 2
def test_stage2_matches_fixture_and_windows(client, stub, tmp_path):
    path = write_video(tmp_path / "fight.avi", n_frames=54, fps=10.0)  # 5.4 s like the fixture
    resp = post_file(client, STAGE2, "fight.avi", path.read_bytes(), ctype="video/x-msvideo")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage2_video.json"))
    check_envelope(body, "x3d_s")
    assert body["class_names"] == ["non_fight", "fight"]
    assert body["frames_per_window"] == 13 and body["window_s"] == 2.0
    src = body["source"]
    assert (src["width"], src["height"], src["frame_count"]) == (64, 48, 54)
    assert src["fps"] == pytest.approx(10.0) and src["duration_s"] == pytest.approx(5.4)

    spans = [(w["start_s"], w["end_s"]) for w in body["windows"]]
    assert spans == pytest.approx([(0.0, 2.0), (2.0, 4.0), (4.0, 5.4)])
    assert [w["index"] for w in body["windows"]] == [0, 1, 2]
    for w, fight in zip(body["windows"], STUB_FIGHT):
        assert w["probs"]["fight"] == pytest.approx(fight, abs=1e-4)
        assert w["probs"]["non_fight"] + w["probs"]["fight"] == pytest.approx(1.0, abs=2e-4)
    # 0.0868 -> non_fight, 0.5 tie -> non_fight, 0.7915 -> fight
    assert [w["label"] for w in body["windows"]] == ["non_fight", "non_fight", "fight"]
    assert body["summary"] == {"max_fight_prob": pytest.approx(0.7915, abs=1e-4), "fight_windows": 1,
                               "total_windows": 3}

    # the registry got 13 BGR frames per window, taken from the contract's frame ranges
    (clips,) = stub.stage2_calls
    assert len(clips) == 3
    for clip, (start, end) in zip(clips, [(0.0, 2.0), (2.0, 4.0), (4.0, 5.4)]):
        assert len(clip) == 13
        # section 8 only fixes "FRAMES BGR frames"; the API may pre-letterbox them
        assert all(f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8 for f in clip)
        first, last = round(start * 10), min(round(end * 10), 54) - 1
        assert [gray_index(f) for f in clip] == [int(i) for i in sample_indices(first, last)]


def test_stage2_clips_preprocess_like_original_frames(client, stub, tmp_path):
    """Whatever the API hands the registry must give the same X3D-S tensor as the decoded source frames."""
    import cv2
    import torch

    from cctv_safety.stage2 import preprocess_frames

    rng = np.random.default_rng(3)
    path = write_video(tmp_path / "noise.avi", n_frames=35, fps=10.0, width=160, height=120,
                       frame_fn=lambda i: rng.integers(0, 256, (120, 160, 3), dtype=np.uint8))
    cap = cv2.VideoCapture(str(path))
    originals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        originals.append(frame)
    cap.release()
    assert len(originals) == 35
    resp = post_file(client, STAGE2, "noise.avi", path.read_bytes())
    assert resp.status_code == 200, resp.text
    (clips,) = stub.stage2_calls
    assert len(clips) == 2  # 0-2 s and 2-3.5 s
    for clip, (first, last) in zip(clips, [(0, 19), (20, 34)]):
        want = preprocess_frames([originals[i] for i in sample_indices(first, last)])
        got = preprocess_frames(clip)
        assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("n_frames,fps,spans", [
    (45, 10.0, [(0.0, 2.0), (2.0, 4.0)]),               # trailing 0.5 s dropped
    (30, 10.0, [(0.0, 2.0), (2.0, 3.0)]),               # trailing exactly 1.0 s kept
    (29, 10.0, [(0.0, 2.0)]),                           # trailing 0.9 s dropped
    (20, 10.0, [(0.0, 2.0)]),                           # exactly 2 s -> one window
    (12, 10.0, [(0.0, 1.2)]),                           # < 2 s -> one window over the whole clip
    (3, 10.0, [(0.0, 0.3)]),                            # very short clip still one window
    (85, 10.0, [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]),
], ids=["4.5s", "3.0s", "2.9s", "2.0s", "1.2s", "0.3s", "8.5s"])
def test_stage2_windowing_rules(client, stub, tmp_path, n_frames, fps, spans):
    path = write_video(tmp_path / "w.mp4", n_frames=n_frames, fps=fps)
    resp = post_file(client, STAGE2, "w.mp4", path.read_bytes())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    got = [(w["start_s"], w["end_s"]) for w in body["windows"]]
    assert got == pytest.approx(spans)
    assert body["summary"]["total_windows"] == len(spans)
    assert len(stub.stage2_calls[0]) == len(spans)
    assert all(len(clip) == 13 for clip in stub.stage2_calls[0])


def test_stage2_thirty_windows_allowed(client, stub, tmp_path):
    path = write_video(tmp_path / "long.mp4", n_frames=300, fps=5.0, width=32, height=24)  # 60 s
    resp = post_file(client, STAGE2, "long.mp4", path.read_bytes())
    assert resp.status_code == 200, resp.text
    assert resp.json()["summary"]["total_windows"] == 30


def test_stage2_more_than_thirty_windows_is_422(client, stub, tmp_path):
    path = write_video(tmp_path / "toolong.mp4", n_frames=305, fps=5.0, width=32, height=24)  # 61 s
    resp = post_file(client, STAGE2, "toolong.mp4", path.read_bytes())
    assert_error(resp, 422, "video_too_long")
    assert not stub.stage2_calls


def test_stage2_image_is_422_video_required(client, stub):
    resp = post_file(client, STAGE2, "photo.jpg", image_bytes(), ctype="image/jpeg")
    body = assert_error(resp, 422, "video_required")
    assert_structure(body, load_fixture("error_stage2_image.json"))
    assert not stub.stage2_calls


# ---------------------------------------------------------------- errors
@pytest.fixture
def small_limit(monkeypatch):
    """Shrink the 100 MB upload limit wherever webapp holds it instead of sending 100 MB."""
    patched = 0
    for name, module in list(sys.modules.items()):
        if name == "webapp" or name.startswith("webapp."):
            for attr, value in list(vars(module).items()):
                if attr.isupper() and isinstance(value, int) and value == DEFAULT_LIMIT:
                    monkeypatch.setattr(module, attr, 1000)
                    patched += 1
    assert patched, "no 100 MB upload limit constant found in webapp.*"
    return 1000


@pytest.mark.parametrize("url,name,data", [(STAGE1, "big.png", {"model": "yolov8n"}), (STAGE2, "big.mp4", {})])
def test_oversize_upload_is_413(client, stub, small_limit, url, name, data):
    resp = post_file(client, url, name, b"\x00" * 5000, data)
    assert_error(resp, 413, "file_too_large")
    assert not stub.stage1_calls and not stub.stage2_calls


def test_upload_under_patched_limit_is_accepted(client, small_limit):
    content = image_bytes(16, 16, fmt="PNG")
    assert len(content) < small_limit
    resp = post_file(client, STAGE1, "tiny.png", content, {"model": "yolov8n"})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("url,data", [(STAGE1, {"model": "yolov8n"}), (STAGE2, {})])
@pytest.mark.parametrize("name", ["notes.txt", "anim.gif", "noext", "clip.webm", "x.jpg.exe"])
def test_bad_extension_is_415(client, url, data, name):
    resp = post_file(client, url, name, image_bytes(), data, "image/jpeg")
    assert_error(resp, 415, "unsupported_media_type")


@pytest.mark.parametrize("url,data", [(STAGE1, {"model": "yolov8n"}), (STAGE2, {})])
def test_missing_file_is_400(client, url, data):
    resp = client.post(url, data=data)
    assert_error(resp, 400, "missing_file")


@pytest.mark.parametrize("url,data", [(STAGE1, {"model": "yolov8n"}), (STAGE2, {})])
def test_empty_file_is_400(client, url, data):
    resp = post_file(client, url, "empty.jpg", b"", data)
    assert_error(resp, 400, "missing_file")


@pytest.mark.parametrize("data", [
    {},
    {"model": ""},
    {"model": "yolov8x"},
    {"model": "x3d_s"},
    {"model": "yolov8n", "max_frames": "0"},
    {"model": "yolov8n", "max_frames": "61"},
    {"model": "yolov8n", "max_frames": "-3"},
    {"model": "yolov8n", "max_frames": "abc"},
    {"model": "yolov8n", "max_frames": "1.5"},
], ids=["no_model", "empty_model", "yolov8x", "x3d_s", "mf0", "mf61", "mf_neg", "mf_abc", "mf_float"])
def test_invalid_parameters_are_400(client, stub, data):
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), data)
    assert_error(resp, 400, "invalid_parameter")
    assert not stub.stage1_calls


def test_invalid_parameter_checked_before_media_type(client):
    resp = post_file(client, STAGE1, "notes.txt", b"hello", {"model": "nope"})
    assert_error(resp, 400, "invalid_parameter")


@pytest.mark.parametrize("url,name,data", [
    (STAGE1, "broken.jpg", {"model": "yolov8n"}),
    (STAGE1, "broken.mp4", {"model": "yolov8n"}),
    (STAGE2, "broken.mp4", {}),
])
def test_undecodable_file_is_422(client, url, name, data):
    resp = post_file(client, url, name, b"this is not media" * 50, data)
    assert_error(resp, 422, "decode_failed")


def test_weights_missing_stage1_image_is_503(missing_client):
    resp = post_file(missing_client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n"})
    body = assert_error(resp, 503, "weights_missing")
    assert_structure(body, load_fixture("error_weights_missing.json"))


def test_weights_missing_stage1_video_is_503(missing_client, tmp_path):
    path = write_video(tmp_path / "v.mp4", n_frames=10, fps=10.0)
    resp = post_file(missing_client, STAGE1, "v.mp4", path.read_bytes(), {"model": "yolov8s"})
    assert_error(resp, 503, "weights_missing")


def test_weights_missing_stage2_is_503(missing_client, tmp_path):
    path = write_video(tmp_path / "v.mp4", n_frames=30, fps=10.0)
    resp = post_file(missing_client, STAGE2, "v.mp4", path.read_bytes())
    assert_error(resp, 503, "weights_missing")


def test_video_required_checked_before_weights(missing_client):
    resp = post_file(missing_client, STAGE2, "a.png", image_bytes(fmt="PNG"))
    assert_error(resp, 422, "video_required")


# ---------------------------------------------------------------- real weights smoke test
def test_real_weights_smoke(tmp_path):
    require_weights(*YOLO_WEIGHTS.values(), X3D_WEIGHTS)
    client = TestClient(create_app(weights_dir=WEIGHTS_DIR))

    health = client.get("/api/health").json()
    assert set(health["models"]) == {"yolov8n", "yolov8s", "x3d_s"}
    assert all(v in ("available", "loaded") for v in health["models"].values())

    resp = post_file(client, STAGE1, "scene.jpg", image_bytes(320, 240, (90, 120, 150)), {"model": "yolov8n"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_image.json"))
    check_envelope(body, "yolov8n")
    for det in body["frames"][0]["detections"]:
        assert det["confidence"] >= body["thresholds"][det["class_name"]]

    video = write_video(tmp_path / "real.mp4", n_frames=75, fps=25.0, width=160, height=120,
                        frame_fn=lambda i: np.full((120, 160, 3), (i * 3) % 256, np.uint8))
    resp = post_file(client, STAGE1, "real.mp4", video.read_bytes(), {"model": "yolov8s", "max_frames": "3"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_video.json"))
    assert [f["index"] for f in body["frames"]] == [0, 37, 74]

    resp = post_file(client, STAGE2, "real.mp4", video.read_bytes())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage2_video.json"))
    check_envelope(body, "x3d_s")
    assert [(w["start_s"], w["end_s"]) for w in body["windows"]] == pytest.approx([(0.0, 2.0), (2.0, 3.0)])
    for w in body["windows"]:
        assert w["probs"]["non_fight"] + w["probs"]["fight"] == pytest.approx(1.0, abs=2e-4)
        assert w["label"] in ("non_fight", "fight")
    assert client.get("/api/health").json()["models"] == {"yolov8n": "loaded", "yolov8s": "loaded", "x3d_s": "loaded"}


# ---------------------------------------------------------------- model_load_failed (real ModelRegistry)
def _wrong_schema_yolo(path: Path) -> Path:
    from ultralytics import YOLO

    YOLO("yolov8n.yaml").save(str(path))  # 80 generic classes, built offline
    return path


@pytest.fixture
def broken_weights_dir(tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    _wrong_schema_yolo(wdir / "YOLOv8n_best.pt")                            # wrong class schema
    (wdir / "YOLOv8s_best.pt").write_bytes(b"not a torch checkpoint" * 20)  # corrupt
    (wdir / "X3D-S_best.pt").write_bytes(b"\x80\x04corrupt" * 50)           # corrupt
    return wdir


@pytest.mark.parametrize("model", ["yolov8n", "yolov8s"], ids=["wrong_schema", "corrupt"])
def test_stage1_model_load_failed_is_503(broken_weights_dir, model):
    client = TestClient(create_app(weights_dir=broken_weights_dir))
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": model})
    assert_error(resp, 503, "model_load_failed")


def test_stage2_model_load_failed_is_503(broken_weights_dir, tmp_path):
    client = TestClient(create_app(weights_dir=broken_weights_dir))
    video = write_video(tmp_path / "v.mp4", n_frames=30, fps=10.0)
    resp = post_file(client, STAGE2, "v.mp4", video.read_bytes())
    assert_error(resp, 503, "model_load_failed")


def test_real_registry_weights_missing_is_503(tmp_path):
    empty = tmp_path / "empty_weights"
    empty.mkdir()
    client = TestClient(create_app(weights_dir=empty))
    assert client.get("/api/health").json()["models"] == {"yolov8n": "missing", "yolov8s": "missing",
                                                          "x3d_s": "missing"}
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n"})
    assert_error(resp, 503, "weights_missing")


# ---------------------------------------------------------------- ModelRegistry unit tests
class _FakeYolo:
    names = {i: n for i, n in enumerate(CLASS_NAMES)}

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def counting_registry(tmp_path, monkeypatch):
    """Real ModelRegistry over dummy weight files; the section-8 loaders are patched to count calls."""
    import threading
    import time

    import torch

    from cctv_safety import detector as det_mod
    from cctv_safety import stage2 as s2_mod
    from webapp.models import ModelRegistry

    wdir = tmp_path / "weights"
    wdir.mkdir()
    for fname in ("YOLOv8n_best.pt", "YOLOv8s_best.pt", "X3D-S_best.pt"):
        (wdir / fname).write_bytes(b"dummy")
    calls = {"yolo": 0, "x3d": 0}
    lock = threading.Lock()

    def fake_load_yolo(path):
        with lock:
            calls["yolo"] += 1
        time.sleep(0.2)  # widen the race window
        return _FakeYolo()

    def fake_load_x3d(path, device="cpu"):
        with lock:
            calls["x3d"] += 1
        time.sleep(0.2)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(det_mod, "load_yolo", fake_load_yolo)
    monkeypatch.setattr(det_mod, "detect", lambda model, frame, th: [])
    monkeypatch.setattr(s2_mod, "load_x3d_s", fake_load_x3d)
    monkeypatch.setattr(s2_mod, "predict_probs", lambda model, batch: [(0.6, 0.4)] * len(batch))
    return ModelRegistry(wdir, thresholds()), calls


def _models_of(status: dict) -> dict:
    models = status.get("models", status)
    return {k: models[k] for k in ("yolov8n", "yolov8s", "x3d_s")}


def test_registry_status_never_loads(counting_registry):
    reg, calls = counting_registry
    for _ in range(5):
        st = reg.status()
    assert _models_of(st) == {"yolov8n": "available", "yolov8s": "available", "x3d_s": "available"}
    assert "device" in st
    assert calls == {"yolo": 0, "x3d": 0}


def test_registry_loads_lazily_and_caches(counting_registry):
    reg, calls = counting_registry
    frame = np.zeros((48, 64, 3), np.uint8)
    assert calls["yolo"] == 0
    assert reg.stage1("yolov8n", [frame]) == [[]]
    assert calls["yolo"] == 1
    assert _models_of(reg.status()) == {"yolov8n": "loaded", "yolov8s": "available", "x3d_s": "available"}
    reg.stage1("yolov8n", [frame, frame])
    assert calls["yolo"] == 1  # cached
    reg.stage1("yolov8s", [frame])
    assert calls["yolo"] == 2
    assert calls["x3d"] == 0
    assert [tuple(p) for p in reg.stage2([[frame] * 13])] == [(0.6, 0.4)]
    assert calls["x3d"] == 1
    reg.stage2([[frame] * 13, [frame] * 13])
    assert calls["x3d"] == 1
    assert _models_of(reg.status()) == {"yolov8n": "loaded", "yolov8s": "loaded", "x3d_s": "loaded"}


def test_registry_concurrent_first_use_loads_once(counting_registry):
    import threading

    reg, calls = counting_registry
    frame = np.zeros((48, 64, 3), np.uint8)
    n = 8
    barrier = threading.Barrier(n)
    errors = []

    def worker(i):
        try:
            barrier.wait()
            if i % 2:
                reg.stage1("yolov8n", [frame])
            else:
                reg.stage2([[frame] * 13])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert calls == {"yolo": 1, "x3d": 1}


def test_registry_missing_and_failing_weights(tmp_path, monkeypatch):
    from cctv_safety import detector as det_mod
    from webapp.models import ModelLoadFailed, ModelRegistry

    wdir = tmp_path / "w"
    wdir.mkdir()
    (wdir / "YOLOv8s_best.pt").write_bytes(b"dummy")
    reg = ModelRegistry(wdir, thresholds())
    assert _models_of(reg.status())["yolov8n"] == "missing"
    frame = np.zeros((48, 64, 3), np.uint8)
    with pytest.raises(WeightsMissing):
        reg.stage1("yolov8n", [frame])

    def boom(path):
        raise RuntimeError("cannot unpickle")

    monkeypatch.setattr(det_mod, "load_yolo", boom)
    with pytest.raises(ModelLoadFailed):
        reg.stage1("yolov8s", [frame])
    assert _models_of(reg.status())["yolov8s"] == "available"


# ---------------------------------------------------------------- temp-file cleanup and 500 / 404
class RaisingRegistry(StubRegistry):
    def stage1(self, name, frames_bgr):
        raise RuntimeError("secret-internal-detail boom")

    def stage2(self, clips):
        raise RuntimeError("secret-internal-detail boom")


@pytest.fixture
def isolated_tmp(tmp_path, monkeypatch):
    import tempfile

    d = tmp_path / "systmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _leftovers(d: Path):
    return sorted(p.name for p in d.glob("cctv_upload_*"))


def test_temp_files_removed_after_success(client, isolated_tmp, tmp_path):
    assert post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n"}).status_code == 200
    video = write_video(tmp_path / "v.mp4", n_frames=30, fps=10.0)
    assert post_file(client, STAGE1, "v.mp4", video.read_bytes(), {"model": "yolov8n"}).status_code == 200
    assert post_file(client, STAGE2, "v.mp4", video.read_bytes()).status_code == 200
    assert _leftovers(isolated_tmp) == []


def test_temp_files_removed_after_decode_failed(client, isolated_tmp):
    assert_error(post_file(client, STAGE1, "bad.jpg", b"garbage" * 100, {"model": "yolov8n"}), 422, "decode_failed")
    assert_error(post_file(client, STAGE1, "bad.mp4", b"garbage" * 100, {"model": "yolov8n"}), 422, "decode_failed")
    assert_error(post_file(client, STAGE2, "bad.mp4", b"garbage" * 100), 422, "decode_failed")
    assert _leftovers(isolated_tmp) == []


def test_temp_files_removed_after_video_too_long(client, isolated_tmp, tmp_path):
    video = write_video(tmp_path / "long.mp4", n_frames=305, fps=5.0, width=32, height=24)
    assert_error(post_file(client, STAGE2, "long.mp4", video.read_bytes()), 422, "video_too_long")
    assert _leftovers(isolated_tmp) == []


@pytest.fixture
def raising_client():
    return TestClient(create_app(registry=RaisingRegistry()), raise_server_exceptions=False)


def _assert_internal_error(resp):
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == "internal_error"
    assert "Traceback" not in resp.text and 'File "' not in resp.text and ".py" not in resp.text


def test_inference_runtime_error_is_500_and_cleans_temp(raising_client, isolated_tmp, tmp_path):
    _assert_internal_error(post_file(raising_client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n"}))
    video = write_video(tmp_path / "v.mp4", n_frames=30, fps=10.0)
    _assert_internal_error(post_file(raising_client, STAGE1, "v.mp4", video.read_bytes(), {"model": "yolov8s"}))
    _assert_internal_error(post_file(raising_client, STAGE2, "v.mp4", video.read_bytes()))
    assert _leftovers(isolated_tmp) == []


@pytest.mark.parametrize("url", ["/api/xyz", "/api/stage3/analyze", "/api/"])
def test_unknown_api_route_is_404_not_found(client, url):
    resp = client.get(url)
    assert resp.status_code == 404, resp.text
    body = resp.json()
    assert set(body) == {"error"} and body["error"]["code"] == "not_found"
    assert isinstance(body["error"]["message"], str)


# ---------------------------------------------------------------- stage 1 dense mode (contract section 2, round 2)
def dense_indices(frame_count: int, fps: float, sample_fps: float) -> list[int]:
    """Contract rule, verbatim: t_k = k/sample_fps while t_k < duration; index = min(int(round(k/sample_fps*fps)),
    frame_count-1); de-duplicated keeping the first occurrence."""
    duration = frame_count / fps
    out, k = [], 0
    while k / sample_fps < duration:
        index = min(int(round(k / sample_fps * fps)), frame_count - 1)
        if index not in out:
            out.append(index)
        k += 1
    return out


def dense_frames_received(stub, model: str) -> list[np.ndarray]:
    """Dense mode may call registry.stage1 in batches (webapp.config.DENSE_BATCH); join them in order."""
    from webapp import config as web_config

    assert stub.stage1_calls
    for name, frames in stub.stage1_calls:
        assert name == model
        assert 0 < len(frames) <= web_config.DENSE_BATCH
    return [fr for _, frames in stub.stage1_calls for fr in frames]


def test_dense_rule_helper_matches_contract_example():
    got = dense_indices(240, 30.0, 10.0)  # 8.0 s at 30 fps
    assert len(got) == 80 and got[:3] == [0, 3, 6] and got[-1] == 237


def test_stage1_dense_matches_fixture(client, stub, tmp_path):
    path = write_video(tmp_path / "d.avi", n_frames=10, fps=10.0, width=160, height=120)
    resp = post_file(client, STAGE1, "d.avi", path.read_bytes(),
                     {"model": "yolov8s", "mode": "dense", "sample_fps": "10"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_video_dense.json"))
    check_envelope(body, "yolov8s")
    assert body["mode"] == "dense" and body["sample_fps"] == 10.0
    assert body["media_type"] == "video"
    assert [f["index"] for f in body["frames"]] == list(range(10))
    for f in body["frames"]:
        assert f["image_jpeg_b64"] is None
        assert f["time_s"] == pytest.approx(f["index"] / 10.0, abs=1e-4)
        assert len(f["detections"]) == 5
        assert f["ppe"] == [{**row, "person_confidence": pytest.approx(row["person_confidence"], abs=1e-4)}
                            for row in assess_ppe(stub_detections(np.zeros((120, 160, 3), np.uint8)),
                                                  body["ppe_min_confidence"])]
    frames = dense_frames_received(stub, "yolov8s")
    assert [gray_index(fr) for fr in frames] == list(range(10))


@pytest.mark.parametrize("n_frames,fps,sample_fps", [
    (40, 20.0, "10"),     # every 2nd frame
    (40, 20.0, "3"),      # non-integer stride, no ties
    (25, 10.0, "4"),      # x.5 ties -> round half to even
    (35, 20.0, "7"),
    (30, 10.0, "12.5"),   # sample_fps above video fps -> every frame
    (30, 10.0, "30"),     # upper bound -> every frame
    (25, 10.0, "1"),      # lower bound
    (3, 10.0, "1"),       # very short clip -> one sample
], ids=["20fps@10", "20fps@3", "10fps@4_ties", "20fps@7", "10fps@12.5", "10fps@30", "10fps@1", "short"])
def test_stage1_dense_sampling_rule(client, stub, tmp_path, n_frames, fps, sample_fps):
    path = write_video(tmp_path / "s.avi", n_frames=n_frames, fps=fps)
    resp = post_file(client, STAGE1, "s.avi", path.read_bytes(),
                     {"model": "yolov8n", "mode": "dense", "sample_fps": sample_fps, "max_frames": "2"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    want = dense_indices(n_frames, fps, float(sample_fps))
    assert [f["index"] for f in body["frames"]] == want  # max_frames ignored in dense mode
    assert body["sample_fps"] == pytest.approx(float(sample_fps))
    if float(sample_fps) >= fps:
        assert want == list(range(n_frames))
    for f in body["frames"]:
        assert f["time_s"] == pytest.approx(f["index"] / fps, abs=1e-4)
        assert f["image_jpeg_b64"] is None
    frames = dense_frames_received(stub, "yolov8n")
    assert [gray_index(fr) for fr in frames] == want


@pytest.mark.parametrize("extra", [{}, {"sample_fps": ""}], ids=["omitted", "empty"])
def test_stage1_dense_default_sample_fps_is_10(client, tmp_path, extra):
    path = write_video(tmp_path / "s.avi", n_frames=40, fps=20.0)
    resp = post_file(client, STAGE1, "s.avi", path.read_bytes(), {"model": "yolov8n", "mode": "dense", **extra})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sample_fps"] == 10.0
    assert [f["index"] for f in body["frames"]] == dense_indices(40, 20.0, 10.0)


def test_stage1_dense_ignores_invalid_max_frames(client, tmp_path):
    path = write_video(tmp_path / "s.avi", n_frames=20, fps=10.0)
    resp = post_file(client, STAGE1, "s.avi", path.read_bytes(),
                     {"model": "yolov8n", "mode": "dense", "max_frames": "999"})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["frames"]) == 20


@pytest.mark.parametrize("extra", [{}, {"mode": "frames"}, {"mode": ""}, {"mode": "frames", "sample_fps": "999"}],
                         ids=["default", "explicit", "empty", "sample_fps_ignored"])
def test_stage1_frames_mode_unchanged(client, stub, tmp_path, extra):
    path = write_video(tmp_path / "f.avi", n_frames=30, fps=10.0)
    resp = post_file(client, STAGE1, "f.avi", path.read_bytes(), {"model": "yolov8n", "max_frames": "5", **extra})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_video.json"))
    assert body["mode"] == "frames" and body["sample_fps"] is None
    assert [f["index"] for f in body["frames"]] == expected_stage1_indices(30, 5)
    assert all(isinstance(f["image_jpeg_b64"], str) for f in body["frames"])


def test_stage1_image_is_frames_mode(client):
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n"})
    body = resp.json()
    assert body["mode"] == "frames" and body["sample_fps"] is None


def test_stage1_dense_image_is_422_video_required(client, stub):
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n", "mode": "dense"})
    assert_error(resp, 422, "video_required")
    assert not stub.stage1_calls


def test_stage1_dense_sixty_seconds_allowed(client, stub, tmp_path):
    path = write_video(tmp_path / "long.mp4", n_frames=300, fps=5.0, width=32, height=24)  # 60 s
    resp = post_file(client, STAGE1, "long.mp4", path.read_bytes(), {"model": "yolov8n", "mode": "dense"})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["frames"]) == 300  # sample_fps 10 > 5 fps -> every frame
    assert len(dense_frames_received(stub, "yolov8n")) == 300


def test_stage1_dense_too_long_is_422(client, stub, tmp_path):
    path = write_video(tmp_path / "toolong.mp4", n_frames=305, fps=5.0, width=32, height=24)  # 61 s
    resp = post_file(client, STAGE1, "toolong.mp4", path.read_bytes(), {"model": "yolov8n", "mode": "dense"})
    assert_error(resp, 422, "video_too_long")
    assert not stub.stage1_calls


def test_stage1_frames_mode_long_video_still_allowed(client, tmp_path):
    path = write_video(tmp_path / "toolong.mp4", n_frames=305, fps=5.0, width=32, height=24)  # 61 s
    resp = post_file(client, STAGE1, "toolong.mp4", path.read_bytes(), {"model": "yolov8n"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mode"] == "frames"


@pytest.mark.parametrize("data", [
    {"mode": "sparse"},
    {"mode": "DENSE"},
    {"mode": "dense", "sample_fps": "0"},
    {"mode": "dense", "sample_fps": "0.5"},
    {"mode": "dense", "sample_fps": "30.01"},
    {"mode": "dense", "sample_fps": "31"},
    {"mode": "dense", "sample_fps": "-5"},
    {"mode": "dense", "sample_fps": "abc"},
    {"mode": "dense", "sample_fps": "nan"},
    {"mode": "dense", "sample_fps": "inf"},
], ids=["sparse", "upper", "sf0", "sf0.5", "sf30.01", "sf31", "sf_neg", "sf_abc", "sf_nan", "sf_inf"])
def test_stage1_dense_invalid_parameters_are_400(client, stub, tmp_path, data):
    path = write_video(tmp_path / "v.mp4", n_frames=10, fps=10.0)
    resp = post_file(client, STAGE1, "v.mp4", path.read_bytes(), {"model": "yolov8n", **data})
    assert_error(resp, 400, "invalid_parameter")
    assert not stub.stage1_calls


def test_stage1_dense_invalid_parameter_before_video_required(client):
    resp = post_file(client, STAGE1, "a.jpg", image_bytes(), {"model": "yolov8n", "mode": "dense", "sample_fps": "99"})
    assert_error(resp, 400, "invalid_parameter")


def test_temp_files_removed_in_dense_mode(client, isolated_tmp, tmp_path):
    ok = write_video(tmp_path / "ok.mp4", n_frames=20, fps=10.0)
    assert post_file(client, STAGE1, "ok.mp4", ok.read_bytes(), {"model": "yolov8n", "mode": "dense"}).status_code == 200
    long = write_video(tmp_path / "long.mp4", n_frames=305, fps=5.0, width=32, height=24)
    assert_error(post_file(client, STAGE1, "long.mp4", long.read_bytes(), {"model": "yolov8n", "mode": "dense"}),
                 422, "video_too_long")
    assert_error(post_file(client, STAGE1, "bad.mp4", b"garbage" * 100, {"model": "yolov8n", "mode": "dense"}),
                 422, "decode_failed")
    assert _leftovers(isolated_tmp) == []


def test_temp_files_removed_in_dense_mode_on_internal_error(raising_client, isolated_tmp, tmp_path):
    ok = write_video(tmp_path / "ok.mp4", n_frames=20, fps=10.0)
    _assert_internal_error(post_file(raising_client, STAGE1, "ok.mp4", ok.read_bytes(),
                                     {"model": "yolov8n", "mode": "dense"}))
    assert _leftovers(isolated_tmp) == []


def test_real_weights_dense_smoke(tmp_path):
    require_weights(*YOLO_WEIGHTS.values())
    client = TestClient(create_app(weights_dir=WEIGHTS_DIR))
    video = write_video(tmp_path / "real.mp4", n_frames=75, fps=25.0, width=160, height=120,
                        frame_fn=lambda i: np.full((120, 160, 3), (i * 3) % 256, np.uint8))
    resp = post_file(client, STAGE1, "real.mp4", video.read_bytes(),
                     {"model": "yolov8n", "mode": "dense", "sample_fps": "5"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_structure(body, load_fixture("stage1_video_dense.json"))
    check_envelope(body, "yolov8n")
    assert body["mode"] == "dense" and body["sample_fps"] == 5.0
    assert [f["index"] for f in body["frames"]] == dense_indices(75, 25.0, 5.0) == list(range(0, 75, 5))
    for f in body["frames"]:
        assert f["image_jpeg_b64"] is None
        for det in f["detections"]:
            assert det["confidence"] >= body["thresholds"][det["class_name"]]
