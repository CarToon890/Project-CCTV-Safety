"""Stage 1 detector contract."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from cctv_safety import detector
from cctv_safety.ppe import Detection
from cctv_safety.schema import CLASS_NAMES, CLASS_NAMES_V1
from conftest import REPO_ROOT, YOLO_WEIGHTS, require_weights

GOOD_THRESHOLDS = {name: 0.2 for name in CLASS_NAMES}


# ---------- validate_runtime_contract ----------
def test_contract_accepts_list_tuple_and_dict_names():
    detector.validate_runtime_contract(list(CLASS_NAMES), GOOD_THRESHOLDS)
    detector.validate_runtime_contract(tuple(CLASS_NAMES), GOOD_THRESHOLDS)
    detector.validate_runtime_contract({i: n for i, n in enumerate(CLASS_NAMES)}, GOOD_THRESHOLDS)


@pytest.mark.parametrize("names", [
    list(CLASS_NAMES_V1),                                   # legacy 7-class schema with fight
    list(CLASS_NAMES[:-1]),                                 # missing a class
    ["helmet", "person", "vest", "fall", "fire", "smoke"],  # reordered
    list(reversed(CLASS_NAMES)),
    ["person", "helmet", "vest", "fall", "fire", "Smoke"],  # renamed
    [str(i) for i in range(80)],                            # untrained/COCO-style head
], ids=["legacy_v1", "missing", "swap", "reversed", "renamed", "coco80"])
def test_contract_rejects_wrong_names(names):
    with pytest.raises(ValueError):
        detector.validate_runtime_contract(names, GOOD_THRESHOLDS)
    with pytest.raises(ValueError):
        detector.validate_runtime_contract({i: n for i, n in enumerate(names)}, GOOD_THRESHOLDS)


@pytest.mark.parametrize("thresholds", [
    {n: 0.2 for n in CLASS_NAMES[:-1]},                     # missing key
    {**GOOD_THRESHOLDS, "fight": 0.2},                      # extra key
    {**{n: 0.2 for n in CLASS_NAMES[:-1]}, "Smoke": 0.2},   # misspelt key
    {},
], ids=["missing", "extra", "misspelt", "empty"])
def test_contract_rejects_threshold_key_mismatch(thresholds):
    with pytest.raises(ValueError):
        detector.validate_runtime_contract(list(CLASS_NAMES), thresholds)


# ---------- load_thresholds ----------
def test_load_thresholds_default_config():
    path = REPO_ROOT / "configs" / "thresholds.yaml"
    got = detector.load_thresholds(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(got) == set(CLASS_NAMES)
    assert all(isinstance(v, float) for v in got.values())
    assert got == {k: float(v) for k, v in raw.items()}
    detector.validate_runtime_contract(list(CLASS_NAMES), got)


# ---------- load_yolo rejects a model with the wrong schema (no real weights needed) ----------
def test_load_yolo_rejects_wrong_class_names(tmp_path):
    from ultralytics import YOLO

    path = tmp_path / "coco_untrained.pt"
    YOLO("yolov8n.yaml").save(str(path))  # 80 generic classes, built offline
    with pytest.raises(ValueError):
        detector.load_yolo(path)


# ---------- real weights ----------
@pytest.fixture(scope="module", params=["yolov8n", "yolov8s"])
def real_yolo(request):
    path = YOLO_WEIGHTS[request.param]
    require_weights(path)
    return detector.load_yolo(path)


def _names(model):
    names = model.names
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def test_real_yolo_names_equal_schema(real_yolo):
    assert _names(real_yolo) == list(CLASS_NAMES)


def _scene():
    frame = np.full((480, 640, 3), 90, np.uint8)
    rng = np.random.default_rng(7)
    frame[200:460, 250:330] = (40, 40, 160)
    frame[180:215, 265:315] = (0, 220, 255)
    frame[50:150, 400:600] = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)
    return frame


@pytest.mark.parametrize("thresholds", [
    {n: 0.001 for n in CLASS_NAMES},
    {"person": 0.05, "helmet": 0.01, "vest": 0.3, "fall": 0.002, "fire": 0.02, "smoke": 0.5},
    {n: 0.99 for n in CLASS_NAMES},
], ids=["very_low", "mixed", "very_high"])
def test_real_detect_respects_per_class_thresholds_and_order(real_yolo, thresholds):
    frame = _scene()
    dets = detector.detect(real_yolo, frame, thresholds)
    assert isinstance(dets, list)
    confs = [d.confidence for d in dets]
    assert confs == sorted(confs, reverse=True)
    for d in dets:
        assert isinstance(d, Detection)
        assert d.class_name in CLASS_NAMES
        assert d.confidence >= thresholds[d.class_name]
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 480  # source-frame pixels


def test_real_detect_low_threshold_yields_boxes(real_yolo):
    # Sanity check that the thresholding is not vacuous: at 0.001 the model emits candidates.
    dets = detector.detect(real_yolo, _scene(), {n: 0.001 for n in CLASS_NAMES})
    assert len(dets) > 0
