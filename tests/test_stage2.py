"""Stage 2 (X3D-S) preprocessing and loading.

The reference functions below re-implement ``notebooks/stage2_kaggle_training.ipynb`` cell 5
independently, so ``cctv_safety.stage2`` is checked against the training-time preprocessing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from cctv_safety import stage2
from conftest import X3D_WEIGHTS, require_weights

FRAMES, SIZE = 13, 224


# ---------- independent reference (notebook cell 5, verbatim logic) ----------
def ref_indices(start, end, n=FRAMES):
    return [int(i) for i in np.linspace(start, end, n).round().astype(int)]


def ref_letterbox(rgb):
    h, w = rgb.shape[:2]
    scale = SIZE / max(h, w); nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA); canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    y, x = (SIZE - nh) // 2, (SIZE - nw) // 2; canvas[y:y + nh, x:x + nw] = im
    return canvas


def ref_preprocess(frames_bgr):
    frames = [ref_letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) for bgr in frames_bgr]
    a = np.stack(frames).astype(np.float32) / 255.; a = (a - .45) / .225
    return torch.from_numpy(a).permute(3, 0, 1, 2).contiguous()


# ---------- constants ----------
def test_constants():
    assert tuple(stage2.CLASS_NAMES) == ("non_fight", "fight")
    assert stage2.FRAMES == 13 and stage2.SIZE == 224
    assert stage2.MEAN == pytest.approx(0.45) and stage2.STD == pytest.approx(0.225)


# ---------- sample_indices ----------
@pytest.mark.parametrize("start,end", [(0, 99), (0, 134), (0, 19), (20, 39), (40, 53), (0, 6), (0, 12), (3, 9)])
def test_sample_indices_matches_rounded_linspace(start, end):
    got = stage2.sample_indices(start, end)
    assert len(got) == 13
    assert [int(i) for i in got] == ref_indices(start, end)
    assert all(isinstance(i, (int, np.integer)) for i in got)
    assert got[0] == start and got[-1] == end  # end is inclusive


@pytest.mark.parametrize("start,end", [(0, 0), (5, 5), (0, 3), (10, 14)])
def test_sample_indices_short_clips_repeat_frames(start, end):
    got = [int(i) for i in stage2.sample_indices(start, end)]
    assert got == ref_indices(start, end)
    assert len(got) == 13 and got == sorted(got)
    assert set(got) <= set(range(start, end + 1))


def test_sample_indices_custom_n():
    assert [int(i) for i in stage2.sample_indices(0, 49, 4)] == [0, 16, 33, 49]


# ---------- letterbox ----------
@pytest.mark.parametrize("h,w", [(100, 200), (120, 160), (300, 150), (480, 270), (50, 50), (224, 224), (100, 300)])
def test_letterbox_centred_aspect_zero_pad(h, w):
    rgb = np.full((h, w, 3), 200, np.uint8)
    out = stage2.letterbox(rgb)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8

    scale = 224 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y, x = (224 - nh) // 2, (224 - nw) // 2
    assert max(nh, nw) == 224  # long side fills the canvas
    assert abs(nw / nh - w / h) < 0.05  # aspect ratio kept
    content = out[y:y + nh, x:x + nw]
    assert (content == 200).all()
    mask = np.zeros((224, 224), bool); mask[y:y + nh, x:x + nw] = True
    assert (out[~mask] == 0).all()  # zero padding
    # centred: left/right (or top/bottom) padding differ by at most one pixel
    assert abs(x - (224 - x - nw)) <= 1 and abs(y - (224 - y - nh)) <= 1


@pytest.mark.parametrize("h,w", [(120, 160), (300, 150), (77, 333)])
def test_letterbox_equals_notebook_on_random_content(h, w):
    rgb = np.random.default_rng(h * w).integers(0, 256, (h, w, 3), dtype=np.uint8)
    np.testing.assert_array_equal(stage2.letterbox(rgb), ref_letterbox(rgb))


# ---------- preprocess_frames ----------
def test_preprocess_shape_dtype_and_normalisation():
    frames = [np.full((120, 160, 3), 255, np.uint8) for _ in range(13)]
    t = stage2.preprocess_frames(frames)
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (3, 13, 224, 224) and t.dtype == torch.float32
    white = (1.0 - 0.45) / 0.225
    pad = (0.0 - 0.45) / 0.225
    # 160x120 -> 224x168, rows 28..195 are content
    assert torch.allclose(t[:, :, 28:196, :], torch.tensor(white), atol=1e-5)
    assert torch.allclose(t[:, :, :28, :], torch.tensor(pad), atol=1e-5)
    assert torch.allclose(t[:, :, 196:, :], torch.tensor(pad), atol=1e-5)


def test_preprocess_converts_bgr_to_rgb():
    blue_bgr = np.zeros((224, 224, 3), np.uint8); blue_bgr[..., 0] = 255  # pure blue in BGR
    t = stage2.preprocess_frames([blue_bgr] * 13)
    hi, lo = (1.0 - 0.45) / 0.225, (0.0 - 0.45) / 0.225
    assert torch.allclose(t[2], torch.tensor(hi), atol=1e-5)  # channel 2 = B in RGB order
    assert torch.allclose(t[0], torch.tensor(lo), atol=1e-5)  # R
    assert torch.allclose(t[1], torch.tensor(lo), atol=1e-5)  # G


def test_preprocess_frame_order_is_time_axis():
    frames = [np.full((64, 64, 3), i * 10, np.uint8) for i in range(13)]
    t = stage2.preprocess_frames(frames)
    for i in range(13):
        expected = (i * 10 / 255.0 - 0.45) / 0.225
        assert torch.allclose(t[:, i], torch.tensor(expected, dtype=torch.float32), atol=1e-5)


@pytest.mark.parametrize("h,w", [(120, 160), (360, 240), (97, 131)])
def test_preprocess_matches_notebook_reference(h, w):
    rng = np.random.default_rng(42 + h)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(13)]
    got = stage2.preprocess_frames(frames)
    want = ref_preprocess(frames)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-6)


def test_preprocess_batch_shape_for_model():
    frames = [np.zeros((48, 64, 3), np.uint8)] * 13
    batch = stage2.preprocess_frames(frames).unsqueeze(0)
    assert tuple(batch.shape) == (1, 3, 13, 224, 224)


# ---------- architecture / checkpoint loading (no real weights needed) ----------
@pytest.fixture(scope="module")
def fresh_state_dict():
    return stage2.build_x3d_s().state_dict()


def test_build_x3d_s_head_is_two_class_raw_logits():
    model = stage2.build_x3d_s().eval()
    head = model.blocks[-1]
    assert isinstance(head.proj, torch.nn.Linear)
    assert head.proj.in_features == 2048 and head.proj.out_features == 2
    assert head.activation is None
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 13, 224, 224))
    assert tuple(out.shape) == (1, 2)


def test_load_x3d_s_accepts_valid_checkpoint(tmp_path, fresh_state_dict):
    path = tmp_path / "ok.pt"
    torch.save({"model": fresh_state_dict, "frames": 13, "size": 224, "epoch": 0}, path)
    model = stage2.load_x3d_s(path)
    assert not model.training


def test_load_x3d_s_is_strict(tmp_path, fresh_state_dict):
    sd = dict(fresh_state_dict)
    sd.pop(next(iter(sd)))
    path = tmp_path / "missing_key.pt"
    torch.save({"model": sd, "frames": 13, "size": 224}, path)
    with pytest.raises(Exception):
        stage2.load_x3d_s(path)
    sd = dict(fresh_state_dict); sd["unexpected.weight"] = torch.zeros(1)
    torch.save({"model": sd, "frames": 13, "size": 224}, path)
    with pytest.raises(Exception):
        stage2.load_x3d_s(path)


@pytest.mark.parametrize("frames,size", [(16, 224), (13, 160)])
def test_load_x3d_s_checks_frames_and_size(tmp_path, fresh_state_dict, frames, size):
    path = tmp_path / "bad_cfg.pt"
    torch.save({"model": fresh_state_dict, "frames": frames, "size": size}, path)
    with pytest.raises(Exception):
        stage2.load_x3d_s(path)


# ---------- real weights ----------
@pytest.fixture(scope="module")
def real_x3d():
    require_weights(X3D_WEIGHTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return stage2.load_x3d_s(X3D_WEIGHTS, device=device)


def test_real_checkpoint_loads_strict(real_x3d):
    ckpt = torch.load(X3D_WEIGHTS, map_location="cpu", weights_only=False)
    assert ckpt["frames"] == 13 and ckpt["size"] == 224
    loaded = real_x3d.state_dict()
    assert set(loaded) == set(ckpt["model"])
    for key in ("blocks.5.proj.weight", "blocks.5.proj.bias"):
        assert torch.equal(loaded[key].cpu(), ckpt["model"][key].cpu())
    assert not real_x3d.training


def test_real_logits_shape(real_x3d):
    device = next(real_x3d.parameters()).device
    with torch.inference_mode():
        out = real_x3d(torch.zeros(1, 3, 13, 224, 224, device=device))
    assert tuple(out.shape) == (1, 2)


def test_real_predict_probs_pairs_sum_to_one(real_x3d):
    rng = np.random.default_rng(0)
    clips = [[rng.integers(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(13)] for _ in range(2)]
    batch = torch.stack([stage2.preprocess_frames(c) for c in clips])
    probs = stage2.predict_probs(real_x3d, batch)
    assert len(probs) == 2
    for pair in probs:
        assert len(pair) == 2
        assert all(0.0 <= p <= 1.0 for p in pair)
        assert sum(pair) == pytest.approx(1.0, abs=1e-4)
