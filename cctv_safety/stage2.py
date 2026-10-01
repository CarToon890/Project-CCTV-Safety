"""Stage 2 X3D-S fight classifier: architecture, checkpoint loading and preprocessing.

Reproduces ``notebooks/stage2_kaggle_training.ipynb`` (cell 5) exactly:
``round(linspace(0, n-1, 13))`` frame indices, BGR->RGB, aspect-preserving
``INTER_AREA`` resize so the long side is 224, centred zero pad to 224x224,
``/255`` then ``(x - 0.45) / 0.225``, tensor layout ``(C=3, T=13, H, W)``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ("non_fight", "fight")
FRAMES = 13
SIZE = 224
MEAN = 0.45
STD = 0.225


def sample_indices(start: int, end: int, n: int = FRAMES) -> list[int]:
    """``round(linspace(start, end, n))`` with ``end`` inclusive (repeats allowed)."""
    return [int(value) for value in np.linspace(start, end, n).round().astype(int)]


def letterbox(rgb: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aspect-preserving INTER_AREA resize (long side = ``size``), centred on a zero canvas."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def preprocess_frames(frames_bgr: list[np.ndarray]) -> torch.Tensor:
    """FRAMES BGR uint8 frames -> float32 tensor ``(3, FRAMES, SIZE, SIZE)``."""
    if len(frames_bgr) != FRAMES:
        raise ValueError(f"Expected {FRAMES} frames, got {len(frames_bgr)}")
    frames = [letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) for bgr in frames_bgr]
    array = np.stack(frames).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def build_x3d_s() -> nn.Module:
    """pytorchvideo ``x3d_s`` (no pretrained download) with a 2-class raw-logit head."""
    from pytorchvideo.models.hub import x3d_s

    model = x3d_s(pretrained=False)
    head = model.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, len(CLASS_NAMES))
    head.activation = None
    return model


def load_x3d_s(path, device="cpu") -> nn.Module:
    """Load the training checkpoint (``checkpoint["model"]``) with ``strict=True``."""
    # Local, trusted training checkpoint; it also stores numpy RNG state, so
    # weights_only=True cannot unpickle it.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path}: not an X3D-S training checkpoint (missing 'model')")
    if checkpoint.get("frames") != FRAMES or checkpoint.get("size") != SIZE:
        raise ValueError(
            f"{path}: preprocessing mismatch, expected frames={FRAMES} size={SIZE}, "
            f"got frames={checkpoint.get('frames')} size={checkpoint.get('size')}"
        )
    model = build_x3d_s()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def predict_probs(model: nn.Module, batch: torch.Tensor) -> list[tuple[float, float]]:
    """Softmax ``(non_fight, fight)`` per clip; ``batch`` is ``(B, 3, FRAMES, SIZE, SIZE)``."""
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(batch.to(device))
        probs = torch.softmax(logits.float(), dim=1).cpu().tolist()
    return [(float(row[0]), float(row[1])) for row in probs]
