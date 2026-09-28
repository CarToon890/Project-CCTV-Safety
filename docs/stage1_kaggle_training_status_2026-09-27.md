# Stage 1 Kaggle training status — updated 2026-09-28

This record supersedes its original 2026-09-27 snapshot. The original 81-epoch
YOLOv8n experiment details are retained below; the later YOLOv8s results are
added for a current Stage 1 picture. See [`project_status.md`](project_status.md)
for the concise project-wide status.

## YOLOv8n

- Six Stage 1 classes: `person`, `helmet`, `vest`, `fall`, `fire`, `smoke`.
- The recorded run requested 100 epochs but contains epochs 1–81 only; it did
  not complete 100 epochs.
- Best validation mAP@50–95: 0.45772 at epoch 51; epoch 81: 0.45293.
- Held-out test metrics: precision 0.81932, recall 0.77343, mAP@50 0.84176,
  mAP@50–95 0.43772.
- The current local handoff is
  [`../artifacts/model_handoff/stage1/yolov8n/`](../artifacts/model_handoff/stage1/yolov8n/)
  and includes the checkpoint, metrics, training arguments, and epoch log;
  model weights remain excluded from Git.
- An older, separate 47-epoch YOLOv8n package is a different historical run.
  Do not treat it as the 81-epoch experiment or overwrite either record.

## YOLOv8s

- Separate fresh experiment from pretrained `yolov8s.pt`, with the same six
  class names. It reached epoch 86 of 100 and then early-stopped; it did not
  complete all 100 requested epochs.
- Held-out test metrics: precision 0.858168, recall 0.785335, mAP@50 0.869818,
  mAP@50–95 0.457331.
- Per-class metrics, weights, and logs are in
  [`../artifacts/model_handoff/stage1/yolov8s/`](../artifacts/model_handoff/stage1/yolov8s/);
  the original download can remain as a backup.
- Before comparing models, confirm that both runs used the same dataset
  manifest, split, and evaluation settings. These pilot metrics alone do not
  establish a winner or production readiness.

## Limitations and possible follow-up

- The unified Stage 1 dataset is provisional. Its documented limitations
  include box-boundary warnings, perceptual near-duplicate candidates, and
  per-class error review; see [`project_status.md`](project_status.md).
- Fall/Fire/Smoke confusion-matrix values are threshold-dependent diagnostics;
  review examples before changing labels or thresholds.
- If the owner decides to pursue a full 100-epoch YOLOv8n run, resume that
  experiment from its own `last.pt`, not from epoch 1 or from YOLOv8s. Training
  approval is a separate explicit decision.
