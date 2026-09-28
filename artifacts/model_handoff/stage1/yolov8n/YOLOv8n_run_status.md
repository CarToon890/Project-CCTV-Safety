# Stage 1 Kaggle run status

> This file preserves the status of the YOLOv8n run captured below. It is not
> the project-wide current status; the later YOLOv8s run and remaining work are
> summarized in `docs/project_status.md`.

Captured: 2026-09-27

- Model: YOLOv8n, six Stage 1 classes (`person`, `helmet`, `vest`, `fall`, `fire`, `smoke`).
- The run configuration requested 100 epochs, but `results.csv` contains epochs 1–81 only. Do not describe this run as a completed 100-epoch run.
- Best validation mAP@50–95 in `results.csv`: 0.45772 at epoch 51. Epoch 81: 0.45293.
- Test metrics recorded in `yolov8n_test_metrics.json`: precision 0.81932, recall 0.77343, mAP@50 0.84176, mAP@50–95 0.43772.
- The metrics file marks the dataset as provisional. These are baseline/pilot results, not production or dataset-finalization approval.
- This run stopped at epoch 81 of 100; the project owner has not authorized a
  continuation. If a continuation is approved later, start from this run's
  `YOLOv8n_last.pt`, not from epoch 1 or the YOLOv8s checkpoint. The selected
  best-validation checkpoint is `YOLOv8n_best.pt`.
- This folder is the current local handoff copy; original downloads may remain
  as backups. Large weights are intentionally excluded from Git.
