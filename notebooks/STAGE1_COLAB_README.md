# Stage 1 Colab package

> Historical Colab reproduction instructions. The recorded Stage 1 YOLOv8n/
> YOLOv8s pilots ran on Kaggle, and the Stage 2 SCFD/X3D-S pilot was evaluated.
> The dataset remains provisional; this package is not authorization for a new
> run. See `docs/project_status.md` for current results and remaining work.

This bundle contains the provisional six-class Unified Stage 1 dataset and a Colab notebook. It does not train or upload anything by itself.

## Files

- `stage1_unified_colab.zip` — 23,169 image/label pairs with split metadata, the six-class manifest, and data YAML (3.15 GiB / 3.38 GB, stored without recompression because the images are already compressed; ZIP paths are normalized for Linux/Colab).
- `stage1_colab_training.ipynb` — Colab workflow for T4 GPU, local runtime extraction, preflight checks, YOLOv8n/YOLOv8s, and Drive-backed run outputs.
- `validation_report.md` — validation findings and known limitations.

## Upload and open

1. Upload `stage1_unified_colab.zip` to `MyDrive/CCTV-Safety-Stage1/` (create the folder if necessary).
2. Upload/open `stage1_colab_training.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**. Run the setup, GPU, extraction, and preflight code cells individually through the cell that prints `Preflight PASS`. Do not use **Run all** while the approval guard is still off; the guard intentionally stops execution before training.
4. Review the validation report and preflight output. It checks CUDA, image/label pairing in both directions, metadata-to-image filename mapping, class counts against the manifest, the six-class schema, and split-group separation. It also counts boxes extending past image edges; the current package has 284 such known warnings. The notebook's `TRAINING_APPROVED` guard is `False` by default. Only after you decide to start training, change it to `True`, run that cell, then run the training cell.

## Training settings and outputs

The notebook trains `yolov8n.pt` and then `yolov8s.pt`, each with up to 100 epochs, `imgsz=640`, `batch=8`, `patience=30`, and seed 42. Batch 8 is the conservative starting point for the available T4. Test-split metrics are computed after each model finishes. Runs/checkpoints are written under `MyDrive/CCTV-Safety-Stage1/training_runs/stage1_first_training/`.

The run tag is intentionally stable. Ultralytics writes `last.pt` and `best.pt` at epoch boundaries; if Colab disconnects after at least one completed epoch, reopen the same notebook, keep the run tag unchanged, run setup/preflight cells again, then rerun the training cell. It detects an unfinished model and resumes from its Drive `last.pt`; completed training is marked so a disconnect during test evaluation does not attempt to train it again, and completed models with saved test metrics are skipped. A stop before the first completed epoch has no checkpoint to resume.

Drive-backed checkpoint writes may be slower than writing to Colab's temporary disk, but they persist across runtime loss. Colab runtime availability and GPU allocation can vary. The notebook pins Ultralytics `8.4.163`; Colab's existing CUDA-enabled PyTorch is retained.

## Important limitations

- This is a provisional dataset package that was used in past pilot work. The
  notebook's approval guard defaults to `False`; a new run still requires an
  explicit owner decision.
- Fall pilot boxes remain unchanged per owner decision; 284 non-fatal boundary warnings and the cross-split perceptual-hash matches are documented in the validation report.
- `fight` is not in Stage 1. Stage 2 has a separate small pilot but is not
  integrated into this detector workflow.
- The archive excludes raw data, QA overlays, contact sheets, and temporary/scratch files.

## Integrity

SHA-256: `EF6C35A39B95F945432E116DBAA956C19E517F8C47A1B36F74C3B50CA9AFEE36`
