# Model and local runtime manifest

Snapshot: 2026-10-04. This records what is available locally; it does not imply
that training runs are bit-for-bit reproducible or that weights have been
evaluated on target CCTV. Large weights, data, and video remain outside Git.

## Local checkpoints

SHA-256 values are for files currently in the ignored local `weights/` folder.
Verify before comparing or replacing a model.

| Runtime model | Local file | SHA-256 | Evaluation record |
|---|---|---|---|
| YOLOv8n | `weights/YOLOv8n_best.pt` | `e78634dfcd288f11193ca1a67c57d45491b4fcf306967ef82f2c8423b8156245` | `artifacts/model_handoff/stage1/yolov8n/yolov8n_test_metrics.json`; provisional 6-class dataset, P .8193 / R .7734 / mAP50 .8418 / mAP50-95 .4377. Run record says 81/100 requested epochs. |
| YOLOv8s | `weights/YOLOv8s_best.pt` | `754cf224f43c30d5c6a268498bced13be4741d3e39a026c0cf27c18d0b159f61` | `artifacts/model_handoff/stage1/yolov8s/yolov8s_heldout_test_metrics.json`; provisional 6-class dataset, P .8582 / R .7853 / mAP50 .8698 / mAP50-95 .4573. Run record says 86/100 requested epochs. |
| X3D-S | `weights/X3D-S_best.pt` | `e4f07f808ef10ee66c439eb77b2b8c8fa59bf0478d26bdbcf82efdd9973d118f` | `artifacts/model_handoff/stage2/x3d_s/X3D-S_test_metrics.json`; SCFD pilot dataset manifest SHA-256 `4aaec038080a1318cc769bac4a6e6ae15a254e4fc9df0ca704af420dc4e58310`; 45 held-out clips, accuracy .80, macro-F1 .7984, balanced accuracy .8024. |
| YuNet face detector | `weights/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | OpenCV Zoo YuNet model; face detector only, not identity recognition. |

### Dataset/split provenance

The archived `data/processed/stage1_colab_bundle/stage1_unified_colab.zip`
contains the same `dataset_manifest.json` and split CSV bytes as the checked-in
local Stage 1 dataset:

| File | SHA-256 | Archive/local comparison |
|---|---|---|
| `dataset_manifest.json` | `45f47484c42ec44de9e724da9ab9cd96fe2c03eb9120a9efd54fac7f945320e9` | Exact match |
| `metadata/train.csv` | `b25199a17f6a4b00adbab62ea3857fe3b3f668a497dfe75d33b9209ccda980d6` | Exact match; 1,013,315 bytes |
| `metadata/val.csv` | `6db2b186c432a242670522e7a9109f02ac13515c56433ba1537b0505897eb063` | Exact match; 101,553 bytes |
| `metadata/test.csv` | `24af4637b299b6a89a1b27eb3c07881e98b3868c95516ec84f42da666dc97958` | Exact match; 173,875 bytes |

Both checked-in Stage 1 Kaggle training notebooks use this archive as their
dataset source (YOLOv8n and YOLOv8s). This verifies the repository's intended
training source and that the archived bundle matches the local manifest/splits.
The exported Kaggle metrics/arguments do not themselves embed these hashes, so
they cannot cryptographically prove which bytes were mounted in each remote
run. Treat same-split comparability as supported by the recorded workflow and
matching archive, not as a fully attested remote-run provenance chain. Stage 2
records its own dataset-manifest hash.

## Runtime used for this review

The local `.venv-test` interpreter reported Python 3.11.9, FastAPI 0.142.2,
Uvicorn 0.54.0, PyTorch 2.14.1, torchvision 0.29.1, Ultralytics 8.4.171,
opencv-python 5.0.0.93, NumPy 2.3.5, Pillow 12.3.0, pytorchvideo 0.1.5,
and pytest 9.1.1. These are an observed local environment, not a portable lock.
`requirements.txt` still specifies broad lower bounds; future reproducible
installation should pin platform-specific inference dependencies and record an
SBOM without assuming CPU and CUDA wheels are interchangeable.

## Reproduction record to retain for future runs

For each approved model evaluation, archive: checkpoint SHA-256; model code
revision; Python and dependency lock; training/inference config; dataset
manifest and split hashes; seed; device/runtime; preprocessing; thresholds;
held-out predictions; aggregate and per-class/event metrics; and human QA
decision. Do not place identifiable raw CCTV into Git or a public artifact.
