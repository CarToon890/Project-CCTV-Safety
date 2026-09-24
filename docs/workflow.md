# Reproducible AI Baseline Workflow

## 1. Source approval

Copy `configs/datasets.example.yaml` to the git-ignored
`configs/datasets.local.yaml`. For every source, record its homepage, immutable
identifier/version when available, license, annotation format and class mapping.
Set `license_approved: true` only after a human verifies that the intended use is
permitted. The downloader refuses unapproved sources.

Stock-photo search pages are discovery leads, not automatically approved
datasets. Credentials for Kaggle or other services remain outside the repo.

## 2. Exhaustive annotation

Convert each source to YOLO format in a source-local directory. Each source must
have a `metadata.csv` with `image,label,group_id`. `group_id` is a video, scene or
camera session—not a frame. Review every image against all six canonical spatial
classes and mark `exhaustive_labels: true` only when missing-label penalties have
been addressed. A source mapping to `fight` must be rejected because Fight is a
Stage 2 temporal event, not a detector label.

Recommended hard negatives include normal walking, sitting, maintenance on the
floor, steam, fog, orange lighting, hugs, sports and collaborative work.

## 3. Prepare and validate

`scripts/prepare_dataset.py` remaps source labels, prefixes filenames to avoid
collisions, assigns whole groups to a single split, writes split metadata and
creates `dataset_manifest.json`. The fixed seed makes the split reproducible.

`scripts/validate_dataset.py` checks directory shape, paired labels, canonical
class IDs, normalized boxes, cross-split group leakage and exact duplicates.
Near-duplicate detection is optional because it is quadratic and intended for
curated batches. Any error blocks training; near-duplicate matches require human
review rather than automatic deletion.

## 4. Train and compare

`scripts/train_compare.py` runs YOLOv8n and YOLOv8s using the same Dataset,
augmentations, seed and test split. It writes best weights, Ultralytics plots,
per-class JSON, a comparison CSV and dynamic-shape ONNX exports. Thresholds are
not selected using the test split; tune them on validation results with Recall
as the primary objective.

## 5. Inference and limitations

`scripts/infer.py` emits detector boxes and derived `no_helmet`/`no_vest` states.
The geometric PPE association is an explicit baseline and must be evaluated on
crowded and occluded scenes. Fall detections are frame-level hypotheses and need
temporal confirmation. Fight is not emitted by this detector; its Stage 2
temporal pipeline remains blocked pending an approved video dataset.

This workflow uses public data only until a target-camera holdout set exists.
Reports must carry that limitation and must not claim production readiness.
