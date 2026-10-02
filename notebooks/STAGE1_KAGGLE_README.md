# Stage 1 Kaggle continuation

> **Historical guide — not the current run procedure.** This describes an
> earlier YOLOv8n continuation attempt and refers to a 47-epoch checkpoint
> archive. The later YOLOv8n (81 epochs) and YOLOv8s (86 epochs) pilots have
> completed; their current local handoff artifacts are under
> `artifacts/model_handoff/stage1/`. Do not follow these old steps to resume or
> start training. See [`docs/project_status.md`](../docs/project_status.md).

This package prepares continuation of the existing YOLOv8n Stage 1 run on Kaggle. It does not start training automatically.

## What is ready locally

- `notebooks/stage1_kaggle_training.ipynb` — Kaggle-specific notebook. The training guard is `False` by default.
- `data/processed/stage1_colab_bundle/stage1_unified_colab.zip` — provisional six-class dataset, about 3.14 GiB.
- A locally downloaded Colab checkpoint archive (`yolov8n-20260927T090240Z-1-001.zip`) containing `last.pt`, `best.pt`, `args.yaml`, and `results.csv`. The result log reaches epoch 47/100; run args show batch 16, image size 640, workers 2, and one GPU.
- `data/processed/stage1_colab_bundle/validation_report.md` — provisional dataset validation and known limitations.

## Upload to Kaggle

1. In Kaggle, create **Private Dataset(s)** for the two ZIP files below. Because the files are in different local folders, easiest is one private dataset for each. Keep filenames unchanged. This avoids public exposure of the project's dataset/checkpoint.
2. Create a Kaggle Notebook by importing `stage1_kaggle_training.ipynb`, then attach both private datasets with **Add Input**. Kaggle may unpack uploaded ZIPs automatically; this notebook accepts either the original ZIP or its expanded folder contents. For later resumptions, also attach the notebook output saved from the previous session.
3. Choose **GPU T4 x2** when ready to run the GPU preflight and resume. Kaggle shows the account's remaining weekly quota in its quota monitor.
4. Run cells individually in order. Do not use **Run All**. The first cells extract to `/kaggle/temp`, validate the six-class package, and stage the newest YOLOv8n checkpoint in `/kaggle/working/stage1_runs/stage1_first_training/yolov8n`.
5. Confirm the training library is Ultralytics `8.4.163`. **Skip** the optional install cell if the version already matches. If needed, the install cell uses `--no-deps` so it will not replace Kaggle's PyTorch/CUDA; it requires Internet access.
6. Training remains disabled until you explicitly change `TRAINING_APPROVED = False` to `True`. With approval, the notebook resumes the existing YOLOv8n run using data from Kaggle and output under `/kaggle/working`; it requests both T4s and preserves the total batch size at 16. If two-GPU startup fails, set `DEVICE = '0'` and rerun after reviewing the error.

## Checkpoint preservation

Ultralytics writes `last.pt` at completed epoch boundaries. Before ending a session, use **Save Version → Quick Save → Advanced Settings → save output files**. Wait for the save to finish. For the next session, attach that saved notebook output through **Add Input → Notebook Output Files**. The notebook chooses the checkpoint with the latest recorded `results.csv` epoch. Dataset extraction is in `/kaggle/temp` so Quick Save output contains training artifacts rather than a second copy of the dataset.

Do not use **Save & Run All** merely to save a checkpoint; it reruns the notebook. Kaggle documents that Quick Save can include output files, and that notebook outputs can be attached to later notebooks. See [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

## Scope and caveats

- Stage 1 remains exactly `person`, `helmet`, `vest`, `fall`, `fire`, `smoke`; Fight stays outside Stage 1.
- This continues YOLOv8n from the supplied checkpoint. It does not train YOLOv8s; there is no YOLOv8s checkpoint in the downloaded archive.
- The package is still provisional. The validation report records 284
  non-fatal box-boundary warnings, cross-split perceptual-hash matches, and
  unchanged Fall pilot boxes. These caveats apply when interpreting the
  completed pilot; a new run requires explicit owner approval.
- Kaggle's T4 x2 does not guarantee twice the speed. The notebook keeps total batch 16 and uses DDP on devices `0,1`; compare measured epoch time with the previous single-GPU run.
- Nothing here uploads data to Kaggle or starts a training session. The owner must create/attach the private dataset and explicitly enable the training guard.
