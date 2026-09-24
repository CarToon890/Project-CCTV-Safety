# Data Schema — 7 Classes (YOLO Format)

> **Deprecated historical schema (v1).** Do not use this schema to prepare,
> train or run the current detector. The active contract is
> [Detector Data Schema v2 — 6 Spatial Classes](data_schema_6classes.md).
> Class `fight` moved to the Stage 2 temporal pipeline, which is currently
> `BLOCKED / PENDING DATA APPROVAL`.

## Canonical class mapping

| ID | Class | Annotation target |
|---:|---|---|
| 0 | `person` | Full visible body of every person |
| 1 | `helmet` | Each safety helmet being worn |
| 2 | `vest` | Each high-visibility/safety vest being worn |
| 3 | `fall` | Full body of a person who has fallen or is lying abnormally |
| 4 | `fire` | Visible flame region |
| 5 | `smoke` | Visible smoke plume |
| 6 | `fight` | One group box covering the people actively involved |

This ID order documents the retired v1 contract only. It intentionally does not
match the active v2 configuration.

## Annotation rules

- One YOLO label file per image: `<class_id> <x_center> <y_center> <width> <height>`.
- Coordinates are normalized to `0.0–1.0`; boxes must have positive width and height.
- Annotate every visible canonical class, even when the image originated from a
  single-purpose dataset. Images that have not received exhaustive annotation
  must not enter the training set.
- `person` and `fall` may overlap. A fallen person receives both a `person` box
  and a `fall` box so PPE association and incident detection remain independent.
- `helmet` and `vest` mean worn PPE. Loose equipment is excluded unless a future
  class explicitly models it.
- `fire` and `smoke` remain separate. If a source provides one combined class,
  it must be manually relabelled; it must not be guessed during conversion.
- `fight` is a group box for the image-based baseline. Ordinary contact, hugs,
  sports and collaborative work are hard-negative examples.
- Occluded/truncated objects are labelled only when a reviewer can identify the
  class and draw a meaningful visible-extent box. Ambiguous cases are flagged
  for review rather than silently guessed.

## PPE compliance semantics

`no_helmet` and `no_vest` are derived alert states, not detector classes. During
inference, detected PPE is associated with a person using containment of the PPE
centre in configurable head/torso regions. Missing or ambiguous PPE should be
reported with the association reason and confidence; it is not ground truth
that a person violated policy when the relevant body region is occluded.

## Required dataset layout

```text
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── metadata/{train,val,test}.csv
└── dataset_manifest.json
```

The metadata CSV records at least `image`, `source_id`, and `group_id`.
`group_id` represents the source video, scene, or camera session and must occur
in only one split to prevent leakage.
