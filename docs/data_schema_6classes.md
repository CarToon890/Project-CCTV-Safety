# Detector Data Schema v2 — 6 Spatial Classes (YOLO Format)

This is the canonical schema for the Stage 1 YOLO detector.

| ID | Class | Annotation target |
|---:|---|---|
| 0 | `person` | Full visible body of every person |
| 1 | `helmet` | Each safety helmet being worn |
| 2 | `vest` | Each high-visibility/safety vest being worn |
| 3 | `fall` | Full body of a person who has fallen or is lying abnormally |
| 4 | `fire` | Visible flame region |
| 5 | `smoke` | Visible smoke plume |

The order is a public contract and must exactly match `configs/classes.yaml`,
`configs/data.yaml`, model checkpoint names and confidence-threshold keys.
`fight` is not a detector class. It is reserved as a Stage 2 temporal event,
which remains blocked pending an approved video dataset.

## Annotation rules

- Use one YOLO label per image: `<class_id> <x_center> <y_center> <width> <height>`.
- Coordinates are normalized to `0.0–1.0`; width and height must be positive.
- Annotate every visible spatial class, even for single-purpose source datasets.
- A fallen person receives overlapping `person` and `fall` boxes.
- `helmet` and `vest` mean worn PPE; loose equipment is excluded.
- Keep `fire` and `smoke` separate. Manually relabel combined source classes.
- Flag ambiguous or heavily occluded cases for human review.
- Reject class ID 6 and any source mapping to `fight`; never drop it silently.

## Derived and temporal events

`no_helmet`, `no_vest` and `fall_confirmed` are derived states, not detector
classes. PPE association uses configurable person regions and must not treat an
occluded body region as proof of non-compliance.

`fight` belongs to the separate temporal pipeline. Its implementation and
training remain `BLOCKED / PENDING DATA APPROVAL` until all Stage 2 entry gates
in `two_stage_architecture_migration.md` pass.

## Required dataset layout

```text
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── metadata/{train,val,test}.csv
└── dataset_manifest.json
```

Each metadata row contains at least `image`, `source_id` and `group_id`.
Groups represent a source video, scene or camera session and must not cross
splits.
