# Construction-PPE Sample Audit

> Audit date: 25 September 2026
>
> Scope: local technical and visual audit only; no model training performed
>
> Source: official Ultralytics release referenced by `construction-ppe.yaml`
>
> Usage policy: education/research only

## Decision

**HOLD FOR TRAINING.** The archive is structurally usable, but its published
train/validation/test split contains visually related frames from the same actor
and scene across different splits. Source-image provenance is also not detailed
enough in the official documentation to pass the project's full approval gate.

The dataset may remain locally under `data/raw/` for further audit. It must not
be merged into the canonical dataset or used for benchmark claims yet.

## Acquisition record

- Official archive: `construction-ppe.zip`
- Download size: 178,415,813 bytes
- SHA-256: `bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc`
- Local data and archive are excluded from Git under `data/raw/`.
- Bundled license file: GNU AGPL v3.

## Inventory

| Split | Images | Label files | Observation |
|---|---:|---:|---|
| train | 1,132 | 1,142 | 10 orphan label files |
| validation | 143 | 143 | paired |
| test | 141 | 141 | paired |
| **Total** | **1,416** | **1,426** | **10 labels have no image** |

All 1,426 label files were parsed. Every non-empty row had five YOLO fields,
class IDs were within `0..10`, coordinates were within `0..1`, widths/heights
were positive, and no label file was empty. No byte-identical image duplicates
were found.

The ten orphan train labels are:

`image940(1).txt`, `image941(1).txt`, `image944(1).txt`, `image945(1).txt`,
`image946(1).txt`, `image947(1).txt`, `image948(1).txt`, `image949(1).txt`,
`image95(1).txt`, and `image950(1).txt`.

## Source class distribution

| Source ID | Source class | Instances | Canonical action |
|---:|---|---:|---|
| 0 | helmet | 1,750 | map to `helmet` |
| 1 | gloves | 1,461 | exclude only after exhaustive-label QA |
| 2 | vest | 1,632 | map to `vest` |
| 3 | boots | 1,613 | exclude only after exhaustive-label QA |
| 4 | goggles | 526 | exclude only after exhaustive-label QA |
| 5 | none | 800 | inspect semantics; do not map automatically |
| 6 | Person | 2,265 | map to `person` |
| 7 | no_helmet | 485 | exclude as detector class; retain as audit evidence |
| 8 | no_goggle | 411 | exclude only after QA |
| 9 | no_gloves | 556 | exclude only after QA |
| 10 | no_boots | 115 | exclude only after QA |

## Confirmed split leakage

Visual inspection found the same worker, rooftop, camera setup, clothing and
recording sequence distributed across all three published splits:

- train: `images/train/image1001.jpg`
- validation: `images/val/image1010.jpg`
- test: `images/test/image1003.jpg`

Related numbered frames also occur across train and test in the `image1000` to
`image1019` range. Exact-file hashing cannot detect this because the frames are
different moments from the same sequence. The official split therefore cannot
be used for an unbiased benchmark.

## Required remediation before reconsideration

1. Recover or construct scene/sequence group IDs for all 1,416 images.
2. Re-split entire groups, not individual images, into train/validation/test.
3. Remove or explain the ten orphan labels.
4. Perform human QA for `Person`, `helmet`, and `vest` completeness on every
   retained image; dropping unrelated source classes must not hide missing
   canonical labels.
5. Define the semantics of source class `none` from publisher evidence.
6. Record image-level or collection-level provenance and confirm that the
   publisher has authority to distribute the underlying real-world imagery.
7. Re-run near-duplicate detection after grouping and report group counts and
   actor/scene diversity.

Until these conditions pass, the status remains **HOLD**, not approved.
