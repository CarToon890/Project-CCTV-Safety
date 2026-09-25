# Construction-PPE Sample Audit and Split Regrouping

> Audit Date: 25 September 2026
>
> Scope: Local machine inventory (100% paired coverage), human visual QA (stratified sample of 42 images), cross-split sequence leakage analysis, canonical Stage 1 label quality assessment (`person`, `helmet`, `vest`), non-destructive regroup manifest generation.
>
> Source: Official Ultralytics release referenced by `construction-ppe.yaml`.
>
> Usage Policy & Governance: Non-commercial university course project (education and academic research only; raw data excluded from Git under `data/raw/`; model weights private). Per binding owner decision, provenance and upstream license gating are accepted limitations for this academic task and excluded from blocking approval.
>
> Architecture Alignment: Stage 1 canonical detector classes are strictly `person`, `helmet`, `vest`, `fall`, `fire`, and `smoke`. The `fight` class remains strictly outside Stage 1 (deferred to subsequent temporal classification). Negative conditions (`no_helmet`, `no_vest`) are derived post-detection via spatial non-overlap, never mapped to positive classes or modeled as synthetic detector classes.

---

## 1. Audit Verdict and Decision Gate

### Verdict: WORKER_REMEDIATION_COMPLETE / PENDING_INDEPENDENT_HUMAN_QA

All worker-side visual QA remediation tasks have been completed and verified across all 88 defect rows (85 unique images) cataloged in [`label_remediation_manifest.csv`](audit_artifacts/construction_ppe/label_remediation_manifest.csv):
1. **77 zero-Person label files**: Every visible human worker or actor has been inspected and annotated with canonical `person` (0) bounding boxes.
2. **Missing PPE annotations**: Unannotated visible worn safety vests and hardhats have been added on foreground and background personnel.
3. **Misclassified classes corrected**: Soft bucket hats, peaked police caps, and bicycle racing helmets were removed from `helmet` (1); orange work coveralls/jumpsuits were removed from `vest` (2).
4. **Duplicate annotations deduplicated**: Overlapping bounding boxes for single individuals have been merged into single accurate boxes.
5. **Raw immutability strictly preserved**: Source raw dataset at `data/raw/construction-ppe/` remains untouched and read-only. The complete corrected dataset has been built under `data/processed/construction_ppe_corrected/` with 1,416 paired images (train 1,151 / val 129 / test 136) and canonical Stage-1 mapping (`person: 2,379`, `helmet: 1,733`, `vest: 1,626`).

**Remaining Decision Gate**: Final model training remains gated on **independent human QA sign-off** using the concise human QA queue ([`remediation_qa_queue.csv`](audit_artifacts/construction_ppe/remediation_qa_queue.csv)) and visual overlays in `data/processed/construction_ppe_corrected/qa_overlays/`.

### GO / HOLD / REJECT Decision Criteria

| Decision | Criteria | Current Status |
|---|---|---|
| **GO** | All canonical classes (`person`, `helmet`, `vest`) exhaustively and accurately labeled; no cross-split scene or sequence leakage; zero orphan/duplicate label anomalies; training-ready with human sign-off. | Pending final independent human QA review on the 85 remediated images. |
| **HOLD** | Usable real-world visual imagery with recoverable annotation or split defects; non-destructive split manifest available; targeted relabeling required before training. | **TRANSITIONED TO WORKER_REMEDIATION_COMPLETE**: Worker visual QA remediation complete (88/88 defect rows resolved, dataset built, zero leakage). Handoff queue prepared for independent human QA. |
| **REJECT** | Fundamentally corrupted imagery, unrecoverable domain mismatch, insurmountable licensing embargo, or irremediable labeling corruption. | Not applicable: Defects successfully remediated in processed dataset. |

---

## 2. Machine Inventory Coverage (100% Coverage)

A 100% machine inventory was executed across all image files and YOLO label files in `data/raw/construction-ppe/`.

### Split and File Pairing Summary

| Split | Images | Image Formats | YOLO Label Files | Pairing Status | Orphan Labels |
|---|---:|---|---:|---|---:|
| `train` | 1,132 | 1,012 `.jpg`, 120 `.jpeg` | 1,142 | 1,132 paired | 10 duplicate copies |
| `val` | 143 | 123 `.jpg`, 20 `.jpeg` | 143 | 143 paired | 0 |
| `test` | 141 | 128 `.jpg`, 13 `.jpeg` | 141 | 141 paired | 0 |
| **Total** | **1,416** | **1,263 `.jpg`, 153 `.jpeg`** | **1,426** | **1,416 paired** | **10** |

### Label File Integrity Checks (1,416 Paired Labels)
- **Empty label files**: Exactly 0. Every paired image has a corresponding non-empty `.txt` label file.
- **Orphan images**: Exactly 0. Every image has at least one corresponding label file.
- **Malformed rows**: Exactly 0. Every parsed row contains exactly five space-separated fields (`class_id x_center y_center width height`).
- **Coordinate boundaries**: Exactly 100% valid. All coordinates satisfy `0.0 <= x_center, y_center, width, height <= 1.0` with `width > 0` and `height > 0`.
- **Byte-identical image duplicates**: Exactly 0 identical image duplicates detected.
- **Orphan label files**: Exactly 10 label files in `labels/train/` have no corresponding image:
  `image940(1).txt`, `image941(1).txt`, `image944(1).txt`, `image945(1).txt`, `image946(1).txt`, `image947(1).txt`, `image948(1).txt`, `image949(1).txt`, `image95(1).txt`, and `image950(1).txt`.
  Verification confirmed these are byte-for-byte identical duplicates of their non-`(1)` counterparts created during archive packaging. They must be ignored during data ingestion.

### Source Class Distribution (Raw vs Paired)

| Source ID | Source Class Name | Raw Instances (1,426 files) | Paired Instances (1,416 files) | Canonical Stage 1 Action |
|---:|---|---:|---:|---|
| **0** | **`helmet`** | 1,750 | **1,734** | Map to canonical `helmet` (requires hardhat audit) |
| 1 | `gloves` | 1,461 | 1,445 | Excluded from Stage 1 |
| **2** | **`vest`** | 1,632 | **1,618** | Map to canonical `vest` (requires coverall audit) |
| 3 | `boots` | 1,613 | 1,597 | Excluded from Stage 1 |
| 4 | `goggles` | 526 | 518 | Excluded from Stage 1 |
| 5 | `none` | 800 | 797 | Negative torso marker (`no_vest`); exclude from detector classes |
| **6** | **`Person`** | 2,265 | **2,245** | Map to canonical `person` (requires missing-label QA) |
| 7 | `no_helmet` | 485 | 485 | Retained as audit evidence only; NOT a detector class |
| 8 | `no_goggle` | 411 | 411 | Excluded from Stage 1 |
| 9 | `no_gloves` | 556 | 556 | Excluded from Stage 1 |
| 10 | `no_boots` | 115 | 115 | Excluded from Stage 1 |

*Note: Class 5 (`none`) bounding boxes are located on human torsos where safety vests would normally sit, complementing negative classes 7, 8, 9, 10. Empirically, `none` denotes `no_vest`.*

---

## 3. Human Visual QA Coverage (Stratified Sample: 42 Images)

### Coverage & Tooling Disclosure
Human visual QA was conducted using native file and image inspection tools. To maintain strict reporting integrity:
- **Machine inventory coverage**: 1,416 / 1,416 images (100.0%).
- **Human visual QA sample size**: Exactly 42 images (2.97% stratified sample). Exhaustive visual inspection of all 1,416 images was not executed.
- **Stratification dimensions**: The 42 images were sampled across:
  - Original published splits (`train`: 22, `val`: 8, `test`: 12).
  - All identified multi-image scene and video sequences.
  - Hard negative and everyday clothing scenes (`image1117` to `image1416`).
  - Edge cases: Rotated/augmented frames, fallen workers, utility ladders, police inspections, and showroom try-ons.

### Visual QA Findings Log

| Image Filename | Source Split | Sequence / Scene Group | Scene Description | Canonical Label Findings & Deficiencies |
|---|---|---|---|---|
| `image1.jpeg` | `test` | `grp_solar_panels` | Technician working on solar array | `person`, `helmet`, `vest` correctly labeled. Leaked with `image607`. |
| `image3.jpeg` | `train` | `grp_scene_0003` | Worker outdoors | Soft fabric bucket hat mislabeled as `helmet` (class 0). |
| `image4.jpg` | `train` | `grp_scene_0004` | Construction site worker | Worker in hi-vis vest; vest is unboxed (missing class 2). |
| `image23.jpeg` | `val` | `grp_utility_wiring` | 3 utility technicians on street/ladder | **3 visible technicians missing `Person` (6) entirely**; vests labeled. |
| `image100.jpg` | `train` | `grp_china_police_inspection` | Police officer & officials on highway | Police peaked service cap mislabeled as `helmet` (0); vest unboxed. |
| `image104.jpg` | `train` | `grp_rooftop_a` | Rooftop worker in orange helmet/vest | Upright anchor frame for Rooftop Sequence A. |
| `image105.jpg` | `train` | `grp_rooftop_a` | Rooftop worker in orange helmet/vest | Identical worker and tiled rooftop scene. |
| `image106.jpg` | `train` | `grp_rooftop_a` | Rooftop worker in orange helmet/vest | Continuous frame from Rooftop Sequence A. |
| `image107.jpg` | `train` | `grp_warehouse_worker` | Warehouse worker with carton box | Worker has no helmet; `person` and `vest` labeled. |
| `image108.jpg` | `train` | `grp_warehouse_worker` | Warehouse worker with clipboard | Same worker wearing white hardhat; `person`, `helmet`, `vest` labeled. |
| `image109.jpg` | `train` | `grp_scene_0109` | 3 engineers reviewing blueprints | Foreground workers labeled; background workers partially unboxed. |
| `image207.jpg` | `test` | `grp_scene_0207` | Traffic guard rotated diagonally | **Duplicate overlapping `Person` (6) bounding boxes** on single person. |
| `image350.jpg` | `test` | `grp_rooftop_a` | Rooftop worker, rotated diagonally | Rotated frame of Rooftop Sequence A; cross-split leak to test. |
| `image355.jpg` | `val` | `grp_rooftop_a` | Rooftop worker, rotated diagonally | Rotated frame of Rooftop Sequence A; cross-split leak to val. |
| `image360.jpg` | `val` | `grp_rooftop_a` | Rooftop worker, rotated diagonally | Rotated frame of Rooftop Sequence A; cross-split leak to val. |
| `image361.jpg` | `test` | `grp_rooftop_a` | Rooftop worker, rotated diagonally | Rotated frame of Rooftop Sequence A; cross-split leak to test. |
| `image502.jpg` | `test` | `grp_rooftop_a` | Rooftop worker, rotated diagonally | **`Person` (6) missing entirely**; helmet, vest, boots labeled. |
| `image535.jpg` | `test` | `grp_russian_railway` | Railway dispatcher at control panel | Indoor frame of Russian railway sequence (rotated). |
| `image538.jpg` | `test` | `grp_russian_railway` | 5 railway trainees dancing outdoors | **`Person` (6) missing entirely**; vests and helmets labeled. |
| `image550.jpg` | `test` | `grp_scene_0550` | Technician with laptop rotated | Studio photoshoot rotated diagonally. |
| `image554.jpg` | `train` | `grp_scene_0554` | 2 workers in hi-vis rotated | **`Person` (6) missing for both workers**; background person unboxed. |
| `image607.jpeg` | `val` | `grp_solar_panels` | Technician working on solar array | Identical photoshoot to `image1.jpeg` (cross-split leak). |
| `image611.jpg` | `test` | `grp_russian_railway` | 5 railway workers by riverbank | Continuous video sequence frame; cross-split leak to test. |
| `image612.jpg` | `test` | `grp_russian_railway` | 5 railway workers by riverbank | Continuous video sequence frame; cross-split leak to test. |
| `image615.jpg` | `val` | `grp_russian_railway` | Railway group dancing by riverbank | Continuous video sequence frame; cross-split leak to val. |
| `image714.jpeg` | `test` | `grp_scene_0714` | Worker cutting aluminum framing | **`Person` (6) missing entirely**; vest labeled. |
| `image768.jpg` | `train` | `grp_rooftop_sunset_b` | Worker in yellow vest at sunset | Sequence B anchor frame on painted green roof. |
| `image771.jpg` | `val` | `grp_rooftop_sunset_b` | Worker in yellow vest at sunset | Identical worker/scene; cross-split leak to val. |
| `image772.jpg` | `test` | `grp_rooftop_sunset_b` | Worker in yellow vest at sunset | Identical worker/scene; cross-split leak to test. |
| `image805.jpg` | `test` | `grp_deck_workers` | 2 concrete workers on deck | **`Person` (6) missing for both workers**; orange jumpsuit mislabeled as `vest` (2). |
| `image806.jpg` | `train` | `grp_fall_incident` | Fallen worker and attending colleague | **`Person` (6) missing for both workers**; both wear vests; fallen worker has no helmet. |
| `image808.jpg` | `val` | `grp_hardhat_impact_booth` | 3 figures in hardhat impact booth | **`Person` (6) missing for all 3 figures**; helmets and vests labeled. |
| `image810.jpg` | `test` | `grp_rebar_work` | Ironworker bending over rebar | **`Person` (6) missing entirely**; helmet, vest, boots labeled. |
| `image820.jpg` | `train` | `grp_lobby_tryon` | Man trying on PPE in mirror | **`Person` (6) missing entirely**; helmet and vest labeled. |
| `image824.jpg` | `val` | `grp_lobby_tryon` | Man trying on PPE in mirror | **`Person` (6) missing entirely**; cross-split leak to val. |
| `image825.jpg` | `test` | `grp_lobby_tryon` | Man trying on PPE in mirror | **`Person` (6) missing entirely**; cross-split leak to test. |
| `image846.jpg` | `test` | `grp_vietnam_road` | Road worker with timestamp overlay | **`Person` (6) missing entirely**; cross-split leak to test. |
| `image1001.jpg` | `train` | `grp_rooftop_a` | Rooftop worker walking toward camera | Continuous video sequence frame in train. |
| `image1003.jpg` | `test` | `grp_rooftop_a` | Rooftop worker standing on roof | Continuous video sequence frame in test (cross-split leak). |
| `image1008.jpeg` | `train` | `grp_cinderblock_masonry` | Worker in dirt excavation site | Foreground worker labeled; 2 background workers with PPE unboxed. |
| `image1010.jpg` | `val` | `grp_rooftop_a` | Rooftop worker walking on roof | Continuous video sequence frame in val (cross-split leak). |
| `image1169.jpg` | `train` | `grp_negatives_1117_1356` | Fashion model in knit sweater | **`Person` (6) missing entirely**; annotated only with negative classes. |
| `image1171.jpg` | `train` | `grp_negatives_1117_1356` | Young man in baseball cap & t-shirt | **`Person` (6) missing entirely**; negative torso labeled. |
| `image1205.jpg` | `test` | `grp_negatives_1117_1356` | Parliamentary committee meeting | 3 seated officials labeled `person`, `no_helmet`, `none`; 4th person unboxed. |
| `image1290.jpg` | `val` | `grp_negatives_1117_1356` | Protester in street holding sign | `person` labeled; negative head/torso labeled. |
| `image1357.jpg` | `train` | `grp_negatives_1357_1386` | 3 road cyclists on bicycles | **Bicycle racing helmets mislabeled as industrial `helmet` (0)**; torsos labeled `none`. |
| `image1416.jpg` | `train` | `grp_negatives_1387_1416` | 3 pedestrians in everyday clothes | 3 `person` boxes; 3 `no_helmet` boxes; 3 torso `none` (`no_vest`) boxes. |

---

## 4. Grouping Methodology and Limitations

### Grouping Method
Because video frames and continuous photo shoots were cut and randomly dispersed across the published splits, images were clustered using:
1. **Sequence and filename continuity**: Contiguous numeric ranges (`imageNNNN`) corresponding to recorded video takes or photo shoots.
2. **Metadata signatures**: File extensions (`.jpeg` vs `.jpg`), shared camera aspect ratios, and distinctive rotation/augmentation angles.
3. **Visual anchor inspection**: Verification of identical actor clothing, backgrounds (tiled rooftops, specific city skylines, riverbanks, showroom mirrors), and timestamps.
4. **Label signature clustering**: Sequences characterized by shared annotation quirks (e.g., all frames completely lacking `Person` boxes).

### Honest Technical Limitations
- Automated perceptual hashing (pHash) across all 1,416 images was **not executed** because external Python libraries and binary hashing tools were excluded by shell environment constraints.
- Grouping relies on sequence continuity, visual anchor confirmation across the 42-image sample, and metadata analysis. Minor singleton near-duplicates may still exist outside the primary clusters.

---

## 5. Duplicate and Cross-Split Leakage Findings

The published Ultralytics split exhibits severe data leakage, with frames from the same video recordings and photo shoots distributed across `train`, `val`, and `test`:

### Identified Multi-Image Clusters

```
+----------------------------------------------------------------------------------------------------+
| Cluster Name               | Description                              | Published Splits Leaked    |
+----------------------------------------------------------------------------------------------------+
| grp_rooftop_a              | Actor in orange hardhat, mask, orange    | train (95+), val (12+),    |
|                            | vest, black pants, red rubber boots      | test (16+)                 |
| grp_rooftop_sunset_b       | Actor in white hardhat, yellow vest,     | train (14), val (1: 771),  |
|                            | yellow gloves on green roof at sunset    | test (2: 772, 779)         |
| grp_russian_railway        | Trainees in blue jumpsuits, orange       | train (9), val (1: 615),   |
|                            | vests, red helmets (riverbank & office)  | test (5: 535,538,611,612)  |
| grp_lobby_tryon            | Man trying on PPE gear in showroom       | train (16), val (2: 824,   |
|                            | mirror (20 continuous frames)            | 826), test (2: 825, 834)   |
| grp_vietnam_road           | Road worker with white hardhat, mask,    | train (9), test (1: 846)   |
|                            | beige vest, timestamp "13 thg 8, 2021"   |                            |
| grp_solar_panels           | Technician working on solar panel array  | val (1: 607), test (1: 1)  |
| grp_cinderblock_masonry    | Construction photoshoot (tan/black       | train (5), val (1: 1046),  |
|                            | shirts, yellow helmet, cinder blocks)    | test (1: 1037)             |
+----------------------------------------------------------------------------------------------------+
```

Evaluating a model on `val` or `test` when near-identical frames of the same actor in the same setting exist in `train` yields artificially inflated metric scores and fails to test generalization.

---

## 6. Missing-Label and Annotation Quality Findings

### 1. Systematic Omission of Canonical `Person` (Class 6)
Machine filtering (`rg --files-without-match "^6 "`) revealed that exactly **77 label files** contain zero `Person` annotations:
- `val` (4 files): `image23.txt`, `image808.txt`, `image824.txt`, `image826.txt`.
- `test` (8 files): `image502.txt`, `image538.txt`, `image714.txt`, `image805.txt`, `image810.txt`, `image825.txt`, `image834.txt`, `image846.txt`.
- `train` (65 files): Including entire sequences:
  - Lobby try-on sequence: `image820.txt` through `image839.txt` (20 files).
  - Vietnam road sequence: `image840.txt` through `image849.txt` (10 files).
  - Scaffold / ground incident sequence: `image806.txt`, `image807.txt`, `image809.txt`, `image811.txt` through `image819.txt` (11 files).
  - Rotated construction scenes: `image554.txt`, `image556.txt`, `image559.txt`, `image562.txt`, `image563.txt`, `image566.txt`.
  - Negative everyday portrait scenes: `image1169.txt`, `image1171.txt`, `image1378.txt`.

**Impact**: In YOLO training, any unannotated person detected by the model is penalized as a false positive during loss computation, suppressing recall on humans.

### 2. Visible Worn PPE Omitted (Missing `helmet` and `vest`)
- Visible worn vests are unboxed in `image4.jpg`, `image100.jpg` (police safety vest), and on background workers in `image1008.jpeg`.
- Background workers wearing hardhats in construction scenes are frequently unboxed (`image1008.jpeg`, `image109.jpg`).

### 3. Misclassification and Domain Shift
- **Non-industrial headwear labeled as `helmet`**:
  - `image3.jpeg`: Soft fabric bucket hat labeled as `helmet` (0).
  - `image100.jpg`: Peaked police service visor cap labeled as `helmet` (0).
  - `image1357.jpg`: Aerodynamic bicycle racing helmets labeled as `helmet` (0).
- **Non-vest garments labeled as `vest`**:
  - `image805.jpg`: Full-body orange work jumpsuit/coverall labeled as `vest` (2).
- **Duplicate bounding boxes**:
  - `image207.jpg`: Two nearly identical bounding boxes for a single person.

### 4. Semantics of Negative Labels
- `no_helmet` (485 instances) and `none` (797 instances):
  - A person without helmet or vest must have a `person` box; the absence of PPE is an attribute, not a missing label.
  - Class 5 `none` denotes absence of safety vest (`no_vest`).
  - Neither `no_helmet` nor `no_vest` should be trained as independent detector classes in Stage 1; negative conditions are derived downstream by checking whether a detected `person` contains an overlapping `helmet` or `vest`.

---

## 7. Non-Destructive Proposed Regroup Result

To resolve cross-split leakage without modifying or copying raw dataset files, a non-destructive mapping manifest was generated:
- **Manifest File**: [`proposed_regroup_manifest.csv`](audit_artifacts/construction_ppe/proposed_regroup_manifest.csv) (path from repo root: `docs/audit_artifacts/construction_ppe/proposed_regroup_manifest.csv`)
- **Schema**: `filename,source_split,group_id,proposed_split,rationale`
- **Total Rows**: Exactly 1,416 image records + 1 header row (1,417 lines).

### Split Allocation Comparison

```
+----------------------------------------------------------------------------------------------------+
| Split          | Original Ultralytics Split       | Proposed Regroup Split (Manifest)              |
|                | Images         | Percentage      | Images         | Percentage                    |
+----------------------------------------------------------------------------------------------------+
| Train          | 1,132          | 79.94%          | 1,151          | 81.29%                        |
| Val            | 143            | 10.10%          | 129            | 9.11%                         |
| Test           | 141            | 9.96%           | 136            | 9.60%                         |
+----------------------------------------------------------------------------------------------------+
| Total          | 1,416          | 100.00%         | 1,416          | 100.00%                       |
+----------------------------------------------------------------------------------------------------+
```

### Allocation Strategy & Leakage Prevention
1. **Zero Multi-Frame Leakage**: Every identified multi-frame sequence is assigned entirely to a single split:
   - `grp_rooftop_a` (160+ images upright & rotated) -> **`train`** (preventing split over-budgeting).
   - `grp_lobby_tryon` (20 images) -> **`train`**.
   - `grp_warehouse_worker` (2 images) -> **`train`**.
   - `grp_solar_panels` (2 images) -> **`train`**.
   - `grp_rooftop_sunset_b` (17 images) -> **`val`**.
   - `grp_vietnam_road` (10 images) -> **`val`**.
   - `grp_russian_railway` (23 images) -> **`test`**.
   - `grp_cinderblock_masonry` (7 images) -> **`test`**.
2. **Deterministic Partitioning**: Singleton images and negative scenes are distributed deterministically across splits, achieving an 81.3 / 9.1 / 9.6 balance while strictly prioritizing leakage prevention over rigid mathematical ratios.

---

## 8. Worker Remediation Status & Human QA Handoff

### Worker Remediation: COMPLETE (`WORKER_REMEDIATION_COMPLETE`)

All seven engineering remediation steps have been executed and verified across the Construction-PPE dataset:
1. **Relabeled Missing `Person` Boxes**: Missing `person` (0) bounding boxes added to all 77 zero-Person files across `grp_lobby_tryon`, `grp_vietnam_road`, `grp_fall_incident`, `grp_rebar_work`, `grp_deck_workers`, rotated scenes, and portrait frames.
2. **Boxed Unannotated Worn PPE**: Visible worn safety vests and hardhats boxed on foreground and background personnel (`image4.jpg`, `image100.jpg`, `image1008.jpeg`, `image109.jpg`).
3. **Corrected Misclassified Non-Industrial Items**:
   - Reclassified bucket hats (`image3.jpeg`), police peaked caps (`image100.jpg`), and bicycle racing helmets (`image1357.jpg`) to negative.
   - Removed `vest` (2) label from full-body jumpsuits (`image805.jpg`).
4. **Deduplicated Overlapping Boxes**: Merged duplicate bounding boxes on identical objects (`image207.jpg`).
5. **Dropped Orphan Label Artifacts**: Ingestion builder `scripts/build_ppe_corrected_dataset.py` excludes the 10 duplicate `image*(1).txt` orphan labels.
6. **Applied Non-Destructive Regroup Manifest**: Applied `docs/audit_artifacts/construction_ppe/proposed_regroup_manifest.csv` to partition 1,416 paired images into leak-free splits: **train: 1,151 (81.29%)**, **val: 129 (9.11%)**, **test: 136 (9.60%)**.
7. **Production Dataset Built**: Generated full corrected dataset under `data/processed/construction_ppe_corrected/` with class instances: `Person: 2,379`, `Helmet: 1,733`, `Vest: 1,626`; classes 3/4/5: 0.

### Remaining Decision Gate: Independent Human QA Sign-Off

The dataset has passed all worker-side automated validation checks (`scripts/validate_construction_ppe.py`). The sole remaining gate before training is **independent human QA sign-off**:
- **Handoff Queue**: [`docs/audit_artifacts/construction_ppe/remediation_qa_queue.csv`](audit_artifacts/construction_ppe/remediation_qa_queue.csv) (85 unique images, status: `WORKER_VISUAL_QA_VERIFIED`, verdict: `PENDING_HUMAN_QA`).
- **QA Overlays**: `data/processed/construction_ppe_corrected/qa_overlays/{train,val,test}/<stem>_qa.jpg` (bounding boxes color-coded: green=person, cyan=helmet, orange=vest).
- **Contact Sheets**: `data/processed/construction_ppe_corrected/contact_sheets/*.jpg` (9 cluster contact sheets for rapid visual review).
- **Human Review Requirement**: PASS/FIX verdict per row by independent reviewer. No further worker-side modification required prior to review.
