# D-Fire Dataset Machine Inventory & Worker Discrepancy Log

> **Audit Date:** 25 September 2026  
> **Audited Dataset:** D-Fire Dataset (`gaia-solutions-on-demand/DFireDataset`)  
> **Source Directory:** `data/raw/dfire/data/` (raw data preserved immutably)  
> **Processed Directory:** `data/processed/dfire_corrected/`  
> **Canonical Schema:** Canonical Stage 1 Detector Schema v2 (6 spatial classes: `0:person`, `1:helmet`, `2:vest`, `3:fall`, `4:fire`, `5:smoke`)  
> **Governance Notice:** Worker visual QA completed. Independent human QA remains the final approval gate before model training.

---

## 1. Class Schema & Mapping Verification

Raw D-Fire configuration `data/raw/dfire/data.yaml` defines:
```yaml
names: ['smoke', 'fire']
nc: 2
```
- **Raw Class 0:** `smoke`
- **Raw Class 1:** `fire`

Under the Canonical Stage 1 Detector Schema v2 ([data_schema_6classes.md](../../data_schema_6classes.md)):
- **Canonical Class 4:** `fire` (Visible flame region)
- **Canonical Class 5:** `smoke` (Visible smoke plume)

### Authoritative Mapping Matrix
| Raw Source ID | Raw Class Name | Canonical Class ID | Canonical Class Name | Justification |
|:---:|---|:---:|---|---|
| `0` | `smoke` | **`5`** | `smoke` | Visible smoke plume; mapped to canonical index 5 |
| `1` | `fire` | **`4`** | `fire` | Visible flame body; mapped to canonical index 4 |
| — | — | `0` | `person` | Unboxed persons flagged in QA; absent from raw annotations |
| — | — | `1`, `2`, `3` | `helmet`, `vest`, `fall` | Strictly 0 instances in D-Fire |

---

## 2. Machine Inventory & Raw Dataset Verification

The comprehensive machine inventory executed by `scripts/inventory_dfire.py` on the 21,527 raw image-label pairs yielded:

- **Total Images:** 21,527 (.jpg)
- **Total Labels:** 21,527 (.txt)
- **1:1 Pairing Check:** 100% paired (0 missing images, 0 missing labels)
- **Raw Splits:**
  - `train`: 14,122 images (65.6%)
  - `val`: 3,099 images (14.4%)
  - `test`: 4,306 images (20.0%)
- **Negative (Empty) Labels:** 9,838 images (train: 6,458, val: 1,375, test: 2,005)
- **Positive Categories:**
  - `fire_only`: 1,164 images (train: 770, val: 174, test: 220)
  - `smoke_only`: 5,867 images (train: 3,836, val: 845, test: 1,186)
  - `both`: 4,658 images (train: 3,058, val: 705, test: 895)
- **Source Instance Counts:**
  - Raw Class 1 (`fire`): 14,692 instances (train: 9,638, val: 2,176, test: 2,878)
  - Raw Class 0 (`smoke`): 11,865 instances (train: 7,794, val: 1,756, test: 2,315)
  - Total raw bounding boxes: 26,557
- **Data Integrity:**
  - Corrupt / Unreadable Images: 0
  - Label Syntax Errors: 0
  - Exact Image Duplicate Groups (SHA-256): 0

---

## 3. Catalog of Box Defects & Remediations

The machine inventory identified exactly **26 bounding box defects** and **379 boundary edge crossings**.

### 3.1 Degenerate Zero-Dimension Boxes (18 cases)
Eighteen bounding boxes have width $w = 0.0$ and/or height $h = 0.0$. A bounding box with zero area is non-physical (representing a dimensionless click artifact or rounding truncation). These boxes cannot be clipped or clamped and must be dropped.

| # | Split | File | Line | Raw Class | xc | yc | w | h | Action Taken |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | train | `AoF05470.txt` | 2 | 0 (`smoke`) | 0.55859375 | 0.46666667 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 2 | train | `AoF06155.txt` | 2 | 0 (`smoke`) | 0.81250000 | 0.33750000 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 3 | train | `AoF06348.txt` | 2 | 0 (`smoke`) | 0.75312500 | 0.25833333 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 4 | train | `AoF06439.txt` | 2 | 0 (`smoke`) | 0.71250000 | 0.27500000 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 5 | train | `AoF06456.txt` | 2 | 0 (`smoke`) | 0.14765625 | 0.43750000 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 6 | train | `PublicDataset00056.txt` | 5 | 1 (`fire`) | 0.32968750 | 0.81527778 | 0.0 | 0.00277778 | Dropped zero-width box; lines 1-4 valid fire preserved |
| 7 | train | `WEB04243.txt` | 5 | 1 (`fire`) | 0.26855469 | 0.41666667 | 0.0 | 0.0 | Dropped degenerate box; lines 1-4 valid fire preserved |
| 8 | train | `WEB04527.txt` | 2 | 1 (`fire`) | 0.55300000 | 0.73309609 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid fire preserved |
| 9 | train | `WEB04742.txt` | 5 | 1 (`fire`) | 0.38750000 | 0.49166667 | 0.0 | 0.0 | Dropped degenerate box; lines 1-4 valid fire preserved |
| 10 | train | `WEB04849.txt` | 5 | 1 (`fire`) | 0.14843750 | 0.42222222 | 0.0 | 0.0 | Dropped degenerate box; lines 1-4 valid fire preserved |
| 11 | train | `WEB05882.txt` | 7 | 1 (`fire`) | 0.65625000 | 0.70555556 | 0.0 | 0.0 | Dropped degenerate box; lines 1-6 valid fire preserved |
| 12 | train | `WEB06900.txt` | 8 | 1 (`fire`) | 0.85390625 | 0.23333333 | 0.0 | 0.0 | Dropped degenerate box; lines 1-7 valid fire preserved |
| 13 | train | `WEB07998.txt` | 2 | 0 (`smoke`) | 0.64687500 | 0.47159091 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 14 | val | `WEB08540.txt` | 2 | 0 (`smoke`) | 0.05871560 | 0.01729107 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 15 | test | `AoF07743.txt` | 3 | 0 (`smoke`) | 0.61718750 | 0.49861111 | 0.0 | 0.0 | Dropped degenerate box; lines 1-2 valid smoke preserved |
| 16 | test | `AoF07774.txt` | 3 | 0 (`smoke`) | 0.44335938 | 0.36666667 | 0.00078125 | 0.0 | Dropped zero-height box; lines 1-2 valid smoke preserved |
| 17 | test | `AoF08348.txt` | 2 | 0 (`smoke`) | 0.98750000 | 0.24166667 | 0.0 | 0.0 | Dropped degenerate box; line 1 valid smoke preserved |
| 18 | test | `WEB10669.txt` | 5 | 0 (`smoke`) | 0.60000000 | 0.26666667 | 0.0 | 0.0 | Dropped degenerate box; lines 1-4 valid smoke preserved |

### 3.2 Coordinate Out-of-Bounds Boxes ($w > 1.0$ or $h > 1.0$) (8 cases)
Eight bounding boxes in `test` exceed normalized coordinate limits ($w > 1.0$ or $h > 1.0$). These are valid large smoke plumes spanning the full camera view that slightly exceeded $1.0$ due to rounding. They are clipped to image boundaries $[0.0, 1.0]$:

| # | Split | File | Line | Raw Class | Raw Box [xc, yc, w, h] | Clipped Box [xc, yc, w, h] | Action Taken |
|:---:|:---:|---|:---:|:---:|---|---|---|
| 19 | test | `WEB10769.txt` | 2 | 0 (`smoke`) | `[0.50546875, 0.36250000, 1.02968750, 0.70833333]` | `[0.50000000, 0.36250000, 1.00000000, 0.70833333]` | Clipped $w$ to 1.0; centered at 0.5 |
| 20 | test | `WEB10770.txt` | 2 | 0 (`smoke`) | `[0.50703125, 0.36250000, 1.00781250, 0.71388889]` | `[0.50000000, 0.36250000, 1.00000000, 0.71388889]` | Clipped $w$ to 1.0; centered at 0.5 |
| 21 | test | `WEB10775.txt` | 3 | 0 (`smoke`) | `[0.49687500, 0.27083333, 1.01562500, 0.57500000]` | `[0.50000000, 0.27083333, 1.00000000, 0.57500000]` | Clipped $w$ to 1.0; centered at 0.5 |
| 22 | test | `WEB10821.txt` | 3 | 0 (`smoke`) | `[0.49843750, 0.33888889, 1.00937500, 0.61666667]` | `[0.50000000, 0.33888889, 1.00000000, 0.61666667]` | Clipped $w$ to 1.0; centered at 0.5 |
| 23 | test | `WEB11090.txt` | 1 | 0 (`smoke`) | `[0.54375000, 0.50138889, 0.53750000, 1.00277778]` | `[0.54375000, 0.50000000, 0.53750000, 1.00000000]` | Clipped $h$ to 1.0; centered at 0.5 |
| 24 | test | `WEB11598.txt` | 1 | 0 (`smoke`) | `[0.49921875, 0.40138889, 1.03593750, 0.81388889]` | `[0.50000000, 0.40138889, 1.00000000, 0.81388889]` | Clipped $w$ to 1.0; centered at 0.5 |
| 25 | test | `WEB11600.txt` | 1 | 0 (`smoke`) | `[0.49687500, 0.42222222, 1.05625000, 0.83888889]` | `[0.50000000, 0.42222222, 1.00000000, 0.83888889]` | Clipped $w$ to 1.0; centered at 0.5 |
| 26 | test | `WEB11606.txt` | 1 | 0 (`smoke`) | `[0.25781250, 0.49861111, 0.50625000, 1.00277778]` | `[0.25781250, 0.50000000, 0.50625000, 1.00000000]` | Clipped $h$ to 1.0; centered at 0.5 |

### 3.3 Boundary Edge Crossings (379 cases)
379 bounding boxes extend slightly beyond normalized boundaries ($x_1 < 0$ or $x_2 > 1$ or $y_1 < 0$ or $y_2 > 1$) due to fire flames or smoke billowing out of the camera view. In `scripts/build_dfire_corrected.py`, all edge crossings are clipped non-destructively:
$$x_1 = \max(0.0, x_c - w/2), \quad x_2 = \min(1.0, x_c + w/2)$$
$$y_1 = \max(0.0, y_c - h/2), \quad y_2 = \min(1.0, y_c + h/2)$$
$$w_{\text{new}} = x_2 - x_1, \quad h_{\text{new}} = y_2 - y_1, \quad x_{c,\text{new}} = x_1 + w_{\text{new}}/2, \quad y_{c,\text{new}} = y_1 + h_{\text{new}}/2$$

---

## 4. Upstream Sequence & Group Leakage Analysis

Upstream D-Fire files originate from three distinct source series:
1. `AoF` (Alert on Fire): 8,384 images (`AoF00000`–`AoF08383`) from surveillance video sequences.
2. `PublicDataset`: 1,336 images (`PublicDataset00000`–`PublicDataset01335`) from experimental test runs.
3. `WEB`: 11,807 images (`WEB00000`–`WEB11806`) scraped from the web, containing burst photo series.

### Upstream Partitioning Defect
While the upstream creators held out the tail as raw `test` (4,306 images), they sampled `train` (14,122) and `val` (3,099) in an interleaved manner across video and web burst sequences. Validated leakage analysis confirms that **762 upstream leaking groups affect 10,504 images (48.8% of the dataset)** across splits.

### Clustering & Group Isolation Resolution
Using sequence continuity linking and perceptual hash clustering:
- Exactly **10,010 scene/incident groups** were identified.
- **762 groups (10,504 images)** leaked across splits in upstream D-Fire.
- In `scripts/analyze_dfire_leakage.py` and `scripts/build_dfire_corrected.py`, entire groups are partitioned intact into `train` (17,248), `val` (1,488), and `test` (2,791), achieving **strictly ZERO cross-split group leakage**.

---

## 5. Visual Sample Audit & Handoff Queue

A stratified 106-frame sample audit was conducted using `scripts/audit_dfire_samples.py`:
- 25 Fire-only frames (`4: fire`)
- 25 Fire + Smoke frames (`4: fire`, `5: smoke`)
- 15 Smoke-only frames (`5: smoke`)
- 15 Hard negatives (headlights, high-bay lamps, sunrise/sunset, campfire ashes)
- 26 Defect reviews (18 zero-dimension + 8 OOB)

### Visual Findings
1. **Flame Tightness:** Fire bounding boxes tightly enclose visible flame bodies with accurate margins.
2. **False Light Rejection:** Streetlamps, sodium floodlights, and sunsets in negatives are correctly unboxed.
3. **Bystander / Person Screen:** Across the audited frames, visible bystanders and ground personnel were screened; no unboxed persons were found in critical proximity to active flame boxes.
4. **Handoff Queue:** All 106 audited frames (80 stratified samples + 26 defect cases) are cataloged in [dfire_audit_handoff_queue.csv](dfire_audit_handoff_queue.csv) with status `WORKER_VISUAL_QA_VERIFIED` and verdict `PENDING_HUMAN_QA`. Independent human QA remains pending for all 106 handoff rows.

---

## 6. Educational Prototype License & Risk Note

- **Repository:** [gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset)
- **Maintainers:** Pedro Vinicius A. B. Venâncio et al.
- **Operative License:** **CC0 1.0 Universal** on dataset collection and YOLO bounding box annotations.
- **Source Rights Limitation:** Maintainers disclaim copyright over underlying third-party images harvested from the public web.
- **Course-Project Risk Note:** Under the 25 September 2026 binding owner decision for a non-commercial university course project submitted to an instructor:
  - This is documented as an educational prototype risk note.
  - Raw images are excluded from Git redistribution.
  - Model weights trained on D-Fire data must remain confidential (no public release).
  - Proper citation of the original creators is mandatory.
  - This license limitation does not block the bounded educational prototype, but prohibits commercial deployment claims.
