# D-Fire Corrected Dataset — Worker Audit & Machine QA Report

> **Audit Date:** 25 September 2026  
> **Dataset Scope:** D-Fire Dataset (`gaia-solutions-on-demand/DFireDataset`), 21,527 images  
> **Target Schema:** Canonical Stage 1 Detector Schema v2 (6 spatial classes: `0:person`, `1:helmet`, `2:vest`, `3:fall`, `4:fire`, `5:smoke`)  
> **Source Directory:** `data/raw/dfire/data/` (immutable raw)  
> **Artifacts Location:** `data/processed/dfire_corrected/` and `docs/audit_artifacts/dfire/`  
> **Readiness Verdict:** **`WORKER_AUDIT_COMPLETE` / `PENDING_INDEPENDENT_HUMAN_QA` (Immutable corrected build complete; zero group leakage; independent human sign-off remains final gate)**

---

## 1. Executive Summary & Verification Metrics

A decision-complete, fully corrected Stage 1 bounding box build and visual audit for the entire D-Fire dataset (21,527 images) has been completed.

The audit and build pipeline comprises four production scripts:
1. `scripts/inventory_dfire.py`: 100% machine inventory of raw files, pairing, syntax, bounds, and SHA-256 digests.
2. `scripts/analyze_dfire_leakage.py`: Sequence continuity and perceptual dHash clustering identifying 10,010 scene groups (remediating 762 upstream leaking groups affecting 10,504 images) and eliminating multi-frame split leakage.
3. `scripts/build_dfire_corrected.py`: Non-destructive, immutable-raw build creating `data/processed/dfire_corrected` with canonical class mapping (0$\to$5, 1$\to$4), dropping 18 degenerate zero-area boxes, and clipping 379 boundary-exceeding boxes.
4. `scripts/audit_dfire_samples.py`: Worker visual sample audit covering 106 frames (80 stratified web/incident scenes + 26 defect cases), emitting QA overlays, contact sheets, and the human QA handoff queue.
5. `scripts/validate_dfire.py`: Comprehensive validation verifying 1:1 pairing, canonical class compliance, coordinate bounds, and zero cross-split group leakage.

### Quantitative Summary Table

| Metric Category | Count / Value | Notes |
|---|---:|---|
| **Total Raw Images** | 21,527 images | All readable, zero corrupt images |
| **Total Raw Labels** | 21,527 labels | Exact 1:1 pairing (diff = 0) |
| **Negative (Empty) Images** | 9,838 images | Normal lighting, headlights, sun glare |
| **Fire-Only Images** | 1,164 images | Flame regions only |
| **Smoke-Only Images** | 5,867 images | Smoke plumes only |
| **Both Fire + Smoke Images** | 4,658 images | Co-occurring flame and smoke |
| **Raw Bounding Boxes** | 26,557 boxes | Class 1 (`fire`): 14,692; Class 0 (`smoke`): 11,865 |
| **Degenerate Zero-Area Boxes Dropped** | 18 boxes | Non-physical $w=0$ or $h=0$ click artifacts |
| **OOB & Boundary Crossings Clipped** | 379 boxes | Clipped strictly to $[0.0, 1.0]$ bounds |
| **Corrected Bounding Boxes** | 26,539 boxes | Class 4 (`fire`): 14,685; Class 5 (`smoke`): 11,854 |
| **Classes 0, 1, 2, 3 Instances** | 0 instances | Strictly zero false-positive PPE/fall boxes |
| **Total Groups Formed** | 10,010 groups | Clustered via sequence continuity + 64-bit dHash |
| **Upstream Leaking Groups** | 762 groups | 762 upstream leaking groups affecting 10,504 images in upstream D-Fire |
| **Cross-Split Group Leakage (Corrected)** | **Exactly 0 groups** | 100% split isolation achieved in `dfire_corrected` |
| **Corrected Split Allocation** | Train: 17,248 / Val: 1,488 / Test: 2,791 | Group-isolated allocation (raw splits: train 14,122 / val 3,099 / test 4,306) |
| **Worker Visual Audit Coverage** | 106 frames | 80 stratified + 26 defect reviews with color overlays |
| **Independent Human QA Status** | `PENDING_HUMAN_QA` | Handoff queue logged for all 106 rows in `dfire_audit_handoff_queue.csv` |

---

## 2. Canonical Schema Mapping & Justified Box Corrections

### 2.1 Mapping Rule
Raw configuration `data.yaml` maps `0: smoke` and `1: fire`.
In accordance with Canonical Detector Schema v2 ([data_schema_6classes.md](../../data_schema_6classes.md)):
$$\text{Raw Class } 0 \ (\text{smoke}) \longrightarrow \text{Canonical Class } \mathbf{5} \ (\text{smoke})$$
$$\text{Raw Class } 1 \ (\text{fire}) \longrightarrow \text{Canonical Class } \mathbf{4} \ (\text{fire})$$

### 2.2 Justified Defect Remediation
1. **18 Degenerate Zero-Dimension Boxes:** Dropped because a box with $w=0.0$ or $h=0.0$ has zero spatial area and represents spurious annotation clicks or coordinate export errors. Surrounding valid boxes on the same images are preserved.
2. **8 Out-of-Bounds Boxes ($w > 1.0$ or $h > 1.0$):** All located in `test` (`WEB10769`, `WEB10770`, `WEB10775`, `WEB10821`, `WEB11090`, `WEB11598`, `WEB11600`, `WEB11606`). Clipped to $[0.0, 1.0]$ bounds ($w=1.000$ or $h=1.000$, centered at $0.500$).
3. **379 Boundary Edge Crossings:** Boxes extending slightly beyond image borders due to flame/smoke expansion are clipped strictly to $[0.0, 1.0]$ boundaries.
4. **Zero Pseudo-Labeling:** No synthetic boxes or algorithmic hallucinations were introduced.

---

## 3. Upstream Sequence Interleaving & Group Isolation

### 3.1 Upstream Defect Diagnosis
Upstream D-Fire files comprise:
- `AoF`: 8,384 frames (`AoF00000`–`AoF08383`)
- `PublicDataset`: 1,336 frames (`PublicDataset00000`–`PublicDataset01335`)
- `WEB`: 11,807 images (`WEB00000`–`WEB11806`)

In the raw dataset, while the tail was held out as raw `test` (4,306 images), `train` (14,122) and `val` (3,099) were sampled in an interleaved manner across video and web burst sequences. Validated leakage analysis confirms that **762 upstream leaking groups affect 10,504 images (48.8% of the dataset)** across splits.

### 3.2 Resolution via Group Partitioning
Using `scripts/analyze_dfire_leakage.py`:
- 10,010 discrete scene groups were formed.
- Groups were allocated intact to splits (`train`: 17,248, `val`: 1,488, `test`: 2,791).
- Cross-split group leakage in `dfire_corrected` is verified at **exactly 0 groups**.

---

## 4. Worker Visual Sample Audit & Handoff Queue

### 4.1 Visual Sample Inspection
Under the protocol defined in `docs/fall_fire_replacement_dataset_search.md` Section 7, exactly 106 frames were sampled and visually reviewed via `scripts/audit_dfire_samples.py`:
- **25 Fire-only frames:** Verified tight flame boundaries, distinct flame cores, and absence of ambient lighting false positives.
- **25 Fire + Smoke frames:** Verified accurate multi-class discrimination between luminous flames and rising smoke envelopes.
- **15 Smoke-only frames:** Verified plume boundaries and confirmed zero unboxed active flames within smoke regions.
- **15 Hard negatives:** Verified that headlights, high-bay factory lights, sodium streetlamps, sunset glare, and campfire ashes contain zero false-positive boxes.
- **26 Defect reviews:** Verified all 18 dropped degenerate boxes and 8 clipped OOB boxes against color-coded QA overlays.

### 4.2 Independent Human QA Handoff Queue
All 106 audit frames (80 stratified visual samples + 26 defect review cases) are cataloged in `docs/audit_artifacts/dfire/dfire_audit_handoff_queue.csv`.
- Each entry records: `filename`, `source_split`, `proposed_split`, `group_id`, `issue_types`, `target_classes`, `evidence`, `worker_remediated_boxes`, `worker_review_status`, `qa_overlay_path`, `human_qa_verdict`, `human_qa_notes`.
- Worker status: `WORKER_VISUAL_QA_VERIFIED`.
- Human verdict: `PENDING_HUMAN_QA` for all 106 rows.

> **Independent Human QA Gate:** In accordance with project policy, independent human sign-off on all 106 rows in `dfire_audit_handoff_queue.csv` remains the final approval gate before any model training. Worker visual verification cannot bypass human sign-off.

---

## 5. Educational Prototype Scope & Risk Note

- **Operative License:** **CC0 1.0 Universal** on dataset collection and YOLO annotations by maintainers Pedro Vinicius A. B. Venâncio et al.
- **Source Rights Limitation:** Maintainers disclaim copyright over underlying third-party images harvested from the public web.
- **Course-Project Risk Baseline:** Under the 25 September 2026 binding owner decision for a non-commercial university course project submitted to an instructor:
  1. *Git Hygiene:* Raw dataset files must never be committed to Git.
  2. *Model Weight Privacy:* Checkpoints trained on D-Fire data must remain confidential (no public release).
  3. *Mandatory Citation:* Original maintainers must be cited in deliverables.
  4. *Likeness Protection:* Any human face appearing in public documentation must be blurred.
  5. *Legal Boundary:* Non-commercial education does not cure third-party copyright; commercial deployment remains prohibited.
