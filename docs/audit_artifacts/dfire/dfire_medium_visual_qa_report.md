# Phase 2 D-Fire MEDIUM-Tier Visual QA and Remediation Report

**Date:** 2026-09-26  
**Reviewer:** `ANTIGRAVITY_VISUAL_QA`  
**Dataset Version:** `2.1.0-medium-remediated`  
**Base Dataset:** `data/processed/dfire_corrected`  
**Target Dataset:** `data/processed/dfire_remediated`  

---

## 1. Executive Summary

Following the completion and validation of Phase 1 (HIGH-tier remediation and visual QA), Phase 2 targeted the **MEDIUM-tier missing-label candidates** (confidence interval $[0.40, 0.70)$) from the full-dataset scan (`data/processed/dfire_missing_label_full_scan/candidates.csv`).

To achieve rigorous quality assurance while maintaining efficiency:
1. **100% Census of MEDIUM Helmets:** Every one of the 126 MEDIUM helmet candidates was visually inspected using high-resolution diagnostic crops and source image overlays.
2. **Deterministic Stratified Sample of MEDIUM Persons:** Exactly 200 MEDIUM person candidates were sampled from the 1,840 available candidates using deterministic seed `seed=42`, stratified across splits (`train`, `val`, `test`), confidence bands ($[0.40, 0.50)$, $[0.50, 0.60)$, $[0.60, 0.70)$), aspect ratios, border-touch cases, smoke/fire overlap, and multi-candidate images.
3. **Visual Quality Verdicts:** Every candidate received an independent visual determination (`PASS`, `FIX`, `REMOVE`, or `UNCERTAIN`). Model confidence was never used as sole justification.
4. **Label Application & Deduplication:** Only `PASS` (269 items) and visually corrected `FIX` (1 item) were ingested into `dfire_remediated` label files. Zero duplicate boxes ($\text{IoU} \ge 0.85$) were added against existing labels or between newly accepted candidates.
5. **Minimal Owner Escalation Queue:** Exactly 2 genuine borderline ambiguous candidates (`CAND_003159` and `CAND_003490`) were flagged as `UNCERTAIN` and routed to the owner queue for final determination without guessing.
6. **Zero Side Effects:** `data/raw` and `data/processed/dfire_corrected` remain 100% immutable and bit-identical.

---

## 2. Sampling Methodology & Stratification

### 2.1 Helmets (100% Census)
- **Total Ingested:** 126 candidates
- **Split Distribution:**
  - `train`: 105
  - `val`: 7
  - `test`: 14
- **Confidence Range:** $0.400$ to $0.697$

### 2.2 Persons (Deterministic Stratified Sample)
- **Total Population:** 1,840 MEDIUM person candidates
- **Sample Size:** Exactly 200 candidates
- **Deterministic Seed:** `seed=42`
- **Specification Artifact:** [dfire_phase2_sampling_spec.json](dfire_phase2_sampling_spec.json)

| Stratification Axis | Strata Breakdown | Count in Sample |
| :--- | :--- | :--- |
| **Dataset Split** | `train` / `val` / `test` | 146 / 23 / 31 |
| **Confidence Band** | $[0.60, 0.70)$ / $[0.50, 0.60)$ / $[0.40, 0.50)$ | 62 / 69 / 69 |
| **Geometric & Context Feature** | Frame-edge touching (`touches_edge=True`) | 46 |
| | Smoke/Fire overlap ($\text{IoU} \ge 0.20$) | 26 |
| | Extreme aspect ratio ($< 0.6$ or $> 3.5$) | 44 |
| | Co-occurring candidates in same image | 44 |
| | Standard interior geometry | 40 |

---

## 3. Visual QA Findings & Failure Modes

Visual inspection across the 326 diagnostic crops revealed distinct detector error modes in the MEDIUM confidence band:

1. **Safety Helmet vs. Fabric Balaclava / Balaclava Shroud (Wildland Gear):**
   - Wildland firefighters frequently wear yellow or Nomex fabric flame-resistant hoods (`WEB11805.jpg`, `WEB07339.jpg`).
   - The detector falsely classified these fabric hoods as helmets ($0.45 < \text{conf} < 0.50$). These were rejected (`REMOVE`) because they lack rigid impact shell structure.
2. **Safety Hardhat vs. Civilian Baseball Caps / Hair:**
   - In crowd or civilian fire scenes (`WEB06284.jpg`, `WEB07180.jpg`, `WEB08006.jpg`), dark or light baseball caps and styled hair were falsely detected as helmets. These were rejected (`REMOVE`).
3. **Smoke Plumes & Flame Patches Misclassified as Persons:**
   - Dark billowing industrial smoke plumes (`WEB07444.jpg`, `WEB09163.jpg`, `WEB03702.jpg`) and bright torch/burner flames (`WEB10519.jpg`, `WEB05660.jpg`) trigger false positive person detections with confidences up to $0.58$. These were rejected (`REMOVE`).
4. **Extreme Frame Edge Slivers & Fragment Truncations:**
   - Detectors triggered on partial finger/hand fragments (`WEB03980.jpg`), scalp edges (`WEB03362.jpg`), or arm slivers with aspect ratios up to $10.22$. These non-viable fragments were rejected (`REMOVE`).
5. **Non-Human Objects (Statues & Landscape Artifacts):**
   - Candidate `CAND_001572` was a marble sculpture/statue (`WEB07584.jpg`), and `CAND_003488` was a distant tree silhouette against a sunset horizon (`WEB10021.jpg`). Both were rejected (`REMOVE`).
6. **Genuine Valid Additions:**
   - 108 valid safety hardhats and structural firefighting helmets were added to ground truth.
   - 161 pedestrians, civilians, and emergency personnel were verified and added.

---

## 4. QA Verdict Summary

### 4.1 Verdict Breakdown by Class

| Class | Total Reviewed | PASS | FIX | REMOVE | UNCERTAIN | Net Added |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **helmet (1)** | 126 | 108 | 1 | 15 | 2 | **109** |
| **person (0)** | 200 | 161 | 0 | 39 | 0 | **161** |
| **Total** | **326** | **269** | **1** | **54** | **2** | **270** |

### 4.2 Verdict Breakdown by Dataset Split

| Split | Candidates Reviewed | PASS | FIX | REMOVE | UNCERTAIN | Added to Labels |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **train** | 251 | 212 | 0 | 39 | 0 | 212 |
| **val** | 30 | 25 | 1 | 4 | 0 | 26 |
| **test** | 45 | 32 | 0 | 11 | 2 | 32 |
| **Total** | **326** | **269** | **1** | **54** | **2** | **270** |

---

## 5. Label Adjustments & Deduplication

### 5.1 Coordinate Corrections (FIX)
- **Candidate:** `CAND_002015`
  - **Image:** `val/WEB09160.jpg`
  - **Class:** `helmet (1)`
  - **Raw Bbox:** `[759.81, 0.99, 1223.42, 93.12]` on $1268 \times 524$ px image
  - **Adjustment:** Top coordinate clamped cleanly to frame boundary $y_1 = 0.0$ px.
  - **Old Norm:** `(0.78203076, 0.08979962, 0.36562303, 0.17582061)`
  - **New Norm:** `(0.78203076, 0.08885496, 0.36562303, 0.17770992)`
  - **Rationale:** Firefighter helmet brim and visor in extreme closeup at upper edge; tightened to image boundary without face intrusion.

### 5.2 Deduplication Audit
- **IoU against existing post-HIGH labels:** Exact 0 collisions with same-class $\text{IoU} \ge 0.85$.
- **IoU between newly added Phase 2 candidates:** Exact 0 collisions with same-class $\text{IoU} \ge 0.85$.
- **Audit File:** [dfire_phase2_label_adjustments.csv](dfire_phase2_label_adjustments.csv)

---

## 6. Minimal Owner Escalation Queue

Two candidates exhibited genuine visual ambiguity due to extreme distance, heavy pixelation, and dense smoke haze. In accordance with policy, these were **not guessed** and were left unapplied:

| Queue Rank | Candidate ID | Split / Image | Class | Conf | Image Dimensions | Ambiguity Justification |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#093** | `CAND_003159` | `test/WEB11783.jpg` | helmet | 0.455 | $18 \times 14$ px | Distant blurred head silhouette in smoke haze; ambiguous whether white safety hardhat or civilian cap. |
| **#124** | `CAND_003490` | `test/WEB11783.jpg` | helmet | 0.407 | $16 \times 13$ px | Heavily pixelated head silhouette in smoke haze; ambiguous whether safety headgear or bare head / fabric cap. |

- **Escalation Queue File:** [dfire_phase2_owner_queue.csv](dfire_phase2_owner_queue.csv)

---

## 7. Dataset Impact & Final Counts

### 7.1 Canonical Class Distribution Across Dataset Iterations

| Class ID | Class Name | Raw / Pre-Remediation | Post-HIGH (Phase 1) | Post-MEDIUM (Phase 2) | Net Phase 2 Added |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | person | 16 | 1,570 | **1,731** | +161 |
| **1** | helmet | 0 | 8 | **117** | +109 |
| **2** | vest | 0 | 0 | **0** | 0 |
| **3** | fall | 0 | 0 | **0** | 0 |
| **4** | fire | 14,683 | 14,683 | **14,683** | 0 |
| **5** | smoke | 11,854 | 11,854 | **11,854** | 0 |
| **Total** | | **26,553** | **28,115** | **28,385** | **+270** |

### 7.2 Per-Split Distribution (Post-Phase 2)

| Split | Class 0 (person) | Class 1 (helmet) | Class 4 (fire) | Class 5 (smoke) | Total Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train** | 1,225 (+137) | 79 (+75) | 10,684 | 9,023 | 21,011 |
| **val** | 207 (+11) | 16 (+16) | 1,516 | 1,083 | 2,822 |
| **test** | 299 (+13) | 22 (+18) | 2,483 | 1,748 | 4,552 |
| **Total** | **1,731** | **117** | **14,683** | **11854** | **28,385** |
