# Fall Detection Corrected Pilot — Human & Machine QA Audit Report

> **Audit Date:** 25 September 2026  
> **Pilot Scope:** 8 CVAT-annotated clips (6 `standing_to_fall`, 2 `sleeping_to_fall`) from Fall Detection Dataset  
> **Target Schema:** Canonical Stage 1 Detector Schema v2 (6 classes: `0:person`, `1:helmet`, `2:vest`, `3:fall`, `4:fire`, `5:smoke`)  
> **Artifacts Location:** `data/processed/fall_corrected_pilot/`  
> **Readiness Verdict:** **PILOT_READY (Approved for pipeline dry-runs & integration testing) — NOT TRAINING_READY (Remaining 46 unannotated clips blocked)**

---

## 1. Executive Summary & Verification Metrics

A decision-complete, fully corrected Stage 1 bounding box pilot was generated and validated for all 8 CVAT-annotated clips in `data/raw/fall_detection_dataset/`.

The build was executed by `scripts/build_fall_corrected_pilot.py` and validated by `scripts/validate_fall_pilot.py` under the following verified technical constraints:
- **Canonical 6-Class Mapping:** Pre-fall normal postures (`standing`, `Sleeping`) are mapped strictly to `0: person`. Active falling motion (`falling`, `Falling`) and fallen postures (`fallen`, `Fallen`) are mapped to co-occurring dual boxes (`0: person` and `3: fall`) sharing exact verified coordinates. No helmet (1), vest (2), fire (4), or smoke (5) boxes are created.
- **Conservative Tail & Gap Truncation:** Frame exports truncate strictly at the last verified tracked frame (`frame <= last_annotated_frame`). All unannotated tails (453 frames) and internal gaps (119 frames) are excluded; zero pseudo-labeling or extrapolation is performed.
- **Deterministic Temporal Decimation:** Near-duplicate frames are pruned using 64-bit dHash (Hamming distance $\le 3$), while strictly preserving all CVAT keyframes, state boundaries, dense active falling motion (stride 3), and representative fallen rest frames (stride 10).
- **Actor/Session Group Isolation:** Clips are allocated across `train`, `val`, and `test` strictly according to `fall_actor_grouping.csv`. Exactly 0 actor groups cross splits.
- **100% Machine Validation:** Every exported pair was validated with zero coordinate out-of-bounds, zero invalid classes, zero empty labels, and exact 1:1 image-label pairing.

### Quantitative Summary Table

| Metric Category | Count / Value | Notes |
|---|---:|---|
| **Total Clips Processed** | 8 clips | All CVAT-annotated clips in the dataset |
| **Total Video Frames in Raw Clips** | 2,771 frames | 50.0 to 60.02 FPS Full HD (1920x1080) |
| **Total Annotated Frames** | 2,199 frames | Ground-truth CVAT XML tracks |
| **Excluded Unannotated Tail Frames** | 453 frames | Truncated at last verified frame across 8 clips |
| **Excluded Internal Gap Frames** | 119 frames | Gaps between state tracks and unannotated mid-sections |
| **Decimated Near-Duplicate Frames** | 1,972 frames | Pruned via dHash temporal decimation rule |
| **Total Retained Pilot Pairs** | **227 pairs** | Exported under `data/processed/fall_corrected_pilot/` |
| **Train Split Pairs** | 132 pairs (58.1%) | Clips `FD0035`, `FD0044`, `FD0024`, `FD0052` (Actors A, B, D) |
| **Val Split Pairs** | 50 pairs (22.0%) | Clips `FD0049`, `FD0027` (Actors C, E) |
| **Test Split Pairs** | 45 pairs (19.8%) | Clips `FD0051`, `FD0054` (Actors F, G) |
| **Class 0 (`person`) Instances** | 227 instances | Exactly 1 person box on 100% of retained frames |
| **Class 3 (`fall`) Instances** | 141 instances | Exactly 1 fall box on falling (97) + fallen (44) frames |
| **Classes 1, 2, 4, 5 Instances** | 0 instances | Strictly zero false-positive PPE/fire/smoke boxes |
| **Actor Group Cross-Split Leakage** | Exactly 0 groups | 100% split isolation confirmed |
| **Machine Validation Pass Rate** | 100.0% (0 errors) | Validated via `scripts/validate_fall_pilot.py` |

---

## 2. Decimation & Frame Selection Statistics

The exact frame selection rule enforced by `scripts/build_fall_corrected_pilot.py` operates as follows:
1. Retain all state boundary frames (track start and track end).
2. Retain all CVAT keyframes (`keyframe="1"`).
3. Retain active falling motion densely (`frame_idx % 3 == 0`).
4. Candidate standing / sleeping and fallen resting frames (`frame_idx % 10 == 0`).
5. Prune consecutive candidate frames with Hamming distance $\le 3$ unless protected as a keyframe or state boundary.
6. Strictly truncate tail frames (`frame_idx > last_annotated_frame`).

### Per-Clip Decimation and Class Instance Breakdown

| Clip ID | Class / Posture | Split | Actor Group | Raw Frames | Annotated | Tail Drop | Candidate | Retained | Pre-Fall (0) | Falling (0,3) | Fallen (0,3) | Person Boxes | Fall Boxes |
|---|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FD0035` | standing_to_fall | train | `grp_session_20260216_actorA_dark_top` | 579 | 547 | 30 | 73 | 50 | 34 | 13 | 3 | 50 | 16 |
| `FD0044` | standing_to_fall | train | `grp_session_20260216_actorB_green_polo` | 474 | 407 | 65 | 60 | 50 | 20 | 19 | 11 | 50 | 30 |
| `FD0024` | sleeping_to_fall | train | `grp_session_20260223_actorD_pink_shirt` | 346 | 252 | 94 | 42 | 18 | 3 | 11 | 4 | 18 | 15 |
| `FD0052` | standing_to_fall | train | `grp_session_20260223_actorD_pink_shirt` | 208 | 105 | 101 | 24 | 14 | 3 | 8 | 3 | 14 | 11 |
| **Train Subtotal** | | | **3 Groups** | **1,607** | **1,311** | **290** | **199** | **132** | **60** | **51** | **21** | **132** | **72** |
| `FD0049` | standing_to_fall | val | `grp_session_20260216_actorC_male_glasses` | 197 | 184 | 12 | 31 | 24 | 8 | 10 | 6 | 24 | 16 |
| `FD0027` | sleeping_to_fall | val | `grp_session_20260223_actorE_black_graphic_top` | 437 | 293 | 36 | 39 | 26 | 11 | 8 | 7 | 26 | 15 |
| **Val Subtotal** | | | **2 Groups** | **634** | **477** | **48** | **70** | **50** | **19** | **18** | **13** | **50** | **31** |
| `FD0051` | standing_to_fall | test | `grp_session_20260223_actorF_green_polo` | 177 | 174 | 1 | 37 | 28 | 4 | 21 | 3 | 28 | 24 |
| `FD0054` | standing_to_fall | test | `grp_session_20260223_actorG_pattern_blouse` | 353 | 237 | 114 | 34 | 17 | 3 | 7 | 7 | 17 | 14 |
| **Test Subtotal** | | | **2 Groups** | **530** | **411** | **115** | **71** | **45** | **7** | **28** | **10** | **45** | **38** |
| **GRAND TOTAL** | | | **7 Groups** | **2,771** | **2,199** | **453** | **340** | **227** | **86** | **97** | **44** | **227** | **141** |

---

## 3. Conservative Tail Truncation and Gap Exclusion

All 8 clips terminate annotations prior to the video stream conclusion. Truncation boundaries are cataloged in `docs/audit_artifacts/fall/fall_tail_exclusion_log.csv`:

| Clip ID | Total Frames | Last Annotated Frame | Excluded Tail Range | Excluded Tail Count | Internal Gaps | Total Excluded Frames |
|---|---:|---:|:---:|---:|:---:|---:|
| `FD0035` | 579 | 548 | 549..578 | 30 frames | 464, 514 (2 frames) | 32 frames |
| `FD0044` | 474 | 408 | 409..473 | 65 frames | 218, 303 (2 frames) | 67 frames |
| `FD0049` | 197 | 184 | 185..196 | 12 frames | 144 (1 frame) | 13 frames |
| `FD0024` | 346 | 251 | 252..345 | 94 frames | None (0 frames) | 94 frames |
| `FD0027` | 437 | 400 | 401..436 | 36 frames | 0..2, 235, 247..350 (108 frames) | 144 frames |
| `FD0051` | 177 | 175 | 176..176 | 1 frames | 81, 156 (2 frames) | 3 frames |
| `FD0052` | 208 | 106 | 107..207 | 101 frames | 37, 90 (2 frames) | 103 frames |
| `FD0054` | 353 | 238 | 239..352 | 114 frames | 135, 165 (2 frames) | 116 frames |
| **TOTAL** | **2,771** | — | — | **453 frames** | **119 frames** | **572 frames** |

Zero frames from any excluded range were exported.

---

## 4. Human Visual QA Inspection

### 4.1 Visual Artifacts Inspected
Visual artifacts rendered under `data/processed/fall_corrected_pilot/` were examined:
- **16 multi-frame contact sheets:** 8 full-clip contact sheets (`qa_contact_sheets/*_contact_sheet.png`) and 8 transition-focused sheets (`qa_contact_sheets/*_transition_sheet.png`).
- **227 individual color-coded overlays:** Rendered under `qa_overlays/{train,val,test}/*_overlay.jpg`.
- **Inspected Count:** Exactly **227 retained frames** and **16 key state transitions** (8 onset transitions + 8 impact transitions).
- **Inspection Disclosure:** Exhaustive human inspection is claimed only for the 227 retained pilot frames and their 16 transitions. The 1,972 decimated frames and 453 tail frames were verified programmatically via machine checks and dHash logs.

### 4.2 State Transition Verification
Visual inspection confirmed semantic alignment at state boundaries:
1. **`FD0024` (sleeping_to_fall):**
   - Onset: Frame 120 (pre-fall `Sleeping`, class 0) $\rightarrow$ Frame 121 (active `Falling`, classes 0+3).
   - Impact: Frame 190 (`Falling`, classes 0+3) $\rightarrow$ Frame 191 (`Fallen`, classes 0+3).
2. **`FD0027` (sleeping_to_fall):**
   - Onset: Frame 180 (pre-fall `Sleeping`, class 0) $\rightarrow$ Frame 181 (active `Falling`, classes 0+3).
   - Impact: Frame 234 (`Falling`, classes 0+3) $\rightarrow$ Frame 236 (`Fallen`, classes 0+3).
3. **`FD0035` (standing_to_fall):**
   - Onset: Frame 463 (pre-fall `standing`, class 0) $\rightarrow$ Frame 465 (active `falling`, classes 0+3).
   - Impact: Frame 513 (`falling`, classes 0+3) $\rightarrow$ Frame 515 (`fallen`, classes 0+3).
4. **`FD0044` (standing_to_fall):**
   - Onset: Frame 217 (pre-fall `standing`, class 0) $\rightarrow$ Frame 219 (active `falling`, classes 0+3).
   - Impact: Frame 302 (`falling`, classes 0+3) $\rightarrow$ Frame 304 (`fallen`, classes 0+3).
5. **`FD0049` (standing_to_fall):**
   - Onset: Frame 113 (pre-fall `standing`, class 0) $\rightarrow$ Frame 114 (active `falling`, classes 0+3).
   - Impact: Frame 143 (`falling`, classes 0+3) $\rightarrow$ Frame 145 (`fallen`, classes 0+3).
6. **`FD0051` (standing_to_fall):**
   - Onset: Frame 80 (pre-fall `standing`, class 0) $\rightarrow$ Frame 82 (active `falling`, classes 0+3).
   - Impact: Frame 155 (`falling`, classes 0+3) $\rightarrow$ Frame 157 (`fallen`, classes 0+3).
7. **`FD0052` (standing_to_fall):**
   - Onset: Frame 36 (pre-fall `standing`, class 0) $\rightarrow$ Frame 38 (active `falling`, classes 0+3).
   - Impact: Frame 89 (`falling`, classes 0+3) $\rightarrow$ Frame 91 (`fallen`, classes 0+3).
8. **`FD0054` (standing_to_fall):**
   - Onset: Frame 134 (pre-fall `standing`, class 0) $\rightarrow$ Frame 136 (active `falling`, classes 0+3).
   - Impact: Frame 164 (`falling`, classes 0+3) $\rightarrow$ Frame 166 (`fallen`, classes 0+3).

---

## 5. Known Limitations & Technical Discrepancies

1. **Loose / Oversized Keyframe Bounding Boxes (Upstream Finding):**
   - In clips `FD0044`, `FD0049`, `FD0035`, `FD0024`, `FD0027`, `FD0051`, `FD0052`, `FD0054`, original CVAT annotators drew wide keyframe bounding boxes enclosing the subject plus large areas of surrounding furniture or blackboard (width > 1000 px).
   - In accordance with the prompt ("using the same verified coordinates... never pseudo-label or extend boxes beyond verified annotation"), these boxes were preserved without synthetic guessing.
   - **Remediation Recommendation:** A subsequent pass of polygon contour clamping or manual keyframe tightening is recommended before final training.
2. **Coverage Scope (8 of 54 clips):**
   - The pilot dataset covers only the 8 CVAT-annotated clips.
   - The remaining 46 clips (sitting-to-fall, ADL negative controls, and other standing/sleeping variations) have zero bounding box annotations.
   - Full training requires either annotating these 46 clips or formally excluding them under a justified dataset adequacy protocol.

---

## 6. Readiness Decision: PILOT_READY vs TRAINING_READY

| Readiness Tier | Status | Rationale |
|:---:|:---:|---|
| **PILOT_READY** | **APPROVED** | The 8 CVAT-annotated clips have been parsed, corrected to Stage 1 canonical dual-box semantics, truncated to exclude unannotated tails, decimated to remove redundancy, grouped to eliminate split leakage, and 100% machine-validated with valid `data.yaml`. Approved for model pipeline dry-runs and integration testing. |
| **TRAINING_READY** | **BLOCKED / NOT READY** | The remaining 46 unannotated clips cannot be ingested without annotations. Model training on the 8 pilot clips alone would suffer from lack of negative ADL controls (causing false positive falls on sitting/bending persons) and loose bounding boxes. Full training readiness requires resolving the remaining clips and satisfying all Stage 1 pre-training gates. |
