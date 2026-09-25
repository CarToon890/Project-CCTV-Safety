# Fall Detection Targeted Annotation Campaign — Human & Machine QA Audit Report

> **Audit Date:** 25 September 2026  
> **Campaign Scope:** 10 Targeted Raw Clips (all 6 `adl_no_fall` negative controls + exactly 4 diverse `sitting_to_fall` clips)  
> **Target Schema:** Canonical Stage 1 Detector Schema v2 (6 classes: `0:person`, `1:helmet`, `2:vest`, `3:fall`, `4:fire`, `5:smoke`)  
> **Campaign Work Queue:** `data/processed/fall_annotation_campaign/`  
> **Pilot Extension Artifacts:** `data/processed/fall_corrected_pilot_extension/`  
> **Readiness Verdict:** **PILOT_EXTENSION_READY (20 verified labels) — NOT TRAINING_READY (156 frames pending manual bbox)**

---

## 1. Executive Summary & Campaign Architecture

In accordance with owner authorization and Stage 1 data quality requirements, a targeted annotation campaign was planned, executed, and audited across 10 unannotated clips from the [Fall Detection Dataset (samruddhi-2308/FallDetectionDataset)](https://github.com/samruddhi-2308/FallDetectionDataset). Raw video files remain strictly immutable.

The campaign selected:
1. **All 6 ADL / No-Fall Clips (`FD0001`–`FD0006`):** Providing critical hard-negative non-fall controls across ordinary walking, sitting, lying down, and bending postures to eliminate false-positive fall alarms.
2. **Exactly 4 Diverse Sitting-to-Fall Clips (`FD0007`, `FD0010`, `FD0014`, `FD0020`):** Providing realistic fall transitions from chairs and beds across multiple actors, indoor environments, and both landscape (1920×1080) and portrait (1080×1920) camera orientations.

### Key Governance & Methodology Decisions:
- **Zero Fabrication Principle:** The 10 clips have `annotation_status: not_provided` in raw data. Bounding boxes are **never guessed, hallucinated, or fabricated**. Every retained frame was visually inspected via full-clip contact sheets. A focused subset of 20 exemplar frames was manually measured, visually verified, and second-reviewed under `data/processed/fall_corrected_pilot_extension/`. All remaining 156 retained frames are cataloged as `PENDING_MANUAL_BBOX` with verified action/state metadata and kept strictly out of the training-ready processed dataset.
- **Actor/Session Group Isolation:** Realigned identity groups in [`fall_actor_grouping.csv`](fall_actor_grouping.csv) based on visual evidence, completely eliminating cross-split actor leakage between `train`, `val`, and `test`.
- **Strict Canonical Semantics:**
  - **ADL Clips:** Class 0 (`person`) always; Class 3 (`fall`) strictly zero across ordinary sitting, standing, walking, lying down, and bending.
  - **Sitting-to-Fall Clips:** Class 0 (`person`) always; Class 3 (`fall`) co-occurring only during verified transition, impact, and fallen-on-floor states.
  - **Classes 1, 2, 4, 5:** Exactly 0 instances (no false-positive helmet, vest, fire, smoke).

---

## 2. Quantitative Summary & Coverage Metrics

| Metric Category | Count / Value | Notes |
|---|---:|---|
| **Targeted Clips Selected** | 10 clips | All 6 ADL + 4 diverse Sitting-to-Fall clips |
| **Total Video Frames in Raw Clips** | 2,829 frames | 50.0 to 60.02 FPS, 1920×1080 & 1080×1920 |
| **Candidate Frames Sampled** | 235 frames | Sampled across pre-fall, action transitions, and resting states |
| **Pruned Near-Duplicate Frames** | 59 frames | Pruned via 64-bit dHash (Hamming distance $\le 3$) |
| **Total Retained Frames Inspected** | **176 frames** | 100% human visual QA inspected via contact sheets |
| **Completed Verified Labels** | **20 frames** | 100% manual bounding box verified + second-reviewed |
| **Pending Frames (`PENDING_MANUAL_BBOX`)** | **156 frames** | State-classified; excluded from training dataset |
| **Train Split Retained Frames** | 68 frames | Clips `FD0001`, `FD0002`, `FD0003`, `FD0004`, `FD0007` |
| **Val Split Retained Frames** | 76 frames | Clips `FD0005`, `FD0010`, `FD0020` |
| **Test Split Retained Frames** | 32 frames | Clips `FD0006`, `FD0014` |
| **Class 0 (`person`) Instances (Extension)** | 21 instances | Exact 1:1 person coverage (2 persons in `FD0003`) |
| **Class 3 (`fall`) Instances (Extension)** | 5 instances | Active falling transition (2) + fallen rest (3) |
| **Classes 1, 2, 4, 5 Instances** | 0 instances | Strictly zero false-positive PPE/fire/smoke |
| **Actor Group Cross-Split Leakage** | Exactly 0 groups | 100% split isolation verified across all 54 clips |
| **Automated Validation Pass Rate** | 100.0% (0 errors) | Validated via `scripts/validate_fall_campaign.py` |

---

## 3. Targeted Clip Selection & Diversity Matrix

| Clip ID | Class Label | Split | Actor Group | Resolution | FPS | Raw Frames | Retained | Completed | Pending | Setting & Furniture | Key Visual Characteristics |
|---|---|:---:|---|:---:|---:|---:|---:|---:|---:|---|---|
| `FD0001` | adl_no_fall | train | `grp_session_20260216_actorA_dark_top` | 1920×1080 | 60.02 | 326 | 19 | 2 | 17 | Classroom chair | Actor A (dark top, yellow lanyard), sitting, rising, walking |
| `FD0002` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1920×1080 | 58.85 | 101 | 6 | 2 | 4 | Classroom open space | Female actor in pink top, standing upright, bending down to reach |
| `FD0003` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1080×1920 | 50.00 | 93 | 4 | 1 | 3 | Classroom chalkboard | Multi-person scene (portrait): male foreground + female background |
| `FD0004` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1920×1080 | 60.02 | 166 | 11 | 2 | 9 | Classroom chair | Male actor in black tee, seated, leaning forward, rising |
| `FD0005` | adl_no_fall | val | `grp_session_20260223_actorE_black_graphic_top` | 1920×1080 | 59.88 | 448 | 25 | 2 | 23 | Studio metal cot/bed | Actor E (black graphic shirt), lying on bed, sitting upright, stretching |
| `FD0006` | adl_no_fall | test | `grp_session_20260223_actor_session2_studio` | 1920×1080 | 50.00 | 223 | 15 | 2 | 13 | Studio metal cot/bed | Female actor in plaid shirt, seated on bed edge, bending forward |
| `FD0007` | sitting_to_fall | train | `grp_session_20260216_actorB_green_polo` | 1920×1080 | 59.55 | 384 | 27 | 3 | 24 | Classroom plastic chair | Actor B (teal/green polo), chair sitting, tipping fall, fallen on floor |
| `FD0010` | sitting_to_fall | val | `grp_session_20260216_actorC_male_glasses` | 1920×1080 | 59.62 | 301 | 26 | 3 | 23 | Classroom plastic chair | Actor C (male with glasses, striped tee), slipping backward/left onto floor |
| `FD0014` | sitting_to_fall | test | `grp_session_20260223_actor_session2_studio` | 1920×1080 | 60.02 | 362 | 18 | 2 | 16 | Studio metal cot/bed | Female actor in plaid shirt, sitting on bed edge, falling forward onto mattress |
| `FD0020` | sitting_to_fall | val | `grp_session_20260223_actorE_black_graphic_top` | 1080×1920 | 60.02 | 425 | 25 | 1 | 24 | Studio metal cot/bed | Actor E (black graphic shirt), portrait orientation, toppling forward from bed |
| **TOTAL** | — | — | **5 Unique Groups** | — | — | **2,829** | **176** | **20** | **156** | — | — |

---

## 4. Leakage Elimination & Grouping Revisions

Visual inspection revealed that several clips previously assigned coarse group labels contained actors present in other splits:
1. **`FD0005` (ADL) & `FD0020` (Sitting-to-Fall):** Contain Actor E (female in black "PACMAN YOU WIN" graphic top). In the pilot dataset, Actor E is present in `FD0027` (`grp_session_20260223_actorE_black_graphic_top`, assigned to `val`). Previously, coarse grouping had assigned studio clips to `test`. Assigning `FD0005` and `FD0020` to `grp_session_20260223_actorE_black_graphic_top` (`val`) prevents cross-split actor leakage between `val` and `test`.
2. **`FD0010` (Sitting-to-Fall):** Contains Actor C (male with glasses in striped t-shirt and sandals). In the pilot dataset, Actor C is present in `FD0049` (`grp_session_20260216_actorC_male_glasses`, assigned to `val`). Previously, coarse grouping assigned `FD0010` to `train`. Assigning `FD0010` to `grp_session_20260216_actorC_male_glasses` (`val`) eliminates train-val leakage.
3. **`fall_actor_grouping.csv` Integrity:** Verified with `scripts/validate_fall_campaign.py`. Exactly 9 actor groups across 54 clips; zero groups span multiple splits.

---

## 5. Human Visual QA Inspection & Discrepancies Log

### 5.1 Inspection Methodology
- **Contact Sheets:** 10 full-clip visual contact sheets (`data/processed/fall_annotation_campaign/contact_sheets/*_campaign_sheet.png`) and 4 transition sheets were generated and examined.
- **100% Frame Coverage:** All 176 retained frames were inspected for scene context, actor visibility, posture classification, and action phase.
- **Overlay Verification:** All 20 completed frames were verified with native visual tool rendering (`data/processed/fall_corrected_pilot_extension/qa_overlays/*_qa.jpg`) and second-reviewed.

### 5.2 Key QA Discrepancies & Resolutions
1. **Background Subtraction / Heuristic Contour Failure:**
   - *Finding:* Initial automated contour experiments on seated frames (`FD0007 f100`) grabbed the stationary desk, chalkboard frames, and floor shadows, missing the person's torso and head entirely.
   - *Resolution:* Automated contour bbox generation was strictly rejected. Adhered to the Zero Fabrication Principle: only manually measured, visually verified coordinates are written as completed; all other 156 frames remain `PENDING_MANUAL_BBOX`.
2. **Splayed Limb Cutoff on Fallen Postures:**
   - *Finding:* On `FD0007 f350` (fallen on floor), initial coordinate estimation centered on the torso, clipping the actor's splayed footwear on the left.
   - *Resolution:* Manually extended `xmin` from 700 to 390 to encompass the entire body from head to toe.
3. **Multi-Person Scene Handling (`FD0003`):**
   - *Finding:* Frame contains two actors simultaneously (foreground male + background female).
   - *Resolution:* Exported exactly two Class 0 (`person`) bounding boxes sharing the same image, validating multi-person detection in portrait CCTV view.
4. **Tail Truncation Enforcement:**
   - *Finding:* Sitting-to-fall clips conclude with actors rising from the floor or stepping forward to stop the camera (`FD0007` frames 376..383, `FD0010` frames 286..300, `FD0014` frames 359..361, `FD0020` frames 401..424).
   - *Resolution:* Excluded all tail recovery frames from training candidates, preventing unannotated or ambiguous post-fall recovery states from entering the dataset.

---

## 6. Pre-Training Readiness Verdict

### Overall Campaign Verdict: PILOT_EXTENSION_READY — NOT TRAINING_READY

1. **Approved Work:**
   - 10 targeted clips selected with full actor/session diversity and zero split leakage.
   - 176 frames extracted, temporally decimated, and cataloged in the work queue.
   - 20 exemplar frames completed with verified tight YOLO dual-box labels and second-reviewed.
   - 10 contact sheets and 4 transition sheets committed to processed storage.
2. **Blockers Preventing Full Model Training:**
   - Exactly **156 retained frames remain `PENDING_MANUAL_BBOX`**. Full training ingestion requires completing manual bounding box annotation across the work queue or formal exclusion.
   - Bounding boxes in the original 8 pilot clips require manual tightening (loose boxes spanning >1000px).
   - Pre-training Gates across PPE (77 zero-person remediation), Fire, and Smoke remain open.
   - Provenance and upstream CC BY-NC 4.0 license limitations are recorded as a course-project risk note (under owner educational prototype baseline), not the present execution blocker.

Model training must remain on hold until all pre-training gates are satisfied.
