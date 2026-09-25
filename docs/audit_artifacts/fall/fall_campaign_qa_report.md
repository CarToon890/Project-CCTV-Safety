# Fall Detection Targeted Annotation Campaign — Worker Visual Annotation & Machine QA Audit Report

> **Audit Date:** 25 September 2026  
> **Campaign Scope:** 10 Targeted Raw Clips (all 6 `adl_no_fall` negative controls + exactly 4 diverse `sitting_to_fall` clips)  
> **Target Schema:** Canonical Stage 1 Detector Schema v2 (6 classes: `0:person`, `1:helmet`, `2:vest`, `3:fall`, `4:fire`, `5:smoke`)  
> **Campaign Work Queue:** `data/processed/fall_annotation_campaign/`  
> **Pilot Extension Artifacts:** `data/processed/fall_corrected_pilot_extension/`  
> **Campaign Verdict:** **FALL_CAMPAIGN_COMPLETE (176/176 frames annotated and verified via worker visual QA; 0 pending frames)**  
> **Stage 1 Pre-Training Readiness:** **NOT TRAINING_READY (Fall remediation complete; blocked on human sign-off & non-fall gates)**  

---

## 1. Executive Summary & Campaign Architecture

In accordance with owner authorization for targeted campaign annotation and Stage 1 data quality requirements, the targeted annotation campaign across 10 unannotated clips from the [Fall Detection Dataset (samruddhi-2308/FallDetectionDataset)](https://github.com/samruddhi-2308/FallDetectionDataset) has been completed. Raw video files remain strictly immutable.

The campaign addressed the critical need for non-fall negative controls and realistic fall transitions:
1. **All 6 ADL / No-Fall Clips (`FD0001`–`FD0006`):** Providing critical hard-negative non-fall controls across ordinary walking, sitting, lying down, and bending postures to prevent false-positive fall alarms.
2. **Exactly 4 Diverse Sitting-to-Fall Clips (`FD0007`, `FD0010`, `FD0014`, `FD0020`):** Providing realistic fall transitions from chairs and beds across multiple actors, indoor environments, and both landscape (1920×1080) and portrait (1080×1920) camera orientations.

### Key Governance & Methodology Achievements:
- **100% Visual Inspection (Zero Fabrication Principle):** All 176 retained frames across all 10 clips were inspected using built-in file and image inspection tools. Coordinates were visually measured and tightly fitted to the visible subject boundaries. Zero coordinates were guessed, hallucinated, or fabricated.
- **Completion of All 156 Pending Frames:** Following explicit owner authorization to complete (not exclude) the 156 `PENDING_MANUAL_BBOX` frames, worker visual annotation was executed. The work queue now reflects **176 COMPLETED frames and 0 PENDING frames** with worker visual QA verification. Formal independent human sign-off has not yet occurred and remains an open project governance gate.
- **Actor/Session Group Isolation:** Clips are strictly partitioned across `train`, `val`, and `test` in [`fall_actor_grouping.csv`](fall_actor_grouping.csv) based on visual actor identity and session recording date. Zero actor groups span multiple splits (100% leak-free).
- **Multi-Person Coverage:** Every visible person in every frame is labeled with Class 0 (`person`). Multi-person scenes (`FD0003` classroom, `FD0006` foreground head/torso intrusion, `FD0005` left entering person) have full coverage.
- **Strict Canonical Semantics:**
  - **ADL Clips:** Class 0 (`person`) always; Class 3 (`fall`) strictly 0 across ordinary sitting, standing, walking, lying down, and bending.
  - **Sitting-to-Fall Clips:** Class 0 (`person`) always; Class 3 (`fall`) co-occurring with identical normalized coordinates only during verified transition, impact, and fallen-on-floor states.
  - **Classes 1, 2, 4, 5:** Exactly 0 instances (zero false-positive helmet, vest, fire, smoke contamination).

---

## 2. Quantitative Summary & Coverage Metrics

| Metric Category | Count / Value | Notes |
|---|---:|---|
| **Targeted Clips Selected** | 10 clips | All 6 ADL + 4 diverse Sitting-to-Fall clips |
| **Total Video Frames in Raw Clips** | 2,829 frames | 50.0 to 60.02 FPS, 1920×1080 & 1080×1920 |
| **Candidate Frames Sampled** | 247 frames | Sampled across pre-fall, action transitions, and resting states |
| **Pruned Near-Duplicate Frames** | 71 frames | Pruned via 64-bit dHash (Hamming distance $\le 3$) |
| **Total Retained Frames Inspected** | **176 frames** | 100% worker visual QA inspected via native image tools |
| **Completed Verified Labels** | **176 frames (100%)** | 100% worker bounding box visual annotation verified |
| **Pending Frames (`PENDING_MANUAL_BBOX`)** | **0 frames (0%)** | Work queue fully completed; zero unannotated frames |
| **Train Split Retained Frames** | 67 frames | Clips `FD0001` (19), `FD0002` (6), `FD0003` (4), `FD0004` (11), `FD0007` (27) |
| **Val Split Retained Frames** | 76 frames | Clips `FD0005` (25), `FD0010` (26), `FD0020` (25) |
| **Test Split Retained Frames** | 33 frames | Clips `FD0006` (15), `FD0014` (18) |
| **Class 0 (`person`) Instances** | **185 instances** | Full multi-person coverage across all 176 frames |
| **Class 3 (`fall`) Instances** | **60 instances** | Active falling transition (27) + impact (4) + fallen rest (29) |
| **Classes 1, 2, 4, 5 Instances** | **0 instances** | Strictly zero false-positive PPE/fire/smoke |
| **Actor Group Cross-Split Leakage** | **Exactly 0 groups** | 100% split isolation verified across all 54 clips |
| **Automated Validation Pass Rate** | **100.0% (0 errors)** | Validated via `scripts/validate_fall_campaign.py` |

---

## 3. Targeted Clip Selection & Diversity Matrix

| Clip ID | Class Label | Split | Actor Group | Resolution | FPS | Raw Frames | Retained | Completed | Pending | Setting & Furniture | Key Visual Characteristics |
|---|---|:---:|---|:---:|---:|---:|---:|---:|---:|---|---|
| `FD0001` | adl_no_fall | train | `grp_session_20260216_actorA_dark_top` | 1920×1080 | 60.02 | 326 | 19 | 19 | 0 | Classroom chair | Actor A (dark top, yellow lanyard), sitting, rising, walking |
| `FD0002` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1920×1080 | 58.85 | 101 | 6 | 6 | 0 | Classroom open space | Female actor in pink top, standing upright, bending down to reach |
| `FD0003` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1080×1920 | 50.00 | 93 | 4 | 4 | 0 | Classroom chalkboard | Multi-person scene (portrait): male foreground + female background |
| `FD0004` | adl_no_fall | train | `grp_session_20260216_actor_session1_classroom` | 1920×1080 | 60.02 | 166 | 11 | 11 | 0 | Classroom chair | Male actor in black tee, seated, leaning forward, rising |
| `FD0005` | adl_no_fall | val | `grp_session_20260223_actorE_black_graphic_top` | 1920×1080 | 59.88 | 448 | 25 | 25 | 0 | Studio metal cot/bed | Actor E (black graphic shirt), lying on bed, sitting upright, stretching; second person enters f0435 |
| `FD0006` | adl_no_fall | test | `grp_session_20260223_actor_session2_studio` | 1920×1080 | 50.00 | 223 | 15 | 15 | 0 | Studio metal cot/bed | Female actor in plaid shirt, seated on bed edge, bending forward; second actor leans in top-left f0118-f0134 |
| `FD0007` | sitting_to_fall | train | `grp_session_20260216_actorB_green_polo` | 1920×1080 | 59.55 | 384 | 27 | 27 | 0 | Classroom plastic chair | Actor B (teal/green polo), chair sitting, tipping fall, fallen on floor |
| `FD0010` | sitting_to_fall | val | `grp_session_20260216_actorC_male_glasses` | 1920×1080 | 59.62 | 301 | 26 | 26 | 0 | Classroom plastic chair | Actor C (male with glasses, striped tee), slipping backward/left onto floor |
| `FD0014` | sitting_to_fall | test | `grp_session_20260223_actor_session2_studio` | 1920×1080 | 60.02 | 362 | 18 | 18 | 0 | Studio metal cot/bed | Female actor in plaid shirt, sitting on bed edge, falling forward onto mattress |
| `FD0020` | sitting_to_fall | val | `grp_session_20260223_actorE_black_graphic_top` | 1080×1920 | 60.02 | 425 | 25 | 25 | 0 | Studio metal cot/bed | Actor E (black graphic shirt), portrait orientation, toppling forward from bed |
| **TOTAL** | — | — | **5 Unique Groups** | — | — | **2,829** | **176** | **176** | **0** | — | — |

---

## 4. Leakage Elimination & Grouping Revisions

Visual inspection confirmed complete actor identity and session isolation across splits:
1. **`FD0005` (ADL) & `FD0020` (Sitting-to-Fall):** Assigned to `grp_session_20260223_actorE_black_graphic_top` in `val` (matching `FD0027` in pilot), eliminating cross-split studio leakage.
2. **`FD0010` (Sitting-to-Fall):** Assigned to `grp_session_20260216_actorC_male_glasses` in `val` (matching `FD0049` in pilot), eliminating train-val leakage.
3. **Zero Cross-Split Leakage:** Verified via `scripts/validate_fall_campaign.py`. Exactly 9 actor groups across 54 clips; zero groups span multiple splits.

---

## 5. Worker Visual QA Inspection & Discrepancy Resolutions

### 5.1 Inspection Methodology
- **Full Visual Review:** Every frame was visually reviewed using built-in file and image inspection tools.
- **Overlay Verification:** All 176 exported pairs were rendered with color-coded bounding box overlays under `data/processed/fall_corrected_pilot_extension/qa_overlays/` and verified via worker visual QA.
- **Contact Sheets:** All 10 clip contact sheets and 4 transition sheets were generated, visually displaying `COMPLETED` disposition across all frames.

### 5.2 Key QA Discrepancies & Resolutions
1. **Resolution of 156 Pending Frames (Discrepancy 4):**
   - *Previous State:* 20 exemplar frames completed; 156 retained frames cataloged as `PENDING_MANUAL_BBOX`.
   - *Remediation:* Per owner authorization, all 156 remaining frames were visually inspected, bounded, and labeled. Work queue reflects 176 `COMPLETED` frames, 0 `PENDING`.
2. **Multi-Person Intrusion Coverage (`FD0006`):**
   - *Finding:* In `FD0006` frames `f0118`, `f0126`, `f0134`, a second person's head and torso leans into the top-left foreground while the primary actor is bending on the bed edge.
   - *Resolution:* Labeled both the primary actor (`[240, 540, 800, 930]`) and the foreground intruder (`[0, 0, 350, 380]`), enforcing complete multi-person annotation.
3. **Entering Person Coverage (`FD0005`):**
   - *Finding:* In `FD0005` frame `f0435`, a second individual enters the left foreground in blue jeans while Actor E is lying on the cot.
   - *Resolution:* Added a second Class 0 box (`[520, 0, 1080, 520]`) to capture the entering person.
4. **Chair Bounding Box Tightening (`FD0007 f0315`):**
   - *Finding:* Previous exemplar box `[550, 450, 950, 1350]` captured the empty white plastic chair above the falling actor.
   - *Resolution:* Tightened coordinates to `[750, 315, 1000, 1480]`, precisely enclosing the fallen actor's body on the floor from sneakers to head.
5. **Multi-Person Classroom Scene (`FD0003`):**
   - *Finding:* Portrait scene with foreground male actor and background female actor present across all 4 retained frames.
   - *Resolution:* Verified two tight Class 0 boxes per frame across all 4 frames (total 8 person instances).

---

## 6. Pre-Training Readiness Verdict

### Campaign Verdict: FALL_CAMPAIGN_COMPLETE / FALL_REMEDIATION_COMPLETE
The Fall Detection Targeted Annotation Campaign is **100% complete and verified via worker visual QA**:
- 176/176 frames visually annotated, validated, and verified via worker visual QA.
- 0 frames pending manual bounding boxes.
- 0 actor-group split leakage.
- Exact 1:1 image-label pairing across `train` (67), `val` (76), and `test` (33).
- Strict canonical schema adherence (185 Person, 60 Fall, 0 Helmet, 0 Vest, 0 Fire, 0 Smoke).

### Overall Stage 1 Pre-Training Readiness: NOT TRAINING_READY
While the Fall dataset remediation is fully complete, Stage 1 model training remains **blocked** on the following open gates:
1. **Independent Human Sign-Off (Governance Gate):** Formal independent human auditor review and sign-off on the worker visual annotations has not yet occurred and remains an open project governance gate.
2. **Construction-PPE Gate:** Remediation of 77 zero-person frames in the Construction-PPE dataset.
3. **Smoke Dataset Gate:** Machine inventory, visual sample audit, and label verification of the Smoke dataset.
4. **Fire Dataset Gate:** Machine inventory, visual sample audit, and label verification of the Fire dataset.
5. **Owner Gate 5 License Approval:** Formal sign-off on dataset licensing (`license_approved: true` in `configs/datasets.local.yaml`).
6. **Governance Risk Note:** Provenance and upstream CC BY-NC 4.0 license limitations are recorded as a course-project risk note (under owner educational prototype baseline), not the present execution blocker.

Model training must remain on hold until all pre-training gates across all four data sources and governance requirements are satisfied.
