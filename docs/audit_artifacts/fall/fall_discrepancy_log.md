# Fall Detection Dataset — Sample Audit Discrepancy Log and Technical Analysis

> **Audit Date:** 25 September 2026
> **Audit Target:** Fall Detection Dataset (State-to-Fall + ADL)
> **Scope:** Machine QA across 100% of clips (54 clips) and CVAT XMLs (8 files); Stratified frame inspection of 10 sampled clips (487 frames); Visual contact sheet inspection across all 10 sampled clips.
> **Operative Framework:** Non-commercial educational prototype; Canonical Detector Schema v2 (6 classes: `person`, `helmet`, `vest`, `fall`, `fire`, `smoke`).
> **Decision Status:** **GO (Passed Sample Audit) — Approved to proceed to corrected label build / curation pilot only (NOT training approval).**

---

## 1. Executive Summary & Audit Decision

A rigorous machine inventory and human visual QA audit was conducted on the **Fall Detection Dataset (State-to-Fall + ADL)** in accordance with the decision-complete sample audit protocol defined in `docs/fall_fire_replacement_dataset_search.md` Section 5.

### Audit Verdict: GO (Proceed to Corrected Label Build / Curation Pilot)

The dataset successfully satisfies the sample-audit gate for proceeding to annotation remediation and pipeline integration. Specifically:
1. **XML-to-Video Mapping Verification:** 100% verified. All 8 CVAT XML files match the exact frame count, resolution (1920x1080), and temporal sequence of their mapped MP4 video clips (`diff = 0` across all files). The `mapped_auto_exact_frame_count` entries in `annotations/cvat/annotation_manifest.csv` are correct.
2. **Real Surveillance/CCTV Relevance:** The video clips capture realistic full-body human falls and everyday activities in indoor settings (classroom and studio/bed environments).
3. **Zero Corrupted Video Files:** All 54 raw MP4 clips are intact, fully decodable via standard OpenCV codecs, and have verified unique SHA-256 digests (0 duplicate video uploads).
4. **Split Isolation Feasibility:** Actor identities, clothing, and room settings partition into distinct clusters (`session_20260216` classroom vs `session_20260223` studio), allowing complete elimination of identity and environment split leakage.

> [!IMPORTANT]
> **Boundary Warning:** This **GO** decision authorizes **only readiness to build corrected labels and a curated processed dataset**. It is **NOT approval for model training**. Unrestricted training remains gated until human label corrections are completed, verified, and approved by the project owner under Gate 5.

---

## 2. Machine QA Verification Results

### 2.1 100% Clip Inventory & Readability
- **Total clips in repository:** 54 MP4 video clips.
- **Total duration:** 363.3 seconds (~6.05 minutes of multi-state video).
- **Resolutions:** 100% uniform 1920×1080 full HD.
- **Frame rates:** 50.0 FPS to 60.02 FPS.
- **Codec:** H.264 / AVC in MP4 container.
- **Exact hash uniqueness:** 54 unique SHA-256 digests. Zero duplicated video files.

### 2.2 CVAT XML Mapping & Syntax Integrity
| Clip ID | Class Label | Video Filename | Frames | FPS | XML Filename | XML Size | Frame Count Match | Tracks in XML |
|---|---|---|---:|---:|---|---:|:---:|---|
| `FD0035` | `standing_to_fall` | `video_20260216_152357.mp4` | 579 | 60.02 | `video_20260216_152357.xml` | 579 | True | `standing(0..463) -> falling(465..513) -> fallen(515..548)` |
| `FD0044` | `standing_to_fall` | `video_20260216_154044.mp4` | 474 | 59.89 | `video_20260216_154044.xml` | 474 | True | `standing(None..None) -> standing(0..217) -> falling(219..302) -> fallen(304..408)` |
| `FD0049` | `standing_to_fall` | `video_20260216_155034.mp4` | 197 | 50.0 | `video_20260216_155034.xml` | 197 | True | `standing(0..113) -> falling(114..143) -> fallen(145..184)` |
| `FD0024` | `sleeping_to_fall` | `video_20260223_150715.mp4` | 346 | 60.02 | `video_20260223_150715.xml` | 346 | True | `Sleeping(0..120) -> Falling(121..190) -> Fallen(191..251)` |
| `FD0027` | `sleeping_to_fall` | `video_20260223_150939.mp4` | 437 | 60.02 | `video_20260223_150939.xml` | 437 | True | `Sleeping(3..180) -> Falling(181..234) -> Fallen(236..246) -> Fallen(351..400) -> Falling(181..181)` |
| `FD0051` | `standing_to_fall` | `video_20260223_152105.mp4` | 177 | 50.0 | `video_20260223_152105.xml` | 177 | True | `standing(0..80) -> falling(82..155) -> fallen(157..175)` |
| `FD0052` | `standing_to_fall` | `video_20260223_152210.mp4` | 208 | 60.01 | `video_20260223_152210.xml` | 208 | True | `standing(0..36) -> falling(38..89) -> fallen(91..106)` |
| `FD0054` | `standing_to_fall` | `video_20260223_152521.mp4` | 353 | 50.0 | `video_20260223_152521.xml` | 353 | True | `standing(0..134) -> falling(136..164) -> fallen(166..238)` |
| `FD0007` | `sitting_to_fall` | `video_20260216_151241.mp4` | 384 | 59.55 | `None` | N/A | None | `none` |
| `FD0001` | `adl_no_fall` | `video_20260216_151903.mp4` | 326 | 60.02 | `None` | N/A | None | `none` |

---

## 3. Key Technical Discrepancies and Remediations Required

While the candidate passed the sample audit, the audit revealed four systematic discrepancies between the raw CVAT XML annotations and the canonical Stage 1 detector requirements that **must be remediated before any training ingestion**:

### Discrepancy 1: Class Name & Semantic Schema Mismatch
- **Raw CVAT XML Classes:** The XML annotations use action/state track labels: `standing`, `falling`, `fallen` (in session 1) and `Sleeping`, `Falling`, `Fallen` (in session 2).
- **Canonical Stage 1 Requirement:** Stage 1 detector requires spatial bounding boxes from the canonical 6-class schema: `0: person`, `1: helmet`, `2: vest`, `3: fall`, `4: fire`, `5: smoke`.
- **Dual-Box Semantics for Falls:**
  - Pre-fall (`standing`, `Sleeping`): Must be mapped to **`0: person` only**.
  - Fall Transition (`falling`): Must be mapped to **co-occurring dual boxes: `0: person` AND `3: fall`**.
  - Fallen Posture (`fallen`): Must be mapped to **co-occurring dual boxes: `0: person` AND `3: fall`**.
- **Remediation Action:** Programmatic mapping transformation script that converts CVAT state tracks into canonical Stage 1 dual-box YOLO labels.

### Discrepancy 2: Unannotated Tail Frames (Premature Track Termination)
- **Observation:** In 7 of the 8 annotated clips, the CVAT tracks terminate before the video ends:
  - `FD0035`: Video has 579 frames; annotations stop at frame 549 (29 unannotated tail frames).
  - `FD0044`: Video has 474 frames; annotations stop at frame 408 (65 unannotated tail frames).
  - `FD0049`: Video has 197 frames; annotations stop at frame 184 (12 unannotated tail frames).
  - `FD0024`: Video has 346 frames; annotations stop at frame 251 (94 unannotated tail frames).
  - `FD0027`: Video has 437 frames; gap 247..350 has no boxes; tail 401..436 has no boxes (36 tail + 103 mid unannotated frames).
  - `FD0052`: Video has 208 frames; annotations stop at frame 106 (101 unannotated tail frames).
  - `FD0054`: Video has 353 frames; annotations stop at frame 238 (114 unannotated tail frames).
- **Root Cause:** In the raw collection, annotators stopped tracking the actor once the fall was completed, while the actor remained on the floor or got up to turn off the camera.
- **Risk:** Ingesting tail frames without boxes would penalize the model for detecting visible persons on the ground.
- **Remediation Action:** Truncate frame extraction to the annotated range `[0, last_annotated_frame]` or extend the final fallen bounding box until the actor starts standing up / exits.

### Discrepancy 3: Loose / Oversized Interpolated Bounding Boxes
- **Observation:** In `video_20260216_154044.xml` (FD0044), `video_20260216_155034.xml` (FD0049), and `video_20260223_150715.xml` (FD0024), some interpolated bounding boxes span > 1000 pixels in width (e.g. `w = 1698.9 px`), enclosing the actor plus wide sections of the background chalkboard/wall.
- **Root Cause:** Keyframe placement in CVAT used wide initial boxes or linear interpolation between standing and fallen keyframes that drifted outward across horizontal boundaries.
- **Remediation Action:** Tighten bounding box coordinates during label generation by clamping to the actor's contour or manual keyframe refinement.

### Discrepancy 4: Partial Annotation Coverage (46 Unannotated Clips)
- **Observation:** Only 8 of the 54 clips have CVAT XML exports (6 standing, 2 sleeping). Zero sitting-to-fall clips and zero ADL clips have annotations.
- **Remediation Action:** A targeted annotation campaign is needed for sitting-to-fall (to prevent false fall alarms on chairs) and ADL clips (to provide negative examples of person without fall).

---

## 4. Human Visual QA Inspection Coverage

### 4.1 Inspection Coverage Disclosure
- **Machine Inventory Coverage:** 54 / 54 clips (100.0%).
- **Sampled Video Clips Inspected:** Exactly 10 clips (8 annotated + 1 sitting_to_fall + 1 ADL hard negative).
- **Exact Stratified Frames Extracted & Inspected:** Exactly 487 frames.
- **Visual Artifacts Inspected:** 10 multi-frame visual contact sheets and 487 color-coded overlay images rendered under `data/processed/fall_sample_audit/`.
- **Exhaustive Frame Inspection:** Not claimed. Visual QA explicitly targeted the 487 stratified frames across pre-fall, falling transition, fallen rest, and unannotated tail sections.

### 4.2 Six-Class Missing-Label and Safety Risk Assessment
1. **`0: person` Completeness:** In all annotated frames, the primary actor is labeled by the state track. However, during the unannotated tail frames (> 450 total frames across clips), the actor remains visible on the floor but has zero bounding boxes. These frames must be truncated or annotated before training.
2. **`1: helmet` & `2: vest` False Positives:** Zero instances of PPE exist in the footage. Actors wear everyday college clothing (polo shirts, T-shirts, jeans, sweaters). Confirming that no everyday clothing is misidentified as PPE.
3. **`3: fall` Semantic Precision:** CVAT `falling` and `fallen` tracks cleanly match the physical dynamics of falling. In ADL clip `FD0001` (coughing/sitting) and sitting clip `FD0007` (sitting upright), the person remains seated without falling—providing essential negative controls.
4. **`4: fire` & `5: smoke` False Positives:** Background environments (chalkboards, softboxes, green screens, white walls) contain zero flame or smoke artifacts.

---

## 5. Duplicate, Near-Duplicate, and Leakage Analysis

### 5.1 Perceptual dHash Near-Duplicate Analysis
- A locally implemented 64-bit difference hash (dHash) was calculated across all 413 stratified frames.
- Consecutive frames during pre-fall standing (e.g. actor waiting for countdown) and post-fall resting exhibit Hamming distance $\le 3$, indicating near-identical temporal content.
- **Decimation Recommendation:** A temporal sampling interval of $\Delta t = 0.2\text{s}$ (every 10th to 12th frame) effectively eliminates near-duplicate frame redundancy while capturing all transitional poses.

### 5.2 Actor and Session Grouping (Leakage Elimination)
- In accordance with Rule #25, random frame-level or clip-level splitting is strictly prohibited.
- Video clips were clustered into 7 actor/session groups based on recording date, room setting, and actor clothing/appearance:
  - `grp_session1_actorA_classroom` (FD0001, FD0035, etc.) -> Allocated to `train`
  - `grp_session1_actorB_classroom` (FD0007, FD0044, etc.) -> Allocated to `train`
  - `grp_session1_actorC_classroom` (FD0049, etc.) -> Allocated to `val`
  - `grp_session2_actorD_studio` (FD0024, FD0052, etc.) -> Allocated to `train`
  - `grp_session2_actorE_studio` (FD0027, etc.) -> Allocated to `val`
  - `grp_session2_actorF_studio` (FD0051, etc.) -> Allocated to `test`
  - `grp_session2_actorG_studio` (FD0054, etc.) -> Allocated to `test`
- This group structure guarantees **zero actor identity leakage** and **zero room environment leakage** across splits.

---

## 6. Next Steps & Precise Remaining Work

The corrected-label pilot for the 8 CVAT-annotated clips has been successfully executed and validated via `scripts/build_fall_corrected_pilot.py` and `scripts/validate_fall_pilot.py` (see [fall_pilot_qa_report.md](fall_pilot_qa_report.md)):
- **Dual-box conversion:** Completed (227 `0:person`, 141 `3:fall` instances across 227 frames).
- **Tail truncation & Gap exclusion:** Completed (453 tail frames and 119 internal gap frames excluded; 0 unannotated frames exported).
- **Temporal decimation:** Completed (1,972 frames pruned using 64-bit dHash Hamming distance $\le 3$, preserving keyframes and state boundaries).
- **Split isolation:** Completed (train=132, val=50, test=45; zero actor group cross-split leakage).
- **Status:** **PILOT_READY (Approved for pipeline dry-runs & integration testing)**.

### Targeted Annotation Campaign & Pilot Extension (25 Sep 2026)
In response to Discrepancy 4 (unannotated ADL and sitting clips), a targeted annotation campaign was planned and executed:
- **10 Targeted Clips Selected:** All 6 ADL negative controls (`FD0001`–`FD0006`) plus 4 diverse sitting-to-fall clips (`FD0007`, `FD0010`, `FD0014`, `FD0020`).
- **176 Retained Review Frames:** 235 candidate frames decimated via 64-bit dHash (59 duplicates pruned), leaving 176 frames 100% human-inspected via contact sheets.
- **20 Completed Human Bbox Frames / 156 Pending:** 20 exemplar keyframes verified with tight manual bounding boxes under `data/processed/fall_corrected_pilot_extension/`; 156 frames remain `PENDING_MANUAL_BBOX` without fabricated labels.
- **Extension Class Counts:** Person 21, Fall 5, all other Stage-1 classes strictly 0 (no false-positive PPE/fire/smoke).
- **Split Isolation:** Regrouped in `fall_actor_grouping.csv` across 9 actor groups with **zero actor-group split leakage** (`scripts/validate_fall_campaign.py` passes 100%).
- **Readiness Verdict:** **PILOT_EXTENSION_READY but NOT TRAINING_READY** (due to 156 pending manual bbox frames).
- **Governance Risk Note:** Provenance and upstream CC BY-NC 4.0 licensing are documented as a course-project risk note (under owner educational prototype baseline), not the present execution blocker.

The remaining tasks prior to declaring the dataset **TRAINING_READY** are:
1. **Completion or Formal Exclusion of 156 Pending Frames:** Complete manual bounding box annotation for the 156 frames cataloged in `fall_annotation_work_queue.csv` or formally exclude them under a justified dataset adequacy protocol.
2. **Bounding Box Tightening:** Clamp oversized upstream keyframe boxes (width > 1000px in FD0044, FD0049, FD0035, etc.) via polygon contour clamping or manual keyframe adjustment.
3. **Owner Gate 5 Sign-off:** Secure formal `license_approved: true` in `configs/datasets.local.yaml` for Stage 1 pre-training.
