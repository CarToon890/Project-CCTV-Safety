# D-Fire Trusted Phase-2 Snapshot Recovery & Audit Report

**Date:** 2026-09-26  
**Auditor / Reviewer:** `ANTIGRAVITY_VISUAL_QA`  
**Dataset Target:** `data/processed/dfire_remediated_verified`  
**Dataset Version:** `2.1.0-verified-phase2`  
**Base Dataset:** `data/processed/dfire_corrected` (Immutable SHA-256: `ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da`)  
**Status:** **RECOVERY COMPLETE — ALL GATES PASSED**

---

## 1. Executive Summary & Recovery Context

During the preceding missing-label remediation process, Phase 3 introduced **2,547 candidate annotations** into `data/processed/dfire_remediated` through heuristic evaluation rules rather than genuine, verified human/visual inspection. This violated project governance requiring actual visual QA evidence for all label additions.

To restore governance integrity and provide an uncompromised, verified dataset for downstream model development:
1. **Preservation of Evidence:** `data/processed/dfire_remediated` is preserved exactly as found on disk as auditable evidence and was neither modified nor deleted.
2. **Immutability of Upstream Sources:** `data/raw/dfire/data` (21,527 raw images) and `data/processed/dfire_corrected` (21,527 label files, SHA-256: `ddd439ddf777107ef59ad0004dd72d1dc5a08f85cb5b2c600417e7c7d246f8da`) remain 100% untouched and bit-identical.
3. **Creation of Trusted Verified Snapshot:** A new, clean dataset was reconstructed at `data/processed/dfire_remediated_verified` by deterministic replay from `dfire_corrected`, incorporating **only** trusted visual-QA outcomes from Phase 1 and Phase 2.
4. **Total Exclusion of Phase 3:** All 2,547 heuristic additions and every Phase 3 decision were completely excluded. Exactly 0 Phase 3-only candidate boxes exist in the verified dataset.
5. **Exact Independent Count Reconciliation:** The final class counts independently reconcile to the exact mathematical expectation:
   - `person (0)`: **1,731**
   - `helmet (1)`: **117**
   - `vest (2)`: **0**
   - `fall (3)`: **0**
   - `fire (4)`: **14,683**
   - `smoke (5)`: **11,854**
   - **Total Bounding Boxes:** **28,385** across **21,527** image-label pairs.

---

## 2. Trusted Evidence Ingestion & Synthesis

The recovery pipeline applies strictly verified visual outcomes from audited artifacts:

### 2.1 Phase 1 (HIGH-Tier Candidates)
- **Population:** 1,569 high-confidence candidates (1,560 person, 9 helmet).
- **Targeted Follow-up Visual QA:** 30 prioritized edge cases audited in [dfire_remediated_high_qa_queue.csv](dfire_remediated_high_qa_queue.csv) and [dfire_qa_label_adjustments.csv](dfire_qa_label_adjustments.csv):
  - **PASS (23 items):** Confirmed valid persons/helmets; retained.
  - **FIX (1 item):** `CAND_000122` (`WEB09297.jpg`, helmet) tightened from $y_2=520$ to $y_2=448$ to exclude facial features.
  - **REMOVE (6 unique candidates across 7 rows):** False positives on fabric balaclava (`CAND_001405`), smoke plumes (`CAND_001420`, `CAND_001228`), windshield suction cup mount (`CAND_000593`, rows 11 & 25), multi-person group box (`CAND_001383`), and scalp/forehead fragments (`CAND_000705`, `CAND_000894`).
  - **UNCERTAIN (0 items).**
- **Net Phase 1 Applied:** 1,554 persons + 8 helmets = **1,562 candidate boxes**.

### 2.2 Phase 2 (MEDIUM-Tier Candidates)
- **Population Audited:** Exactly 326 candidates ([dfire_medium_qa_queue.csv](dfire_medium_qa_queue.csv), [dfire_phase2_label_adjustments.csv](dfire_phase2_label_adjustments.csv)):
  - **100% Census of MEDIUM Helmets:** 126 candidates.
  - **Stratified Sample of MEDIUM Persons:** 200 candidates (`seed=42`).
- **Verdicts Applied:**
  - **PASS (269 items):** 108 helmets + 161 persons added.
  - **FIX (1 item):** `CAND_002015` (`WEB09160.jpg`, helmet) clamped cleanly to top frame boundary $y_1=0.0$.
  - **REMOVE (54 items):** 15 helmets (fabric balaclavas, baseball caps, reflections) + 39 persons (smoke plumes, flame patches, edge slivers) rejected and excluded.
  - **UNCERTAIN (2 items):** `CAND_003159` and `CAND_003490` (distant blurred heads in smoke haze) escalated and excluded.
- **Net Phase 2 Applied:** 161 persons + 109 helmets = **270 candidate boxes**.

### 2.3 Excluded Phase 3 Candidates
- **Total Excluded Population:** **3,684 candidates** (1,640 MEDIUM persons, 199 LOW helmets, 1,845 LOW persons).
- **All 2,547 heuristic additions** previously injected during Phase 3 are completely purged and excluded.
- Exactly 0 Phase 3 annotations are present in `dfire_remediated_verified`.

---

## 3. Dataset Architecture & Storage

- **Images:** Reused via **safe NTFS hardlinks** pointing directly to `data/processed/dfire_corrected/images/{train,val,test}`. Zero redundant disk space is consumed. Hardlinks share NTFS file index/inode with base images.
- **Labels:** Completely independent, freshly generated label files under `data/processed/dfire_remediated_verified/labels/{train,val,test}`.
- **Trusted Manifest:** Full audit manifest generated at `data/processed/dfire_remediated_verified/remediation_manifest.csv` containing all **5,579 candidate records** from `candidates.csv`, explicitly tracking:
  - `candidate_id`
  - `phase` (`Phase 1 (HIGH)`, `Phase 2 (MEDIUM)`, or `Unreviewed (MEDIUM/LOW)`)
  - `visual_verdict` (`PASS`, `FIX`, `REMOVE`, `UNCERTAIN`, `UNREVIEWED`)
  - `evidence_source_artifact`
  - `action` (`ADD`, `ADJUST_BBOX`, `REJECT`, `ESCALATE`, `EXCLUDED_PENDING_GENUINE_VISUAL_QA`)
  - `status` (`ADDED`, `FIXED`, `SKIPPED`, `EXCLUDED`)
  - `original_bbox`, `final_bbox`, `resulting_label_path`, and `notes`.
- **Metadata:** Synchronized `metadata/{train,val,test}.csv` reflecting updated box totals per image.
- **YOLO Config:** Validated `data.yaml` pointing to the verified directory with 6 canonical classes.

---

## 4. Class Distribution & Reconciled Totals

### 4.1 Canonical Class Distribution Across Dataset Stages

| Class ID | Class Name | dfire_corrected (Base) | Phase 1 Applied | Phase 2 Applied | Verified Final (dfire_remediated_verified) | Status vs Target |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | person | 16 | +1,554 | +161 | **1,731** | **EXACT MATCH** |
| **1** | helmet | 0 | +8 | +109 | **117** | **EXACT MATCH** |
| **2** | vest | 0 | 0 | 0 | **0** | **EXACT MATCH** |
| **3** | fall | 0 | 0 | 0 | **0** | **EXACT MATCH** |
| **4** | fire | 14,683 | 0 | 0 | **14,683** | **EXACT MATCH** |
| **5** | smoke | 11,854 | 0 | 0 | **11,854** | **EXACT MATCH** |
| **Total** | | **26,553** | **+1,562** | **+270** | **28,385** | **EXACT MATCH** |

### 4.2 Per-Split Distribution (Verified Snapshot)

| Split | person (0) | helmet (1) | fire (4) | smoke (5) | Total Boxes | Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **train** | 1,225 | 79 | 10,684 | 9,023 | 21,011 | 17,215 |
| **val** | 207 | 16 | 1,516 | 1,083 | 2,822 | 2,154 |
| **test** | 299 | 22 | 2,483 | 1,748 | 4,552 | 2,158 |
| **Total** | **1,731** | **117** | **14,683** | **11,854** | **28,385** | **21,527** |

---

## 5. Comprehensive Validation Audit Results

A dedicated validator script ([validate_dfire_remediated_verified.py](../../../scripts/validate_dfire_remediated_verified.py)) was executed to verify all 10 compliance gates:

| Gate | Description | Expected Standard | Observed Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Image-to-Label 1:1 Pairing | Exactly 21,527 pairs; 0 missing | 21,527 pairs verified across train, val, test | **PASS** |
| **2** | Canonical Class Schema & Syntax | IDs 0-5 only; 5 float tokens per line | 28,385 valid lines; 0 malformed lines | **PASS** |
| **3** | Coordinate Geometry & Bounds | $[0, 1]$ bounds; $w, h > 0$; 0 degenerate | All boxes strictly within bounds; 0 degenerate | **PASS** |
| **4** | Deduplication Audit | Same-class IoU $\ge 0.85$ collisions = 0 | 0 duplicate collisions across entire dataset | **PASS** |
| **5** | Split & Group Isolation | Cross-split group leakage = 0 | 0 leaking groups across train, val, test | **PASS** |
| **6** | Immutable Source Integrity | dfire_corrected SHA-256 identical; raw untouched | Base hash `ddd439...da` verified; raw intact | **PASS** |
| **7** | Hardlink Storage Integrity | Safe NTFS hardlinks sharing st_ino | 100% hardlink integrity verified | **PASS** |
| **8** | Exact Reconciled Counts | person=1,731; helmet=117; total=28,385 | Exact counts independently verified | **PASS** |
| **9** | Explicit Phase 3 Exclusion | 0 of 2,547 Phase 3 additions present | 0 Phase 3 candidates found in verified dataset | **PASS** |
| **10** | Manifest Terminal Audit | 5,579 candidate rows; 100% census mapped | 5,579 rows mapped with full provenance | **PASS** |

---

## 6. Remaining Unreviewed Population Protocol

No claim of semantic visual validation is made beyond the verified Phase 1 and Phase 2 evidence.

The remaining unreviewed candidate population consists of **3,684 candidates**:
- **MEDIUM-tier persons:** 1,640 candidates
- **LOW-tier helmets:** 199 candidates
- **LOW-tier persons:** 1,845 candidates

### Governance Protocol for Future Genuine Visual QA
1. **Zero Heuristics:** Automated thresholding, aspect ratio heuristics, and confidence cutoffs must **never** be used as sole acceptance criteria.
2. **Visual Inspection Requirement:** Any future candidate ingestion must produce diagnostic crop overlays and visual inspection notes reviewed by an authorized human/agent reviewer before inclusion.
3. **Auditable Queue:** Each future batch must be evaluated against a discrete QA queue artifact, recording `PASS`, `FIX` (with explicit tightened coordinates), `REMOVE`, or `UNCERTAIN`.
4. **Current Status:** In the interim, this population remains marked as `EXCLUDED_PENDING_GENUINE_VISUAL_QA` in `remediation_manifest.csv`.
