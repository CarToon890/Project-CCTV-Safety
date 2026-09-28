# Phase 3 D-Fire Final Candidate Resolution and Visual QA Report

**Date:** 2026-09-26  
**Reviewer:** `ANTIGRAVITY_VISUAL_QA`  
**Dataset Version:** `3.0.0-final-remediated`  
**Base Dataset:** `data/processed/dfire_corrected`  
**Target Dataset:** `data/processed/dfire_remediated`  

---

## 1. Executive Summary

Phase 3 concludes the comprehensive missing-label remediation of the D-Fire dataset. Every candidate from the full dataset scan (`data/processed/dfire_missing_label_full_scan/candidates.csv`) has now been fully resolved with 100% terminal auditability.

Key outcomes of Phase 3:
1. **Owner Resolution Applied:** Both ambiguous tiny helmets from the Phase 2 owner queue (`CAND_003159` and `CAND_003490`) were conservatively resolved as **REMOVE/SKIP** under the owner policy: *"insufficient pixels to verify safety helmet"*.
2. **100% Candidate Pool Coverage:** All remaining 1,640 MEDIUM persons and all 2,044 LOW candidates (199 helmets, 1,845 persons) were systematically audited via visual QA in 15 deterministic checkpointed batches.
3. **Checkpointed Deterministic Execution:** All 15 batches (size 250) were evaluated and checkpointed to `phase3_batch_checkpoints.json`, guaranteeing replayability and crash-safety.
4. **Verdicts & Quality Controls:** 2,353 candidates were **PASS** (added), 194 candidates were **FIX** (boundary clamped / adjusted and added), 1,137 candidates were **REMOVE** (conservative rejection of false positives on smoke, fire, civilian caps/hair, background structures, and tiny ambiguous silhouettes), and 0 items were left unresolved.
5. **Deduplication:** Absolute zero same-class duplicate boxes (IoU $\ge 0.85$) were ingested.
6. **Immutable Sources:** `data/raw` and `data/processed/dfire_corrected` remain 100% untouched and bit-identical.

---

## 2. Checkpointed Batch Progress

| Batch ID | Index Range | Candidates | PASS | FIX | REMOVE | UNCERTAIN | Checkpoint Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `batch_01` | 0..250 | 250 | 226 | 21 | 3 | 0 | Completed |
| `batch_02` | 250..500 | 250 | 236 | 13 | 1 | 0 | Completed |
| `batch_03` | 500..750 | 250 | 222 | 26 | 2 | 0 | Completed |
| `batch_04` | 750..1000 | 250 | 218 | 26 | 6 | 0 | Completed |
| `batch_05` | 1000..1250 | 250 | 220 | 22 | 8 | 0 | Completed |
| `batch_06` | 1250..1500 | 250 | 219 | 19 | 12 | 0 | Completed |
| `batch_07` | 1500..1750 | 250 | 217 | 15 | 18 | 0 | Completed |
| `batch_08` | 1750..2000 | 250 | 207 | 16 | 27 | 0 | Completed |
| `batch_09` | 2000..2250 | 250 | 195 | 12 | 43 | 0 | Completed |
| `batch_10` | 2250..2500 | 250 | 193 | 8 | 49 | 0 | Completed |
| `batch_11` | 2500..2750 | 250 | 147 | 15 | 88 | 0 | Completed |
| `batch_12` | 2750..3000 | 250 | 12 | 0 | 238 | 0 | Completed |
| `batch_13` | 3000..3250 | 250 | 18 | 1 | 231 | 0 | Completed |
| `batch_14` | 3250..3500 | 250 | 14 | 0 | 236 | 0 | Completed |
| `batch_15` | 3500..3684 | 184 | 9 | 0 | 175 | 0 | Completed |

---

## 3. QA Verdicts by Tier and Class

| Tier | Class | Reviewed | PASS | FIX | REMOVE | UNCERTAIN | Net Added |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MEDIUM** | person | 1,640 | 1468 | 134 | 38 | 0 | **1602** |
| **LOW** | helmet | 199 | 135 | 2 | 62 | 0 | **137** |
| **LOW** | person | 1,845 | 750 | 58 | 1037 | 0 | **808** |
| **Total** | | **3,684** | **2353** | **194** | **1137** | **0** | **2547** |

---

## 4. Final Dataset Counts

| Class ID | Class Name | Raw / Pre-Remediation | Post-HIGH (Phase 1) | Post-MEDIUM (Phase 2) | Final (Phase 3) | Net Added (Phase 3) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | person | 16 | 1,570 | 1,731 | **4141** | +2410 |
| **1** | helmet | 0 | 8 | 117 | **254** | +137 |
| **2** | vest | 0 | 0 | 0 | **0** | 0 |
| **3** | fall | 0 | 0 | 0 | **0** | 0 |
| **4** | fire | 14,683 | 14,683 | 14,683 | **14,683** | 0 |
| **5** | smoke | 11,854 | 11,854 | 11,854 | **11,854** | 0 |
| **Total** | | **26,553** | **28,115** | **28,385** | **30932** | **+2547** |

### Per-Split Distribution (Final Version 3.0.0)

| Split | Class 0 (person) | Class 1 (helmet) | Class 4 (fire) | Class 5 (smoke) | Total Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **train** | 2958 | 170 | 10,684 | 9,023 | 22835 |
| **val** | 485 | 35 | 1,516 | 1,083 | 3119 |
| **test** | 698 | 49 | 2,483 | 1,748 | 4978 |
| **Total** | **4141** | **254** | **14,683** | **11,854** | **30932** |

---

## 5. Owner Queue Resolution

| Candidate ID | Split / Image | Class | Old Status | Terminal Action | Owner Policy Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CAND_003159` | `test/WEB11783.jpg` | helmet | UNCERTAIN | REMOVE / SKIP | Insufficient pixels to verify safety helmet. |
| `CAND_003490` | `test/WEB11783.jpg` | helmet | UNCERTAIN | REMOVE / SKIP | Insufficient pixels to verify safety helmet. |

**Final Owner Queue Items Remaining:** Exactly **0** unresolved items.
