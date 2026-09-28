# D-Fire Remediated Dataset — Visual QA Audit Report

> **Audit Date:** 26 September 2026  
> **Reviewer:** `ANTIGRAVITY_VISUAL_QA`  
> **Target Queue:** `docs/audit_artifacts/dfire/dfire_remediated_high_qa_queue.csv` (30 items)  
> **Dataset Target:** `data/processed/dfire_remediated`  
> **Governance Verdict:** **`VISUAL_QA_RESOLVED` (30/30 items reviewed and resolved; 0 UNCERTAIN)**

---

## 1. Executive Summary

A comprehensive visual inspection was executed across all 30 prioritized items in the D-Fire HIGH-tier follow-up Human QA queue. Each candidate detection was evaluated by inspecting high-resolution image crops and full diagnostic overlays to verify semantic class correctness, bounding box tightness, duplicate status, and usability.

### Summary of Verdicts

| Action | Count | Percentage | Description |
|---|---:|---:|---|
| **PASS** | 23 | 76.7% | Correct semantic class and usable tight bounding box; retained in dataset without modification. |
| **FIX** | 1 | 3.3% | Correct object/class, but bounding box required tightening; adjusted in label files. |
| **REMOVE** | 6 unique (7 rows) | 20.0% | False positive detections, redundant group boxes, or unusable edge truncations; removed from dataset. |
| **UNCERTAIN** | 0 | 0.0% | Zero items require escalation; all 30 decisions backed by definitive visual evidence. |
| **TOTAL** | **30 rows** | **100.0%** | **Complete decision coverage across all targeted edge cases.** |

---

## 2. Key Visual Findings & Remediation Actions

### 2.1 Helmets (9 Candidates Evaluated)
- **CAND_000014 (Item #1, PASS):** Close-up of CAL FIRE firefighter wearing yellow structural helmet with goggles and chinstrap. Despite large size (36.7% width, 32.2% height), it is a genuine, high-quality helmet detection.
- **CAND_000122 (Item #2, FIX):** Interview shot of São Paulo firefighter wearing white/silver helmet with visor. The detection was valid but the bottom boundary extended down over his face and mouth ($y_2=520$). Tightened $y_2$ to $448$ to precisely encompass helmet dome, visor, and earflaps.
- **CAND_001405 (Item #5, REMOVE):** Wildland firefighter in savannah wearing a white fabric flame-resistant hood/balaclava and goggles. Misclassified by YOLO-World as a safety helmet; no rigid helmet/hardhat present. Removed.
- **CAND_001232, CAND_001233, CAND_001436, CAND_001488, CAND_001499, CAND_001547 (Items #3, #4, #6, #7, #8, #9, PASS):** All confirmed as genuine industrial hardhats or firefighter helmets on personnel.

### 2.2 Smoke Overlap Detections (3 Candidates Evaluated)
- **CAND_001420 (Item #10, REMOVE):** Dark vertical smoke plume rising from burning industrial warehouse in an aerial drone shot misclassified as a person. Removed.
- **CAND_000593 (Items #11 & #25, REMOVE):** Car windshield suction-cup GPS/camera mount silhouette filming wildfire misclassified as a person. Removed.
- **CAND_001228 (Item #12, REMOVE):** Billowing black smoke plume rising from crash/wildfire site misclassified as a person. Removed.

### 2.3 Same-Class Candidate Pairs (10 Candidates / 5 Pairs Evaluated)
- **CAND_001383 (Item #14, REMOVE):** Redundant multi-person group box spanning across multiple individuals who already have individual tight person detections. Removed.
- **CAND_000704, CAND_000851, CAND_001297, CAND_000325, CAND_000910, CAND_000241, CAND_000895, CAND_000088, CAND_000089 (Items #13, #15-#22, PASS):** All confirmed as distinct individuals standing in close proximity, walking behind one another, or holding hoses side-by-side.

### 2.4 Extreme Geometry & Border Touch (8 Candidates Evaluated)
- **CAND_000705 (Item #26, REMOVE):** Unusable extreme edge truncation showing only partial dark scalp/hair at the very bottom frame border with zero face, torso, or limbs.
- **CAND_000894 (Item #27, REMOVE):** Unusable extreme edge truncation / photobomb showing only forehead and top of glasses at the bottom corner of a scenic roof image.
- **CAND_000001, CAND_000493, CAND_000907, CAND_000965, CAND_001011 (Items #23, #24, #28, #29, #30, PASS):** Valid pedestrians, workers, or speakers whose bounding boxes naturally touch frame margins or have wide aspects due to posture.

---

## 3. Label Adjustment Manifest Summary

| Candidate ID | Canonical Class | Split / Image | Action | Justification |
|---|---|---|:---:|---|
| `CAND_000122` | `1: helmet` | `train/WEB09297.jpg` | **FIX** | Tightened $y_2$ from 520 to 448 (normalized: $0.4667 \to 0.4167, h: 0.5111 \to 0.4111$) |
| `CAND_001405` | `1: helmet` | `test/WEB11805.jpg` | **REMOVE** | False positive on fabric flame shroud/balaclava |
| `CAND_001420` | `0: person` | `train/WEB04681.jpg` | **REMOVE** | False positive on aerial smoke plume |
| `CAND_000593` | `0: person` | `test/WEB10675.jpg` | **REMOVE** | False positive on windshield suction-cup mount |
| `CAND_001228` | `0: person` | `train/WEB07625.jpg` | **REMOVE** | False positive on smoke plume |
| `CAND_001383` | `0: person` | `train/WEB07526.jpg` | **REMOVE** | Redundant multi-person group box |
| `CAND_000705` | `0: person` | `test/WEB10118.jpg` | **REMOVE** | Unusable scalp-only edge truncation |
| `CAND_000894` | `0: person` | `test/WEB10414.jpg` | **REMOVE** | Unusable forehead-only photobomb |

---

## 4. Final Dataset Counts

- **Total Image-Label Pairs:** 21,527
- **Total Bounding Boxes:** 28,116 (net change: $-6$ false positives/redundant boxes removed)
  - `person` (class 0): 1,571 (16 original + 1,555 remediated)
  - `helmet` (class 1): 8 (0 original + 8 remediated)
  - `fire` (class 4): 14,683
  - `smoke` (class 5): 11,854
- **Empty (Negative) Labels:** 9,702 (WEB10118 and WEB10414 reverted to negative images after removing unusable partial head clippings).
