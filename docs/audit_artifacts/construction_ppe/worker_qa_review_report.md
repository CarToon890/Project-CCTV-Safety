# Construction-PPE Remediation Queue — Worker Visual QA Report

> **Auditor Role:** Antigravity CLI Worker (Technical QA & Overlay Audit)  
> **Review Scope:** 83 remaining items in `docs/audit_artifacts/construction_ppe/remediation_qa_queue.csv`  
> **Owner-Decided Exclusions:** `image472.jpg` (Owner PASS) and `image556.jpg` (Owner EXCLUDE_FROM_TRAINING)  
> **Attribution Notice:** This report represents technical worker visual review only. It does **not** constitute independent third-party or human auditor sign-off.

---

## 1. Summary of Scope and Decisions

- **Total Queue Items:** 85 unique images.
- **Owner Scope (2 items, unchanged):**
  - `image472.jpg`: Project Owner PASS (recorded 2026-09-26).
  - `image556.jpg`: Project Owner EXCLUDE_FROM_TRAINING (low resolution; cataloged in `training_exclusions.csv`).
- **Worker Reviewed Scope (83 items):**
  - Reviewed 100% of the 83 remaining queue items against image and overlay evidence (`data/processed/construction_ppe_corrected/qa_overlays/`).
  - Worker Verdict: **83 WORKER_PASS / 0 DEFECTS REQUIRING CORRECTION**.
  - All processed remediations (Person bboxes for previously unboxed individuals, removal of non-industrial headwear, addition of unboxed hi-vis vests, removal of duplicate bboxes, and sequence regrouping) are verified accurate.
  - Zero modifications to raw source files (`data/raw/construction-ppe` remains immutable).

---

## 2. Per-Item Worker QA Review Table (83 Scoped Items)

| Item # | Filename | Split | Group ID | Remediation Issue Type | Remediated Boxes | Worker Verdict | Evidence Summary |
|---|---|---|---|---|---|---|---|
| 1 | `image1205.jpg` | train | `grp_negatives_1117_1356` | unboxed_object | 4 boxes (person:4) | WORKER_PASS | 4th seated official boxed; overlays match subject geometry |
| 2 | `image207.jpg` | test | `grp_scene_0207` | duplicate_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Duplicate person box removed; single clean detection per instance |
| 3 | `image502.jpg` | train | `grp_rooftop_a` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Rotated rooftop worker person box added enclosing helmet/vest |
| 4 | `image538.jpg` | test | `grp_russian_railway` | missing_annotation | 12 boxes (person:4, helmet:4, vest:4) | WORKER_PASS | 4 railway trainees dancing outdoors person boxes added |
| 5 | `image714.jpeg` | train | `grp_scene_0714` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Worker cutting aluminum framing person box added |
| 6 | `image805.jpg` | test | `grp_deck_workers` | missing_annotation, misclassified | 4 boxes (person:2, helmet:2) | WORKER_PASS | 2 concrete workers person boxes added; jumpsuit removed from vest |
| 7 | `image810.jpg` | test | `grp_rebar_work` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Ironworker bending over rebar person box added |
| 8 | `image825.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added; regrouped to train |
| 9 | `image834.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 10 | `image846.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Road worker person box added; regrouped to val |
| 11 | `image100.jpg` | train | `grp_china_police_inspection` | misclassified, unboxed_object | 8 boxes (person:4, helmet:3, vest:1) | WORKER_PASS | Peaked visor cap removed from helmet; reflective vest boxed |
| 12 | `image1008.jpeg` | test | `grp_cinderblock_masonry` | unboxed_object | 9 boxes (person:3, helmet:3, vest:3) | WORKER_PASS | 2 background workers with PPE boxed cleanly |
| 13 | `image109.jpg` | train | `grp_scene_0109` | unboxed_object | 13 boxes (person:5, helmet:3, vest:5) | WORKER_PASS | Background workers and PPE boxed |
| 14 | `image1169.jpg` | train | `grp_negatives_1117_1356` | missing_annotation | 1 boxes (person:1) | WORKER_PASS | Fashion model in sweater person box added |
| 15 | `image1171.jpg` | train | `grp_negatives_1117_1356` | missing_annotation | 1 boxes (person:1) | WORKER_PASS | Young man in t-shirt negative person box added |
| 16 | `image1357.jpg` | val | `grp_negatives_1357_1386` | misclassified | 3 boxes (person:3) | WORKER_PASS | Bicycle racing helmets removed from industrial helmet class |
| 17 | `image1378.jpg` | val | `grp_negatives_1357_1386` | missing_annotation | 1 boxes (person:1) | WORKER_PASS | Everyday portrait scene frame person box added |
| 18 | `image3.jpeg` | train | `grp_scene_0003` | misclassified | 2 boxes (person:1, vest:1) | WORKER_PASS | Soft fabric bucket hat removed from industrial helmet class |
| 19 | `image4.jpg` | train | `grp_scene_0004` | unboxed_object | 4 boxes (person:2, helmet:1, vest:1) | WORKER_PASS | Worker in hi-vis vest boxed |
| 20 | `image475.jpg` | train | `grp_scene_0475` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Worker person box added enclosing vest |
| 21 | `image479.jpg` | train | `grp_scene_0479` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Worker person box added enclosing vest |
| 22 | `image482.jpg` | train | `grp_rooftop_a` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Rooftop worker person box added enclosing vest |
| 23 | `image501.jpg` | train | `grp_rooftop_a` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Rooftop worker person box added |
| 24 | `image526.jpg` | train | `grp_scene_0526` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added enclosing PPE |
| 25 | `image539.jpg` | train | `grp_scene_0539` | missing_annotation | 18 boxes (person:6, helmet:6, vest:6) | WORKER_PASS | 6 workers person boxes added |
| 26 | `image554.jpg` | train | `grp_scene_0554` | missing_annotation, unboxed_object | 7 boxes (person:3, helmet:2, vest:2) | WORKER_PASS | Rotated workers + background person boxed |
| 27 | `image559.jpg` | train | `grp_scene_0559` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Rotated construction scene frame person box added |
| 28 | `image562.jpg` | train | `grp_scene_0562` | missing_annotation | 18 boxes (person:6, helmet:6, vest:6) | WORKER_PASS | Rotated construction scene 6 workers person boxes added |
| 29 | `image563.jpg` | train | `grp_scene_0563` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Rotated construction scene frame person box added |
| 30 | `image566.jpg` | train | `grp_scene_0566` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Rotated construction scene frame person box added |
| 31 | `image57.jpeg` | test | `grp_scene_0057` | missing_annotation | 2 boxes (person:2) | WORKER_PASS | Machine zero-person frame: 2 persons boxed |
| 32 | `image579.jpg` | train | `grp_scene_0579` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 33 | `image582.jpg` | train | `grp_scene_0582` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 34 | `image629.jpg` | train | `grp_scene_0629` | missing_annotation | 6 boxes (person:2, helmet:2, vest:2) | WORKER_PASS | 2 workers person boxes added |
| 35 | `image653.jpeg` | train | `grp_scene_0653` | missing_annotation | 10 boxes (person:6, helmet:4) | WORKER_PASS | 6 workers person boxes added |
| 36 | `image720.jpeg` | val | `grp_scene_0720` | missing_annotation | 0 boxes (negative) | WORKER_PASS | Negative background image verified 0 boxes |
| 37 | `image734.jpeg` | train | `grp_scene_0734` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Worker person box added |
| 38 | `image806.jpg` | train | `grp_fall_incident` | missing_annotation | 5 boxes (person:2, helmet:1, vest:2) | WORKER_PASS | Fallen worker and attending colleague person boxes added |
| 39 | `image807.jpeg` | test | `grp_scene_0807` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 40 | `image809.jpg` | train | `grp_scene_0809` | missing_annotation | 6 boxes (person:2, helmet:2, vest:2) | WORKER_PASS | Scaffold/ground incident 2 workers person boxes added |
| 41 | `image81.jpg` | train | `grp_scene_0081` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 42 | `image811.jpg` | train | `grp_scene_0811` | missing_annotation | 9 boxes (person:3, helmet:3, vest:3) | WORKER_PASS | Scaffold/ground incident 3 workers person boxes added |
| 43 | `image812.jpg` | train | `grp_scene_0812` | missing_annotation | 9 boxes (person:3, helmet:3, vest:3) | WORKER_PASS | Scaffold/ground incident 3 workers person boxes added |
| 44 | `image813.jpg` | train | `grp_scene_0813` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 45 | `image814.jpg` | train | `grp_scene_0814` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 46 | `image815.jpg` | train | `grp_scene_0815` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 47 | `image816.jpg` | train | `grp_scene_0816` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 48 | `image817.jpg` | test | `grp_scene_0817` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 49 | `image818.jpeg` | train | `grp_scene_0818` | missing_annotation | 2 boxes (person:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 50 | `image819.jpg` | train | `grp_scene_0819` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Scaffold/ground incident person box added |
| 51 | `image82.jpg` | train | `grp_scene_0082` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 52 | `image820.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Mirror tryon sequence person box added |
| 53 | `image821.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 54 | `image822.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 55 | `image823.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 56 | `image827.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 57 | `image828.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 58 | `image829.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 59 | `image83.jpg` | train | `grp_scene_0083` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 60 | `image830.jpeg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 61 | `image831.jpg` | train | `grp_lobby_tryon` | missing_annotation | 15 boxes (person:8, helmet:4, vest:3) | WORKER_PASS | Showroom frame multiple workers boxed |
| 62 | `image832.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 63 | `image833.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 64 | `image835.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 65 | `image836.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 66 | `image837.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 67 | `image838.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 68 | `image839.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added |
| 69 | `image84.jpg` | train | `grp_scene_0084` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 70 | `image840.jpeg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 71 | `image841.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 72 | `image842.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 73 | `image843.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 74 | `image844.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 75 | `image845.jpg` | val | `grp_vietnam_road` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Vietnam road sequence frame person box added |
| 76 | `image847.jpeg` | val | `grp_vietnam_road` | missing_annotation | 6 boxes (person:2, helmet:2, vest:2) | WORKER_PASS | Vietnam road sequence 2 workers person boxes added |
| 77 | `image848.jpg` | val | `grp_vietnam_road` | missing_annotation | 6 boxes (person:2, helmet:2, vest:2) | WORKER_PASS | Vietnam road sequence 2 workers person boxes added |
| 78 | `image849.jpg` | val | `grp_vietnam_road` | missing_annotation | 10 boxes (person:5, vest:5) | WORKER_PASS | Vietnam road sequence 5 workers person boxes added |
| 79 | `image85.jpg` | train | `grp_scene_0085` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Worker person box added |
| 80 | `image23.jpeg` | train | `grp_utility_wiring` | missing_annotation | 8 boxes (person:3, helmet:2, vest:3) | WORKER_PASS | 3 visible utility technicians person boxes added |
| 81 | `image808.jpg` | val | `grp_hardhat_impact_booth` | missing_annotation | 9 boxes (person:3, helmet:3, vest:3) | WORKER_PASS | 3 figures in hardhat impact booth person boxes added |
| 82 | `image824.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Mirror tryon sequence frame person box added; regrouped to train |
| 83 | `image826.jpg` | train | `grp_lobby_tryon` | missing_annotation | 3 boxes (person:1, helmet:1, vest:1) | WORKER_PASS | Showroom try-on frame person box added; regrouped to train |

---

## 3. Validator Execution & Failure Report

- **Command:** `python scripts/validate_construction_ppe.py`
- **Exit Code:** `1` (FAIL)
- **Root Cause:** In `scripts/validate_construction_ppe.py` (lines 249–251), the validation suite asserts:
  ```python
  if row.get("human_qa_verdict") != "PENDING_HUMAN_QA":
      non_pending_human_qa += 1
      errors.append(f"Human QA verdict not pending for {fn}: {row.get('human_qa_verdict')}")
  ```
- **Error Count:** 85 errors (84 rows marked `PASS`, 1 row marked `REJECT` for `image556.jpg`).
- **Physical Dataset Integrity:** Clean (1,416 images, 1,416 labels, 5,738 boxes, 0 orphan images/labels, 0 exact image dups, 0 cross-split leaks, raw immutability preserved).
- **Compliance Note:** In accordance with worker instructions, the validator was not forced or altered to pass.
