# Project Owner Human QA Sign-off — 2026-09-26

This record documents decisions explicitly provided by the project owner in the conversation on 2026-09-26. It records owner sign-off; it does not claim independent third-party review. Raw source datasets remain unchanged.

## Construction PPE

- `image472.jpg`: **PASS** per project owner.
- `image556.jpg`: **EXCLUDE FROM TRAINING** per project owner because its low resolution makes the detection/annotation quality inadequate for training. Preserve the raw source and existing corrected audit build; downstream training packaging must omit this image and its label. This decision is not a request to delete source files.
- Machine-readable exclusion: [`construction_ppe/training_exclusions.csv`](construction_ppe/training_exclusions.csv). The corrected audit build remains intact; the exclusion must be applied when producing the eventual training package.
- Project owner confirms they personally QA-reviewed and sign off **PASS on the other 83 PPE queue entries**. The queue records this in `human_qa_notes`; this is project-owner human QA, not an independent third-party audit.

## Fall

- Project owner authorizes recording human sign-off for the Fall sample represented by all 176 rows in `fall_annotation_work_queue.csv`; those rows currently carry worker visual QA and PASS verdicts.
- Scope is the 176-frame campaign queue only. This does **not** sign off the separate 8-clip CVAT pilot bounding-box tightening issue identified in the Fall readiness report.
- For the 8-clip CVAT pilot, the project owner decides that current evidence is insufficient to redefine the bounding-box boundaries. Keep the existing annotations unchanged. This is a no-change disposition, not a claim that the boxes have been tightened or are fully accurate for training.

## D-Fire

- Project owner authorizes recording human sign-off for the 106-row sample represented by `dfire_audit_handoff_queue.csv`; those rows currently carry worker visual QA and PASS verdicts.
- This applies to the sample audit only. It does not authorize training or waive remaining dataset-packaging, validation, provenance, or owner training-authorization gates.

## Smoke-source decision

- Boreal Smoke is not part of the planned training dataset; use D-Fire for the Smoke class. No Boreal acquisition or audit is required for this project scope.

## SCFD Stage 2 pilot — owner QA decision (2026-09-27)

- **All 300 SCFD Clips Owner Human Visual QA Approved (`OWNER_HUMAN_QA_APPROVED`):**
  - **Five Edge Cases (2026-09-27 initial sign-off):** The project owner reviewed the worker-flagged edge cases and confirms `scfd_non_fight_nofi011`, `scfd_non_fight_nofi052`, `scfd_non_fight_nofi095`, `scfd_non_fight_nofi115`, and `scfd_non_fight_nofi116` as **Non-Fight** under a clip-level physical-fight definition.
  - **Remaining 295 Clips (2026-09-27 subsequent owner inspection):** The project owner confirms they personally inspected all remaining 295 SCFD clips and they passed. Genuine owner human visual QA **PASS** is recorded for these 295 clips (`OWNER_HUMAN_QA_APPROVED`), superseding the prior intermediate status `OWNER_ACCEPTED_WORKER_QA_PILOT`. This applies strictly to these 295 clips with labels as currently recorded.
  - **Total Human Visual Approvals:** All 300 SCFD pilot clips (150 fight, 150 non-fight) now carry genuine owner human visual QA approval (**`OWNER_HUMAN_QA_APPROVED`**).
- **Seven Atomic Cross-Label Scene Groups:** The project owner approves the seven proposed cross-label scene groups as atomic grouping units for leakage-safe partitioning. All clips within each group must be assigned together to one split; none of these groups may be divided across train/validation/test:
  - Cluster A: `fi046`, `fi047`, `fi049`, `fi050`, `nofi023`, `nofi024`.
  - Cluster B: `fi065`–`fi067`, `nofi097`–`nofi098`.
  - Cluster C: `fi073`–`fi074`, `nofi147`.
  - Cluster D: `fi059`, `nofi148`.
  - Cluster E: `fi107`, `nofi027`.
  - Cluster F: `fi116`, `nofi051`.
  - Cluster G: `fi122`–`fi123`, `nofi055`–`nofi056`.
- **Scope:** Owner human visual QA records clip event-label decisions; it is not a third-party rights audit or production/deployment sign-off. The owner later accepted the documented source limitations for the educational pilot and separately authorized pilot training. This record does not authorize future runs or deployment.

## Training status

**Current status (2026-09-28):** These sign-offs were followed by preparation,
validation, and owner-authorized educational pilot training. Stage 1 remains
exactly `person`, `helmet`, `vest`, `fall`, `fire`, `smoke`; `fight` remains in
Stage 2. The Stage 1 YOLOv8n/YOLOv8s and Stage 2 SCFD/X3D-S pilot runs have
completed and been evaluated. This dated sign-off is evidence of QA decisions,
not approval for a new training run or production deployment.
