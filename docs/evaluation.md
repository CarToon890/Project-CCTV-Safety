# Evaluation Protocol

## Model comparison

Use the identical test split for YOLOv8n and YOLOv8s. Report overall and
per-class Precision, Recall, F1, mAP50 and mAP50-95. Include model size, end-to-end
training time, peak GPU allocation and inference speed from the same hardware.
Do not select a winner from aggregate mAP alone; document regressions for every
class and emphasize Recall for safety-event classes.

## Error analysis

Review representative false positives and false negatives for:

- Fall versus sitting, crouching and maintenance work.
- Fire/smoke versus steam, fog, reflections and orange lighting.
- PPE with complete equipment, missing helmet, missing vest, multiple nearby
  people, truncation and occlusion.

For PPE, score detector quality and derived compliance separately. Record cases
where the body region is not visible; absence of a detection is not proof of a
violation in an occluded region.

Fight evaluation is outside the Stage 1 detector protocol. When Stage 2 is
unblocked, evaluate it at clip/event level against hugs, sports and close
collaborative work, including event Recall and false alerts per camera-hour.

## Baseline acceptance

This iteration has no fixed pass/fail threshold because source quality is not
yet known. Preserve the complete comparison and error-analysis results, then set
numeric targets for the next iteration. A future target-camera set should add
false alerts per camera-hour, event-level Recall, detection-to-alert latency,
duplicate-alert rate and day/night breakdowns.
