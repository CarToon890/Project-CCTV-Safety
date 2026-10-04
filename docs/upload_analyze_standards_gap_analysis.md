# Upload & Analyze — standards-oriented gap analysis

**Scope:** local educational pilot, reviewed 2026-10-04. This is a risk and
engineering checklist, not a certification, conformity assessment, or claim of
PDPA compliance. No model training was performed as part of this review.

## Risk register

| Area / reference | Current evidence | Gap and risk | Pilot action | Before production |
|---|---|---|---|---|
| AI governance / [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | Pilot disclaimer; held-out metrics and limitations are recorded in `project_status.md`; X3D has 45 held-out clips and missed 7/23 Fight clips. | No approved intended-use statement, operating threshold calibration, event-level target-camera evaluation, or monitoring/rollback process. | Keep human review, shadow triggers, and explicit pilot label. Record this as an open risk; do not treat a model label as an incident finding. | Assign accountable owner, define use boundaries and human escalation, monitor drift/error rates, establish incident response and change control. |
| Detection reliability / [ISO/IEC 25010](https://www.iso.org/standard/78176.html) | YOLO output provides boxes and confidence; PPE is derived from person/helmet/vest boxes. Direct model inference on the sample reproduces three broad/overlapping boxes before the web overlay, so this issue originates in model predictions. IoU 0.70→0.50 did not change the duplicate count. | Multiple person boxes can be mistaken for multiple people. No identity tracking or event-level precision/recall. The available clip is not enough ground truth to justify suppressing boxes safely. | UI labels the number as a maximum count of person boxes per frame, warns that duplicate boxes can inflate it, and labels Fall as a reviewable model signal. Preserve raw detections; do not present detections as verified unique people. | Evaluate box duplicate rate and event metrics on a representative, consented, camera-specific set; select tracking/NMS/temporal logic using measured quality and latency. Do not tune thresholds from a single clip. |
| Fight and fall evaluation / NIST AI RMF | Stage 2 uses X3D-S in 2-second windows. Recorded held-out confusion matrix is `[[20,2],[7,16]]`; raw X3D runs every window in shadow mode. | A window label is not a verified event. The pilot has small public-data coverage; Fall boxes are frame-level and can be false positives. | Conflicting Fight/Fall signal is explicitly marked for review; standalone Fall wording is “model signal · review”. Keep X3D evaluating every window so a trigger miss is observable. | Establish clip-level and event-level recall, false alerts per camera-hour, camera/site split evaluation, confidence calibration, and an operator-confirmation workflow. |
| Privacy / PDPA considerations | YuNet scans each decoded video frame. The original is used in memory for inference; the UI receives anonymized preview pixels and face masks. Temp uploads are removed in `finally`; anonymization incompleteness returns an error. | YuNet can miss small, side-view, occluded, blurred, or low-light faces. Full-frame scanning is slow (observed about 49 s on a 3.54 s/177-frame 1080p sample). Local code alone cannot decide lawful basis, notices, retention, access, or controller responsibilities. | Keep every-frame scanning and fail-closed behavior. Record face-detection and preview timings. Never optimize by reducing scan frequency or falling back to original video without a separately reviewed quality evaluation. | Complete a context-specific privacy assessment, access/retention/deletion policy, secure storage and transport, face-miss human review, and organizational/legal review. |
| API security / [OWASP ASVS](https://owasp.org/projects/asvs), [OWASP API Security Top 10](https://owasp.org/projects/api-security-project) | Upload extensions/size are checked; model settings have server-side bounds; temp files are removed; generic errors avoid returning tracebacks. | No authentication/authorization, CSRF boundary, audit identity, or network-access policy is built in. Uploads can be large and CPU/GPU intensive. | New decoded-pixel bounds and a one-analysis-per-process gate return 413/429 instead of allowing unconstrained decoded image memory or concurrent inference. Bind the server to loopback (`127.0.0.1`); there is no authentication. | If reachable from another device, add authentication, authorization, CSRF/origin policy, TLS, quotas, secure temp storage, audit controls, and threat-model review. Do not expose this pilot directly to a network. |
| Reliability and performance / ISO/IEC 25010 | Dense sampling and X3D windows are bounded; the API returns per-stage timings for decode, YuNet, YOLO, X3D, preview encoding, and total pipeline time. | Only a few local clips and one CPU are represented; timings vary by machine, codec, model load, and resolution. Faster sampling can miss short events and is not enabled as an optimization. | Treat timings as diagnostic observations; compare the same clip, hardware, and settings. Keep model input original and anonymization scan complete. | Set service-specific latency/error budgets after a representative benchmark; test overload, cancellation, malformed codecs, and sustained use on deployment hardware. |
| Data/model reproducibility | X3D held-out manifest hash is recorded in the model metrics. Stage 1 notebooks reference the archived unified dataset; archive manifest and train/val/test CSV hashes match local files. Current local weight checksums are in `model_evaluation_manifest.md`. | Kaggle metrics do not embed the archive hashes, so the exact bytes used in each remote run are not cryptographically attested. `requirements.txt` uses broad lower bounds. | Record the matching archive/split hashes and retain the remote-run attestation limitation; do not overstate provenance. | Embed dataset/split hashes and exact config in every experiment result; use platform-compatible dependency locks and provenance/SBOM for deployed builds. |
| Quality process / ASVS + ISO/IEC 25010 | Automated tests cover API validation, privacy helpers, detector schema, PPE association, stage-2 preprocessing and frontend structure. | CI, hardware-level load tests, and statistically meaningful model regressions are not set up. | The supported pytest suite is scoped to `tests/`; legacy scripts under `scripts/test_*.py` are manual data-QA tools requiring custom arguments, not pytest tests. | Establish CI with deterministic tests, security dependency scanning, release gates, regression datasets with approved handling, and operational monitoring. |

## Local benchmark observations

Measured 2026-10-04 on this Windows host, `.venv-test` / Python 3.11.9,
CPU, with the local checkpoints listed in `model_evaluation_manifest.md`.
These are diagnostic pilot samples, not a statistically valid evaluation.

| Clip / call | Result | Time |
|---|---|---|
| Same fall clip and settings, before caching YuNet's unchanged input size | YuNet scanned all 177 frames and found 194 face boxes; YOLO max person boxes/frame = 3; Fall detections on 10 sampled frames; X3D Fight windows = 0. | API `total_ms` 57,182; YuNet 48,783; decode 1,790; YOLO 1,579; X3D 675; preview encode 14. |
| Same fall clip and settings, after caching input size | Same 177-frame scan; 194 face boxes; max person boxes/frame = 3; Fall detections on 10 sampled frames. All selected analysis settings were unchanged. | API `total_ms` 24,087; YuNet 18,619; decode 1,654; YOLO 1,360; X3D 367; preview encode 6. |
| SCFD `fi008.mp4` (640×360, 20 fps, 44 frames, 2.20 s), Stage 2 endpoint | 1 window; model predicted `non_fight`, Fight probability 0.2155 (miss on this known Fight sample). | Client wall time 346 ms. |
| SCFD `nofi003.mp4` (480×360, 25 fps, 60 frames, 2.40 s), Stage 2 endpoint | 1 window; model predicted `non_fight`, Fight probability 0.2497. | Client wall time 423 ms. |

The fall clip reproduces the known duplicate person-box problem. Skipping
redundant calls to `FaceDetectorYN.setInputSize` reduced recorded combined time
by about 58% and YuNet time by about 62%, without reducing the frame scan. A
separate paired scan compared the old per-frame reset path with the cached path:
all 194 padded face boxes matched exactly on all 177 frames. This establishes
equivalence for this clip/runtime only, not broad privacy recall. The Fight
sample also reproduces a false negative; changing thresholds cannot correct a
low raw Fight score without trading off false positives. Review more held-out
clips and compare visual miss/false-alarm rates before selecting any further
speed or threshold change.

## Verification completed 2026-10-04

- Full suite: **234 passed**, with two upstream deprecation/future warnings, in
  `.venv-test` (Python 3.11.9). `pip check` reported no broken requirements.
- Regression coverage includes invalid/oversized uploads, decoded-pixel limits,
  decode failures, temporary-file cleanup, busy-analysis 429, YuNet fail-closed
  behavior, and API settings validation. The fail-closed regression confirms a
  detector exception returns an error without any frame/preview field and
  removes the temporary upload.
- Held-out X3D rerun used all 45 clips in `test_manifest.csv` and the repository's
  training preprocessing. It reproduced the stored results exactly:
  accuracy 0.8000, macro-F1 0.7984, balanced accuracy 0.8024, confusion matrix
  `[[20, 2], [7, 16]]`. Fight recall is 16/23 (69.6%); non-Fight false alarms
  are 2/22 (9.1%). CPU runtime for this evaluation was 20.6 seconds. These are
  public pilot clips, not target-camera performance.
- A temporary loopback HTTP server exercised the combined endpoint on the
  3.54-second fall clip: HTTP 200, complete face scan, 18 YOLO sample frames,
  10 sampled frames with a Fall box, up to 3 person boxes/frame, and 0 Fight
  windows. Timings were decode 1,509 ms, YuNet 32,488 ms, YOLO 2,028 ms, X3D
  419 ms, preview encoding 11 ms, total 39,055 ms. The server was stopped after
  the checks. Repeated timings varied substantially, so the faster earlier
  run is not a dependable latency guarantee.
- Visual spot check of six anonymized snapshots from that clip found one
  covered face in each sampled frame, including standing and falling motion.
  This is a narrow manual check of visible output; it does not measure missed
  faces across all frames, viewpoints, or lighting conditions.
- Direct Stage 2 HTTP checks reproduced `fi008.mp4` at Fight probability 0.2155
  and `nofi003.mp4` at 0.2497; both were classified non-Fight. This agrees with
  the known miss on the Fight example.
- `pip-audit` 2.10.1 reported no known vulnerabilities in the installed
  `.venv-test` environment or in a resolution of `requirements.txt` on this
  date; `pip check` also found no broken requirements. This is a point-in-time
  result, not a guarantee about future package releases or another deployment
  environment. The broad lower-bound declarations remain a reproducibility gap;
  production builds still need platform-specific lockfiles.

## Operational boundaries

- Start the demo with an explicit loopback bind: `python -m uvicorn webapp.api:app --host 127.0.0.1 --port 8000`.
- This endpoint has no login. Do not bind to `0.0.0.0`, port-forward it, or expose it to a shared network.
- A person count in this page means detection boxes in sampled frames, not unique people. PPE counts are derived from those boxes and inherit their errors.
- Track/proximity/motion values are experimental shadow signals; X3D still evaluates every window. They are not a validated alarm policy.
- The standards above are references for identifying gaps. This pilot is not certified to NIST, OWASP, ISO, or declared compliant with Thai PDPA.

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — voluntary AI risk-management framework.
- [OWASP ASVS](https://owasp.org/projects/asvs) and [OWASP API Security Project](https://owasp.org/projects/api-security-project) — application and API verification references.
- [ISO/IEC 25010:2023](https://www.iso.org/standard/78176.html) — product quality model.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/42001) — organization-level AI management-system standard, not a source-code checklist.
- [Thailand Personal Data Protection Act B.E. 2562 (2019)](https://www.mdes.go.th/law/detail/3577-Personal-Data-Protection-Act-B-E--2562--2019).
