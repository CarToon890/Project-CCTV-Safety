# Stage 1 Dataset Primary-Source Audit

## Project CCTV Safety — Spatial Detector (6 Classes)

> **Audit date:** 25 September 2026
>
> **Scope:** Candidate datasets for `person`, `helmet`, `vest`, `fall`, `fire`, and `smoke`
>
> **Method:** Read-only review of publisher-owned dataset cards, repositories, license files, and official documentation. No dataset archive or image was downloaded.
>
> **Important:** This is a technical provenance screen, not legal advice.

> **Project-owner usage decision (25 September 2026):** This baseline is for
> education and research only. Commercial use is out of scope. Non-commercial
> candidates may therefore enter sample audit, but their attribution,
> share-alike, source-platform, and likeness restrictions remain mandatory.

## Executive decision

The earlier `APPROVED` labels in [dataset_evaluation.md](dataset_evaluation.md) are **not supported by the current primary-source evidence**. There is presently **no candidate approved for full download or training**.

The most important corrections are:

1. **D-Fire is CC0, not CC BY 4.0.** Its license also says the maintainers do not own copyright in the underlying images and disclaim third-party rights clearance. This blocks training approval until image-level provenance is audited.
2. **Beyza PPE Detection V1 is marked MIT by Kaggle, not CC0.** Its card says the data came from Roboflow, but does not establish that the Kaggle uploader can license all source imagery under MIT.
3. **Uttej Fall does not provide blanket ODbL permission for the images.** Kaggle reports `Database: Open Database, Contents: © Original Authors`, while the uploader says the images were gathered from various sources.
4. **The two Simuletic candidates are CC BY-NC-SA 4.0 in Kaggle metadata, not CC BY 4.0.** The Fire/Smoke card text and the Simuletic/Hugging Face pages contradict that metadata, creating a material license conflict.
5. **Ironwolf is Apache 2.0 only**, not “Apache 2.0 / CC BY 4.0”. Its card does not identify the owners or source footage.

### Status vocabulary

- **GO — sample audit only:** It is reasonable to inspect a small, non-training sample after the project owner accepts the stated license constraints. This does **not** authorize training.
- **HOLD:** Do not download the full dataset or train. Resolve license authority/provenance and then conduct sample annotation QA.
- **REJECT:** Do not use for the Stage 1 detector because of a blocking license, availability, task-format, or scope mismatch.

## Recommended shortlist

| Priority | Candidate | Category | Status now | Why |
|---:|---|---|---|---|
| 1 | Ultralytics Construction-PPE | PPE | **SAMPLE AUDITED → HOLD FOR TRAINING** | Structure and labels parse, but related frames from the same actor/scene cross train/val/test; 10 orphan labels and unresolved source-image provenance also require remediation. |
| 2 | SH17 | PPE | **GO — research-only sample audit (owner accepted non-commercial scope)** | Author repository and paper document 8,099 images, 75,994 instances, 17 classes, and source URLs; CC BY-NC-SA 4.0, attribution/share-alike and Pexels terms remain binding. |
| 3 | D-Fire | Fire/Smoke | **HOLD** | Excellent detection format and scale, but the official license explicitly says the publisher does not own the source images and does not clear third-party rights. |
| 4 | Uttej Fall Detection | Fall | **HOLD** | Useful YOLO boxes and three postures, but image contents remain © original authors and were gathered from unspecified sources. |
| 5 | Shlok PPE COCO | PPE | **HOLD** | Apache 2.0 is stated on Kaggle, but the card provides no source-image provenance or license authority. |

**Training gate:** A candidate may move from this shortlist to training only after (1) primary license/authority verification, (2) source provenance review, (3) sample annotation audit, (4) duplicate/sequence-leakage audit, and (5) confirmation that every visible canonical class is labeled or remediated.

## Candidate findings

### A. PPE: `person`, `helmet`, `vest`

#### 1. `beyzakucuk/ppe-detection-v1`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/beyzakucuk/ppe-detection-v1)
- **Availability:** Available; version 1 metadata reported approximately 1.06 GB at audit time.
- **License evidence:** Kaggle metadata says **MIT**, while the description says the upstream Roboflow dataset is CC0. The project report currently records only CC0; that is inaccurate.
- **Format/classes:** The card describes JPG images, YOLO bounding boxes, train/validation/test splits, and six classes: `Helmet`, `No-Helmet`, `Goggles`, `No-Goggles`, `Vest`, and `Person`.
- **Authority/provenance:** The Kaggle publisher points to a Roboflow project but provides no source-by-source rights trail. MIT on the Kaggle copy does not by itself prove authority over the upstream images.
- **Decision:** **HOLD** pending identification of the exact upstream Roboflow version, its image provenance, and the governing license for that version.

#### 2. `shlokraval/ppe-dataset`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/shlokraval/ppe-dataset)
- **Availability:** Available; version 1 metadata reported approximately 2.54 GB.
- **License evidence:** Kaggle metadata says **Apache 2.0**.
- **Format/classes:** The card states COCO JSON bounding-box annotations for helmets, gloves, goggles, face shields, safety vests, and other PPE. It does not give a complete authoritative category list or state whether all people are annotated.
- **Authority/provenance:** The dataset card does not identify the image sources or establish that the uploader controls the image copyrights.
- **Decision:** **HOLD** pending provenance, full category extraction, and a sample check for `person` missing labels.

#### 3. `niravnaik/safety-helmet-and-reflective-jacket`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/niravnaik/safety-helmet-and-reflective-jacket)
- **Availability:** Available; 10,500 images are claimed, with 7,350/1,575/1,575 train/test/validation images and approximately 546 MB total.
- **License evidence:** Kaggle metadata says **Apache 2.0**.
- **Format/classes:** YOLOv7 bounding boxes for only `helmet` and `reflective jacket`.
- **Blocker:** No `person` class is described. Training it directly in the unified detector would create a severe person missing-label penalty. The card also provides no source-image provenance.
- **Decision:** **HOLD**. Consider only if provenance is cleared and people are exhaustively relabeled by humans; pseudo-labeling alone is not sufficient approval.

#### 4. `mugheesahmad/sh17-dataset-for-ppe-detection` (SH17)

- **Official sources:** [Author repository](https://github.com/ahmadmughees/SH17dataset), [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/mugheesahmad/sh17-dataset-for-ppe-detection), and [author paper](https://arxiv.org/abs/2407.04590)
- **Availability/size:** 8,099 images, 75,994 instances, 17 classes; Kaggle metadata reported approximately 14.24 GB.
- **License evidence:** Kaggle says **CC BY-NC-SA 4.0**. The authors state that images were scraped from Pexels and direct users to the [Pexels license](https://www.pexels.com/license/).
- **Format/classes:** YOLO and Pascal VOC labels; includes `Person`, `Safety-vest`, and `Helmet` plus body parts and other PPE.
- **Restrictions:** Non-commercial and share-alike restrictions apply. Image-likeness and Pexels restrictions remain relevant. The author repository has no separate root LICENSE file, so the Kaggle terms and dataset disclaimer must be preserved in the provenance record.
- **Decision:** **GO — research-only sample audit.** The project owner accepted an education/research-only scope on 25 September 2026. Preserve CC BY-NC-SA attribution/share-alike requirements, Pexels source records, and likeness/privacy review; this decision does not authorize commercial use or full training before QA.

#### 5. New candidate: Ultralytics Construction-PPE

- **Official sources:** [Ultralytics dataset documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe/) and [official YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml)
- **Availability/size:** 1,416 images: 1,132 train, 143 validation, and 141 test; downloadable through the official YAML.
- **License evidence:** **AGPL-3.0**, stated in the official documentation and YAML.
- **Format/classes:** Ultralytics YOLO; 11 classes including `helmet`, `vest`, `Person`, `no_helmet`, and other PPE/missing-PPE labels.
- **Integration issue:** The project must ignore/remap source `no_*` categories consistently because its canonical `no_helmet` and `no_vest` are post-processing events, not detector classes. Visible canonical objects still require completeness QA.
- **Decision:** **SAMPLE AUDITED → HOLD FOR TRAINING.** The owner accepted an education/research-only scope, but visual audit confirmed scene/actor leakage across the published splits and found ten orphan labels. Underlying curation provenance also remains unresolved. See [construction_ppe_sample_audit.md](construction_ppe_sample_audit.md).

### B. Fall

#### 6. `uttejkumarkandagatla/fall-detection-dataset`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/uttejkumarkandagatla/fall-detection-dataset)
- **Availability/size:** Available; 485 images are described (374 train, 111 validation), approximately 52.6 MB.
- **License evidence:** Kaggle says **Database: Open Database, Contents: © Original Authors**. This is not blanket permission to reuse every image.
- **Format/classes:** YOLO-style bounding boxes for `Fall Detected`, `Walking`, and `Sitting`.
- **Authority/provenance:** The uploader explicitly says images were gathered “from various sources” without identifying or licensing those sources.
- **Decision:** **HOLD/BLOCKED** pending source-level permissions. Technically promising, but not cleared for training.

#### 7. `simuletic/cctv-incident-dataset-fall-and-lying-down-detection`

- **Official sources:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/simuletic/cctv-incident-dataset-fall-and-lying-down-detection) and [Simuletic dataset catalogue](https://www.simuletic.com/datasets)
- **Availability/size:** Available; approximately 172 MB. Simuletic's catalogue reports a 113-image open sample.
- **License conflict:** Kaggle platform metadata says **CC BY-NC-SA 4.0**, while the card body says CC BY 4.0. Platform and prose do not agree.
- **Format/classes:** YOLO Pose with bounding boxes and 17 keypoints; `laying` and `standing`. The publisher says roughly 95% of subjects are lying/fallen.
- **Blockers:** The license conflict must be resolved in writing. The sample is highly positive-skewed, and prior Simuletic provenance contradictions in the Fight audit require visual/human verification rather than accepting the “fully synthetic” claim automatically.
- **Decision:** **HOLD** pending written license clarification and provenance/human QA.

#### 8. `soumicksarker/multiple-cameras-fall-dataset`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/soumicksarker/multiple-cameras-fall-dataset)
- **Availability/size:** Available; approximately 3.79 GB.
- **License evidence:** `Other (specified in description)`, but the card description provides no operative reuse license.
- **Format:** Multi-camera video of simulated falls and daily activities; no Stage 1 bounding-box annotations are established by the card.
- **Decision:** **REJECT for Stage 1** due to unclear license and task mismatch. It could only be reconsidered for future temporal research after written rights clarification.

#### 9. `payutch/fall-video-dataset`

- **Official source:** The previously recorded Kaggle candidate was not accessible through the public metadata endpoint during this audit.
- **Format issue:** The project record describes video plus keypoint CSV rather than canonical detection boxes.
- **Decision:** **REJECT for Stage 1** until an accessible publisher source and license can be verified; even then it would require a separate conversion and human annotation project.

### C. Fire and Smoke

#### 10. D-Fire (`gaia-solutions-on-demand/DFireDataset`)

- **Official sources:** [Publisher repository](https://github.com/gaia-solutions-on-demand/DFireDataset), [README](https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/README.md), and [LICENSE](https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE)
- **Availability/size:** Available through links maintained in the official repository; 21,527 images.
- **Format/classes:** YOLO bounding boxes; 1,164 fire-only images, 5,867 smoke-only, 4,658 with both, and 9,838 negatives; 14,692 fire boxes and 11,865 smoke boxes.
- **License evidence:** **CC0 1.0**, not CC BY 4.0.
- **Critical authority blocker:** The LICENSE says the maintainers do not own copyright in the images, applies CC0 to their collection, and disclaims responsibility for clearing third-party rights. This is not sufficient evidence that every source image is reusable.
- **Decision:** **HOLD/BLOCKED** pending an image-source manifest or a defensible source-level rights audit. Do not approve full training solely because the repository contains a CC0 file.

#### 11. `simuletic/cctv-smoke-and-fire-emergency-detection-dataset`

- **Official sources:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/simuletic/cctv-smoke-and-fire-emergency-detection-dataset), [Simuletic catalogue](https://www.simuletic.com/datasets), [Simuletic specification article](https://simuletic.com/blog/smoke-fire-dataset), and [Hugging Face dataset card](https://huggingface.co/datasets/Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset/blob/main/README.md)
- **Availability/size:** Kaggle metadata reports approximately 341 MB; publisher pages describe a 220-image open sample.
- **Format/classes:** The Simuletic site/article says YOLO boxes for `fire` and `smoke`. The Kaggle card body instead describes a central JSONL categorical metadata file, creating a format inconsistency that must be checked against the actual file inventory.
- **License conflict:** Kaggle platform metadata says **CC BY-NC-SA 4.0**; Hugging Face metadata says **CC BY-NC 4.0**; prose on the cards/site says **CC BY 4.0**. These are materially different permissions.
- **Provenance:** Publisher claims 100% synthetic, but the project has already documented a provenance conflict in a different Simuletic dataset. This candidate therefore requires visual and metadata verification.
- **Decision:** **HOLD/NO-GO** pending written license clarification, actual file-format verification, and human provenance/annotation QA.

#### 12. `ironwolf437/fire-detection-dataset`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/ironwolf437/fire-detection-dataset)
- **Availability/size:** Available; approximately 873 MB.
- **License evidence:** Kaggle says **Apache 2.0 only**.
- **Classes/content:** Card describes `Fire`, `Light`, `Smoke`, and `Non-Fire` from real surveillance footage. The card does not explicitly establish the annotation file format or ownership/source of the footage.
- **Potential value:** `Light` and `Non-Fire` could supply useful hard negatives if the images and annotations are legally and technically cleared.
- **Decision:** **HOLD** pending source-footage authority, actual label-format inspection, and confirmation that hard-negative images do not contain unlabeled canonical objects.

#### 13. `phylake1337/fire-dataset`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/phylake1337/fire-dataset)
- **Availability/size:** Available; 999 images (755 outdoor fire, 244 non-fire), approximately 409 MB.
- **License evidence:** Kaggle says **CC0**.
- **Format:** Binary classification folders; no bounding boxes.
- **Decision:** **REJECT as a positive detection dataset**. The non-fire folder may be reconsidered as background only after provenance and canonical-label completeness checks.

#### 14. `ritupande/fire-detection-from-cctv`

- **Official source:** [Kaggle dataset card/API](https://www.kaggle.com/api/v1/datasets/view/ritupande/fire-detection-from-cctv)
- **Availability/size:** Available; approximately 184 MB.
- **License evidence:** **Unknown**.
- **Format:** Card only says images from CCTV videos and does not establish bounding-box annotations.
- **Decision:** **REJECT** because both permission and required annotations are absent.

### D. Commercial stock discovery leads

The iStock fire and Adobe Stock fall search results are **REJECTED** as dataset sources. They are commercial discovery pages, not dataset grants, and individual stock licenses do not automatically authorize compiling and redistributing a training dataset or its annotations.

## Required corrections to the existing evaluation

Before the next dataset decision, [dataset_evaluation.md](dataset_evaluation.md) should be revised as follows:

| Current claim | Required correction |
|---|---|
| D-Fire: CC BY 4.0 and `APPROVED` | CC0 collection; underlying-image authority expressly not established; **HOLD/BLOCKED**. |
| Beyza PPE: CC0 and `APPROVED` | Kaggle metadata is MIT while upstream prose says CC0; authority unclear; **HOLD**. |
| Uttej Fall: ODbL and `APPROVED` | Database is open but image contents are © original authors; unspecified sources; **HOLD/BLOCKED**. |
| Simuletic Fall/Fire: CC BY 4.0 and `APPROVED` | Kaggle metadata is CC BY-NC-SA 4.0 and conflicts with prose; **HOLD**. |
| Ironwolf: Apache 2.0 / CC BY 4.0 and `APPROVED` | Kaggle metadata says Apache 2.0 only; provenance unclear; **HOLD**. |
| Shlok PPE: ready supplementary source | License label exists, but source authority and person-label completeness are unverified; **HOLD**. |
| SH17: restricted but generally usable | Preserve research-only restriction and add Pexels/source/likeness audit requirements. |

## Recommended next action

1. Keep the corrected `HOLD`/`REJECT` statuses in [dataset_evaluation.md](dataset_evaluation.md) synchronized with future audit evidence.
2. Apply the recorded education/research-only policy: non-commercial datasets may enter sample audit, but each resulting artifact must retain its license and attribution obligations and must not be represented as commercially reusable.
3. Start with a **metadata and small-sample audit**, not a full download:
   - PPE: Construction-PPE has been sampled and is now HOLD pending scene regrouping/provenance; SH17 is the next research-only sample-audit candidate.
   - Fall: no candidate is cleared; seek a first-party academic dataset with explicit content rights or obtain written permission.
   - Fire/Smoke: request D-Fire's source manifest/rights explanation; in parallel search for a first-party dataset with captured/owned imagery and explicit detection annotations.
4. Do not use pseudo-labeling to cure license uncertainty. Pseudo-labeling may accelerate annotation only after image rights are established and must still receive human QA.
