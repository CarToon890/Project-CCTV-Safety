# Replacement Dataset Search: Fall and Fire/Smoke

## Project CCTV Safety — Stage 1 Spatial Detector

> **Review date:** 25 September 2026  
> **Permitted use:** education and research only  
> **Method:** primary-source desk review of creator repositories, institutional
> dataset records, papers, and license records. No dataset was downloaded.  
> **Scope:** replacement candidates for `fall`, `fire`, and `smoke`. A **GO**
> below authorizes only a small sample/inventory audit, not training.

## Executive result

Two candidates are sufficiently documented to enter a sample audit:

1. **Fall Detection Dataset (State-to-Fall + ADL)** — first-party, creator-
   published videos under CC BY-NC 4.0. It is small (54 clips) and only 8 clips
   have CVAT annotations, so it is an annotation pilot rather than a complete
   training source.
2. **Boreal Forest Fire, Subset A** — first-party UAV footage of prescribed
   burns, CC BY 4.0, with 4,954 images and YOLO smoke boxes. It is the strongest
   rights/provenance candidate found, but covers **smoke only** and its aerial
   domain does not match fixed indoor CCTV.

No fire-box dataset found in this pass satisfies all three requirements at
once: explicit dataset license, demonstrated authority over source images, and
ready object-detection labels. `fire` therefore remains **BLOCKED**.

## Decision table

| Priority | Candidate | Target | Decision | Immediate reason |
|---:|---|---|---|---|
| 1 | Fall Detection Dataset (State-to-Fall + ADL) | fall | **GO — sample audit only** | First-party creator statement and CC BY-NC 4.0; partial CVAT boxes require verification and substantial relabeling. |
| 2 | Boreal Forest Fire — Subset A | smoke | **GO — sample audit only** | First-party prescribed-burn capture, CC BY 4.0, human-reviewed YOLO smoke boxes; aerial-only domain. |
| 3 | TsetFall | fall | **HOLD** | Useful human bbox CSV and sequence IDs, but GPL-3.0 is a software license and its scope over media/person likenesses is not explained. |
| 4 | GMDCSA-24 | fall | **HOLD** | First-party videos are downloadable, but Zenodo exposes no operative license and no spatial boxes are documented. |
| 5 | EDF/OCCU | fall | **HOLD** | Strong controlled collection and grouping, but no operative license or Stage 1 bounding boxes; 26.9 GB depth-domain archive. |
| 6 | CQU Annotated Fire-Smoke | fire, smoke | **HOLD** | CC BY 4.0 and YOLO boxes are explicit, but the institutional record calls the depositor an aggregator and gives no image-source/authority manifest. |
| 7 | Indoor Fire Smoke Dataset | fire, smoke | **HOLD** | 5,000 boxed images are claimed, but the Zenodo record has a blank license field and no capture/source protocol. |
| 8 | FIRESENSE | fire, smoke | **HOLD** | Credible EU-project videos and scene grouping, but the Zenodo license field is blank and no bounding boxes are provided. |

## Fall candidates

### 1. Fall Detection Dataset (State-to-Fall + ADL)

- **Primary source:** [creator repository](https://github.com/samruddhi-2308/FallDetectionDataset)
- **License/authority:** the four named creators state that they created the
  dataset and release it under **CC BY-NC 4.0**. This matches the project's
  education/research-only policy. Faces and full bodies are visible, so the
  repository's privacy notes and attribution requirements remain binding.
- **Availability/scale:** 54 downloadable MP4 clips: 16 sitting-to-fall, 10
  sleeping-to-fall, 22 standing-to-fall, and 6 ADL/no-fall. Total byte size is
  not stated by the publisher and must be measured during inventory.
- **Annotations:** clip labels for all videos; CVAT XML for only 8 clips (6
  standing-to-fall and 2 sleeping-to-fall). The creator warns that XML-to-video
  mapping was automatically inferred from exact frame counts and needs a manual
  spot check.
- **Stage 1 fit:** the existing boxes may provide a small spatial pilot, but the
  remaining clips require human boxes for both `person` and `fall`. The meaning
  of `fall` must be defined at frame level: label a person as `fall` only during
  the project-approved fallen/falling state, not every frame of a positive clip.
- **Grouping:** split by participant and recording session, never by extracted
  frame. The public metadata does not guarantee complete participant/session
  IDs, so these must be derived and documented during sample audit.
- **Decision:** **GO — sample audit only.** Inspect repository inventory and all
  8 annotated clips first; do not download/extract all frames or train until
  consent/privacy evidence, exact box semantics, actor/session groups, and
  annotation quality pass human review.

### 2. TsetFall

- **Primary source:** [author/lab repository](https://github.com/ppgia-unifor/TsetFall_dataset)
- **License/authority:** repository-wide **GPL-3.0** is shown, and the authors
  describe a dataset they generated. However, GPL is written for software; the
  repository does not clearly state that the license covers videos, annotations,
  participant likenesses, and redistribution of the decoded archive.
- **Availability/scale:** images and videos are distributed through a MEGA link;
  a Google Form supplies the decoding key. The publisher lists 36 named
  sequences covering falls and difficult negatives in light/dark conditions.
- **Annotations:** `ground_truth.csv` contains human annotations with class
  (`Not Fallen`, `Fallen`, `Falling`, `Confounding`) and `xmin,ymin,xmax,ymax`;
  an extended file contains AI-assisted annotations.
- **Grouping:** filenames carry camera, sequence, and frame identifiers, making
  sequence-level grouping feasible. Actor/session metadata still needs review.
- **Decision:** **HOLD.** Ask the authors to confirm in writing the media/data
  license, participant consent/redistribution basis, archive size, and whether
  the human-only ground truth covers every sequence.

### 3. GMDCSA-24

- **Primary sources:** [Zenodo record](https://zenodo.org/records/11216408) and
  [author repository](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)
- **License/authority:** the author says videos were captured for this dataset,
  which supports source authority, but the Zenodo **License field is blank**.
- **Availability/scale:** 984.7 MB; four actors in three home environments with
  fall and ADL clips.
- **Annotations/grouping:** clip-level action data; no Stage 1 person/fall boxes
  are documented. Actor and home identities support group-level splitting.
- **Decision:** **HOLD** pending an explicit dataset license and confirmation of
  consent/redistribution; even if cleared, it requires a human bbox project.

### 4. EDF and OCCU

- **Primary source:** [University of Texas at Arlington deposit on Zenodo](https://zenodo.org/records/15494102)
- **License/authority:** five subjects were recorded by the research team with
  Kinect cameras, but the Zenodo **License field is blank**.
- **Availability/scale:** 26.9 GB total. EDF has about 50,378 frames and 80
  synchronized-view fall recordings; OCCU about 49,321 frames and 60 falls,
  plus fall-like negatives.
- **Annotations/grouping:** no Stage 1 bbox labels are stated. Subject, event,
  and viewpoint grouping are available, but simultaneous views of one EDF event
  must remain in the same split.
- **Decision:** **HOLD** pending written license/consent terms. The depth/Kinect
  and staged-home domain is also a secondary fit for ordinary RGB CCTV.

## Fire and smoke candidates

### 5. Boreal Forest Fire — Subset A

- **Primary sources:** [institutional dataset record](https://research.aalto.fi/en/datasets/boreal-forest-fire-uav-collected-wildfire-detection-and-smoke-seg/),
  [peer-reviewed data descriptor](https://www.nature.com/articles/s41597-025-05634-0),
  and [Fairdata DOI](https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9).
- **License/authority:** **CC BY 4.0**. The creators captured 4K drone video at
  four prescribed-burning events in Finland. They removed frames containing
  identifying people, plates, or residences before release. This is materially
  stronger source authority than an Internet-image aggregation.
- **Availability/scale:** Subset A contains 4,954 images, of which 256 have
  intentionally empty annotations. Images are grouped by four capture locations:
  Evo (931), Ruokolahti (1,767), Karkkila (1,313), and Heinola (943).
- **Annotations:** human-drawn, image-by-image reviewed YOLO boxes for one class,
  `smoke`. Subset B contains 30-second 4K clips with binary labels; Subset C has
  smoke segmentation masks.
- **Stage 1 fit:** technically ready for remapping to canonical class 5
  (`smoke`). It supplies no canonical `fire` boxes and is aerial/outdoor rather
  than fixed CCTV, so it must be supplementary—not the sole smoke source.
- **Grouping:** recover source-video/event IDs from filenames/metadata and split
  by prescribed-burn event or original video. The four locations alone are too
  few to allocate blindly to 70/20/10; audit the event count before selecting a
  split strategy.
- **Decision:** **GO — sample audit only.** Start with metadata plus 50–100 images
  spanning all four locations, including negatives and dense/thin smoke. Check
  sequence duplicates, bbox consistency, fire/person missing labels, and domain
  mismatch before approving a full download.

### 6. CQU Annotated Fire-Smoke Image Dataset for YOLO

- **Primary source:** [Central Queensland University institutional record](https://researchdata.edu.au/annotated-fire-smoke-using-yolo/3671995),
  DOI `10.25946/28747046.v1`.
- **License:** **CC BY 4.0** is explicitly recorded.
- **Availability/scale:** 11,027 images in YOLO format with fire and smoke boxes;
  published 80/10/10 splits and reported instance counts.
- **Authority/provenance gap:** the record labels Shouthiri Partheepan as
  **“Aggregated by”** and only says the images represent diverse real-world
  scenarios. It does not state who captured the images, list upstream sources,
  or establish the depositor's authority to license every image.
- **Grouping:** no original video/scene/camera group identifiers are documented;
  the published random-looking split must not be trusted until near-duplicate
  and sequence leakage are audited.
- **Decision:** **HOLD.** Request an image-source manifest, upstream permissions,
  collection method, duplicate-removal method, and group IDs. An institutional
  DOI plus CC BY label does not by itself resolve source-image authority.

### 7. Indoor Fire Smoke Dataset

- **Primary source:** [creator deposit on Zenodo](https://zenodo.org/records/15826133)
- **Availability/scale:** 200.5 MB, 5,000 images, reported 3,500/750/750 split,
  and fire/smoke bounding boxes.
- **Blockers:** the record's **License field is blank**. It says images are real
  and captured in indoor settings but provides no camera/session list, capture
  protocol, source manifest, annotation format details beyond boxes, or evidence
  of the depositor's authority over all images.
- **Decision:** **HOLD** pending an explicit data license and a first-party
  capture/provenance statement. Do not infer permission from “Open” access.

### 8. FIRESENSE

- **Primary source:** [FIRESENSE project deposit on Zenodo](https://zenodo.org/records/836749)
- **Authority/relevance:** created by CERTH/Bilkent within an EU-funded fire
  detection project; 11 positive and 16 negative flame videos, plus 13 positive
  and 9 negative smoke videos. Video sequences are relevant for surveillance
  testing and allow correct source-video grouping.
- **Availability/size:** two downloadable archives totaling about 820.3 MB.
- **Blockers:** the Zenodo **License field is blank**, and the record does not
  provide spatial bounding boxes. A third-party index calling it “CC BY” is not
  sufficient to replace operative terms from the publisher.
- **Decision:** **HOLD.** Ask the deposit contact to state reuse/derivative-
  annotation permission. If granted, treat it as a small manually annotated
  supplement and hard-negative source, not a ready detector dataset.

## Excluded leads and rights-resolution paths

- **D-Fire:** remains **HOLD**. A mirror or paper describing it as CC BY/CC0 does
  not override the original repository's statement that maintainers do not own
  underlying images. Concrete resolution requires an image-level source manifest
  and compatible upstream rights, or written confirmation from every source.
- **FASDD / New Fire and Smoke:** large and technically useful, but compiled from
  YouTube/public Internet resources. A license on the paper, annotations, or
  Roboflow export does not establish authority over each source video/image;
  therefore it is not promoted here.
- **Fire Recognition Image Dataset (Mendeley):** classification folders derived
  from YouTube, not object boxes; the authors' CC BY 4.0 declaration does not
  document source-video redistribution rights.
- **Simuletic and Uttej:** statuses remain as recorded in the existing primary-
  source audit and were not reopened.

## Recommended next action

1. **Fall first:** perform a repository-inventory and 8-clip annotation audit of
   Fall Detection Dataset (State-to-Fall + ADL). This is small enough to reject
   cheaply if box semantics, privacy evidence, or grouping are inadequate.
2. **Smoke in parallel:** sample 50–100 Boreal Subset A images across all four
   locations and original videos/events. Do not adopt the publisher's grouping
   without leakage checks.
3. **Keep `fire` blocked:** contact the CQU depositor for a source manifest and
   capture/permission evidence. If unavailable, prefer commissioning/recording a
   small controlled first-party fire dataset rather than relabeling uncertain
   Internet imagery.
4. A GO candidate advances to training only after license archive, human bbox
   QA, exhaustive six-class label review, duplicate/near-duplicate audit, and a
   group-level 70/20/10 split (or a documented alternative when groups are too
   few).

