# รายงานการประเมินชุดข้อมูลปฐมภูมิ (Primary Source Dataset Evaluation)
## โครงการ Project CCTV Safety — Dataset Candidate Evaluation

> **สถานะปัจจุบัน:** ตารางและคะแนนบางส่วนด้านล่างถูกจัดทำภายใต้ detector
> schema v1 จำนวน 7 คลาสและเก็บไว้เป็นประวัติการตัดสินใจเท่านั้น สัญญาที่ใช้งาน
> ปัจจุบันคือ [Detector Schema v2 จำนวน 6 Spatial Classes](data_schema_6classes.md);
> `fight` ย้ายไป Stage 2 และยัง `BLOCKED / PENDING DATA APPROVAL`.
>
> **ขอบเขตการใช้งานที่เจ้าของโครงการกำหนด (25 ก.ย. 2026):** เพื่อการศึกษาและ
> วิจัยเท่านั้น ไม่ใช้เชิงพาณิชย์ จึงอนุญาตให้ชุด Non-commercial เข้าสู่รอบ
> sample audit ได้ แต่ยังต้องปฏิบัติตาม attribution/share-alike, provenance,
> likeness/privacy และ Human QA ก่อน Training

> **วันที่ประเมิน:** 24 กันยายน 2026  
> **เป้าหมายเดิมของรายงาน:** ประเมิน Candidate Datasets ทั้งหมด 21 รายการจาก `docs/dataset.md` ตามข้อมูลจากแหล่งปฐมภูมิ โดยผลสำหรับ 6 spatial classes ใช้ประกอบการคัดเลือก Stage 1 ส่วนผล Fight เป็นประวัติและต้องอ่านร่วมกับ [fight_dataset_evaluation.md](fight_dataset_evaluation.md)

---

### นิยาม Historical Schema v1 (ไม่ใช่ Active Contract)
* `0: person` — ลำตัวคนทั้งตัวที่มองเห็น
* `1: helmet` — หมวกนิรภัยที่กำลังสวมใส่
* `2: vest` — เสื้อสะท้อนแสง/เสื้อนิรภัยที่กำลังสวมใส่
* `3: fall` — ลำตัวคนทั้งตัวที่หกล้มหรือนอนผิดปกติบนพื้น (ได้รับทั้งกรอบ person และ fall)
* `4: fire` — ขอบเขตเปลวไฟที่มองเห็น
* `5: smoke` — กลุ่มควันที่มองเห็น
* `6: fight` — Bounding Box กลุ่ม (Group Box) ครอบคลุมกลุ่มคนที่กำลังปะทะ/วิวาทกัน

*(หมายเหตุ: `no_helmet` และ `no_vest` ไม่ใช่คลาสของตัวตรวจจับ แต่คำนวณจากความสัมพันธ์เชิงเรขาคณิตระหว่าง `person` และ `helmet`/`vest` ผ่าน `cctv_safety/ppe.py`)*

---

## ส่วน A: ตารางเปรียบเทียบ Dataset ทั้ง 21 รายการ (Comparison Matrix)

| # | Dataset Candidate | หมวด | License (ปฐมภูมิ) | สิทธิ์ทางกฎหมาย | ชนิดข้อมูล | งานต้นทาง | Annotation Format | Bbox Usability | เข้ากับ 7 คลาส | CCTV Domain | คะแนน (100) | สถานะ (Approval Status) |
|---|---|---|---|---|---|---|---|---|---|---|:---:|:---:|
| 1 | `phylake1337/fire-dataset` | Fire | CC0: Public Domain | ใช้/แจกจ่าย/พาณิชย์ได้ | รูปภาพ (999 รูป) | Classification | ไม่มี (แยกโฟลเดอร์) | ❌ ไม่มี Bbox | ต่ำ (ไม่แยกควัน) | ต่ำ (ภาพธรรมชาติ) | 35 | **Blocked (No Bbox)** |
| 2 | `iStock Factory Fire` | Fire | All Rights Reserved | ❌ ห้ามแจกจ่าย/มีลิขสิทธิ์ | รูปภาพสต็อก | Discovery Lead | ไม่มี | ❌ ไม่มี | N/A | ปานกลาง | 0 | **REJECTED (License)** |
| 3 | **`gaiasd/DFireDataset`** | Fire/Smoke | **CC0 collection; source-image rights disclaimed** | ⚠️ ผู้รวบรวมระบุว่าไม่ได้เป็นเจ้าของภาพต้นฉบับ | รูปภาพ (21,527 รูป) | Object Detection | **YOLO (.txt)** | ✅ ดีเยี่ยม | ✅ ดีเยี่ยม (fire+smoke) | สูง (อุตสาหกรรม/CCTV) | — | **HOLD — source-rights audit required** |
| 4 | `ritupande/fire-detection-from-cctv` | Fire | **Unknown** | ❌ ไม่ระบุสิทธิ์ชัดเจน | รูปภาพ | Classification | ไม่มี | ❌ ไม่มี Bbox | ต่ำ | สูง (CCTV) | 18 | **REJECTED (License Unknown)** |
| 5 | **`simuletic/cctv-smoke-and-fire`** | Fire/Smoke | **Conflicting: CC BY-NC-SA / CC BY-NC / CC BY** | ❌ สิทธิ์ขัดแย้งกันระหว่างแหล่ง | รูปภาพที่อ้างว่าสังเคราะห์ | Object Detection/JSONL (ขัดแย้ง) | ต้องตรวจไฟล์จริง | ดีเชิงแนวคิด | สูง | — | **HOLD — license/format/provenance conflict** |
| 6 | `ironwolf437/fire-detection-dataset` | Fire/Smoke | Apache 2.0 | ⚠️ ไม่พบ source-footage authority | รูปภาพ | ระบุ 4 คลาส; format ต้องตรวจ | ต้องตรวจ | ปานกลาง | สูง | — | **HOLD — provenance/format audit required** |
| 7 | **`uttejkumarkandagatla/fall-detection`** | Fall | **Open Database; contents © original authors** | ❌ ไม่ครอบคลุมสิทธิ์ภาพทั้งหมด | รูปภาพ (485 รูป) | Object Detection | **YOLO (.txt)** | ✅ มี Bbox | ✅ ดี (Fall/Walk/Sit) | ปานกลาง | — | **HOLD — source permissions required** |
| 8 | `Adobe Stock Fall Accident` | Fall | All Rights Reserved | ❌ ห้ามแจกจ่าย/มีลิขสิทธิ์ | ภาพสต็อก | Discovery Lead | ไม่มี | ❌ ไม่มี | N/A | ปานกลาง | 0 | **REJECTED (License)** |
| 9 | `soumicksarker/multiple-cameras-fall` | Fall | **Unknown / Other** | ❌ ไม่ระบุสิทธิ์ชัดเจน | วิดีโอ (8 มุมกล้อง) | Video Action | ไม่มี Bbox | ❌ ไม่มี Bbox | ปานกลาง | สูง (Multi-camera) | 15 | **REJECTED (License Unknown)** |
| 10 | **`simuletic/cctv-incident-fall`** | Fall | **Conflicting: CC BY-NC-SA / CC BY** | ❌ License conflict | รูปภาพที่อ้างว่าสังเคราะห์ (113-image sample) | YOLO Pose | Bbox + keypoints | ดีเชิงแนวคิด | สูง | — | **HOLD — written clarification and QA required** |
| 11 | `payutch/fall-video-dataset` | Fall | CC0: Public Domain | ใช้/ดัดแปลง/พาณิชย์ได้ | วิดีโอ + CSV Keypoints | Pose Estimation | CSV (2D Keypoints) | ❌ ไม่มี Bbox | ปานกลาง | ปานกลาง (Indoor) | 40 | **Blocked (Needs Frame Bbox)** |
| 12 | **`shlokraval/ppe-dataset`** | PPE | **Apache 2.0 metadata** | ⚠️ ไม่พบ source-image provenance/authority | รูปภาพ | Object Detection | COCO (ตาม card) | ✅ มี Bbox | ต้องตรวจ person completeness | ปานกลาง | — | **HOLD — provenance and sample audit required** |
| 13 | `ndomalau/ppe-dataset` | PPE | **Unknown / Ambiguous** | ❌ สิทธิ์ไม่ชัดเจน | รูปภาพ (4,000+) | Object Detection | YOLO (.txt) | ✅ มี Bbox | ปานกลาง | ปานกลาง | 55 | **Blocked (License Audit Req.)** |
| 14 | **`beyzakucuk/ppe-detection-v1`** | PPE/Person | **MIT metadata; upstream prose says CC0** | ❌ License authority/provenance ไม่ชัด | รูปภาพ (Roboflow copy) | Object Detection | **YOLO (.txt)** | ✅ มี Bbox | Person/Helmet/Vest | สูง | — | **HOLD — upstream version and rights required** |
| 15 | `niravnaik/safety-helmet-vest` | PPE | Apache 2.0 | ใช้/ดัดแปลง/แจกจ่ายได้ | รูปภาพ (10,500 รูป) | Object Detection | YOLOv7 (.txt) | ✅ มี Bbox | ปานกลาง (ขาด Person) | ปานกลาง (งานช่าง) | **75** | **Hold (Missing-label penalty)** |
| 16 | `mugheesahmad/sh17-dataset` | PPE | **CC BY-NC-SA 4.0** | ⚠️ เฉพาะการศึกษา/ห้ามพาณิชย์ | รูปภาพ (8,099 รูป) | Object Detection | YOLO & VOC | ✅ ดีเยี่ยม | ✅ ดีมาก (17 คลาส) | ปานกลาง (ภาพถ่าย Pexels) | **76** | **Restricted (Non-Commercial Only)** |
| 17 | `anbumalar1991/fight-dataset` | Fight | **Other / Unknown** | ❌ ไม่ระบุสิทธิ์ชัดเจน | รูปภาพ (224x224) | Classification | CSV | ❌ ไม่มี Bbox | ต่ำ (Binary) | ต่ำ (ตัดจาก HMDB51) | 20 | **REJECTED (No Bbox & License)** |
| 18 | `magicearth25/video-violence` | Fight | **Unknown** | ❌ ไม่ระบุสิทธิ์ชัดเจน | วิดีโอ | Video Classification | ไม่มี | ❌ ไม่มี Bbox | ต่ำ | ปานกลาง | 12 | **REJECTED (License Unknown)** |
| 19 | `naveenk903/movies-fight` | Fight | **Inaccessible (404)** | ❌ ลิงก์เสีย/เข้าถึงไม่ได้ | N/A | N/A | N/A | ❌ N/A | N/A | N/A | 0 | **REJECTED (Inaccessible)** |
| 20 | `mohamedmustafa/real-life-violence` | Fight | License not confirmed; citation terms | ⚠️ รอตรวจสอบสิทธิ์ | วิดีโอ (2,000 คลิป) | Video Action | ไม่มี Bbox | ❌ ไม่มี Bbox | ปานกลาง | สูง (CCTV Real-life) | 28 | **BLOCKED pending license clarification**|
| 21 | `yash07yadav/project-data` | Fight | MIT (Compiler only; source rights unverified) | ⚠️ รอตรวจสอบสิทธิ์รายแหล่ง | วิดีโอ (MP4) | Video Classification | ไม่มี | ❌ ไม่มี Bbox | ปานกลาง (Fight/NonFight) | ปานกลาง | 27 | **HOLD — Provenance and source-license audit required**|

---

## ส่วน B: รายงานรายละเอียดแยกแต่ละ Dataset (Detailed Evaluation)

### หมวด 1: FIRE & SMOKE

#### 1. `phylake1337/fire-dataset`
* **ข้อมูลทั่วไป:** Fire Dataset โดย phylake1337 บน Kaggle (จัดทำขึ้นสำหรับงาน NASA Space Apps Challenge 2018)
* **Official URL:** `https://www.kaggle.com/datasets/phylake1337/fire-dataset`
* **License:** **CC0: Public Domain** (ยืนยันจากหน้า Kaggle Dataset Card)
* **สิทธิ์การใช้งาน:** ดาวน์โหลดได้ ใช้เพื่อการศึกษา ดัดแปลง ฝึกโมเดล แจกจ่าย และใช้เชิงพาณิชย์ได้สมบูรณ์
* **ประเภทข้อมูล & ขนาด:** รูปภาพนิ่ง 999 รูป (โฟลเดอร์ `fire_images`: 755 รูป, `non_fire_images`: 244 รูป) ขนาดดาวน์โหลด ~387 MB
* **ประเภทงาน & Format:** Image Classification แบบ Binary ไม่ใช่ Object Detection ไม่มีไฟล์ Annotation พิกัด Bounding Box
* **Mapping เข้า 7 คลาส:** ไม่สามารถแมปได้ตรง ต้องคัดแยกรูปไฟไหม้มาตีกรอบ Bounding Box ใหม่
* **ข้อจำกัดเชิงเทคนิค:** ภาพส่วนใหญ่เป็นไฟป่า ภาพถ่ายมุมมองบุคคล (Eye-level) หรือถ่ายจากโดรน ไม่ใช่กล้อง CCTV ติดเพดาน/เสา
* **Missing-label Penalty:** ไม่มี Bounding Box หากนำไปรวมจะใช้ได้เพียงชุด `non_fire_images` เป็น Background/Negative images
* **คะแนน Rubric (35/100):** License (15) | Bbox (0) | 7-Class (5) | Completeness (3) | CCTV (4) | Diversity (5) | Negative (4) | Leakage (3) | Cost (1)
* **สรุปสถานะ:** **BLOCKED สำหรับ Object Detection Baseline** (แต่อนุญาตให้นำเฉพาะ `non_fire_images` มาทำ Negative Samples)

#### 2. `iStock Factory Fire Search Page`
* **ข้อมูลทั่วไป:** หน้าค้นหาภาพสต็อกบน iStockPhoto
* **Official URL:** `https://www.istockphoto.com/...`
* **License:** **Royalty-Bearing / All Rights Reserved** (มีลิขสิทธิ์เชิงพาณิชย์ ห้ามดาวน์โหลดมาเทรนโมเดลหรือแจกจ่ายโดยไม่ซื้อสิทธิ์)
* **คะแนน Rubric (0/100):** License (0) | อื่นๆ (0)
* **สรุปสถานะ:** **REJECTED ทันที** ตามข้อกำหนดใน [docs/workflow.md](workflow.md)

#### 3. `gaiasd/DFireDataset` (D-Fire Dataset) ⭐ [PRIMARY RECOMMENDATION]
* **ข้อมูลทั่วไป:** D-Fire: An Image Dataset for Fire and Smoke Detection โดย Gaia Solutions on Demand (gaiasd)
* **Official URL:** `https://github.com/gaiasd/DFireDataset`
* **License:** **CC0 1.0 สำหรับตัว collection** ไม่ใช่ CC BY 4.0
* **ข้อจำกัดสิทธิ์:** LICENSE ระบุว่าผู้ดูแลไม่ได้เป็นเจ้าของลิขสิทธิ์ภาพต้นฉบับและไม่ได้รับรองสิทธิ์ของบุคคลที่สาม จึงยังไม่อนุมัติให้เทรนจนกว่าจะตรวจ source-level rights
* **ประเภทข้อมูล & ขนาด:** รูปภาพนิ่งจำนวน **21,527 ภาพ** (Train ~14,123, Val ~3,099, Test ~4,306)
* **ประเภทงาน & Format:** **Object Detection** ในฟอร์แมต **YOLO (.txt)** พิกัดแบบ Normalized
* **Class ต้นฉบับ & การ Mapping:**
  * Source Class 0: `smoke` $\rightarrow$ Canonical Class `5: smoke`
  * Source Class 1: `fire` $\rightarrow$ Canonical Class `4: fire`
* **การแยกไฟและควัน:** **แยกเดี่ยวอย่างสมบูรณ์** ตรงตามข้อกำหนดใน [Detector Schema v2](data_schema_6classes.md)
* **Missing-label Penalty:** ในภาพไฟไหม้บางภาพที่มีคนหรือรถยนต์อยู่เบื้องหลัง วัตถุเหล่านั้นไม่ได้ถูก Label หากนำไปรวมกับชุด PPE/Person ต้องตรวจทานเพื่อหลีกเลี่ยง Negative Penalty ต่อคลาส `person`
* **ความใกล้เคียง CCTV:** มีมุมกล้องหลากหลาย รวมถึงภาพจากกล้องวงจรปิดในโกดัง อาคาร ทางเดิน และกลางแจ้ง
* **คะแนน Rubric (89/100):** License (15) | Bbox (15) | 7-Class (14) | Completeness (12) | CCTV (11) | Diversity (9) | Negative (4) | Leakage (4) | Cost (5)
* **สรุปสถานะ:** **HOLD/BLOCKED — เทคนิคเหมาะสม แต่ต้องผ่าน source-rights audit ก่อน**

#### 4. `ritupande/fire-detection-from-cctv`
* **ข้อมูลทั่วไป:** Fire Detection from CCTV โดย Ritu Pande บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/ritupande/fire-detection-from-cctv`
* **License:** **Unknown / Unspecified** บน Kaggle
* **ประเภทงาน & Format:** Image Classification (โฟลเดอร์ภาพเปลวไฟ vs ภาพปกติ) ไม่มี Bounding Box
* **คะแนน Rubric (18/100):** License (0 - Blocked) | Bbox (0) | 7-Class (4) | Completeness (2) | CCTV (8) | Diversity (2) | Negative (2) | Leakage (0) | Cost (0)
* **สรุปสถานะ:** **REJECTED (License เป็น Unknown และไม่มี Bounding Box)**

#### 5. `simuletic/cctv-smoke-and-fire-emergency-detection-dataset` ⭐ [SUPPLEMENTARY]
* **ข้อมูลทั่วไป:** CCTV Smoke & Fire Emergency Detection Dataset โดย Simuletic
* **Official URL:** `https://www.kaggle.com/datasets/simuletic/cctv-smoke-and-fire-emergency-detection-dataset`
* **License:** **ขัดแย้งกัน**: Kaggle metadata ระบุ CC BY-NC-SA 4.0, Hugging Face metadata ระบุ CC BY-NC 4.0 และข้อความผู้เผยแพร่บางแห่งระบุ CC BY 4.0
* **ประเภทข้อมูล:** รูปภาพสังเคราะห์ (Synthetic CCTV) จำลองเหตุการณ์จุดไฟ ควัน และไฟไหม้ขนาดเล็ก
* **ประเภทงาน & Format:** **Object Detection** ในฟอร์แมต **YOLO (.txt)**
* **Mapping เข้า 7 คลาส:** ไฟขนาดเล็ก $\rightarrow$ `fire` (4), ควัน $\rightarrow$ `smoke` (5)
* **ความใกล้เคียง CCTV:** ดีเยี่ยม ออกแบบมาเพื่อมุมมองกล้องวงจรปิดมุมสูง (High-angle Surveillance) โดยเฉพาะ
* **คะแนน Rubric (86/100):** License (15) | Bbox (14) | 7-Class (13) | Completeness (12) | CCTV (14) | Diversity (7) | Negative (3) | Leakage (4) | Cost (4)
* **สรุปสถานะ:** **HOLD — ต้องได้คำชี้แจง license, ตรวจ format จริง และทำ provenance/Human QA**

#### 6. `ironwolf437/fire-detection-dataset` ⭐ [HARD-NEGATIVE SOURCE]
* **ข้อมูลทั่วไป:** Fire Detection Dataset โดย ironwolf437 บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/ironwolf437/fire-detection-dataset`
* **License:** Kaggle metadata ระบุ **Apache 2.0 เท่านั้น**; ยังไม่พบหลักฐานสิทธิ์เหนือ footage ต้นฉบับ
* **ประเภทงาน & Format:** Object Detection (YOLO format)
* **Class ต้นทาง:** `Fire`, `Smoke`, `Light`, `Non-Fire`
* **จุดเด่นพิเศษ:** มีคลาส `Light` (แสงไฟนีออน ไฟหน้ารถ แสงสะท้อน) ซึ่งทำหน้าที่เป็น **Hard Negative ชั้นยอด** เพื่อป้องกันโมเดลสับสนระหว่างเปลวไฟกับหลอดไฟ
* **คะแนน Rubric (78/100):** License (13) | Bbox (12) | 7-Class (12) | Completeness (10) | CCTV (12) | Diversity (7) | Negative (5) | Leakage (3) | Cost (4)
* **สรุปสถานะ:** **HOLD — มีศักยภาพเป็น hard negative แต่ต้องตรวจ provenance และ annotation format ก่อน**

---

### หมวด 2: FALL (คนหกล้ม)

#### 7. `uttejkumarkandagatla/fall-detection-dataset` ⭐ [PRIMARY FALL BASELINE]
* **ข้อมูลทั่วไป:** Fall Detection Dataset โดย Uttej Kumar Kandagatla บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset`
* **License:** Kaggle ระบุ **Database: Open Database, Contents: © Original Authors** จึงไม่ใช่ blanket permission สำหรับภาพทั้งหมด
* **สิทธิ์การใช้งาน:** ผู้เผยแพร่ระบุว่ารวบรวมภาพจากหลายแหล่งโดยไม่แจกแจง จึงต้องตรวจสิทธิ์ระดับแหล่งก่อน
* **ประเภทข้อมูล:** รูปภาพนิ่ง 485 ภาพ (Train: 374, Val: 111) ทำ Annotation ผ่าน makesense.ai
* **ประเภทงาน & Format:** **Object Detection** แบบ **YOLO Bounding Box (.txt)**
* **Class ต้นฉบับ & การ Mapping:**
  * `Fall Detected` $\rightarrow$ Map เป็น **`fall` (ID 3)** และต้องสร้างคู่กรอบ **`person` (ID 0)** ซ้อนทับตามกฎข้อ 25 ของ Schema
  * `Walking`, `Sitting` $\rightarrow$ Map เป็น **`person` (ID 0)** และใช้เป็น **Hard Negative สำหรับ `fall`**
* **คะแนน Rubric (78/100):** License (13) | Bbox (14) | 7-Class (13) | Completeness (10) | CCTV (10) | Diversity (6) | Negative (5) | Leakage (3) | Cost (4)
* **สรุปสถานะ:** **HOLD/BLOCKED — รูปแบบเหมาะ แต่ยังไม่มี source-level permission**

#### 8. `Adobe Stock Fall Down Accident Search Page`
* **ข้อมูลทั่วไป:** หน้าเว็บค้นหาภาพสต็อกของ Adobe Stock
* **License:** **Royalty-Bearing / All Rights Reserved** (มีลิขสิทธิ์)
* **คะแนน Rubric (0/100)**
* **สรุปสถานะ:** **REJECTED ทันที**

#### 9. `soumicksarker/multiple-cameras-fall-dataset`
* **ข้อมูลทั่วไป:** Multiple Cameras Fall Dataset บน Kaggle (ตัดทอนมาจากงานวิจัยของ Université de Montréal)
* **Official URL:** `https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset`
* **License:** **Unknown / Other** บน Kaggle
* **ประเภทข้อมูล & Format:** วิดีโอคลิปดิบ (8 มุมกล้อง) ไม่มีไฟล์ Bounding Box
* **คะแนน Rubric (15/100):** License (0 - Blocked)
* **สรุปสถานะ:** **REJECTED (License ไม่ชัดเจน และไม่มี Bounding Box)**

#### 10. `simuletic/cctv-incident-dataset-fall-and-lying-down-detection` ⭐ [SUPPLEMENTARY FALL]
* **ข้อมูลทั่วไป:** CCTV Incident Dataset - Fall & Lying Down โดย Simuletic
* **Official URL:** `https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection`
* **License:** Kaggle metadata ระบุ **CC BY-NC-SA 4.0** แต่ข้อความ card ระบุ CC BY 4.0 จึงขัดแย้งกัน
* **ประเภทข้อมูล:** รูปภาพสังเคราะห์ CCTV มุมสูง คนล้ม/นอนหมดสติกว่า 100+ ซีน
* **ประเภทงาน & Format:** Object Detection / Pose (YOLO format)
* **Mapping เข้า 7 คลาส:** กรอบคนล้ม $\rightarrow$ `fall` (3) และ `person` (0)
* **คะแนน Rubric (83/100):** License (15) | Bbox (14) | 7-Class (12) | Completeness (11) | CCTV (14) | Diversity (7) | Negative (3) | Leakage (4) | Cost (3)
* **สรุปสถานะ:** **HOLD — ต้องได้คำชี้แจงเป็นลายลักษณ์อักษรและผ่าน provenance/Human QA**

#### 11. `payutch/fall-video-dataset`
* **ข้อมูลทั่วไป:** Fall Video Dataset โดย Payut Ch. บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/payutch/fall-video-dataset`
* **License:** **CC0: Public Domain**
* **ประเภทข้อมูล & Format:** คลิปวิดีโอดิบ + ไฟล์ CSV บันทึก 2D Keypoint Pose (ไม่มี Bounding Box YOLO)
* **งานที่ต้องทำเพิ่ม:** ต้อง Extract Frames และวาดกรอบ Bounding Box ใหม่ทั้งหมด
* **คะแนน Rubric (40/100):** License (15) | Bbox (2) | 7-Class (5) | Completeness (3) | CCTV (11) | Diversity (2) | Negative (1) | Leakage (0) | Cost (1)
* **สรุปสถานะ:** **BLOCKED สำหรับรอบนี้** (เก็บเป็น Data Lake สำรองกรณีต้องการขยายชุดวิดีโอ)

---

### หมวด 3: PPE (อุปกรณ์ความปลอดภัย & บุคคล)

#### 12. `shlokraval/ppe-dataset`
* **ข้อมูลทั่วไป:** PPE Dataset (YOLOv8 & COCO) โดย Shlok Raval บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/shlokraval/ppe-dataset`
* **License:** Kaggle metadata ระบุ **Apache 2.0** แต่ card ไม่แจกแจงที่มาภาพหรือ authority ของผู้เผยแพร่
* **ประเภทงาน & Format:** Card ยืนยัน COCO JSON; ต้องตรวจ file inventory ก่อนสรุปว่ามี YOLOv8 พร้อมใช้
* **Class ต้นฉบับ & การ Mapping:**
  * `helmet` $\rightarrow$ `helmet` (1)
  * `safety vest` $\rightarrow$ `vest` (2)
  * `gloves`, `goggles`, `face shield` $\rightarrow$ `null` (Drop ทิ้ง)
* **คะแนน Rubric (82/100):** License (15) | Bbox (15) | 7-Class (13) | Completeness (12) | CCTV (9) | Diversity (8) | Negative (3) | Leakage (3) | Cost (4)
* **สรุปสถานะ:** **HOLD — ต้องตรวจ provenance, category list และ person-label completeness**

#### 13. `ndomalau/personal-protective-equipment-ppe-dataset`
* **ข้อมูลทั่วไป:** PPE Dataset โดย Fransiscus Rolanda Malau บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset`
* **License:** **Ambiguous / Unknown** (หน้า Kaggle Card ระบุไม่ชัดเจน มีสำเนาหลายเวอร์ชัน)
* **คะแนน Rubric (55/100):** License (4 - Blocked)
* **สรุปสถานะ:** **BLOCKED จนกว่าจะได้รับการตรวจสอบสิทธิ์รายลักษณ์อักษร**

#### 14. `beyzakucuk/ppe-detection-v1` ⭐ [PRIMARY PERSON & PPE]
* **ข้อมูลทั่วไป:** PPE Detection V1 โดย Beyza Kucuk บน Kaggle (จัดทำผ่าน Roboflow)
* **Official URL:** `https://www.kaggle.com/datasets/beyzakucuk/ppe-detection-v1`
* **License:** Kaggle metadata ระบุ **MIT** แต่ข้อความอ้าง upstream Roboflow เป็น CC0 จึงต้องยืนยัน exact upstream version และสิทธิ์ภาพ
* **สิทธิ์การใช้งาน:** ยังไม่อนุมัติ เพราะ license ของ copy ไม่พิสูจน์ authority เหนือภาพต้นฉบับ
* **ประเภทข้อมูล:** รูปภาพนิ่ง แบ่ง Train/Valid/Test พร้อมไฟล์ Annotation แบบ **YOLOv8 (.txt)**
* **Class ต้นฉบับ & การ Mapping:**
  * `Person` $\rightarrow$ **`person` (ID 0)**
  * `Helmet` $\rightarrow$ **`helmet` (ID 1)**
  * `Vest` $\rightarrow$ **`vest` (ID 2)**
  * `No-Helmet`, `No-Goggles`, `Goggles` $\rightarrow$ **`null` (Drop ทิ้ง)**
* **จุดเด่นพิเศษ:** มีคลาส `Person`, `Helmet`, `Vest` อยู่ในภาพเดียวกันอย่างสมบูรณ์แบบ ทำให้ตรงกับโมดูล [cctv_safety/ppe.py](../cctv_safety/ppe.py) โดยไม่เกิด Missing-label penalty ต่อคลาส person!
* **คะแนน Rubric (89/100):** License (15) | Bbox (15) | 7-Class (15) | Completeness (13) | CCTV (10) | Diversity (8) | Negative (4) | Leakage (4) | Cost (5)
* **สรุปสถานะ:** **HOLD — ต้องตรวจ upstream Roboflow version, provenance และ license authority**

#### 15. `niravnaik/safety-helmet-and-reflective-jacket`
* **ข้อมูลทั่วไป:** Safety Helmet and Reflective Jacket โดย Nirav B. Naik บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket`
* **License:** **Apache 2.0**
* **ประเภทข้อมูล & Format:** รูปภาพ 10,500 ภาพ ฟอร์แมต YOLOv7
* **ข้อจำกัดสำคัญ:** มีเฉพาะ Bounding Box ของ `helmet` และ `reflective jacket` แต่ **ไม่มี Bounding Box ของตัวบุคคล (`person`)**
* **Missing-label Penalty:** สูงมาก หากนำไปรวมกับชุดข้อมูลอื่น โมเดลจะเรียนรู้ว่าตัวคนงานคือ Background!
* **คะแนน Rubric (75/100):** License (15) | Bbox (14) | 7-Class (11) | Completeness (9) | CCTV (10) | Diversity (8) | Negative (2) | Leakage (3) | Cost (3)
* **สรุปสถานะ:** **HOLD — พักไว้ก่อน** จะนำเข้าได้ต่อเมื่อรันสคริปต์ตรวจจับและเติมกรอบ `person` แบบ Pseudo-labeling เท่านั้น

#### 16. `mugheesahmad/sh17-dataset-for-ppe-detection`
* **ข้อมูลทั่วไป:** SH17 Dataset โดย Mughees Ahmad บน Kaggle / Zenodo
* **Official URL:** `https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection`
* **License:** **CC BY-NC-SA 4.0 (Non-Commercial)** (ห้ามใช้ในเชิงพาณิชย์โดยเด็ดขาด)
* **ประเภทข้อมูล:** 8,099 ภาพ มี 17 คลาส (Person, Helmet, Vest, ฯลฯ) ภาพความละเอียดสูงมากจาก Pexels
* **ข้อจำกัด:** ภาพเป็นสไตล์ช่างภาพถ่าย Portrait/Close-up ไม่ใช่มุมมองกล้องวงจรปิด CCTV และติดเงื่อนไข Non-commercial
* **คะแนน Rubric (76/100):** License (8 - Restricted) | Bbox (14) | 7-Class (13) | Completeness (14) | CCTV (8) | Diversity (9) | Negative (3) | Leakage (3) | Cost (4)
* **สรุปสถานะ:** **RESTRICTED — ใช้เพื่อการวิจัย/ทดลองในห้องแล็บเท่านั้น ห้ามนำเข้าโมเดลเชิงพาณิชย์**

---

### หมวด 4: FIGHT (การทะเลาะวิวาท)

#### 17. `anbumalar1991/fight-dataset`
* **ข้อมูลทั่วไป:** Fight dataset โดย anbumalar1991 บน Kaggle (ดึงเฟรมจากชุดวิดีโอ HMDB51)
* **Official URL:** `https://www.kaggle.com/datasets/anbumalar1991/fight-dataset`
* **License:** **Other / Unspecified** บน Kaggle
* **ประเภทข้อมูล & Format:** ภาพขนาดเล็ก 224x224 พิกเซล จับคู่ป้ายกำกับผ่าน `dataset.csv` (ไม่มี Bounding Box)
* **คะแนน Rubric (20/100):** License (2) | Bbox (0) | 7-Class (3) | Completeness (2) | CCTV (5) | Diversity (4) | Negative (2) | Leakage (1) | Cost (1)
* **สรุปสถานะ:** **REJECTED (ไม่มี Bounding Box และ License ไม่ชัดเจน)**

#### 18. `magicearth25/video-violence-detection-dataset`
* **ข้อมูลทั่วไป:** Video violence detection dataset บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset`
* **License:** **Unknown**
* **ประเภทงาน:** Video Action Recognition (วิดีโอคลิป ไม่มี Bounding Box)
* **คะแนน Rubric (12/100)**
* **สรุปสถานะ:** **REJECTED (License Unknown และเป็นวิดีโอระดับ Classification)**

#### 19. `naveenk903/movies-fight-detection-dataset`
* **Official URL:** `https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset`
* **สถานะการเข้าถึง:** **HTTP 404 / Inaccessible / Deleted**
* **คะแนน Rubric (0/100)**
* **สรุปสถานะ:** **REJECTED (ลิงก์เข้าถึงไม่ได้)**

#### 20. `mohamedmustafa/real-life-violence-situations-dataset` (RWF-2000)
* **ข้อมูลทั่วไป:** Real Life Violence Situations Dataset โดย Mohamed Mustafa (ชุด RWF-2000)
* **Official URL:** `https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset`
* **License:** **License not confirmed; research access/citation terms observed** (ไม่พบสัญญาอนุญาต Open Source มาตรฐาน ปรากฏเฉพาะข้อกำหนดการอ้างอิงงานวิจัยของผู้จัดทำ)
* **ประเภทข้อมูล:** 2,000 คลิปวิดีโอ CCTV จริง (1,000 Violence, 1,000 Non-Violence)
* **ข้อจำกัด:** เป็นคลิปวิดีโอ ไม่มี Bounding Box และไม่สามารถยืนยันสิทธิ์ในการดัดแปลงหรือแจกจ่าย Model weights ได้
* **คะแนน Rubric (28/100):** License (4) | Bbox (0) | 7-Class (3) | Completeness (2) | CCTV (12) | Diversity (5) | Negative (1) | Leakage (0) | Cost (1)
* **สรุปสถานะ:** **BLOCKED pending license clarification**

#### 21. `yash07yadav/project-data`
* **ข้อมูลทั่วไป:** Violence Detection - Combined โดย Yash07Yadav บน Kaggle
* **Official URL:** `https://www.kaggle.com/datasets/yash07yadav/project-data`
* **License:** **MIT** บนหน้า Kaggle Card (ครอบคลุมเฉพาะงานรวบรวมและโค้ดของผู้จัดทำ แต่ไม่ได้เป็นการรับรองสิทธิ์ในเนื้อหาวิดีโอต้นฉบับที่นำมารวม)
* **ประเภทข้อมูล:** รวมวิดีโอคลิปหลายแหล่ง (Hockey, Movies, Real-life) เป็น Video Classification
* **ข้อจำกัดสำคัญ:** **ห้ามนำวิดีโอนี้ไปสกัดเฟรม ตีกรอบ Annotation หรือฝึกโมเดลโดยเด็ดขาด** จนกว่าจะมีการตรวจสอบและยืนยันสิทธิ์ของวิดีโอต้นฉบับรายแหล่งได้อย่างชัดเจน
* **คะแนน Rubric (27/100):** License (0 - Provisional / Unverified Source Rights) | Bbox (0) | 7-Class (4) | Completeness (3) | CCTV (9) | Diversity (7) | Negative (4) | Leakage (0) | Cost (0)
* **สรุปสถานะ:** **HOLD — Provenance and source-license audit required**

---

## ส่วน C: จัดอันดับ Candidate สำหรับ 6 Spatial Classes และ Fight Research เดิม

### 1. หมวด Person & PPE (Classes 0, 1, 2)
1. **Ultralytics Construction-PPE — SAMPLE AUDITED → HOLD FOR TRAINING**: พบ actor/scene leakage ข้าม split, orphan labels 10 ไฟล์ และ provenance ยังไม่ครบ ดู [รายงาน sample audit](construction_ppe_sample_audit.md)
2. **SH17 — GO สำหรับ research-only sample audit**: มีข้อมูลจากผู้เขียนชัดกว่า แต่ติด CC BY-NC-SA และข้อกำหนด Pexels
3. **Beyza / Shlok / Nirav — HOLD**: รูปแบบข้อมูลน่าสนใจ แต่ license authority, provenance หรือ person-label completeness ยังไม่ผ่าน

### 2. หมวด Fall (Class 3)
ยังไม่มีชุดที่ผ่านสิทธิ์สำหรับ Training แบบไร้เงื่อนไข: **Uttej HOLD** เพราะสิทธิ์ฐานข้อมูลไม่ครอบคลุมภาพต้นฉบับ และ **Simuletic HOLD** เพราะ license ขัดแย้งพร้อมความเสี่ยง provenance; ชุดข้อมูลสำรวจใหม่ **Fall Detection Dataset (State-to-Fall + ADL)** ได้สถานะ **GO สำหรับ sample audit เท่านั้น** ดู [fall_fire_replacement_dataset_search.md](fall_fire_replacement_dataset_search.md)

### 3. หมวด Fire (Class 4) & Smoke (Class 5)
ปรับเปลี่ยนจากนโยบาย absolute fire BLOCKED สู่แบบจำลองสถานะอย่างรอบคอบ (Careful Status Model):
- **D-Fire:** ได้รับสถานะ **CONDITIONAL GO — educational prototype/sample audit only** (มี operative CC0 1.0 license ครอบคลุมชุดข้อมูลและกรอบ Bbox; เข้าสู่การประเมินได้ภายใต้ข้อตกลงโครงการรายวิชามหาวิทยาลัยแบบไม่แสวงหากำไร โดยห้าม commit ข้อมูลดิบเข้า Git, ห้ามเผยแพร่น้ำหนักโมเดลสู่สาธารณะ, อ้างอิงสิทธิ์, เบลอใบหน้าบุคคลในรายงาน, และเปิดเผยข้อจำกัดสิทธิ์ภาพต้นฉบับ; ไม่ใช่การอนุมัติเทรนแบบไร้เงื่อนไขหรือการใช้งานเชิงพาณิชย์)
- **Boreal Forest Fire (Subset A):** ได้สถานะ **GO สำหรับ sample audit เฉพาะควัน (smoke only)** (CC BY 4.0 โดรนป่าไม้ฟินแลนด์ ไม่มีกรอบ fire)
- **ชุดข้อมูลที่มีสิทธิ์ไม่ชัดเจนหรือขัดแย้ง:** **Simuletic HOLD/BLOCKED** (license ขัดแย้งกัน), **CQU และ Indoor Fire Smoke HOLD/BLOCKED** (สิทธิ์ภาพต้นฉบับไม่ได้รับการรับรอง), และ **Ironwolf HOLD** (provenance/format ยังไม่ยืนยัน)
- ดูรายละเอียดใน [fall_fire_replacement_dataset_search.md](fall_fire_replacement_dataset_search.md) และ [stage1_dataset_primary_source_audit.md](stage1_dataset_primary_source_audit.md)

### 4. หมวด Fight (Class 6 ในสถาปัตยกรรมเดิม -> ย้ายไป Stage 2 Temporal Classifier)
> ⚠️ **หมายเหตุสำคัญและเอกสารอ้างอิงหลัก:** การประเมินและสถานะของชุดข้อมูลในหมวด Fight ได้รับการตรวจสอบและวิจัยเชิงลึกแยกเฉพาะในเอกสาร **[docs/fight_dataset_evaluation.md](fight_dataset_evaluation.md)** และแผนการย้ายระบบใน **[docs/two_stage_architecture_migration.md](two_stage_architecture_migration.md)** โดย **รายละเอียดและสถานะในรายงานดังกล่าวมีอำนาจเหนือข้อความสรุปเดิมในส่วนนี้**

* **สรุปมติและสถานะหมวด Fight ล่าสุด:**
  * **มติเจ้าของโครงการ (Project Owner Decision):** **อนุมัติเลือกแนวทาง A: Two-Stage Pipeline อย่างเป็นทางการ** โดย:
    1. **Stage 1 (Spatial Object Detection):** ปรับเป้าหมาย YOLOv8 เป็น **6 Spatial Classes** (`0: person`, `1: helmet`, `2: vest`, `3: fall`, `4: fire`, `5: smoke`)
    2. **Stage 2 (Temporal Event Classification):** แยก `fight` ออกไปใช้ Temporal Video Classifier (เช่น X3D-S / VideoMAE) ทำงานร่วมกับ Tracking (ByteTrack) และ Video Buffer 1.5–3.0s
  * **Primary ready-to-use สำหรับ Spatial YOLO:** **None** (เนื่องจากย้าย Fight ออกจาก Spatial Detector แล้ว)
  * **Simuletic CCTV Aggressive Poses:** **NO-GO FOR PHASE 3 / CLOSED** (ปิด Track 1 ถาวร, คำกล่าวอ้างสังเคราะห์ขัดแย้งกับภาพคนจริง/CCTV จริง, สิทธิ์ CC BY 4.0 เหนือภาพต้นฉบับยังไม่ได้รับการยืนยัน, มี Active Fight เพียง 48 ภาพ, ขาด Human QA — ห้ามสร้าง Training labels และห้ามเทรนโดยเด็ดขาด)
  * **Permission candidate:** **TNUE-Fight Detection** (สถานะ OPTIONAL FUTURE CANDIDATE รอการติดต่อขอสิทธิ์เป็นลายลักษณ์อักษร 4 ประการสำหรับงานวิจัย Stage 2 ในอนาคต โดย**ไม่ถือเป็น Dependency ของ Stage 1**)
  * **Yash Combined (`yash07yadav`):** **HOLD pending provenance audit** (ห้ามนำมาสกัดเฟรมหรือฝึกโมเดลจนกว่าจะตรวจสิทธิ์วิดีโอรายแหล่งสำเร็จ)
  * **Roboflow datasets (เช่น `street-fight`, `ezgis-workspace`):** **ไม่ใช้เป็น Primary Dataset** เนื่องจากขาดเอกสาร Provenance ที่น่าเชื่อถือและมีปัญหาคุณภาพ Annotation
  * *เอกสารอ้างอิงฉบับเต็ม:* [fight_dataset_evaluation.md](fight_dataset_evaluation.md) และ [two_stage_architecture_migration.md](two_stage_architecture_migration.md) (รายงานตรวจ Simuletic ฉบับละเอียดเก็บเป็น local audit artifact ภายใต้ `reports/` ซึ่งไม่ถูก commit)

---

## ส่วน D: การจัดกลุ่มเพื่อนำไปใช้งาน (Categorical Recommendation)

### 1. Primary Datasets
**ยังไม่มีชุดใดได้รับอนุมัติสำหรับ Full Download หรือ Training แบบไร้เงื่อนไข** รายละเอียดหลักฐานปฐมภูมิอยู่ใน [stage1_dataset_primary_source_audit.md](stage1_dataset_primary_source_audit.md)

### 2. Sample-audit Candidates
1. **Ultralytics Construction-PPE:** sample audit เสร็จแล้วและเป็น HOLD จนกว่าจะ regroup split ระดับ scene, แก้ orphan labels และผ่าน provenance/Human QA
2. **SH17:** GO เฉพาะ research-only sample audit; เจ้าของโครงการยอมรับขอบเขต Non-commercial แล้ว แต่ยังต้องรักษา CC BY-NC-SA/Pexels constraints
3. **Fall Detection Dataset (State-to-Fall + ADL):** GO สำหรับ sample audit เท่านั้น (CC BY-NC 4.0 บันทึกจำลองการล้มโดยนักวิจัยปฐมภูมิ)
4. **Boreal Forest Fire (Subset A):** GO สำหรับ sample audit เฉพาะควัน (smoke only; CC BY 4.0)
5. **D-Fire:** CONDITIONAL GO สำหรับ educational prototype / sample audit ภายใต้ข้อจำกัดโครงการการศึกษาแบบไม่แสวงหากำไรอย่างเข้มงวด
6. Candidate อื่นทั้งหมดที่มีสิทธิ์ไม่ชัดเจนหรือขัดแย้งคงสถานะ HOLD หรือ HOLD/BLOCKED จนกว่าจะได้หลักฐานสิทธิ์และ provenance เพิ่มเติม (การศึกษาไม่ลบล้างเงื่อนไขลิขสิทธิ์)

### 3. Hard-Negative Sources
ยังไม่มีชุดใดได้รับอนุมัติ การใช้เฉพาะ negative images ก็ต้องผ่านสิทธิ์ภาพ, provenance และ exhaustive-label QA เช่นเดียวกับ positive images

### 4. Rejected Datasets (ชุดที่ต้องคัดทิ้งออกจากระบบ)
* `iStock Factory Fire` และ `Adobe Stock Fall` (ติดลิขสิทธิ์ All Rights Reserved)
* `ritupande/fire-detection-from-cctv` และ `magicearth25/video-violence` (License เป็น Unknown)
* `naveenk903/movies-fight-detection-dataset` (ลิงก์ 404 เสีย)
* `anbumalar1991/fight-dataset` (License ไม่ชัดเจน และภาพเล็ก 224x224 ไม่มี Bbox)

---

## ส่วน E: รายการ Dataset ที่ควรดาวน์โหลด Sample 50–200 ภาพเพื่อตรวจรอบต่อไป

ยังไม่อนุญาต Full Download ให้เริ่มได้เฉพาะ metadata/file-inventory review และ sample audit ของ:

1. **Ultralytics Construction-PPE:** ตรวจ sample แล้ว; ห้าม Training ด้วย split เดิมเพราะพบ scene leakage ดู [construction_ppe_sample_audit.md](construction_ppe_sample_audit.md)
2. **SH17:** อนุญาตให้ตรวจ sample ในขอบเขต research-only; ต้องเก็บ attribution, share-alike, Pexels source records และตรวจ likeness/privacy
3. **Fall Detection Dataset (State-to-Fall + ADL):** สุ่มตรวจ 10 คลิป (~410–700 เฟรม) ตาม Protocol ใน [fall_fire_replacement_dataset_search.md](fall_fire_replacement_dataset_search.md) เพื่อตรวจ CVAT XML mapping, dual-box fall, ความสมบูรณ์ของคน และ actor grouping
4. **Boreal Forest Fire (Subset A):** สุ่มตรวจ 80 ภาพแบบ stratified sample สำหรับ smoke only ตาม Protocol ใน [fall_fire_replacement_dataset_search.md](fall_fire_replacement_dataset_search.md) เพื่อตรวจคุณภาพกรอบควันและการปนเปื้อนของไฟ (flame contamination)
5. **D-Fire:** สุ่มตรวจ 80 ภาพแบบ stratified sample สำหรับ fire/smoke ภายใต้ CONDITIONAL GO ตาม Protocol ใน [fall_fire_replacement_dataset_search.md](fall_fire_replacement_dataset_search.md) เพื่อตรวจความกระชับของกรอบไฟ, แสงไฟลวง, unboxed persons, และ web burst grouping
6. **`Simuletic CCTV Aggressive Poses & Fight Detection Dataset` (ประวัติการตัดสิน):**
   * *สถานะผลการตรวจล่าสุด:* **NO-GO FOR PHASE 3 / CLOSED (Project Owner Approved Approach A: Two-Stage Pipeline)**
   * *สรุปผลการ Audit:* คำกล่าวอ้างสังเคราะห์ขัดแย้งกับภาพคนจริง/CCTV จริง, สิทธิ์ CC BY 4.0 เหนือภาพต้นฉบับยังไม่ได้รับการยืนยัน, มี Active Fight เพียง 48 ภาพ, ขาด Human QA และมี Overfitting จากเฟรมต่อเนื่องสูง
   * *คำสั่งการ:* **ยุติ Track 1 ถาวร, ห้ามสร้าง Training labels และห้ามเทรนโมเดลจากชุดข้อมูลนี้โดยเด็ดขาด**

---

## ส่วน F: คำถามและประเด็นที่ต้องตรวจสอบด้วยมนุษย์ (Human Review Points)

1. **การดำเนินการสำหรับคลาส `fight` (Two-Stage Pipeline Decision):**
   * เจ้าของโครงการได้ **อนุมัติเลือกแนวทาง A: Two-Stage Pipeline อย่างเป็นทางการ**
   * **Stage 1 Spatial Detector:** กำหนดเป้าหมายตรวจจับ **6 Spatial Classes** (`0: person`, `1: helmet`, `2: vest`, `3: fall`, `4: fire`, `5: smoke`)
   * **Stage 2 Temporal Fight Detection:** แยกการตรวจจับ `fight` ออกไปเป็นโมดูล Temporal Action Classifier (ByteTrack + Multi-signal Trigger + Video Buffer 1.5–3.0s + X3D/VideoMAE)
   * **สถานะ Dataset:** Simuletic ปิดเป็น NO-GO ถาวร (ห้ามสร้าง labels หรือเทรน, Human QA ยังไม่ได้เริ่ม), TNUE-Fight เป็น optional candidate สำหรับงานวิจัย Stage 2 ในอนาคต (ไม่บล็อก Stage 1)
   * **สถานะปัจจุบัน:** Migration สู่ Detector Schema v2 จำนวน 6 คลาสและการปรับ Code/Config เสร็จแล้ว แต่ยังไม่ได้ดาวน์โหลด Candidate Dataset หรือเริ่ม Training; Stage 2 ยังคง `BLOCKED / PENDING DATA APPROVAL`
2. **การทำ Exhaustive Annotation สำหรับคลาส Person ในชุด DFire:**
   - ในชุด `DFireDataset` มีภาพบางส่วนที่มีนักผจญเพลิงหรือประชาชนยืนอยู่ หากนำเข้าเทรนโมเดลรวม อาจทำให้โมเดลคิดว่าคนคือ Background (Missing-label penalty) **การใช้ pseudo-labeling เพียงอย่างเดียวไม่สามารถอนุมัติข้อมูลสำหรับฝึกสอนได้** ต้องมีการตรวจสอบและตีกรอบ bounding box โดยมนุษย์ (Human QA) อย่างครบถ้วนทุกกรณี
3. **การอนุมัติไฟล์คอนฟิก `configs/datasets.local.yaml`:**
   - เมื่อตรวจสอบ License ปฐมภูมิแล้ว ต้องให้ผู้รับผิดชอบโครงการลงลายมือชื่อ/กำหนดค่า `license_approved: true` ในระบบตามระเบียบของ [docs/workflow.md](workflow.md) โดยเป็นการอนุมัติเฉพาะ bounded educational prototype หรือ sample audit เท่านั้น ไม่ใช่การอนุมัติเทรนแบบไร้เงื่อนไข
