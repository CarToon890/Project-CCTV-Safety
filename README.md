# Project CCTV Safety

Public-dataset AI baseline สำหรับตรวจจับเหตุความปลอดภัยจากภาพ CCTV ด้วย YOLOv8

---

## 1. ภาพรวมโครงการ (Project Overview)

Project CCTV Safety พัฒนาขึ้นเพื่อช่วยตรวจจับและเฝ้าระวังเหตุการณ์เสี่ยงภัยหรือพฤติกรรมผิดปกติผ่านกล้องวงจรปิด (CCTV) แบบอัตโนมัติ โดยประมวลผลด้วยโมเดลคอมพิวเตอร์วิทัศน์ YOLOv8 เพื่อลดภาระการเฝ้าหน้าจอของเจ้าหน้าที่ และเพิ่มความรวดเร็วในการรับมือกับอุบัติเหตุหรือสถานการณ์ฉุกเฉิน

---

## 2. สถานการณ์ที่ตรวจจับ (Detection Scenarios)

Stage 1 ใช้โมเดลตรวจจับ 6 คลาสเชิงพื้นที่ และสร้างสถานะการไม่สวม PPE ด้วย post-processing:

| ID | Class | เป้าหมาย Bounding Box |
|---:|---|---|
| 0 | person | บุคคลทั้งตัว |
| 1 | helmet | หมวกนิรภัยที่สวมอยู่ |
| 2 | vest | เสื้อสะท้อนแสง/เสื้อนิรภัยที่สวมอยู่ |
| 3 | fall | ร่างกายของบุคคลที่ล้มหรือนอนผิดปกติ |
| 4 | fire | บริเวณเปลวไฟ |
| 5 | smoke | บริเวณกลุ่มควัน |

`no_helmet` และ `no_vest` ไม่ใช่คลาสของโมเดล แต่คำนวณจากความสัมพันธ์ระหว่าง
`person`, `helmet` และ `vest` ดูกติกาฉบับเต็มใน `docs/data_schema_6classes.md`

`fight` ถูกย้ายไป Stage 2 Temporal Event Classification และยังมีสถานะ
`BLOCKED / PENDING DATA APPROVAL`; ไม่ใช่ YOLO bounding-box class ใน Stage 1

---

## 3. สถาปัตยกรรมและเทคโนโลยีที่ใช้ (Tech Stack & Architecture)

- AI Model: YOLOv8 (เริ่มต้นด้วย `yolov8n` สำหรับ Low-latency Inference และเปรียบเทียบกับ `yolov8s`)
- Frameworks & Libraries: PyTorch, Ultralytics YOLO, OpenCV, Roboflow
- Experiment: YOLOv8n เป็น baseline และ YOLOv8s เป็นตัวเปรียบเทียบภายใต้ config เดียวกัน
- Runtime target: Google Colab GPU; export ผลเป็น ONNX
- Web mockup อยู่นอกขอบเขต AI baseline และยังไม่เชื่อมต่อโมเดล/API

---

## 4. โครงสร้างไดเรกทอรี (Directory Structure)

```text
Project-CCTV-Safety/
├── configs/
│   ├── classes.yaml               # Canonical schema และ derived PPE events
│   ├── data.yaml                  # Dataset config สำหรับ YOLOv8
│   ├── datasets.example.yaml      # Template inventory/license/provenance
│   ├── training.yaml              # Shared config สำหรับ YOLOv8n/yolov8s
│   └── thresholds.yaml            # Confidence threshold แยกรายคลาส
├── cctv_safety/                   # Dataset validation และ PPE association library
├── docs/
│   ├── data_schema_6classes.md    # Canonical detector schema v2
│   ├── data_schema_7classes.md    # Deprecated detector schema v1
│   ├── evaluation.md              # Metrics และ error-analysis protocol
│   └── workflow.md                # End-to-end reproducible workflow
├── mockup/
│   ├── index.html                 # UI Mockup (Dashboard, Live Monitoring, Alert Logs)
│   └── README.md                  # คำอธิบาย Mockup และ Code Map
├── notebooks/
│   └── yolov8_baseline.ipynb      # Jupyter Notebook สำหรับ Pipeline การเทรนและประเมินผล
├── scripts/                        # Download, prepare, validate, train และ inference CLIs
├── tests/                          # Unit tests สำหรับ leakage/labels/PPE association
├── Project-Plan.jpg               # แผนการดำเนินงานโครงการ
├── SWOT.jpg                       # การวิเคราะห์ SWOT ของโครงการ
├── requirements.txt               # รายการ Dependencies สำหรับติดตั้งระบบ
└── README.md                      # เอกสารแนะนำโครงการ
```

---

## 5. การเริ่มต้นใช้งาน (Getting Started)

### ข้อกำหนดเบื้องต้น (Prerequisites)
- Python 3.10 หรือสูงกว่า (โค้ดใช้ type syntax แบบ `X | None`)
- แนะนำสภาพแวดล้อมที่มี GPU (CUDA) หรือใช้งานผ่าน Google Colab / Kaggle

### ขั้นตอนการติดตั้งและเตรียมการ

1. Clone repository:
   ```bash
   git clone https://github.com/CarToon890/Project-CCTV-Safety.git
   cd Project-CCTV-Safety
   ```

2. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. สร้าง source manifest จาก template และกรอกข้อมูลที่ตรวจสอบแล้ว:
   ```bash
   cp configs/datasets.example.yaml configs/datasets.local.yaml
   python scripts/download_datasets.py --manifest configs/datasets.local.yaml
   ```

4. หลังอนุมัติ license และทำ annotation ครบทุกคลาสแล้ว จึงเตรียมและตรวจ Dataset:
   ```bash
   python scripts/prepare_dataset.py --manifest configs/datasets.local.yaml
   python scripts/validate_dataset.py dataset --near-duplicates
   ```

5. เทรนและเปรียบเทียบโมเดลบน Colab GPU:
   ```bash
   python scripts/train_compare.py --config configs/training.yaml
   ```

6. รัน inference พร้อม PPE association:
   ```bash
   python scripts/infer.py reports/model_comparison/runs/yolov8n/weights/best.pt sample.mp4 --save
   ```

---

## 6. แผนการดำเนินงาน (Roadmap & Milestones)

- [x] ปรับเป็น Detector Schema v2 จำนวน 6 Spatial Classes และ PPE association baseline
- [x] จัดทำ reproducible data/training/evaluation pipeline
- [ ] ตรวจ license, ดาวน์โหลด และทำ exhaustive annotation ของ Dataset จริง
- [ ] ฝึกและเปรียบเทียบ YOLOv8n กับ YOLOv8s บน Colab
- [ ] ทดสอบกับข้อมูลกล้อง CCTV เป้าหมายเมื่อมีข้อมูล
> ข้อจำกัด: ผลจาก public datasets ยังไม่ใช่หลักฐานว่าโมเดลพร้อมใช้งานกับกล้องจริง
> Fall ต้องมี temporal confirmation ในระบบจริง ส่วน Fight แยกอยู่ใน Stage 2 ซึ่งยังถูกบล็อกจนกว่าจะมีข้อมูลวิดีโอที่ผ่านการอนุมัติ
