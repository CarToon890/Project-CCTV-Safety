# Project CCTV Safety

ระบบตรวจจับความปลอดภัยและแจ้งเตือนเหตุการณ์ผิดปกติจากกล้องวงจรปิดแบบ Real-time ด้วยโมเดล Deep Learning (YOLOv8)

---

## 1. ภาพรวมโครงการ (Project Overview)

Project CCTV Safety พัฒนาขึ้นเพื่อช่วยตรวจจับและเฝ้าระวังเหตุการณ์เสี่ยงภัยหรือพฤติกรรมผิดปกติผ่านกล้องวงจรปิด (CCTV) แบบอัตโนมัติ โดยประมวลผลด้วยโมเดลคอมพิวเตอร์วิทัศน์ YOLOv8 เพื่อลดภาระการเฝ้าหน้าจอของเจ้าหน้าที่ และเพิ่มความรวดเร็วในการรับมือกับอุบัติเหตุหรือสถานการณ์ฉุกเฉิน

---

## 2. สถานการณ์ที่ตรวจจับ (Detection Scenarios)

โมเดลได้รับการฝึกสอนให้ตรวจจับ 4 สถานการณ์หลัก:

| ID | Class Name | คำอธิบาย | ขอบเขตการตรวจจับ (Bounding Box) |
|:---|:---|:---|:---|
| 0 | fall | บุคคลหกล้ม หรือนอนหมดสติผิดปกติบนพื้น | ตรวจจับบุคคลทั้งตัว (Full Body) |
| 1 | fire_smoke | เปลวไฟ หรือกลุ่มควันหนาแน่น | ขอบเขตเปลวไฟและกลุ่มควันที่มองเห็น |
| 2 | ppe | อุปกรณ์ป้องกันส่วนบุคคล (เช่น หมวกนิรภัย, เสื้อสะท้อนแสง) | ชิ้นอุปกรณ์ป้องกันแต่ละชิ้น |
| 3 | fight | การทะเลาะวิวาท การทำร้ายร่างกาย หรือการปะทะกัน | ครอบคลุมกลุ่มบุคคลที่เกี่ยวข้องทั้งหมด |

---

## 3. สถาปัตยกรรมและเทคโนโลยีที่ใช้ (Tech Stack & Architecture)

- AI Model: YOLOv8 (เริ่มต้นด้วย `yolov8n` สำหรับ Low-latency Inference และเปรียบเทียบกับ `yolov8s`)
- Frameworks & Libraries: PyTorch, Ultralytics YOLO, OpenCV, Roboflow
- Web & Dashboard: Frontend Monitoring Dashboard และ Backend API
- Deployment & Infra: Database (MySQL / PostgreSQL), Server Hosting & FileZilla (FTP)

---

## 4. โครงสร้างไดเรกทอรี (Directory Structure)

```text
Project-CCTV-Safety/
├── configs/
│   └── data.yaml                  # การตั้งค่า Dataset สำหรับ YOLOv8 (Path, Classes)
├── docs/
│   ├── data_schema_4classes.md    # รายละเอียด Data Schema และแนวทางการ Label ข้อมูล
│   └── yolov8_architecture.md     # เอกสารสรุปสถาปัตยกรรมโมเดลและพารามิเตอร์
├── notebooks/
│   └── yolov8_baseline.ipynb      # Jupyter Notebook สำหรับ Pipeline การเทรนและประเมินผล
├── Project-Plan.jpg               # แผนการดำเนินงานโครงการ
├── SWOT.jpg                       # การวิเคราะห์ SWOT ของโครงการ
└── README.md                      # เอกสารแนะนำโครงการ
```

---

## 5. การเริ่มต้นใช้งาน (Getting Started)

### ข้อกำหนดเบื้องต้น (Prerequisites)
- Python 3.8 หรือสูงกว่า
- แนะนำสภาพแวดล้อมที่มี GPU (CUDA) หรือใช้งานผ่าน Google Colab / Kaggle

### ขั้นตอนการติดตั้งและเตรียมการ

1. Clone repository:
   ```bash
   git clone https://github.com/CarToon890/Project-CCTV-Safety.git
   cd Project-CCTV-Safety
   ```

2. ติดตั้ง Dependencies:
   ```bash
   pip install ultralytics roboflow opencv-python torch torchvision
   ```

3. การเทรนโมเดล (Model Training):
   - กำหนดโฟลเดอร์ Dataset ตามโครงสร้างใน `configs/data.yaml`
   - เปิดและรันสคริปต์ใน `notebooks/yolov8_baseline.ipynb` เพื่อเริ่มต้นการ Fine-tune และประเมินค่าความแม่นยำ (mAP)

4. การนำโมเดลไปใช้งาน (Inference):
   ```python
   from ultralytics import YOLO

   # โหลดโมเดลที่ผ่านการเทรน
   model = YOLO("runs/detect/train/weights/best.pt")

   # ทดสอบกับวิดีโอหรือกล้อง CCTV
   results = model.predict(source=0, conf=0.25, show=True)
   ```

---

## 6. แผนการดำเนินงาน (Roadmap & Milestones)

- [x] ออกแบบ Data Schema และกำหนด 4 Classes
- [x] จัดทำเอกสารสถาปัตยกรรม YOLOv8 และจัดเตรียม Baseline Notebook
- [ ] รวบรวม Dataset และดำเนินการเทรนโมเดล YOLOv8n
- [ ] พัฒนาระบบ Web Dashboard และหน้าจอ CCTV Monitoring
- [ ] เชื่อมต่อระบบฐานข้อมูลและการแจ้งเตือนเหตุการณ์
- [ ] ทดสอบและ Deploy ระบบขึ้นสู่สภาพแวดล้อมจริง
