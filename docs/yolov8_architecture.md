# YOLOv8 Architecture Summary (สำหรับ Real-time CCTV)

## ภาพรวม
YOLOv8 (Ultralytics, 2023) เป็น single-stage object detector แบบ **anchor-free**
รองรับหลาย task: Detect / Segment / Classify / Pose / OBB

## องค์ประกอบหลัก 3 ส่วน

| ส่วน | รายละเอียด |
|------|-----------|
| **Backbone** | CSPDarknet ที่ใช้ module **C2f** (ต่อยอดจาก C3 ของ YOLOv5) — gradient flow ดีขึ้น พารามิเตอร์น้อยลง |
| **Neck** | **PAN-FPN** (Path Aggregation Network + Feature Pyramid) — รวม feature หลาย scale |
| **Head** | **Decoupled Head** แยก branch classification กับ regression + **Anchor-free** (ไม่ต้อง tune anchor boxes) |

## Loss Functions
- Classification → Binary Cross-Entropy (BCE)
- Box Regression → **DFL** (Distribution Focal Loss) + **CIoU**

## การจับคู่ label (Assignment)
ใช้ **Task-Aligned Assigner**: ให้คะแนน = cls_score^α × IoU^β เลือก top-k candidate

## ขนาดโมเดล (COCO mAP50-95)

| Model | Params | mAP | เหมาะกับ |
|-------|--------|-----|----------|
| yolov8n | ~3.2M | 37.3 | Edge device / CPU / GPU อ่อน — **แนะนำเริ่มต้น** |
| yolov8s | ~11.2M | 44.9 | GPU ระดับกลาง — **แนะนำถ้า accuracy ต้องสูงขึ้น** |
| yolov8m | ~25.9M | 50.2 | Server GPU |
| yolov8l | ~43.7M | 52.9 | Server GPU แรงสูง |
| yolov8x | ~68.2M | 53.9 | Offline batch processing |

## ทำไมแนะนำ yolov8n / yolov8s สำหรับ Real-time CCTV?
1. **Latency ต่ำ** — yolov8n ทำได้ >100 FPS บน GPU ทั่วไป, yolov8s ~60-80 FPS
2. **Transfer learning ดี** — pretrained COCO weights มี person class อยู่แล้ว ช่วยงาน fall และ PPE; Fight ไม่ใช่คลาสของ detector
3. **Deploy ง่าย** — export เป็น ONNX/TensorRT ได้ทันที รองรับ Jetson/CPU inference
4. **เทรนเร็ว** — เหมาะกับ baseline iteration เร็วๆ ก่อน scale ขึ้น

## แนวทาง Baseline Pipeline

    Pretrained yolov8n.pt → Fine-tune บน custom 6-class spatial dataset → Evaluate → Export ONNX

## Hyperparameters แนะนำเริ่มต้น
- imgsz: 640 (CCTV ควรลอง 1280 ถ้าวัตถุเล็ก)
- epochs: 100–150
- batch: 16 (n) / 8–16 (s)
- optimizer: auto (SGD/AdamW)
- augmentation: mosaic, hsv, flipud=0.0, fliplr=0.5
