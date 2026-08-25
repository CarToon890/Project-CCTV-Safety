# Project CCTV Safety

ระบบ AI ตรวจจับความปลอดภัยจากกล้องวงจรปิดแบบ Real-time ด้วย YOLOv8

## 📋 Features (4 Classes Detection)
- all: ตรวจจับคนล้ม / นอนผิดปกติบนพื้น
- ire_smoke: ตรวจจับเปลวไฟและกลุ่มควัน
- ppe: ตรวจจับอุปกรณ์ความปลอดภัยส่วนบุคคล (Personal Protective Equipment)
- ight: ตรวจจับพฤติกรรมทะเลาะวิวาทหรือการปะทะกัน

## 🛠️ โครงสร้างโปรเจกต์
`
Project-CCTV-Safety/
├── configs/
│   └── data.yaml                  # การตั้งค่า Dataset สำหรับ YOLOv8
├── docs/
│   ├── data_schema_4classes.md    # รายละเอียด Schema และการ Label ข้อมูล 4 คลาส
│   └── yolov8_architecture.md     # สรุปสถาปัตยกรรมและการปรับแต่งโมเดล YOLOv8
├── notebooks/
│   └── yolov8_baseline.ipynb      # Notebook สำหรับฝึกสอนและทดสอบโมเดล Baseline
├── Project-Plan.jpg
├── SWOT.jpg
└── README.md
`

## 🚀 เริ่มต้นใช้งาน
1. ติดตั้ง Dependencies:
   `ash
   pip install ultralytics roboflow opencv-python
   `
2. เปิดและรันคำสั่งใน 
otebooks/yolov8_baseline.ipynb เพื่อเริ่มต้นขั้นตอนการเตรียมข้อมูลและเทรนโมเดล
