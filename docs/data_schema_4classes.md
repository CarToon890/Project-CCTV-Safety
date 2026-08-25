# Data Schema — 4 Classes (YOLO Format)

## Class Mapping

| ID | Class | คำอธิบาย | Bounding Box Target |
|----|-------|---------|---------------------|
| 0  | fall       | คนล้ม / นอนผิดปกติบนพื้น | ตัวคนทั้งตัว (full body) |
| 1  | fire_smoke | ไฟไหม้ หรือควัน | ขอบเขตเปลวไฟ/ควันที่มองเห็น |
| 2  | ppe        | อุปกรณ์ป้องกันส่วนบุคคล (หมวกกันน็อก, เสื้อสะท้อนแสง ฯลฯ) | ชิ้นอุปกรณ์แต่ละชิ้น |
| 3  | fight      | ท่าทางทะเลาะวิวาท | กลุ่มคนที่เกี่ยวข้อง (group bbox) |

> ⚠️ ห้ามเปลี่ยนลำดับ ID — ต้องตรงกับ names ใน data.yaml เสมอ

## Label File Format (.txt)
1 ไฟล์ .txt ต่อ 1 รูป (ชื่อไฟล์ต้องตรงกับรูป) แต่ละบรรทัด:

    <class_id> <x_center> <y_center> <width> <height>

- ค่าทั้งหมด **normalize 0.0–1.0** เทียบกับขนาดรูป
- ตัวอย่าง: `0 0.512 0.634 0.180 0.420`

## โครงสร้าง Dataset

    dataset/
    ├── images/
    │   ├── train/   # ~70%
    │   ├── val/     # ~20%
    │   └── test/    # ~10%
    └── labels/
        ├── train/
        ├── val/
        └── test/

## data.yaml

    path: ./dataset
    train: images/train
    val: images/val
    test: images/test
    nc: 4
    names: ['fall', 'fire_smoke', 'ppe', 'fight']

## ข้อควรระวังเฉพาะ Class
- **fall**: ต้องมี negative samples (คนยืน/เดินปกติ) ปนใน train set ลด false positive
- **fire_smoke**: แยก label ไฟ vs ควันให้ชัดเจนตั้งแต่ต้น ถ้าอนาคตต้องแยก class
- **ppe**: ถ้ามีหลายชนิด (helmet/vest/gloves) พิจารณาแตก subclass ภายหลัง
- **fight**: ใช้ group bbox ครอบคนที่เกี่ยวข้องทั้งหมด ไม่ใช่รายบุคคล
- Class imbalance: เก็บจำนวน instance ต่อ class ให้สมดุล หรือใช้ oversampling
