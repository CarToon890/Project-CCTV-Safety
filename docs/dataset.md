# Dataset Source Inventory (Candidates)

> สถานะปัจจุบัน: รายการด้านล่างเป็นเพียง candidate sources ยังไม่มีแหล่งใดได้รับ
> การยืนยัน license, version, annotation format หรือ exhaustive-label status
> จึงยังห้ามนำเข้า training pipeline จนกว่าจะบันทึกข้อมูลใน
> `configs/datasets.local.yaml` และตั้ง `license_approved: true` โดยผู้ตรวจสอบ

## Checklist ต่อหนึ่งแหล่งข้อมูล

| Field | Required decision |
|---|---|
| Source ID/version | ระบุชื่อและรุ่นที่สร้างซ้ำได้ |
| License + URL | ตรวจสิทธิ์ใช้งาน ดัดแปลง และแจกจ่าย |
| Annotation format | YOLO/COCO/VOC/video-level หรือไม่มี annotation |
| Original classes | บันทึกลำดับ class ID ต้นทาง |
| Canonical mapping | map ไปยัง 7 คลาส หรือกำหนดให้ทิ้ง |
| Group identity | video/scene/camera session สำหรับป้องกัน split leakage |
| Exhaustive review | ตรวจและเติมทุก canonical class ที่ปรากฏในภาพ |
| Domain notes | CCTV/stock/movie, day/night, resolution, viewpoint |

ไฟล์ stock-photo จาก iStock/Adobe เป็นเพียง discovery leads ไม่ใช่ dataset ที่
อนุมัติแล้ว ส่วนชุดที่ระบุ `(Video)` ต้องตรวจว่ามี bounding boxes หรือเป็นเพียง
video-level labels; หากไม่มี bbox ต้อง extract frames และ annotate ใหม่

---

## Candidate links preserved from the original inventory

**dataset**

**DATASET(FIRE, FALL, PPE, Fight)**



**FIRE**

https://www.kaggle.com/datasets/phylake1337/fire-dataset



https://www.istockphoto.com/th/search/2/image-film?family=creative\&mediatype=photography\&phrase=factory%20fire\&orientations=horizontal\&sort=mostpopular\&page=2\&excludenudity=true\&istockcollection=main%2Cvalue



https://github.com/gaia-solutions-on-demand/DFireDataset.git



https://www.kaggle.com/datasets/ritupande/fire-detection-from-cctv



https://www.kaggle.com/datasets/simuletic/cctv-smoke-and-fire-emergency-detection-dataset



https://www.kaggle.com/datasets/ironwolf437/fire-detection-dataset



\----



**FALL**

https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset



https://stock.adobe.com/th/search?k=fall+down+accident



https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset (Video)



https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection



https://www.kaggle.com/datasets/payutch/fall-video-dataset (Video)



\----



**PPE**

https://www.kaggle.com/datasets/shlokraval/ppe-dataset



https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset



https://www.kaggle.com/datasets/beyzakucuk/ppe-detection-v1



https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket



https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection



\----



**Fight**

https://www.kaggle.com/datasets/anbumalar1991/fight-dataset



https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset (Video)



https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset (Video)



https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset (Video)



https://www.kaggle.com/datasets/yash07yadav/project-data (Video)



