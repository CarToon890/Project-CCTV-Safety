# CCTV Safety — Draft Mockup (Handoff Package)


> **สถานะ:** Draft Mockup — เป็น static HTML ไฟล์เดียว ข้อมูลทั้งหมดเป็น mock data
> ที่สุ่มขึ้นในเบราว์เซอร์ **ยังไม่ได้เชื่อมต่อโมเดล YOLOv8, API หรือฐานข้อมูลจริง**

---

## โครงสร้าง Handoff Package

```text
mockup/
├── index.html              
├── README.md               

```



---

##  หน้าจอที่มีใน Mockup

| # | หน้า | สิ่งที่แสดง | สถานะ |
|:--|:---|:---|:---|
| 1 | **Dashboard** | KPI 4 ตัว, กราฟเหตุการณ์รายชั่วโมง, สัดส่วน 4 classes, สถานะกล้อง 14 ตัว, แจ้งเตือนล่าสุด | UI เสร็จ · ข้อมูลจำลอง |
| 2 | **Live CCTV Monitoring** | กล้องจำลอง 7 ตัว พร้อม bounding box ครบ 4 เหตุการณ์, สลับ layout, กรองเฉพาะกล้องที่มีเหตุการณ์, live event stream | UI เสร็จ · ยังไม่มี video stream จริง |
| 3 | **Alert / Logs** | ตาราง 90 รายการ + ตัวกรอง 5 แบบ + แบ่งหน้า + modal รายละเอียด | UI เสร็จ · ตัวกรองทำงานจริงบน mock data |

หน้า **จัดการกล้อง** และ **ตั้งค่าระบบ** มีเมนูไว้แล้วแต่ยังไม่ได้ออกแบบ (ปุ่มกดได้แต่ไม่ทำอะไร)

---

##  แผนที่โค้ด (Code Map)

`index.html` แบ่งเป็น 3 ส่วนใหญ่ — แก้ตรงไหนดูจากตารางนี้

| บรรทัด | ส่วน | หมายเหตุ |
|---:|:---|:---|
| 7–363 | `<style>` ทั้งหมด | |
| **8–18** | `:root` design tokens | **แก้สี/ธีมทั้งระบบที่นี่จุดเดียว** |
| 369–411 | Sidebar + navigation | |
| 415–424 | TopBar (ชื่อหน้า, นาฬิกา, avatar) | |
| 427–493 | HTML หน้า Dashboard | |
| 496–533 | HTML หน้า Live Monitoring | |
| 536–603 | HTML หน้า Alert / Logs | |
| 607–625 | Modal รายละเอียดเหตุการณ์ | |
| 627–1058 | `<script>` ทั้งหมด | |
| **628–681** | **Mock data ทั้งหมด** | `CLS`, `CAMS`, `LOGS` — จุดที่ต้องแทนด้วย API call |
| 629 | `CLS` — นิยาม 4 classes | สี + ชื่อไทย + ระดับความรุนแรง |
| 636 | `CAMS` — กล้อง 14 ตัว | id, โซน, สถานะ, fps |
| 660–681 | สร้าง `LOGS` 90 รายการแบบสุ่ม | **ลบทิ้งทั้งบล็อกตอนต่อ API** |
| 697 | `go(page)` — สลับหน้า | router แบบง่ายๆ |
| 712–787 | กราฟ Dashboard (SVG เขียนเอง) | `spark()`, `bars()`, `donut()` |
| 790–849 | `SCENES` — ฉากกล้องจำลอง | **ลบทิ้งเมื่อมี video stream จริง** |
| 851–877 | `camHTML()` — สร้าง tile กล้อง + bounding box | เก็บ logic bbox ไว้ใช้ต่อได้ |
| 925 | `pushFeed()` — live event stream | เปลี่ยนเป็น WebSocket handler |
| 958 | `filtered()` — ตัวกรองตาราง | ย้ายไป query string ฝั่ง server |
| 978 | `render()` — วาดตาราง + pagination | |
| 1021 | `openModal()` — modal รายละเอียด | |

---

## Mock → Real: ต้องเปลี่ยนอะไรบ้าง

| ตอนนี้ (mock) | ต้องเปลี่ยนเป็น ||
|:---|:---|:---|
| `LOGS` สุ่มในเบราว์เซอร์ (บรรทัด 660) | `GET /api/v1/events` | 
| `CAMS` hardcode 14 ตัว (บรรทัด 636) | `GET /api/v1/cameras` | 
| ตัวเลข KPI hardcode ใน HTML | `GET /api/v1/stats/summary` | 
| กราฟสุ่มใน `bars()` / `donut()` | `GET /api/v1/stats/hourly`, `/by-class` | 
| `SCENES` ฉาก CSS | `<video>` + HLS / WebRTC | 
| bounding box ตำแหน่งคงที่ | WebSocket `detection` frame | 
| `pushFeed()` สุ่มทุก 4.2 วิ | WebSocket `alert` event | 
| `filtered()` กรองใน JS | query params → server | 
| ปุ่ม "รับทราบเหตุการณ์" ไม่ทำอะไร | `PATCH /api/v1/events/{id}` | 

---



