# Contact Angle Vision API

REST API สำหรับวัดมุมสัมผัสซ้าย/ขวาของหยดของเหลวจากภาพด้วย OpenCV (Otsu segmentation และ local tangent fitting) พร้อมหน้า Swagger UI สำหรับทดลองเรียก API

> ขณะนี้เป็น computer vision แบบไม่ใช้โมเดล ML ที่ต้องฝึกเพิ่ม ไม่ใช่โมเดล AI ที่ผ่านการ validate สำหรับงานวิจัย ผลที่ได้เป็นค่าประมาณและขึ้นกับคุณภาพภาพ/เส้นฐาน

## เริ่มใช้งาน

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

เปิด `http://localhost:8000/docs` เพื่อทดลอง API หรือ `http://localhost:8000/health` เพื่อตรวจสถานะ

## Endpoint

`POST /v1/contact-angle` รับ `multipart/form-data`:

| Field | Required | รายละเอียด |
| --- | --- | --- |
| `image` | ใช่ | PNG, JPEG หรือ BMP; ไม่เกิน 12 MB |
| `polarity` | ไม่ | `dark` (ค่าเริ่มต้น) หรือ `light` ตามสีหยดเทียบกับพื้นหลัง |
| `baseline_y` | ไม่ | พิกัด y ของเส้นพื้นผิว แนะนำให้ระบุเพื่อความแม่นยำ หากไม่ระบุจะประมาณจากขอบล่างของคอนทัวร์ |
| `fit_height` | ไม่ | ช่วงความสูงพิกเซลสำหรับ fit เส้นสัมผัส (8–150; ค่าเริ่มต้น 30) |

ตัวอย่างเรียกใช้:

```bash
curl -X POST http://localhost:8000/v1/contact-angle \\
  -F "image=@drop.png" \\
  -F "polarity=dark" \\
  -F "baseline_y=380"
```

ผลลัพธ์มี `left_angle_deg`, `right_angle_deg`, `mean_angle_deg`, พิกัด contact/tangent points และ metadata ของภาพ หากระบุตำแหน่งเส้นฐานผิด ผลมุมจะคลาดเคลื่อน

## ข้อจำกัดของภาพ

- ใช้ภาพหยดเดี่ยวที่เห็นขอบหยดชัด มี contrast กับพื้นหลัง และพื้นผิวเกือบแนวนอน
- ควร crop ให้หยดเป็นวัตถุหลัก และระบุ `baseline_y` ให้ตรงกับผิวสัมผัส
- ใช้แสงสม่ำเสมอ ลดเงาสะท้อน/เงาตกทอด และหลีกเลี่ยงวัตถุอื่นที่มีพื้นที่ใหญ่กว่าหยด
- ระบบนี้ประมาณ silhouette จาก threshold ไม่ได้ฝึกโมเดล AI และไม่รองรับการรับรองค่าทางมาตรวิทยา

## Docker

```bash
docker build -t contact-angle-api .
docker run --rm -p 8000:8000 contact-angle-api
```
