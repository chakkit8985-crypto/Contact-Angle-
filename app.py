from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Contact Angle Vision API",
    description="วิเคราะห์มุมสัมผัสของหยดของเหลวจากภาพด้วย computer vision",
    version="1.0.0",
)

MAX_IMAGE_BYTES = 12 * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "contact-angle-vision-api"}


def _get_contour(gray: np.ndarray, polarity: str) -> np.ndarray:
    # Otsu threshold supports both bright and dark droplets on a contrasting background.
    threshold_type = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise HTTPException(status_code=422, detail="ไม่พบขอบหยดของเหลวในภาพ")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        raise HTTPException(status_code=422, detail="พื้นที่หยดเล็กเกินไป กรุณาใช้ภาพที่ชัดขึ้น")
    return contour[:, 0, :].astype(np.float64)


def _contact_point(contour: np.ndarray, baseline_y: float, side: Literal["left", "right"], band: float) -> tuple[float, float]:
    close = contour[np.abs(contour[:, 1] - baseline_y) <= band]
    if len(close) < 2:
        raise HTTPException(
            status_code=422,
            detail="ไม่พบจุดสัมผัสใกล้เส้นฐาน กรุณากำหนด baseline_y ให้ตรงกับพื้นผิว",
        )
    x = float(np.min(close[:, 0]) if side == "left" else np.max(close[:, 0]))
    # Use the contour location at the selected edge, rather than assuming it lies exactly on the baseline.
    candidates = close[np.abs(close[:, 0] - x) <= 1.5]
    y = float(np.median(candidates[:, 1]))
    return x, y


def _measure_side(
    contour: np.ndarray,
    contact: tuple[float, float],
    side: Literal["left", "right"],
    baseline_y: float,
    fit_height: float,
) -> tuple[float, tuple[float, float]]:
    cx, cy = contact
    # Fit x as a function of height near the contact point. This is stable for steep edges,
    # where fitting y(x) would be poorly conditioned.
    region = contour[(contour[:, 1] <= baseline_y + 1) & (contour[:, 1] >= baseline_y - fit_height)]
    if side == "left":
        region = region[(region[:, 0] >= cx - 2) & (region[:, 0] <= cx + fit_height * 1.5)]
    else:
        region = region[(region[:, 0] <= cx + 2) & (region[:, 0] >= cx - fit_height * 1.5)]
    if len(region) < 8:
        raise HTTPException(status_code=422, detail="จุดขอบหยดไม่ชัดพอสำหรับคำนวณมุม")

    # Fit x = a*y^2 + b*y + c and estimate the tangent at the contact height.
    # Restricting the fit to a local region reduces sensitivity to the upper drop profile.
    ys = region[:, 1]
    xs = region[:, 0]
    try:
        a, b, _ = np.polyfit(ys, xs, 2)
    except (ValueError, np.linalg.LinAlgError):
        raise HTTPException(status_code=422, detail="ไม่สามารถ fit เส้นสัมผัสได้")
    dx_dy = float(2 * a * cy + b)
    # Tangent angle measured from the substrate, pointing from the contact point into the drop.
    # Image y increases downward; use the correct inward direction for each side.
    inward_dx_dy = dx_dy if side == "left" else -dx_dy
    angle = math.degrees(math.atan2(1.0, max(0.0, inward_dx_dy)))
    tangent_x = cx + inward_dx_dy * max(1.0, fit_height * 0.25)
    tangent_y = cy - max(1.0, fit_height * 0.25)
    return round(angle, 2), (round(tangent_x, 2), round(tangent_y, 2))


@app.post("/v1/contact-angle")
async def measure_contact_angle(
    image: UploadFile = File(..., description="หยดบนพื้นผิว; รองรับ PNG, JPEG, BMP"),
    polarity: Literal["dark", "light"] = Form("dark", description="หยดสีเข้มหรือสีอ่อนกว่าพื้นหลัง"),
    baseline_y: float | None = Form(None, description="ตำแหน่งเส้นพื้นผิวในพิกัด y ของภาพ; ถ้าไม่ระบุจะประมาณจากขอบล่างของหยด"),
    fit_height: int = Form(30, ge=8, le=150, description="ความสูงพิกเซลสำหรับ fit เส้นสัมผัส"),
) -> JSONResponse:
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="ไฟล์ภาพว่างเปล่า")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="ขนาดภาพต้องไม่เกิน 12 MB")
    encoded = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise HTTPException(status_code=415, detail="อ่านภาพไม่ได้ กรุณาใช้ไฟล์ PNG, JPEG หรือ BMP")
    height, width = decoded.shape
    if width < 40 or height < 40:
        raise HTTPException(status_code=422, detail="ภาพต้องมีขนาดอย่างน้อย 40 x 40 พิกเซล")

    contour = _get_contour(decoded, polarity)
    used_baseline = float(baseline_y) if baseline_y is not None else float(np.max(contour[:, 1]))
    if not 0 <= used_baseline < height:
        raise HTTPException(status_code=422, detail=f"baseline_y ต้องอยู่ระหว่าง 0 ถึง {height - 1}")
    band = max(3.0, height * 0.015)
    left = _contact_point(contour, used_baseline, "left", band)
    right = _contact_point(contour, used_baseline, "right", band)
    if right[0] - left[0] < 5:
        raise HTTPException(status_code=422, detail="ระยะระหว่างจุดสัมผัสสั้นเกินไป")

    left_angle, left_tangent = _measure_side(contour, left, "left", used_baseline, float(fit_height))
    right_angle, right_tangent = _measure_side(contour, right, "right", used_baseline, float(fit_height))
    return JSONResponse({
        "left_angle_deg": left_angle,
        "right_angle_deg": right_angle,
        "mean_angle_deg": round((left_angle + right_angle) / 2, 2),
        "baseline_y": round(used_baseline, 2),
        "baseline_source": "provided" if baseline_y is not None else "estimated_from_drop_contour",
        "image_size": {"width": width, "height": height},
        "contact_points": {
            "left": {"x": round(left[0], 2), "y": round(left[1], 2)},
            "right": {"x": round(right[0], 2), "y": round(right[1], 2)},
        },
        "tangent_points": {
            "left": {"x": left_tangent[0], "y": left_tangent[1]},
            "right": {"x": right_tangent[0], "y": right_tangent[1]},
        },
        "method": "opencv_otsu_segmentation_local_tangent_fit",
        "note": "ผลลัพธ์เป็นการวิเคราะห์ภาพเบื้องต้น ควรตรวจสอบ baseline และจุดสัมผัสก่อนใช้งานเชิงวิทยาศาสตร์",
    })


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
