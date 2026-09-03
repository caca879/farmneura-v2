import os
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.core.config import settings
from app.models.db_models import Plot, Crop, InspectionRecord
from app.models.schemas import InspectionResponse, QuickScanSaveRequest

from app.services.yolo_service import run_yolo_inference
from app.services.iot_service import fetch_cloud_iot_telemetry
from app.services.llm_service import generate_llm_intervention

router = APIRouter(prefix="/inspections", tags=["Inspections"])

MYT = timezone(timedelta(hours=8))

def get_crop_stage_badge(elapsed_days: int, language_choice: str = "Bahasa Melayu"):
    is_english = "english" in (language_choice or "").lower()
    if is_english:
        if elapsed_days <= 14:
            return "Seedling / Early Emergence Stage"
        elif elapsed_days <= 35:
            return "Vegetative Canopy Expansion Stage"
        elif elapsed_days <= 49:
            return "Flowering & Fruit Set Stage"
        elif elapsed_days <= 55:
            return "Active Harvest Window"
        else:
            return "Late Harvest / Maturation Stage"
    else:
        if elapsed_days <= 14:
            return "Peringkat Anak Benih / Percambahan Awal"
        elif elapsed_days <= 35:
            return "Peringkat Perluasan Kanopi Vegetatif"
        elif elapsed_days <= 49:
            return "Peringkat Berbunga & Pembentukan Buah"
        elif elapsed_days <= 55:
            return "Tempoh Menuai Aktif"
        else:
            return "Peringkat Tuaian Akhir / Kematangan"

@router.post("/diagnose", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
async def create_inspection(
    plot_id: Optional[str] = Form(None),
    crop_id: Optional[str] = Form(None),
    model_preference: str = Form("Auto-Detect"),
    language_choice: str = Form("Bahasa Melayu"),
    field_notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    plot = None
    if plot_id and plot_id != "quick_scan":
        plot = db.query(Plot).filter(Plot.id == plot_id).first()
        
    # Read upload file
    contents = await file.read()
    temp_filepath = os.path.join(settings.UPLOADS_DIR, f"temp_{uuid.uuid4()}.png")
    with open(temp_filepath, "wb") as f:
        f.write(contents)
        
    # 1. Run YOLOv8 ONNX Vision Inference
    leaf_count, diagnosis, annotated_img = run_yolo_inference(
        temp_filepath, 
        model_preference=model_preference, 
        language_choice=language_choice
    )
    
    # Save annotated image
    timestamp_str = datetime.now(MYT).strftime("%Y%m%d_%H%M%S")
    prefix = f"plot_{plot_id}" if plot else "quick_scan"
    image_filename = f"{prefix}_{timestamp_str}.png"
    save_path = os.path.join(settings.UPLOADS_DIR, image_filename)
    if annotated_img:
        annotated_img.save(save_path)
    else:
        os.rename(temp_filepath, save_path)
        
    if os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
        except Exception:
            pass

    # 2. Fetch Live Cloud IoT Sensor Telemetry
    iot_telemetry = fetch_cloud_iot_telemetry(plot_id if plot else "simulated")
    
    # 3. LLM Multimodal Fusion
    intervention = generate_llm_intervention(diagnosis, iot_telemetry=iot_telemetry, language_choice=language_choice)
    
    # 4. Calculate Cycle Day & Stage Badge if plot is linked
    now_dt = datetime.now(MYT)
    cycle_day = 0
    is_en = "english" in (language_choice or "").lower()
    stage_name = "Quick Scan / Diagnostic Mode" if is_en else "Imbas Cepat / Mod Diagnostik"
    if plot:
        try:
            start_dt = datetime.strptime(plot.cycle_start_date, "%Y-%m-%d").date()
            elapsed = (now_dt.date() - start_dt).days
            cycle_day = max(0, elapsed)
            stage_name = get_crop_stage_badge(cycle_day, language_choice=language_choice)
        except Exception:
            pass

    # Return Diagnostic result (NOT saved to database until user explicitly clicks Save Record)
    return InspectionResponse(
        id=f"preview_{uuid.uuid4()}",
        plot_id=plot_id if (plot_id and plot_id != "quick_scan") else None,
        crop_id=crop_id,
        image_url=f"/uploads/{image_filename}",
        leaf_count=leaf_count,
        diagnosis=diagnosis,
        intervention=intervention,
        field_notes=field_notes,
        cycle_day=cycle_day,
        stage_name=stage_name,
        created_at=now_dt.strftime("%Y-%m-%d %H:%M")
    )


@router.post("/save-quick-scan", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
def save_quick_scan_record(req: QuickScanSaveRequest, db: Session = Depends(get_db)):
    plot = db.query(Plot).filter(Plot.id == req.plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Target Plot not found.")
        
    now_dt = datetime.now(MYT)
    cycle_day = 0
    stage_name = "Seedling / Early Emergence Stage"
    try:
        start_dt = datetime.strptime(plot.cycle_start_date, "%Y-%m-%d").date()
        elapsed = (now_dt.date() - start_dt).days
        cycle_day = max(0, elapsed)
        stage_name = get_crop_stage_badge(cycle_day)
    except Exception:
        pass

    record = InspectionRecord(
        plot_id=req.plot_id,
        crop_id=req.crop_id,
        image_url=req.image_url,
        leaf_count=req.leaf_count,
        diagnosis=req.diagnosis,
        intervention=req.intervention,
        field_notes=req.field_notes,
        cycle_day=cycle_day,
        stage_name=stage_name,
        created_at=now_dt
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return InspectionResponse(
        id=record.id,
        plot_id=record.plot_id,
        crop_id=record.crop_id,
        image_url=record.image_url,
        leaf_count=record.leaf_count,
        diagnosis=record.diagnosis,
        intervention=record.intervention,
        field_notes=record.field_notes,
        cycle_day=record.cycle_day,
        stage_name=record.stage_name,
        created_at=record.created_at.strftime("%Y-%m-%d %H:%M")
    )

@router.get("/history/{plot_id}", response_model=List[InspectionResponse])
def get_inspection_history(plot_id: str, db: Session = Depends(get_db)):
    records = db.query(InspectionRecord).filter(InspectionRecord.plot_id == plot_id).order_by(InspectionRecord.created_at.desc()).all()
    
    result = []
    for i, r in enumerate(records):
        interval_str = None
        if i < len(records) - 1:
            prev_r = records[i + 1]
            day_diff = (r.created_at.date() - prev_r.created_at.date()).days
            leaf_diff = r.leaf_count - prev_r.leaf_count
            diff_sign = "+" if leaf_diff >= 0 else ""
            interval_str = f"Uploaded +{day_diff} days after previous photo ({prev_r.created_at.strftime('%Y-%m-%d %H:%M')}). Foliage expansion: {diff_sign}{leaf_diff} leaves count difference."
        else:
            interval_str = "Initial crop inspection recorded for this plot."

        res = InspectionResponse(
            id=r.id,
            plot_id=r.plot_id,
            crop_id=r.crop_id,
            image_url=r.image_url,
            leaf_count=r.leaf_count,
            diagnosis=r.diagnosis,
            intervention=r.intervention,
            field_notes=r.field_notes,
            cycle_day=r.cycle_day,
            stage_name=r.stage_name,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M"),
            interval_tracking=interval_str
        )
        result.append(res)
        
    return result

@router.delete("/{inspection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inspection(inspection_id: str, db: Session = Depends(get_db)):
    record = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Inspection record not found.")
    db.delete(record)
    db.commit()
    return None
