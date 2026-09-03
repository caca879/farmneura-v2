from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from app.core.database import get_db
from app.models.db_models import User, Plot, InspectionRecord, Farm, Crop

router = APIRouter(prefix="/overview", tags=["Overview"])

MYT = timezone(timedelta(hours=8))

@router.get("/summary", response_model=Dict[str, Any])
def get_overview_summary(user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if user_id:
        user_farms = db.query(Farm).filter(Farm.user_id == user_id).all()
        farm_ids = [f.id for f in user_farms]
        plots = db.query(Plot).filter(Plot.farm_id.in_(farm_ids)).all() if farm_ids else []

    else:
        plots = db.query(Plot).all()

    total_plots = len(plots)
    
    healthy_count = 0
    needs_attention = 0
    needs_photo = 0
    
    action_list = []
    today = datetime.now(MYT).date()
    
    for plot in plots:
        farm = db.query(Farm).filter(Farm.id == plot.farm_id).first()
        farm_name = farm.name if farm else "Unknown Farm"
        
        latest_rec = db.query(InspectionRecord).filter(InspectionRecord.plot_id == plot.id).order_by(InspectionRecord.created_at.desc()).first()
        
        is_att = False
        is_over = False
        days_over = 0
        
        if latest_rec:
            d_text = latest_rec.diagnosis.lower()
            if any(w in d_text for w in ["stressed", "diseased", "chlorosis", "spot", "blight", "virus", "mold", "mildew"]):
                is_att = True
                needs_attention += 1
            else:
                healthy_count += 1
                
            days_diff = (today - latest_rec.created_at.date()).days
            if days_diff >= 3:
                is_over = True
                days_over = days_diff
                needs_photo += 1
        else:
            is_over = True
            needs_photo += 1
            days_over = 0
            
        action_list.append({
            "plot_id": plot.id,
            "plot_name": plot.name,
            "farm_id": plot.farm_id,
            "farm_name": farm_name,
            "is_attention": is_att,
            "is_overdue": is_over,
            "days_overdue": days_over,
            "latest_diagnosis": latest_rec.diagnosis if latest_rec else "No Inspection Photo Yet",
            "btn_label": "Take Action ➔" if is_att else ("📷 Scan Now" if is_over else "Inspect ➔")
        })
        
    overall_health_pct = int((healthy_count / total_plots) * 100) if total_plots > 0 else 100
    
    # Calculate Total Harvest Yield (KG) & Total Revenue (RM)
    total_harvest_kg = 0.0
    total_harvest_revenue_myr = 0.0
    try:
        from app.models.db_models import HarvestRecord
        if user_id:
            harvest_records = db.query(HarvestRecord).filter(HarvestRecord.user_id == user_id).all()
        else:
            harvest_records = db.query(HarvestRecord).all()

        total_harvest_kg = round(sum(h.yield_weight_kg for h in harvest_records), 2)
        total_harvest_revenue_myr = round(sum(h.total_revenue_myr for h in harvest_records), 2)
    except Exception as err:
        db.rollback()
        print("Harvest calculation warning:", err)


    # Sort action list: Needs attention first, then overdue photo
    action_list.sort(key=lambda x: (not x["is_attention"], not x["is_overdue"]))
    
    return {
        "overall_health_pct": overall_health_pct,
        "needs_attention_plots": needs_attention,
        "needs_photo_plots": needs_photo,
        "active_plots": total_plots,
        "total_harvest_kg": total_harvest_kg,
        "total_harvest_revenue_myr": total_harvest_revenue_myr,
        "today_action_list": action_list
    }


@router.get("/db-summary", response_model=Dict[str, Any])
def get_live_db_summary(db: Session = Depends(get_db)):
    try:
        users = db.query(User).all()
        farms = db.query(Farm).all()
        plots = db.query(Plot).all()
        crops = db.query(Crop).all()
        inspections = db.query(InspectionRecord).all()

        return {
            "total_users": len(users),
            "total_farms": len(farms),
            "total_plots": len(plots),
            "total_crops": len(crops),
            "total_saved_inspections": len(inspections),
            "registered_users": [
                {"id": getattr(u, 'id', ''), "full_name": getattr(u, 'full_name', ''), "email": getattr(u, 'email', ''), "role": getattr(u, 'role', ''), "created_at": str(getattr(u, 'created_at', ''))}
                for u in users
            ],
            "registered_farms": [
                {"id": getattr(f, 'id', ''), "name": getattr(f, 'name', ''), "user_id": getattr(f, 'user_id', ''), "location": getattr(f, 'location', ''), "size_sq_ft": getattr(f, 'size_sq_ft', 0)}
                for f in farms
            ],
            "registered_plots": [
                {"id": getattr(p, 'id', ''), "name": getattr(p, 'name', ''), "farm_id": getattr(p, 'farm_id', ''), "cycle": f"{getattr(p, 'cycle_start_date', '')} to {getattr(p, 'cycle_end_date', '')}"}
                for p in plots
            ],
            "registered_crops": [
                {"id": getattr(c, 'id', ''), "name": getattr(c, 'name', ''), "variety": getattr(c, 'variety', ''), "plot_id": getattr(c, 'plot_id', '')}
                for c in crops
            ],
            "saved_inspections": [
                {"id": getattr(i, 'id', ''), "plot_id": getattr(i, 'plot_id', ''), "leaf_count": getattr(i, 'leaf_count', 0), "diagnosis": getattr(i, 'diagnosis', ''), "created_at": str(getattr(i, 'created_at', ''))}
                for i in inspections
            ]
        }
    except Exception as err:
        db.rollback()
        return {"error": str(err), "status": "failed_db_query"}


@router.get("/test-groq")
def test_groq_api():
    import os
    import requests
    from app.core.config import settings

    key1 = os.environ.get("GROQ_API_KEY", "")
    key2 = os.environ.get("GROQ_APT_KEY", "")
    settings_key = getattr(settings, "GROQ_API_KEY", "")

    raw_key = (key1 or key2 or settings_key or "").strip().strip('"').strip("'")
    key_preview = f"{raw_key[:6]}...{raw_key[-4:]}" if len(raw_key) > 10 else "EMPTY_KEY"

    if not raw_key:
        return {
            "status": "error",
            "message": "No GROQ_API_KEY or GROQ_APT_KEY found in environment variables.",
            "key_preview": key_preview
        }

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {raw_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Say hello in 3 words"}
        ],
        "max_tokens": 20
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        return {
            "status_code": res.status_code,
            "key_preview": key_preview,
            "groq_response": res.json() if res.status_code == 200 else res.text
        }
    except Exception as e:
        return {
            "status": "exception",
            "key_preview": key_preview,
            "error": str(e)
        }


