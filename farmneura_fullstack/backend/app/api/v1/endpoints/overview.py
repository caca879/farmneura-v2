from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from app.core.database import get_db
from app.models.db_models import Plot, InspectionRecord, Farm

router = APIRouter(prefix="/overview", tags=["Overview"])

MYT = timezone(timedelta(hours=8))

@router.get("/summary", response_model=Dict[str, Any])
def get_overview_summary(user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if user_id:
        user_farms = db.query(Farm).filter((Farm.user_id == user_id) | (Farm.user_id == None)).all()
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
    
    # Sort action list: Needs attention first, then overdue photo
    action_list.sort(key=lambda x: (not x["is_attention"], not x["is_overdue"]))
    
    return {
        "overall_health_pct": overall_health_pct,
        "needs_attention_plots": needs_attention,
        "needs_photo_plots": needs_photo,
        "active_plots": total_plots,
        "today_action_list": action_list
    }
