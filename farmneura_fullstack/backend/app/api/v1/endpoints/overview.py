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
@router.get("/test-llm")
def test_llm_api():
    import os
    import requests
    from app.core.config import settings

    gemini_key = (os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", "") or "").strip().strip('"').strip("'")
    groq_key = (os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_APT_KEY") or getattr(settings, "GROQ_API_KEY", "") or "").strip().strip('"').strip("'")
    openai_key = (os.environ.get("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "") or "").strip().strip('"').strip("'")

    results = {}

    # Test Gemini
    if gemini_key:
        try:
            gemini_model = os.environ.get("GEMINI_MODEL") or getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
            g_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"

            g_res = requests.post(g_url, json={"contents": [{"parts": [{"text": "Hello"}]}]}, timeout=5)
            results["gemini"] = {"model": gemini_model, "status_code": g_res.status_code, "key_preview": f"{gemini_key[:6]}...{gemini_key[-4:]}", "response": g_res.json() if g_res.status_code == 200 else g_res.text[:200]}
        except Exception as e:
            results["gemini"] = {"error": str(e)}
    else:
        results["gemini"] = "No GEMINI_API_KEY provided"

    # Test Groq
    if groq_key:
        groq_tested = False
        for g_cand in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini", "llama-3.3-70b-versatile"]:
            try:
                gr_url = "https://api.groq.com/openai/v1/chat/completions"
                gr_res = requests.post(gr_url, headers={"Authorization": f"Bearer {groq_key}"}, json={"model": g_cand, "messages": [{"role": "user", "content": "Hello"}]}, timeout=5)
                if gr_res.status_code == 200:
                    results["groq"] = {"model": g_cand, "status_code": gr_res.status_code, "key_preview": f"{groq_key[:6]}...{groq_key[-4:]}", "response": gr_res.json()}
                    groq_tested = True
                    break
            except Exception as e:
                results["groq"] = {"error": str(e)}
                break
        if not groq_tested and "groq" not in results:
            results["groq"] = {"status_code": 404, "message": "No active candidate models succeeded"}
    else:
        results["groq"] = "No GROQ_API_KEY provided"

    # Test OpenAI
    if openai_key:
        try:
            o_url = "https://api.openai.com/v1/chat/completions"
            o_res = requests.post(o_url, headers={"Authorization": f"Bearer {openai_key}"}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}, timeout=5)
            results["openai"] = {"status_code": o_res.status_code, "key_preview": f"{openai_key[:6]}...{openai_key[-4:]}", "response": o_res.json() if o_res.status_code == 200 else o_res.text[:200]}
        except Exception as e:
            results["openai"] = {"error": str(e)}
    else:
        results["openai"] = "No OPENAI_API_KEY provided"

    return results



