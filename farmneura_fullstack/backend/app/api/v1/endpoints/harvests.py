from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.db_models import HarvestRecord, Plot, Farm
from app.models.schemas import HarvestCreate, HarvestResponse

router = APIRouter(prefix="/harvests", tags=["Harvests"])

@router.get("", response_model=List[HarvestResponse])
def get_harvests(plot_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(HarvestRecord)
    
    if user_id:
        user_farms = db.query(Farm).filter(Farm.user_id == user_id).all()
        user_farm_ids = [f.id for f in user_farms]
        if not user_farm_ids:
            return []
        user_plots = db.query(Plot).filter(Plot.farm_id.in_(user_farm_ids)).all()
        user_plot_ids = [p.id for p in user_plots]
        if not user_plot_ids:
            return []
        query = query.filter(HarvestRecord.plot_id.in_(user_plot_ids))
        
    if plot_id:
        query = query.filter(HarvestRecord.plot_id == plot_id)
        
    return query.order_by(HarvestRecord.created_at.desc()).all()

@router.post("", response_model=HarvestResponse, status_code=status.HTTP_201_CREATED)
def create_harvest(harvest_in: HarvestCreate, db: Session = Depends(get_db)):
    plot = db.query(Plot).filter(Plot.id == harvest_in.plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Parent Plot not found.")
    
    total_revenue = round(harvest_in.yield_weight_kg * harvest_in.price_per_kg_myr, 2)
    
    new_harvest = HarvestRecord(
        plot_id=harvest_in.plot_id,
        user_id=harvest_in.user_id,
        yield_weight_kg=harvest_in.yield_weight_kg,
        price_per_kg_myr=harvest_in.price_per_kg_myr,
        total_revenue_myr=total_revenue,
        harvest_date=harvest_in.harvest_date,
        notes=harvest_in.notes
    )
    db.add(new_harvest)
    db.commit()
    db.refresh(new_harvest)
    return new_harvest

@router.delete("/{harvest_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_harvest(harvest_id: str, db: Session = Depends(get_db)):
    harvest = db.query(HarvestRecord).filter(HarvestRecord.id == harvest_id).first()
    if not harvest:
        raise HTTPException(status_code=404, detail="Harvest record not found.")
    db.delete(harvest)
    db.commit()
    return None
