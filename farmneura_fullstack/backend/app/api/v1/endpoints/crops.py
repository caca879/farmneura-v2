from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.db_models import Crop, Plot, Farm
from app.models.schemas import CropCreate, CropResponse

router = APIRouter(prefix="/crops", tags=["Crops"])

@router.get("", response_model=List[CropResponse])
def get_crops(plot_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(Crop)
    
    if user_id:
        user_farms = db.query(Farm).filter((Farm.user_id == user_id) | (Farm.user_id == None)).all()
        user_farm_ids = [f.id for f in user_farms]
        if not user_farm_ids:
            return []
        user_plots = db.query(Plot).filter(Plot.farm_id.in_(user_farm_ids)).all()
        user_plot_ids = [p.id for p in user_plots]
        if not user_plot_ids:
            return []
        query = query.filter(Crop.plot_id.in_(user_plot_ids))
        
    if plot_id:
        query = query.filter(Crop.plot_id == plot_id)
        
    return query.all()

@router.post("", response_model=CropResponse, status_code=status.HTTP_201_CREATED)
def create_crop(crop_in: CropCreate, db: Session = Depends(get_db)):
    plot = db.query(Plot).filter(Plot.id == crop_in.plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Parent Plot not found.")
    
    new_crop = Crop(
        plot_id=crop_in.plot_id,
        name=crop_in.name,
        variety=crop_in.variety,
        planting_date=crop_in.planting_date,
        harvest_target_days=crop_in.harvest_target_days
    )
    db.add(new_crop)
    db.commit()
    db.refresh(new_crop)
    return new_crop

@router.delete("/{crop_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_crop(crop_id: str, db: Session = Depends(get_db)):
    crop = db.query(Crop).filter(Crop.id == crop_id).first()
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found.")
    db.delete(crop)
    db.commit()
    return None
