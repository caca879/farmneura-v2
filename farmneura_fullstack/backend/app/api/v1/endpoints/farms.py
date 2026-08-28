from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.db_models import Farm
from app.models.schemas import FarmCreate, FarmResponse

router = APIRouter(prefix="/farms", tags=["Farms"])

@router.get("", response_model=List[FarmResponse])
def get_farms(user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if user_id:
        return db.query(Farm).filter((Farm.user_id == user_id) | (Farm.user_id == None)).all()
    return db.query(Farm).all()

@router.post("", response_model=FarmResponse, status_code=status.HTTP_201_CREATED)
def create_farm(farm_in: FarmCreate, db: Session = Depends(get_db)):
    if farm_in.user_id:
        existing = db.query(Farm).filter(Farm.name == farm_in.name, Farm.user_id == farm_in.user_id).first()
    else:
        existing = db.query(Farm).filter(Farm.name == farm_in.name, Farm.user_id == None).first()

    if existing:
        raise HTTPException(status_code=400, detail=f"Farm with name '{farm_in.name}' already exists in your account.")
    
    new_farm = Farm(
        name=farm_in.name,
        location=farm_in.location,
        size_sq_ft=farm_in.size_sq_ft,
        user_id=farm_in.user_id
    )
    db.add(new_farm)
    db.commit()
    db.refresh(new_farm)
    return new_farm

@router.delete("/{farm_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_farm(farm_id: str, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found.")
    db.delete(farm)
    db.commit()
    return None
