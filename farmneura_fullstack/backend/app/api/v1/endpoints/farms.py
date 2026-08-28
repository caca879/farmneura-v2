from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.db_models import Farm
from app.models.schemas import FarmCreate, FarmResponse

router = APIRouter(prefix="/farms", tags=["Farms"])

@router.get("", response_model=List[FarmResponse])
def get_farms(db: Session = Depends(get_db)):
    return db.query(Farm).all()

@router.post("", response_model=FarmResponse, status_code=status.HTTP_201_CREATED)
def create_farm(farm_in: FarmCreate, db: Session = Depends(get_db)):
    existing = db.query(Farm).filter(Farm.name == farm_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Farm with name '{farm_in.name}' already exists.")
    
    new_farm = Farm(
        name=farm_in.name,
        location=farm_in.location,
        size_sq_ft=farm_in.size_sq_ft
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
