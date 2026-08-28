from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.db_models import Plot, Farm
from app.models.schemas import PlotCreate, PlotResponse

router = APIRouter(prefix="/plots", tags=["Plots"])

@router.get("", response_model=List[PlotResponse])
def get_plots(farm_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Plot)
    if farm_id:
        query = query.filter(Plot.farm_id == farm_id)
    return query.all()

@router.post("", response_model=PlotResponse, status_code=status.HTTP_201_CREATED)
def create_plot(plot_in: PlotCreate, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == plot_in.farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Parent Farm not found.")
    
    new_plot = Plot(
        farm_id=plot_in.farm_id,
        name=plot_in.name,
        size_sq_ft=plot_in.size_sq_ft,
        cycle_start_date=plot_in.cycle_start_date,
        cycle_end_date=plot_in.cycle_end_date,
        cost_budget_myr=plot_in.cost_budget_myr,
        notes=plot_in.notes
    )
    db.add(new_plot)
    db.commit()
    db.refresh(new_plot)
    return new_plot

@router.delete("/{plot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plot(plot_id: str, db: Session = Depends(get_db)):
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found.")
    db.delete(plot)
    db.commit()
    return None
