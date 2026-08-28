from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.db_models import Plot, IoTTelemetryLog
from app.models.schemas import TelemetryResponse
from app.services.iot_service import fetch_cloud_iot_telemetry

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

@router.get("/{plot_id}", response_model=TelemetryResponse)
def get_plot_telemetry(plot_id: str, db: Session = Depends(get_db)):
    plot = db.query(Plot).filter(Plot.id == plot_id).first()
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found.")
    
    # Fetch live simulated cloud telemetry
    telemetry_data = fetch_cloud_iot_telemetry(plot_id)
    
    # Log reading to database
    log_entry = IoTTelemetryLog(
        plot_id=plot_id,
        soil_moisture=telemetry_data["soil_moisture"],
        soil_ec=telemetry_data["soil_ec"],
        soil_ph=telemetry_data["soil_ph"],
        air_temp=telemetry_data["air_temp"],
        air_humidity=telemetry_data["air_humidity"],
        server_status=telemetry_data["server_status"]
    )
    db.add(log_entry)
    db.commit()
    
    return telemetry_data
