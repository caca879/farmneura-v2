from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# --- User & Auth Schemas ---
class UserCreate(BaseModel):
    full_name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# --- Farm Schemas ---

class FarmBase(BaseModel):
    name: str
    location: Optional[str] = None
    size_sq_ft: Optional[float] = 1000.0
    user_id: Optional[str] = None

class FarmCreate(FarmBase):
    pass

class FarmResponse(FarmBase):
    id: str
    user_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Plot Schemas ---
class PlotBase(BaseModel):
    name: str
    farm_id: str
    size_sq_ft: Optional[float] = 1000.0
    cycle_start_date: str
    cycle_end_date: str
    cost_budget_myr: Optional[float] = 0.0
    notes: Optional[str] = None

class PlotCreate(PlotBase):
    pass

class PlotResponse(PlotBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Crop Schemas ---
class CropBase(BaseModel):
    name: str
    plot_id: str
    variety: Optional[str] = None
    planting_date: Optional[str] = None
    harvest_target_days: Optional[int] = 50

class CropCreate(CropBase):
    pass

class CropResponse(CropBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Telemetry Schemas ---
class TelemetryResponse(BaseModel):
    plot_id: str
    soil_moisture: float
    soil_ec: float
    soil_ph: float
    air_temp: float
    air_humidity: float
    server_status: str
    timestamp: str

# --- Inspection Schemas ---
class InspectionCreate(BaseModel):
    plot_id: str
    crop_id: Optional[str] = None
    model_preference: Optional[str] = "Auto-Detect"
    language_choice: Optional[str] = "🇲🇾 Bahasa Melayu"
    field_notes: Optional[str] = None

class InspectionResponse(BaseModel):
    id: str
    plot_id: Optional[str] = None
    crop_id: Optional[str] = None
    image_url: str
    leaf_count: int
    diagnosis: str
    intervention: str
    field_notes: Optional[str] = None
    cycle_day: int
    stage_name: Optional[str] = None
    created_at: str
    interval_tracking: Optional[str] = None

    class Config:
        from_attributes = True

class QuickScanSaveRequest(BaseModel):
    plot_id: str
    crop_id: Optional[str] = None
    image_url: str
    leaf_count: int
    diagnosis: str
    intervention: str
    field_notes: Optional[str] = None

