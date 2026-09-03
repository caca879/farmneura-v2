import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import Column, String, Float, Integer, Text, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

# Malaysia Timezone Helper
MYT = timezone(timedelta(hours=8))

def get_myt_now():
    return datetime.now(MYT)

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="farmer")
    created_at = Column(DateTime, default=get_myt_now)


class Farm(Base):
    __tablename__ = "farms"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    name = Column(String(100), nullable=False)
    location = Column(String(255), nullable=True)
    size_sq_ft = Column(Float, default=1000.0)
    created_at = Column(DateTime, default=get_myt_now)

    user = relationship("User", backref="farms")
    plots = relationship("Plot", back_populates="farm", cascade="all, delete-orphan")



class Plot(Base):
    __tablename__ = "plots"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    farm_id = Column(String(36), ForeignKey("farms.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    size_sq_ft = Column(Float, default=1000.0)
    cycle_start_date = Column(String(10), nullable=False) # YYYY-MM-DD
    cycle_end_date = Column(String(10), nullable=False)   # YYYY-MM-DD
    cost_budget_myr = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_myt_now)

    farm = relationship("Farm", back_populates="plots")
    crops = relationship("Crop", back_populates="plot", cascade="all, delete-orphan")
    telemetry_logs = relationship("IoTTelemetryLog", back_populates="plot", cascade="all, delete-orphan")
    inspection_records = relationship("InspectionRecord", back_populates="plot", cascade="all, delete-orphan")


class Crop(Base):
    __tablename__ = "crops"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plot_id = Column(String(36), ForeignKey("plots.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    variety = Column(String(100), nullable=True)
    planting_date = Column(String(10), nullable=True)
    harvest_target_days = Column(Integer, default=50)
    created_at = Column(DateTime, default=get_myt_now)

    plot = relationship("Plot", back_populates="crops")
    inspection_records = relationship("InspectionRecord", back_populates="crop")


class IoTTelemetryLog(Base):
    __tablename__ = "iot_telemetry_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plot_id = Column(String(36), ForeignKey("plots.id", ondelete="CASCADE"), nullable=False)
    soil_moisture = Column(Float, nullable=False) # %
    soil_ec = Column(Float, nullable=False)       # mS/cm
    soil_ph = Column(Float, nullable=False)       # pH
    air_temp = Column(Float, nullable=False)      # °C
    air_humidity = Column(Float, default=75.0)   # %
    server_status = Column(String(50), default="ONLINE (Connected to Cloud IoT Broker)")
    recorded_at = Column(DateTime, default=get_myt_now)

    plot = relationship("Plot", back_populates="telemetry_logs")


class InspectionRecord(Base):
    __tablename__ = "inspection_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plot_id = Column(String(36), ForeignKey("plots.id", ondelete="CASCADE"), nullable=False)
    crop_id = Column(String(36), ForeignKey("crops.id", ondelete="SET NULL"), nullable=True)
    image_url = Column(Text, nullable=False)
    leaf_count = Column(Integer, default=0)
    diagnosis = Column(Text, nullable=False)
    intervention = Column(Text, nullable=False)
    field_notes = Column(Text, nullable=True)
    cycle_day = Column(Integer, default=0)
    stage_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=get_myt_now)

    plot = relationship("Plot", back_populates="inspection_records")
    crop = relationship("Crop", back_populates="inspection_records")


class HarvestRecord(Base):
    __tablename__ = "harvest_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plot_id = Column(String(36), ForeignKey("plots.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    yield_weight_kg = Column(Float, nullable=False, default=0.0)
    price_per_kg_myr = Column(Float, nullable=False, default=0.0)
    total_revenue_myr = Column(Float, nullable=False, default=0.0)
    harvest_date = Column(String(20), nullable=False)
    notes = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_myt_now)

    plot = relationship("Plot", backref="harvests")

