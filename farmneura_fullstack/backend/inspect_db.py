import sys
import os

# Ensure backend path is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.models.db_models import User, Farm, Plot, Crop, InspectionRecord, IoTTelemetryLog

def print_db_summary():
    db = SessionLocal()
    try:
        print("=========================================================================")
        print("                 FARMNEURA V2 HUMAN-READABLE DATABASE                    ")
        print("=========================================================================")

        # 1. Users
        users = db.query(User).all()
        print(f"\n[1] REGISTERED USERS ({len(users)}):")
        if not users:
            print("   (No users registered yet)")
        for u in users:
            print(f"   * Name: {u.full_name:<18} | Email: {u.email:<25} | Role: {u.role}")

        # 2. Farms & Plots
        farms = db.query(Farm).all()
        print(f"\n[2] REGISTERED FARMS & PLOTS ({len(farms)} Farms):")
        if not farms:
            print("   (No farms registered yet)")
        for f in farms:
            print(f"\n   -> FARM: {f.name} (Location: {f.location})")
            farm_plots = db.query(Plot).filter(Plot.farm_id == f.id).all()
            if not farm_plots:
                print("      +-- (No plots inside this farm)")
            for p in farm_plots:
                plot_crops = db.query(Crop).filter(Crop.plot_id == p.id).all()
                crop_names = ", ".join([f"{c.name} ({c.variety or 'Standard'})" for c in plot_crops]) if plot_crops else "No registered crops"
                print(f"      +-- PLOT: {p.name:<8} | Crops: {crop_names:<25} | Cycle: {p.cycle_start_date} to {p.cycle_end_date}")

        # 3. Inspection Log Records with Human Names
        inspections = db.query(InspectionRecord).all()
        print(f"\n[3] INSPECTION LOG RECORDS ({len(inspections)} Saved Scans):")
        if not inspections:
            print("   (No inspection records saved yet)")
        for i in inspections:
            plot = db.query(Plot).filter(Plot.id == i.plot_id).first()
            plot_name = plot.name if plot else "Unknown Plot"
            farm = db.query(Farm).filter(Farm.id == plot.farm_id).first() if plot else None
            farm_name = farm.name if farm else "Unknown Farm"
            crop = db.query(Crop).filter(Crop.id == i.crop_id).first() if i.crop_id else None
            crop_name = crop.name if crop else "General"

            print(f"   * SCAN #{i.id[:8]} | Farm: {farm_name:<10} | Plot: {plot_name:<6} | Crop: {crop_name:<10} | Detections: {i.leaf_count:<2} | Date: {str(i.created_at)[:19]}")

        print("\n=========================================================================")
    finally:
        db.close()

if __name__ == "__main__":
    print_db_summary()
