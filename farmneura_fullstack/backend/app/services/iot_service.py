import random
from datetime import datetime, timezone, timedelta

MYT = timezone(timedelta(hours=8))

def fetch_cloud_iot_telemetry(plot_id: str):
    """
    Simulates fetching real-time IoT sensor telemetry from a Cloud IoT Server (Firebase/MQTT).
    Returns a dict with sensor readings: timestamp, air_temp, soil_moisture, soil_ec, soil_ph, server_status.
    """
    t_now = datetime.now(MYT)
    # Generate seed based on plot_id string hashing + minute for smooth variation
    seed_val = sum(ord(c) for c in str(plot_id)) + t_now.minute
    random.seed(seed_val)
    
    moisture = round(random.uniform(28.0, 68.0), 1)
    temp = round(random.uniform(28.0, 34.5), 1)
    ec = round(random.uniform(0.9, 2.3), 2)
    ph = round(random.uniform(5.9, 6.7), 2)
    humidity = round(random.uniform(65.0, 88.0), 1)
    
    return {
        "plot_id": str(plot_id),
        "timestamp": t_now.strftime("%Y-%m-%d %H:%M:%S"),
        "air_temp": temp,
        "soil_moisture": moisture,
        "soil_ec": ec,
        "soil_ph": ph,
        "air_humidity": humidity,
        "server_status": "🟢 ONLINE (Connected to Cloud IoT Broker)"
    }
