"""
FarmNeura v2 - Plot Monitoring UI (Phase 1, Module 4)
Scope: Farmer role only.
    - Mobile-friendly, high-contrast, premium farmer UI for field use.
    - Take/upload picture -> run YOLOv8 ONNX (plant count & health diagnosis)
    - View agronomist-style intervention recommendations via Groq API (Llama-4-Scout)
    - Save and view past historical records per plot

Run with: streamlit run farmneura_farmer_draft.py
"""

import streamlit as st
from datetime import datetime
import os
import random
import sqlite3
import pandas as pd

import numpy as np
from PIL import Image, ImageDraw

# Optional ONNX Runtime import for YOLO inference
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

HAS_YOLO_DEPS = HAS_ONNX


try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False
    import requests

# Set page config
st.set_page_config(
    page_title="FarmNeura - Plot Monitoring",
    page_icon="🌱",
    layout="centered", # Centered layout is much more mobile-friendly
    initial_sidebar_state="collapsed" # Fold sidebar by default on mobile screens
)

# ---------------------------------------------------------------------
# CUSTOM STYLING (Agri-Solar Forest Green & Warm Yellow Theme)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Custom App Header Card */
    .header-card {
        background: linear-gradient(135deg, #1b4d3e 0%, #2e7d32 60%, #e65100 100%);
        padding: 1.5rem 1.25rem;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .header-card * {
        color: white !important;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 6px;
    }

    /* Custom Card Layouts */
    .result-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border: 1px solid #e0e0e0;
        border-left: 5px solid #2e7d32;
    }
    .result-card.alert {
        border-left-color: #e65100;
        background-color: #fff8e1;
        border-right: 1px solid #ffe082;
        border-top: 1px solid #ffe082;
        border-bottom: 1px solid #ffe082;
    }
    .result-card.neutral {
        border-left-color: #1565c0;
        background-color: #e3f2fd;
        border-right: 1px solid #90caf9;
        border-top: 1px solid #90caf9;
        border-bottom: 1px solid #90caf9;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1b4d3e !important;
        margin-bottom: 0.4rem;
    }
    .card-text {
        font-size: 0.95rem;
        color: #37474f !important;
        line-height: 1.5;
    }

    /* Metric Layout Card */
    .custom-metric {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .custom-metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2e7d32 !important;
        margin: 0;
    }
    .custom-metric-lbl {
        font-size: 0.85rem;
        color: #558b2f !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2) !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #0d3c12 100%) !important;
    }

    /* Dev Banners */
    .dev-banner {
        background-color: #eceff1;
        border: 1px dashed #b0bec5;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #455a64;
        margin-bottom: 1.5rem;
    }

    /* Overview Plot Card styling to force dark high-contrast colors in dark mode */
    .plot-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .plot-card h4 {
        margin: 0 !important;
        color: #1b4d3e !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
    }
    .plot-card p.farm-subtitle {
        margin: 2px 0 8px 0 !important;
        font-size: 0.85rem !important;
        color: #424242 !important;
        font-weight: 500 !important;
    }
    .plot-card p, .plot-card span, .plot-card strong {
        color: #111111 !important;
        font-size: 0.9rem !important;
    }
    .plot-card p.inspection-time, .plot-card p.inspection-time span {
        color: #333333 !important;
        font-size: 0.85rem !important;
    }
    .plot-card p.progress-label {
        margin: 4px 0 2px 0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# DATABASE PERSISTENCE & SCHEMAS (SQLite)
# ---------------------------------------------------------------------
DATABASE_FILE = "farmneura.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create Farms Table (Module 1)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        location TEXT,
        size_sq_ft REAL
    )
    """)
    
    # 2. Create Plots Table (Module 3)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id INTEGER NOT NULL,
        plot_name TEXT NOT NULL,
        size_sq_ft REAL,
        cycle_start TEXT,
        cycle_end TEXT,
        cost_records REAL,
        notes TEXT,
        FOREIGN KEY (farm_id) REFERENCES farms (id) ON DELETE CASCADE,
        UNIQUE (farm_id, plot_name)
    )
    """)
    
    # 3. Create Plants Table (Module 2)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plot_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (plot_id) REFERENCES plots (id) ON DELETE CASCADE
    )
    """)
    
    # 4. Create Monitoring Records Table (Module 4)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plot_id INTEGER NOT NULL,
        time TEXT NOT NULL,
        bil_pokok INTEGER NOT NULL,
        diagnosis TEXT NOT NULL,
        intervention TEXT NOT NULL,
        notes TEXT,
        canopy_cover_pct REAL,
        image_path TEXT,
        FOREIGN KEY (plot_id) REFERENCES plots (id) ON DELETE CASCADE
    )
    """)
    
    # Auto-migration check for existing databases
    cursor.execute("PRAGMA table_info(monitoring_records)")
    columns = [col[1] for col in cursor.fetchall()]
    if "canopy_cover_pct" not in columns:
        cursor.execute("ALTER TABLE monitoring_records ADD COLUMN canopy_cover_pct REAL")
    if "image_path" not in columns:
        cursor.execute("ALTER TABLE monitoring_records ADD COLUMN image_path TEXT")
        
    conn.commit()

    
    # Pre-populate with default demo data on first-time initialization
    cursor.execute("SELECT COUNT(*) FROM farms")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO farms (name, location, size_sq_ft) VALUES (?, ?, ?)", 
                       ("Farm A - Sekinchan", "Selangor, Malaysia", 50000.0))
        farm_a_id = cursor.lastrowid
        cursor.execute("INSERT INTO farms (name, location, size_sq_ft) VALUES (?, ?, ?)", 
                       ("Farm B - IADA Barat Laut", "Kuala Selangor, Malaysia", 75000.0))
        farm_b_id = cursor.lastrowid
        
        # Plots for Farm A
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_a_id, "Plot 1", 12000.0, "2026-06-01", "2026-08-30", 1500.0, "Under Solar Array Section A"))
        plot_1_id = cursor.lastrowid
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_a_id, "Plot 2", 15000.0, "2026-06-15", "2026-09-15", 1850.0, "Under Solar Array Section B"))
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_a_id, "Plot 3", 10000.0, "2026-07-01", "2026-10-01", 1200.0, "Under Solar Array Section C"))
        
        # Plots for Farm B
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_b_id, "Plot 1", 20000.0, "2026-05-10", "2026-08-10", 2500.0, "Open canopy layout"))
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_b_id, "Plot 2", 25000.0, "2026-05-20", "2026-08-20", 3000.0, "High shade index rows"))
        
        # Crop (Module 2)
        cursor.execute("INSERT INTO plants (plot_id, name) VALUES (?, ?)", (plot_1_id, "Pakchoy"))
        
        # Diagnosis Logs (Module 4) for Plot 1
        cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_pokok, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                       (plot_1_id, "2026-07-28 09:30", 45, "Healthy crops. Robust growth with standard green leaf index.", "Continue normal irrigation schedule. Apply NPK fertilizer at next scheduled cycle.", "Visual check under Solar Array Row B."))
        cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_pokok, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                       (plot_1_id, "2026-07-29 14:15", 42, "Mild leaf chlorosis detected (yellowing of leaves on 10% of crop canopy).", "Verify soil pH levels. Recommend localized nitrogen boost fertilizer to restore chlorophyll content.", "Observed near inverter box #3."))
        
        conn.commit()
    conn.close()

# Initialize Database
init_db()

# DB Query Helpers
def db_get_farms():
    conn = get_db_connection()
    farms = conn.execute("SELECT * FROM farms ORDER BY name").fetchall()
    conn.close()
    return [dict(f) for f in farms]

def db_add_farm(name, location, size):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO farms (name, location, size_sq_ft) VALUES (?, ?, ?)", (name.strip(), location.strip(), size))
        conn.commit()
        conn.close()
        return True, "Farm successfully registered!"
    except sqlite3.IntegrityError:
        return False, f"A farm with name '{name}' already exists."
    except Exception as e:
        return False, str(e)

def db_get_plots(farm_id):
    conn = get_db_connection()
    plots = conn.execute("SELECT * FROM plots WHERE farm_id = ? ORDER BY plot_name", (farm_id,)).fetchall()
    conn.close()
    return [dict(p) for p in plots]

def db_add_plot(farm_id, name, size, start_date, end_date, cost, notes):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (farm_id, name.strip(), size, start_date, end_date, cost, notes.strip()))
        conn.commit()
        conn.close()
        return True, "Plot successfully registered!"
    except sqlite3.IntegrityError:
        return False, f"A plot named '{name}' already exists for this farm."
    except Exception as e:
        return False, str(e)

def db_get_plants(plot_id):
    conn = get_db_connection()
    plants = conn.execute("SELECT * FROM plants WHERE plot_id = ? ORDER BY name", (plot_id,)).fetchall()
    conn.close()
    return [dict(p) for p in plants]

def db_add_plant(plot_id, name):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO plants (plot_id, name) VALUES (?, ?)", (plot_id, name.strip()))
        conn.commit()
        conn.close()
        return True, "Crop successfully registered!"
    except Exception as e:
        return False, str(e)

def db_get_records(plot_id):
    conn = get_db_connection()
    records = conn.execute("SELECT * FROM monitoring_records WHERE plot_id = ? ORDER BY time DESC", (plot_id,)).fetchall()
    conn.close()
    return [dict(r) for r in records]

def save_uploaded_image(image_input, plot_id):
    """
    Saves image input to local uploads/ directory and returns relative file path.
    """
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{plot_id}_{timestamp}.png"
    filepath = os.path.join(uploads_dir, filename)
    
    try:
        if isinstance(image_input, Image.Image):
            image_input.save(filepath)
        else:
            pil_img = Image.open(image_input)
            pil_img.save(filepath)
        return filepath
    except Exception:
        return None

def db_add_record(plot_id, time, bil_pokok, diagnosis, intervention, notes, canopy_cover_pct=None, image_path=None):
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO monitoring_records (plot_id, time, bil_pokok, diagnosis, intervention, notes, canopy_cover_pct, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (plot_id, time, bil_pokok, diagnosis, intervention, notes.strip() if notes else "", canopy_cover_pct, image_path)
        )
        conn.commit()
        conn.close()
        return True, "Record successfully saved!"
    except Exception as e:
        return False, str(e)


def db_get_plots_with_plants():
    conn = get_db_connection()
    # Fetch all plots, with their associated plants (if any) and their farm name
    query = """
    SELECT plots.*, farms.name AS farm_name, plants.name AS plant_name 
    FROM plots 
    JOIN farms ON plots.farm_id = farms.id
    LEFT JOIN plants ON plants.plot_id = plots.id
    ORDER BY farms.name, plots.plot_name
    """
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_latest_record(plot_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM monitoring_records WHERE plot_id = ? ORDER BY time DESC LIMIT 1", 
        (plot_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

# --- CRUD Operations (Update & Delete) ---
def db_update_farm(farm_id, name, location, size):
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE farms SET name = ?, location = ?, size_sq_ft = ? WHERE id = ?",
            (name, location, size, farm_id)
        )
        conn.commit()
        conn.close()
        return True, "Farm details updated successfully."
    except sqlite3.IntegrityError:
        return False, f"Farm name '{name}' already exists. Please choose a unique name."
    except Exception as e:
        return False, f"Error updating farm: {str(e)}"

def db_delete_farm(farm_id):
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM farms WHERE id = ?", (farm_id,))
        conn.commit()
        conn.close()
        return True, "Farm and all associated plots/crops deleted successfully."
    except Exception as e:
        return False, f"Error deleting farm: {str(e)}"

def db_update_plot(plot_id, name, size, start_date, end_date, cost, notes):
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE plots SET plot_name = ?, size_sq_ft = ?, cycle_start = ?, cycle_end = ?, cost_records = ?, notes = ? WHERE id = ?",
            (name, size, start_date, end_date, cost, notes, plot_id)
        )
        conn.commit()
        conn.close()
        return True, "Plot details updated successfully."
    except Exception as e:
        return False, f"Error updating plot: {str(e)}"

def db_delete_plot(plot_id):
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM plots WHERE id = ?", (plot_id,))
        conn.commit()
        conn.close()
        return True, "Plot and all associated crops/logs deleted successfully."
    except Exception as e:
        return False, f"Error deleting plot: {str(e)}"

def db_update_plant(plant_id, name):
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE plants SET name = ? WHERE id = ?",
            (name, plant_id)
        )
        conn.commit()
        conn.close()
        return True, "Crop details updated successfully."
    except Exception as e:
        return False, f"Error updating crop: {str(e)}"

def db_delete_plant(plant_id):
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM plants WHERE id = ?", (plant_id,))
        conn.commit()
        conn.close()
        return True, "Crop removed successfully."
    except Exception as e:
        return False, f"Error removing crop: {str(e)}"

def db_delete_record(record_id):
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM monitoring_records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True, "Monitoring record deleted successfully."
    except Exception as e:
        return False, f"Error deleting record: {str(e)}"

# ---------------------------------------------------------------------
# INTEGRATION FUNCTIONS
# ---------------------------------------------------------------------

def simple_numpy_nms(boxes, scores, iou_threshold=0.45):
    """
    Pure NumPy Implementation of Non-Maximum Suppression (NMS).
    Keeps the code compatible across basic environments without OpenCV or Torch.
    
    boxes: array of shape (N, 4) in [x_center, y_center, w, h] format
    scores: array of shape (N,)
    """
    if len(boxes) == 0:
        return []
    
    # Convert boxes from center format [cx, cy, w, h] to corner coordinates [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep items with IoU less than or equal to threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def run_yolo_count_and_diagnosis(image_file, model_preference="Auto"):
    """
    Runs a YOLOv8 ONNX model to count plants and diagnose their condition.
    Supports Tomato Disease (8 classes), Okra/Bendi Disease (3 classes), and generic detectors.
    
    Returns:
        tuple: (bil_pokok: int, diagnosis: str, annotated_image: PIL.Image)
    """
    # Select appropriate model path
    if "Okra" in model_preference and os.path.exists("models/best_(okra_model).onnx"):
        model_path = "models/best_(okra_model).onnx"
    elif "Tomato" in model_preference and os.path.exists("models/best_(tomato_leaf_model).onnx"):
        model_path = "models/best_(tomato_leaf_model).onnx"
    else:
        # Auto-detect best available model
        model_candidates = [
            "models/best_(tomato_leaf_model).onnx",
            "models/best_(okra_model).onnx",
            "models/yolov8_plant_detector.onnx",
            "models/best_epoch50.onnx",
            "models/best.onnx"
        ]
        model_path = next((p for p in model_candidates if os.path.exists(p)), "models/best_(tomato_leaf_model).onnx")
    
    # 8-Class Tomato Disease Mapping
    TOMATO_CLASSES = {
        0: {"name": "Early Blight", "color": "#ff5722", "is_healthy": False, "ms": "Hawar Awal (Kulat)"},
        1: {"name": "Septoria Leaf Spot", "color": "#e91e63", "is_healthy": False, "ms": "Bintik Daun Septoria"},
        2: {"name": "Healthy Foliage", "color": "#00e676", "is_healthy": True, "ms": "Daun Sihat & Subur"},
        3: {"name": "Bacterial Spot", "color": "#d50000", "is_healthy": False, "ms": "Bintik Daun Bakteria"},
        4: {"name": "Late Blight", "color": "#b71c1c", "is_healthy": False, "ms": "Hawar Lewat (Kulat)"},
        5: {"name": "Mosaic Virus", "color": "#9c27b0", "is_healthy": False, "ms": "Virus Mozek Daun"},
        6: {"name": "Yellow Leaf Virus", "color": "#ffeb3b", "is_healthy": False, "ms": "Virus Daun Kuning / Klorosis"},
        7: {"name": "Leaf Mold", "color": "#ff9800", "is_healthy": False, "ms": "Kulapuk Daun"},
    }

    # 3-Class Okra Disease Mapping
    OKRA_CLASSES = {
        0: {"name": "Downy Mildew", "color": "#ff5722", "is_healthy": False, "ms": "Kulat Kulapuk Berdebu"},
        1: {"name": "Healthy Okra Leaf", "color": "#00e676", "is_healthy": True, "ms": "Daun Bendi Sihat & Subur"},
        2: {"name": "Yellow Vein Mosaic", "color": "#ffeb3b", "is_healthy": False, "ms": "Penyakit Mozek Urat Kuning"},
    }
    
    # Verify environment has necessary libraries and the model file exists
    if not HAS_YOLO_DEPS or not os.path.exists(model_path):
        st.markdown(
            f"""
            <div class="dev-banner">
                <strong>🔧 YOLOv8 Integration Info:</strong> Running in simulated mode.<br/>
                Model targeted: <code>{os.path.abspath(model_path)}</code><br/>
                Dependencies: <code>pip install onnxruntime numpy Pillow</code>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        image = Image.open(image_file).convert("RGB")
        draw = ImageDraw.Draw(image)
        orig_w, orig_h = image.size
        
        simulated_count = random.randint(28, 42)
        num_mock_boxes = min(simulated_count, 12)
        for _ in range(num_mock_boxes):
            box_w = random.randint(int(orig_w * 0.1), int(orig_w * 0.2))
            box_h = random.randint(int(orig_h * 0.1), int(orig_h * 0.2))
            x1 = random.randint(0, orig_w - box_w)
            y1 = random.randint(0, orig_h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            is_healthy = random.random() > 0.2
            color = "#00e676" if is_healthy else "#ff1744"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
        diagnoses = [
            "🟢 Healthy growth. Robust chlorophyll distribution across canopy rows.",
            "⚠️ Mild leaf chlorosis detected (~15% of plants) - suspected nitrogen deficiency.",
            "🔴 Early stage Septoria leaf spot detected on 3 lower-canopy crop clusters.",
            "🟠 Mild wilting / water stress noticed along outer crop border."
        ]
        return simulated_count, random.choice(diagnoses), image
    
    try:
        @st.cache_resource
        def get_onnx_session(path):
            return ort.InferenceSession(path)
            
        session = get_onnx_session(model_path)
        
        # 1. Image Preprocessing
        image = Image.open(image_file).convert("RGB")
        orig_w, orig_h = image.size
        
        # YOLOv8 standard dimensions (640x640)
        resized_img = image.resize((640, 640))
        img_data = np.array(resized_img).astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1)) # HWC to CHW
        img_data = np.expand_dims(img_data, axis=0)  # Shape: (1, 3, 640, 640)
        
        # 2. Run Inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_data})
        
        # 3. Postprocess Output (Shape: 1, 4 + num_classes, 8400)
        output_tensor = np.squeeze(outputs[0])
        output_tensor = np.transpose(output_tensor)  # Shape: (8400, 4 + num_classes)
        
        num_classes = output_tensor.shape[1] - 4
        boxes = output_tensor[:, :4]
        scores = output_tensor[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Confidence threshold
        confidence_threshold = 0.20
        mask = confidences > confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = confidences[mask]
        filtered_class_ids = class_ids[mask]
        
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = simple_numpy_nms(filtered_boxes, filtered_scores, iou_threshold=0.45)
        plant_count = len(keep_indices)
        
        draw = ImageDraw.Draw(image)
        is_normalized = (np.max(filtered_boxes) <= 1.01) if len(filtered_boxes) > 0 else False
        
        healthy_count = 0
        diseased_count = 0
        disease_counts = {}
        
        for idx in keep_indices:
            box = filtered_boxes[idx]
            cid = int(filtered_class_ids[idx])
            conf = float(filtered_scores[idx])
            
            # Map class info based on number of classes detected in model
            if num_classes == 3 and cid in OKRA_CLASSES:
                c_info = OKRA_CLASSES[cid]
                is_healthy = c_info["is_healthy"]
                cls_name = c_info["name"]
                color = c_info["color"]
            elif num_classes == 8 and cid in TOMATO_CLASSES:
                c_info = TOMATO_CLASSES[cid]
                is_healthy = c_info["is_healthy"]
                cls_name = c_info["name"]
                color = c_info["color"]
            elif num_classes == 2:
                is_healthy = (cid == 0)
                cls_name = "Healthy Leaf" if is_healthy else "Diseased / Stressed"
                color = "#00e676" if is_healthy else "#ff1744"
            else:
                is_healthy = (cid == 0)
                cls_name = f"Class {cid}"
                color = "#00e676" if is_healthy else "#ff1744"
                
            if is_healthy:
                healthy_count += 1
            else:
                diseased_count += 1
                disease_counts[cls_name] = disease_counts.get(cls_name, 0) + 1
                
            # Scale coordinates back to original size
            x_center, y_center, w, h = box[0], box[1], box[2], box[3]
            if is_normalized:
                x1 = int((x_center - w / 2) * orig_w)
                y1 = int((y_center - h / 2) * orig_h)
                x2 = int((x_center + w / 2) * orig_w)
                y2 = int((y_center + h / 2) * orig_h)
            else:
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                x1 = int((x_center - w / 2) * scale_x)
                y1 = int((y_center - h / 2) * scale_y)
                x2 = int((x_center + w / 2) * scale_x)
                y2 = int((y_center + h / 2) * scale_y)
            
            # Clamp inside image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw label banner
            label_text = f"{cls_name} ({int(conf * 100)}%)"
            text_w = len(label_text) * 7 + 8
            tag_y1 = max(0, y1 - 20)
            draw.rectangle([x1, tag_y1, min(orig_w, x1 + text_w), y1], fill=color)
            text_fill = "#000000" if color in ["#ffeb3b", "#00e676"] else "#ffffff"
            draw.text((x1 + 4, tag_y1 + 2), label_text, fill=text_fill)
                
        # Generate detailed diagnostic summary
        if plant_count == 0:
            diagnosis = "No leaves/plants detected in frame. Please adjust camera distance or lighting."
        elif diseased_count > 0:
            percentage = int((diseased_count / plant_count) * 100)
            breakdown_list = [f"{cnt}x {dname}" for dname, cnt in disease_counts.items()]
            breakdown_str = ", ".join(breakdown_list)
            diagnosis = f"Detected {diseased_count} stressed/diseased instances (~{percentage}% of detected canopy). Breakdown: {breakdown_str}. ({healthy_count} healthy clusters)."
        else:
            diagnosis = f"Healthy growth. All {plant_count} detected crop clusters appear healthy and vigorous with strong chlorophyll signatures."
            
        return plant_count, diagnosis, image

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")
        return 0, f"Error processing model: {str(e)}", None


def run_intervention_recommendation(diagnosis, language_choice="🇲🇾 Bahasa Melayu"):
    """
    Calls Groq API with Llama 3.1 8B (llama-3.1-8b-instant) to generate an actionable agronomist intervention.
    
    Returns:
        str: Recommended crop action plan
    """
    # Attempt to load the API Key
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    
    # 1. Standard Prompt configuration based on language preference
    if language_choice == "🇲🇾 Bahasa Melayu":
        system_prompt = (
            "Anda adalah ejen pertanian jitu FarmNeura, seorang agronomis profesional yang pakar dalam bidang agrivoltaik (penanaman tanaman di bawah panel solar) dan sayuran daun brassica, terutamanya Sawi Pakchoy.\n"
            "Sediakan senarai cadangan tindakan agronomis yang sangat praktikal dan ringkas untuk petani berdasarkan diagnosis yang diberikan.\n"
            "Ambil kira faktor persekitaran agrivoltaik (contohnya, air larian hujan dari panel solar, kelembapan tanah di bawah teduhan solar, dan pH tanah akibat pembersihan panel).\n"
            "Formatkan jawapan anda dalam bentuk senarai peluru (bullet-point) dalam Bahasa Melayu yang mudah difahami oleh petani tempatan. Hadkan cadangan di bawah 3-4 mata sahaja."
        )
    else:
        system_prompt = (
            "You are FarmNeura's precision agricultural agent, a professional agronomist specializing in agrivoltaics (crops grown under solar panel arrays) and leafy green brassicas, specifically Pakchoy.\n"
            "Provide a highly actionable, concise list of agronomist recommendations for the farmer based on the provided diagnosis.\n"
            "Take into consideration agrivoltaic environmental factors (e.g., panel rain runoff lines, altered soil shade humidity, and panel-washing runoff pH values).\n"
            "Format your answer as a clean bullet-point list in simple farmer-friendly terms. Keep it under 3-4 bullet points."
        )

    # Mock response if Groq API key is missing
    if not api_key:
        st.markdown(
            """
            <div class="dev-banner">
                <strong>🔑 LLM Integration Info:</strong> Groq API key is missing.<br/>
                To enable live LLM recommendations:<br/>
                Add <code>GROQ_API_KEY = "your_key_here"</code> inside your <code>.streamlit/secrets.toml</code> 
                or set it as an environment variable in your terminal.
            </div>
            """, 
            unsafe_allow_html=True
        )
        d_lower = diagnosis.lower()
        if language_choice == "🇲🇾 Bahasa Melayu":
            if "nitrogen" in d_lower or "chlorosis" in d_lower or "yellowing" in d_lower:
                return "- Sembur baja cecair nitrogen (seperti urea atau ekstrak rumpai laut organik) pada waktu senja.\n- Periksa tahap kelembapan tanah pada kedalaman akar.\n- Periksa panel solar di atas untuk memastikan limpahan air tidak menghakis nutrien tanah."
            elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower or "stressed" in d_lower:
                return "- Sembur minyak neem organik atau racun kulat berasaskan tembaga yang disyorkan pada kawasan terjejas.\n- Buang daun bawah yang berpenyakit untuk mencegah penyebaran spora.\n- Pastikan teduhan panel solar tidak perangkap kelembapan berlebihan semalaman."
            else:
                return "- Teruskan pemantauan kelembapan tanah dan pemeriksaan kanopi mingguan.\n- Pastikan panel solar bersih (air basuhan berhabuk boleh mengubah kemasinan tanah berhampiran panel)."
        else:
            if "nitrogen" in d_lower or "chlorosis" in d_lower or "yellowing" in d_lower:
                return "- Apply a nitrogen-rich liquid fertilizer (e.g. urea or organic seaweed extract) at sunset.\n- Test soil moisture levels at root depth.\n- Inspect solar panels above to verify rain runoff isn't causing localized soil leaching."
            elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower or "stressed" in d_lower:
                return "- Spray organic neem oil or recommended copper-based fungicide to affected canopy rows.\n- Prune diseased lower leaves to prevent spore transmission.\n- Ensure solar panel shading is not trapping excessive humidity overnight."
            else:
                return "- Continue regular moisture tracking and weekly canopy inspections.\n- Ensure solar panels are clean (dust runoff can alter soil salinity near panel edges)."

    model_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
        "groq/compound",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]
    
    for model_name in model_candidates:
        try:
            if HAS_GROQ_SDK:
                client = Groq(api_key=api_key)
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Diagnosis: {diagnosis}"}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                if chat_completion.choices and len(chat_completion.choices) > 0:
                    content = chat_completion.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
            else:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Diagnosis: {diagnosis}"}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300
                }
                
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content and content.strip():
                            return content.strip()
        except Exception:
            continue
            
    return "⚠️ Could not retrieve live intervention recommendation from Groq. Please check your API key and connection."


# ---------------------------------------------------------------------
# GROWTH & YIELD ANALYSIS ENGINE (Visual NDVI GLI, DOA S-Curves & Regression)
# ---------------------------------------------------------------------

CROP_PROFILES = {
    "Bendi (Okra)": {
        "harvest_start_day": 50,
        "harvest_end_day": 55,
        "total_days": 60,
        "t0": 28,
        "k": 0.12,
        "cc_max": 85.0,
        "base_weight_g": 200.0,
        "default_density": 4.5,
        "opt_dli": 16.0,
        "opt_temp": 28.0,
        "opt_ec": 1.8
    },
    "Timun (Cucumber)": {
        "harvest_start_day": 35,
        "harvest_end_day": 45,
        "total_days": 50,
        "t0": 20,
        "k": 0.18,
        "cc_max": 90.0,
        "base_weight_g": 400.0,
        "default_density": 5.0,
        "opt_dli": 18.0,
        "opt_temp": 27.0,
        "opt_ec": 2.0
    },
    "Kacang Panjang (Yardlong Bean)": {
        "harvest_start_day": 45,
        "harvest_end_day": 50,
        "total_days": 60,
        "t0": 25,
        "k": 0.14,
        "cc_max": 85.0,
        "base_weight_g": 250.0,
        "default_density": 6.0,
        "opt_dli": 16.0,
        "opt_temp": 28.0,
        "opt_ec": 1.8
    },
    "Pakchoy (Leafy Brassica)": {
        "harvest_start_day": 30,
        "harvest_end_day": 35,
        "total_days": 40,
        "t0": 15,
        "k": 0.20,
        "cc_max": 80.0,
        "base_weight_g": 150.0,
        "default_density": 12.0,
        "opt_dli": 15.0,
        "opt_temp": 26.0,
        "opt_ec": 1.6
    }
}


def process_visual_ndvi_gli(image_input):
    """
    Computes Visual NDVI using Green Leaf Index (GLI):
    GLI = (2*G - R - B) / (2*G + R + B)
    Segments crop leaves from background soil and returns:
    - canopy_cover_pct (float)
    - heatmap_image (PIL.Image)
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = Image.open(image_input).convert("RGB")
            
        img_np = np.array(pil_img).astype(np.float32)
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]
        
        denom = 2.0 * G + R + B + 1e-6
        gli = (2.0 * G - R - B) / denom
        
        # Green canopy threshold (GLI > 0.08 and Green > Red)
        green_mask = (gli > 0.08) & (G > R)
        green_pixel_count = float(np.sum(green_mask))
        total_pixels = float(img_np.shape[0] * img_np.shape[1])
        canopy_cover_pct = (green_pixel_count / total_pixels) * 100.0
        
        # Create vegetation heatmap overlay
        heatmap_np = np.zeros_like(img_np)
        
        # Non-vegetation background: Warm soil tone
        heatmap_np[:, :, 0] = 139  # R
        heatmap_np[:, :, 1] = 90   # G
        heatmap_np[:, :, 2] = 43   # B
        
        # Green canopy pixels mapped by GLI intensity
        normalized_gli = np.clip((gli - 0.08) / 0.4, 0.0, 1.0)
        heatmap_np[green_mask, 0] = (1.0 - normalized_gli[green_mask]) * 40.0
        heatmap_np[green_mask, 1] = 130.0 + normalized_gli[green_mask] * 125.0
        heatmap_np[green_mask, 2] = (1.0 - normalized_gli[green_mask]) * 30.0
        
        # Blend original photo (40%) + Heatmap (60%)
        blended_np = (img_np * 0.4 + heatmap_np * 0.6).astype(np.uint8)
        heatmap_pil = Image.fromarray(blended_np)
        
        return round(float(canopy_cover_pct), 1), heatmap_pil
    except Exception as e:
        default_img = Image.open(image_input).convert("RGB") if not isinstance(image_input, Image.Image) else image_input
        return 38.5, default_img


def calculate_yield_prediction(crop_name, cc_pct, dli, temp, ec, density, plot_size_sqft=1000.0):
    """
    Predicts weight per head (g/plant), area yield (kg/m²), and total plot yield (kg)
    using Monteith's RUE and FAO AquaCrop response multipliers.
    """
    profile = CROP_PROFILES.get(crop_name, CROP_PROFILES["Bendi (Okra)"])
    
    # Environmental multipliers (Gaussian curve response centered at optimal conditions)
    f_dli = np.exp(-((dli - profile["opt_dli"]) ** 2) / (2.0 * (6.0 ** 2)))
    f_temp = np.exp(-((temp - profile["opt_temp"]) ** 2) / (2.0 * (5.0 ** 2)))
    f_ec = np.exp(-((ec - profile["opt_ec"]) ** 2) / (2.0 * (0.8 ** 2)))
    
    f_env = float(f_dli * f_temp * f_ec)
    
    # Weight per head (g/plant)
    cc_factor = (max(cc_pct, 1.0) / 100.0) ** 0.85
    weight_per_head_g = profile["base_weight_g"] * cc_factor * f_env
    
    # Total area yield (kg/m²)
    area_yield_kg_m2 = (weight_per_head_g * density) / 1000.0
    
    # Plot size conversion: sq ft to m² (1 sq ft = 0.092903 m²)
    plot_size_m2 = plot_size_sqft * 0.092903
    total_plot_yield_kg = area_yield_kg_m2 * plot_size_m2
    
    return round(float(weight_per_head_g), 1), round(float(area_yield_kg_m2), 2), round(float(total_plot_yield_kg), 1), profile


def get_crop_stage_badge(elapsed_days, harvest_start_day, harvest_end_day):
    """
    Returns (stage_title, bg_color, text_color, advice_text) based on crop cycle day.
    """
    if elapsed_days <= 14:
        return "🌱 Seedling / Early Emergence Stage", "#e8f5e9", "#2e7d32", "Focus on root establishment, gentle soil irrigation, and weed management."
    elif elapsed_days <= 35:
        return "🌿 Vegetative Canopy Expansion Stage", "#e3f2fd", "#1565c0", "Rapid leaf canopy growth phase. Monitor nitrogen fertigation and soil moisture."
    elif elapsed_days < harvest_start_day:
        return "🌼 Flowering & Fruit Set Stage", "#fff8e1", "#f57f17", "Crops blooming. Optimize phosphorus and potassium fertigation for optimal pod/fruit set."
    elif harvest_start_day <= elapsed_days <= harvest_end_day:
        return "🧺 Active Harvest Window", "#e8f5e9", "#1b5e20", "Optimal harvest window according to DOA guidelines! Pick ripe crop yield every 2 days."
    else:
        return "🍂 Late Harvest / Maturation Stage", "#fafafa", "#616161", "Final harvest cycle. Plan field clearing and land preparation for the next rotation."


def generate_growth_scurve_data(crop_name, plot_id=None, cycle_start_str=None):
    """
    Generates DOA Malaysia Sigmoid S-Curve dataframe for expected CC%
    and overlays actual historical canopy cover trajectory if plot_id is provided.
    """
    profile = CROP_PROFILES.get(crop_name, CROP_PROFILES["Bendi (Okra)"])
    days = list(range(0, profile["total_days"] + 1))
    
    t0 = profile["t0"]
    k = profile["k"]
    cc_max = profile["cc_max"]
    
    expected_cc = [cc_max / (1.0 + np.exp(-k * (d - t0))) for d in days]
    
    chart_data = {
        "Day": days,
        "DOA Baseline (Expected CC%)": [round(float(val), 1) for val in expected_cc]
    }
    
    # Query actual historical records if plot_id and cycle_start_str are provided
    if plot_id and cycle_start_str:
        try:
            start_date = datetime.strptime(cycle_start_str, "%Y-%m-%d")
            records = db_get_records(plot_id)
            
            actual_map = {}
            for r in records:
                if r.get("canopy_cover_pct") is not None:
                    try:
                        r_date = datetime.strptime(r["time"], "%Y-%m-%d %H:%M")
                        elapsed = (r_date - start_date).days
                        if 0 <= elapsed <= profile["total_days"]:
                            actual_map[elapsed] = float(r["canopy_cover_pct"])
                    except Exception:
                        pass
                        
            if actual_map:
                actual_series = [actual_map.get(d, None) for d in days]
                chart_data["Actual Measured CC% (Plot History)"] = actual_series
        except Exception:
            pass
            
    df_chart = pd.DataFrame(chart_data).set_index("Day")
    return df_chart, profile



# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION & SELECTION

st.sidebar.markdown("<h2 style='color:#1b4d3e; margin-top:0;'>🌱 FarmNeura</h2>", unsafe_allow_html=True)
st.sidebar.caption("Farmer Interface v2")

# Main Navigation Menu
view_mode = st.sidebar.radio(
    "📁 Navigation Menu",
    ["📋 Overview", "📷 Plot Monitoring", "⚙️ Registry & Management"],
    index=0
)

selected_farm_obj = None
selected_plot_obj = None

# Show selectors in sidebar only when in Monitoring mode
if view_mode == "📷 Plot Monitoring":
    st.sidebar.markdown("---")
    farms_list = db_get_farms()
    if not farms_list:
        st.sidebar.warning("No farms registered yet. Please go to Registry & Management to add a farm.")
    else:
        selected_farm_obj = st.sidebar.selectbox("Select Farm location", options=farms_list, format_func=lambda x: x["name"])
        
        plots_list = db_get_plots(selected_farm_obj["id"])
        if not plots_list:
            st.sidebar.warning("No plots registered for this farm. Please go to Registry & Management to add a plot.")
        else:
            selected_plot_obj = st.sidebar.selectbox("Select Target Plot", options=plots_list, format_func=lambda x: x["plot_name"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Configuration")
language_choice = st.sidebar.selectbox(
    "Intervention Language",
    ["🇲🇾 Bahasa Melayu", "🇬🇧 English"],
    index=0
)
st.sidebar.caption("⚡ **LLM Engine:** Groq Ultra-Fast LPU")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.85rem; color: #555;">
        <strong>Phase 1 Scope:</strong><br/>
        ✓ Farm Registry (Module 1)<br/>
        ✓ Plant Registry (Module 2)<br/>
        ✓ Plot Registry (Module 3)<br/>
        ✓ Plot Monitoring (Module 4)<br/>
        <br/>
        <span style="color:#d32f2f;">✗ IoT Sensors & Analytics (Phase 2)</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------

# Top branded header
if view_mode == "📋 Overview":
    farm_header_text = "Active Plots"
    plot_header_text = "All Crops"
elif view_mode == "⚙️ Registry & Management":
    farm_header_text = "Registry"
    plot_header_text = "Management"
else:
    farm_header_text = selected_farm_obj["name"] if selected_farm_obj else "No Farm Selected"
    plot_header_text = selected_plot_obj["plot_name"] if selected_plot_obj else "No Plot Selected"

st.markdown(
    f"""
    <div class="header-card">
        <div class="header-title">FarmNeura v2</div>
        <div class="header-subtitle">{view_mode.split()[-1]} • {farm_header_text} • {plot_header_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Render View: OVERVIEW
if view_mode == "📋 Overview":
    st.markdown("### Active Plots Dashboard")
    st.caption("Monitoring health status, crop cycle timelines, and plant count across your active plots.")
    
    plots_data = db_get_plots_with_plants()
    
    if not plots_data:
        st.info("⚠️ No plots registered yet. Go to **Registry & Management** to register your farms and plots.")
    else:
        # Group plots by crop type
        grouped_plots = {}
        for p in plots_data:
            p_name = p["plant_name"] if p["plant_name"] else "Unassigned (No Crop)"
            if p_name not in grouped_plots:
                grouped_plots[p_name] = []
            grouped_plots[p_name].append(p)
            
        for plant_name, plots in grouped_plots.items():
            st.markdown(f"#### 🌿 Crop: {plant_name}")
            
            # Display plots in a 3-column layout
            cols = st.columns(min(len(plots), 3))
            
            for index, plot in enumerate(plots):
                col = cols[index % 3]
                
                latest_rec = db_get_latest_record(plot["id"])
                
                if latest_rec:
                    d_text = latest_rec["diagnosis"].lower()
                    if "stressed" in d_text or "diseased" in d_text or "chlorosis" in d_text:
                        status_badge = "🔴 Diseased"
                        card_bg = "#ffebee"
                        border_color = "#ef5350"
                    else:
                        status_badge = "🟢 Healthy"
                        card_bg = "#e8f5e9"
                        border_color = "#66bb6a"
                    last_time = latest_rec["time"]
                    plant_count_display = f"{latest_rec['bil_pokok']} Plants"
                else:
                    status_badge = "⚪ Pending Inspection"
                    card_bg = "#f5f5f5"
                    border_color = "#bdbdbd"
                    last_time = "N/A"
                    plant_count_display = "No data"
                
                cycle_progress = 0
                cycle_days_str = ""
                try:
                    start_date = datetime.strptime(plot["cycle_start"], "%Y-%m-%d")
                    end_date = datetime.strptime(plot["cycle_end"], "%Y-%m-%d")
                    total_days = (end_date - start_date).days
                    
                    today = datetime.now()
                    elapsed_days = (today - start_date).days
                    
                    if elapsed_days < 0:
                        cycle_progress = 0
                    elif elapsed_days >= total_days:
                        cycle_progress = 100
                    else:
                        cycle_progress = int((elapsed_days / total_days) * 100)
                    
                    cycle_days_str = f"Day {elapsed_days} of {total_days}"
                except Exception:
                    cycle_progress = 0
                    cycle_days_str = "Unknown timeline"
                
                with col:
                    card_html = (
                        f'<div class="plot-card" style="background-color: {card_bg}; border: 2px solid {border_color};">'
                        f'<h4>📍 {plot["plot_name"]}</h4>'
                        f'<p class="farm-subtitle">🚜 {plot["farm_name"]}</p>'
                        f'<hr style="margin: 8px 0; border: 0; border-top: 1px solid rgba(0,0,0,0.15);"/>'
                        f'<p><strong>Status:</strong> <span>{status_badge}</span></p>'
                        f'<p><strong>Last Count:</strong> <span>{plant_count_display}</span></p>'
                        f'<p class="inspection-time"><strong>Last Inspection:</strong> <span>{last_time}</span></p>'
                        f'<hr style="margin: 8px 0; border: 0; border-top: 1px solid rgba(0,0,0,0.15);"/>'
                        f'<p class="progress-label">Cycle Progress ({cycle_days_str}):</p>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    st.progress(cycle_progress / 100.0)
                    
                    footer_html = f'<div style="font-size: 0.8rem; color: #666; margin-top: -8px; margin-bottom: 24px; text-align: center;">📐 Size: {plot["size_sq_ft"]:,} sq ft | 💰 Budget: MYR {plot["cost_records"]:,}</div>'
                    st.markdown(footer_html, unsafe_allow_html=True)

# Render View: REGISTRY & MANAGEMENT
elif view_mode == "⚙️ Registry & Management":
    st.markdown("### Farm & Plot Registry")
    st.caption("Register and manage your agrivoltaic farms, plot boundaries, and crop types.")
    
    sub_tab_farm, sub_tab_plot, sub_tab_plant = st.tabs(["🚜 Register Farm", "📐 Register Plot", "🌿 Register Crops"])
    
    with sub_tab_farm:
        st.markdown("#### Register a New Farm Location")
        with st.form("new_farm_form", clear_on_submit=True):
            f_name = st.text_input("Farm Name (Unique)", placeholder="e.g. Farm C - Rawang")
            f_loc = st.text_input("Location / Coordinate", placeholder="e.g. Selangor, Malaysia")
            f_size = st.number_input("Total Size (sq ft)", min_value=1.0, step=100.0, value=1000.0)
            
            submitted = st.form_submit_button("Register Farm", type="primary")
            if submitted:
                if not f_name.strip():
                    st.error("Farm Name is required.")
                else:
                    success, msg = db_add_farm(f_name, f_loc, f_size)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        
        # Display existing farms in a table
        farms = db_get_farms()
        if farms:
            st.markdown("##### Registered Farms")
            df_farms = pd.DataFrame(farms)
            df_farms.columns = ["ID", "Farm Name", "Location", "Size (sq ft)"]
            st.dataframe(df_farms, use_container_width=True, hide_index=True)
            
            # Edit & Delete Farm Expanders
            col_edit, col_del = st.columns(2)
            with col_edit:
                with st.expander("✏️ Edit Farm Details"):
                    selected_farm_to_edit = st.selectbox(
                        "Select Farm to Edit",
                        options=farms,
                        format_func=lambda x: x["name"],
                        key="edit_farm_select"
                    )
                    if selected_farm_to_edit:
                        with st.form("edit_farm_form"):
                            edit_name = st.text_input("New Farm Name", value=selected_farm_to_edit["name"])
                            edit_loc = st.text_input("New Location", value=selected_farm_to_edit["location"])
                            edit_size = st.number_input("New Size (sq ft)", min_value=1.0, value=float(selected_farm_to_edit["size_sq_ft"]))
                            
                            edit_submit = st.form_submit_button("Save Changes", type="primary")
                            if edit_submit:
                                if not edit_name.strip():
                                    st.error("Farm name cannot be empty.")
                                else:
                                    ok, m = db_update_farm(selected_farm_to_edit["id"], edit_name, edit_loc, edit_size)
                                    if ok:
                                        st.success(m)
                                        st.rerun()
                                    else:
                                        st.error(m)
            with col_del:
                with st.expander("🗑️ Delete Farm Location"):
                    selected_farm_to_delete = st.selectbox(
                        "Select Farm to Delete",
                        options=farms,
                        format_func=lambda x: x["name"],
                        key="delete_farm_select"
                    )
                    if selected_farm_to_delete:
                        st.warning(f"⚠️ Warning: Deleting '{selected_farm_to_delete['name']}' will permanently delete all associated plots, crop registries, and monitoring history records! This action cannot be undone.")
                        confirm_delete = st.checkbox(f"I confirm that I want to delete '{selected_farm_to_delete['name']}'", key="confirm_delete_farm")
                        if st.button("Delete Farm Location", type="primary", disabled=not confirm_delete, key="del_farm_btn"):
                            ok, m = db_delete_farm(selected_farm_to_delete["id"])
                            if ok:
                                st.success(m)
                                st.rerun()
                            else:
                                st.error(m)
            
    with sub_tab_plot:
        st.markdown("#### Register a New Plot inside a Farm")
        farms = db_get_farms()
        if not farms:
            st.warning("Please register a Farm first.")
        else:
            with st.form("new_plot_form", clear_on_submit=True):
                selected_f = st.selectbox("Select Farm", options=farms, format_func=lambda x: x["name"])
                p_name = st.text_input("Plot Name", placeholder="e.g. Plot 4")
                p_size = st.number_input("Plot Size (sq ft)", min_value=1.0, step=100.0, value=1000.0)
                
                col1, col2 = st.columns(2)
                with col1:
                    cycle_start = st.date_input("Cycle Start Date", value=datetime.now())
                with col2:
                    cycle_end = st.date_input("Cycle End Date", value=datetime.now())
                    
                p_cost = st.number_input("Est. Cycle Cost Budget (MYR)", min_value=0.0, step=50.0, value=0.0)
                p_notes = st.text_area("Plot Notes (optional)", placeholder="Shading ratio, solar panel alignment, etc.")
                
                submitted = st.form_submit_button("Register Plot", type="primary")
                if submitted:
                    if not p_name.strip():
                        st.error("Plot Name is required.")
                    else:
                        success, msg = db_add_plot(
                            selected_f["id"], p_name, p_size,
                            cycle_start.strftime("%Y-%m-%d"),
                            cycle_end.strftime("%Y-%m-%d"),
                            p_cost, p_notes
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                            
            # Display existing plots grouped by selected farm
            st.markdown("##### Registered Plots")
            selected_farm_for_list = st.selectbox("View plots in:", options=farms, format_func=lambda x: x["name"], key="view_plots_selector")
            plots = db_get_plots(selected_farm_for_list["id"])
            if plots:
                df_plots = pd.DataFrame(plots)
                df_plots = df_plots.drop(columns=["farm_id"])
                df_plots.columns = ["ID", "Plot Name", "Size (sq ft)", "Cycle Start", "Cycle End", "Budget (MYR)", "Notes"]
                st.dataframe(df_plots, use_container_width=True, hide_index=True)
                
                # Edit & Delete Plot Expanders
                col_edit, col_del = st.columns(2)
                with col_edit:
                    with st.expander("✏️ Edit Plot Details"):
                        selected_plot_to_edit = st.selectbox(
                            "Select Plot to Edit",
                            options=plots,
                            format_func=lambda x: x["plot_name"],
                            key="edit_plot_select"
                        )
                        if selected_plot_to_edit:
                            with st.form("edit_plot_form"):
                                edit_p_name = st.text_input("New Plot Name", value=selected_plot_to_edit["plot_name"])
                                edit_p_size = st.number_input("New Plot Size (sq ft)", min_value=1.0, value=float(selected_plot_to_edit["size_sq_ft"]))
                                try:
                                    start_val = datetime.strptime(selected_plot_to_edit["cycle_start"], "%Y-%m-%d")
                                    end_val = datetime.strptime(selected_plot_to_edit["cycle_end"], "%Y-%m-%d")
                                except:
                                    start_val = datetime.now()
                                    end_val = datetime.now()
                                edit_p_start = st.date_input("New Cycle Start Date", value=start_val)
                                edit_p_end = st.date_input("New Cycle End Date", value=end_val)
                                edit_p_cost = st.number_input("New Budget (MYR)", min_value=0.0, value=float(selected_plot_to_edit["cost_records"]))
                                edit_p_notes = st.text_area("New Notes", value=selected_plot_to_edit["notes"] or "")
                                
                                edit_submit = st.form_submit_button("Save Changes", type="primary")
                                if edit_submit:
                                    if not edit_p_name.strip():
                                        st.error("Plot name cannot be empty.")
                                    else:
                                        ok, m = db_update_plot(
                                            selected_plot_to_edit["id"], edit_p_name, edit_p_size,
                                            edit_p_start.strftime("%Y-%m-%d"),
                                            edit_p_end.strftime("%Y-%m-%d"),
                                            edit_p_cost, edit_p_notes
                                        )
                                        if ok:
                                            st.success(m)
                                            st.rerun()
                                        else:
                                            st.error(m)
                with col_del:
                    with st.expander("🗑️ Delete Plot"):
                        selected_plot_to_delete = st.selectbox(
                            "Select Plot to Delete",
                            options=plots,
                            format_func=lambda x: x["plot_name"],
                            key="delete_plot_select"
                        )
                        if selected_plot_to_delete:
                            st.warning(f"⚠️ Warning: Deleting '{selected_plot_to_delete['plot_name']}' will permanently delete all associated crop registries and monitoring history logs! This action cannot be undone.")
                            confirm_delete = st.checkbox(f"I confirm that I want to delete '{selected_plot_to_delete['plot_name']}'", key="confirm_delete_plot")
                            if st.button("Delete Plot", type="primary", disabled=not confirm_delete, key="del_plot_btn"):
                                ok, m = db_delete_plot(selected_plot_to_delete["id"])
                                if ok:
                                    st.success(m)
                                    st.rerun()
                                else:
                                    st.error(m)
            else:
                st.info("No plots registered under this farm yet.")

    with sub_tab_plant:
        st.markdown("#### Register Crops to a Plot")
        farms = db_get_farms()
        if not farms:
            st.warning("Please register a Farm first.")
        else:
            selected_f = st.selectbox("Select Farm for Crop Registry", options=farms, format_func=lambda x: x["name"], key="crop_farm_selector")
            plots = db_get_plots(selected_f["id"])
            if not plots:
                st.warning("Please register a Plot under this farm first.")
            else:
                with st.form("new_plant_form", clear_on_submit=True):
                    selected_p = st.selectbox("Select Plot", options=plots, format_func=lambda x: x["plot_name"])
                    crop_name = st.text_input("Crop/Plant Name", placeholder="e.g. Pakchoy")
                    
                    submitted = st.form_submit_button("Register Crop to Plot", type="primary")
                    if submitted:
                        if not crop_name.strip():
                            st.error("Crop/Plant name is required.")
                        else:
                            success, msg = db_add_plant(selected_p["id"], crop_name)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                # Show crops
                st.markdown("##### Registered Crops")
                selected_plot_for_crop_list = st.selectbox("View crops in:", options=plots, format_func=lambda x: x["plot_name"], key="view_crops_selector")
                plants = db_get_plants(selected_plot_for_crop_list["id"])
                if plants:
                    df_plants = pd.DataFrame(plants)
                    df_plants = df_plants.drop(columns=["plot_id"])
                    df_plants.columns = ["ID", "Crop/Plant Name"]
                    st.dataframe(df_plants, use_container_width=True, hide_index=True)
                    
                    # Edit & Delete Crop Expanders
                    col_edit, col_del = st.columns(2)
                    with col_edit:
                        with st.expander("✏️ Edit Crop Name"):
                            selected_crop_to_edit = st.selectbox(
                                "Select Crop to Edit",
                                options=plants,
                                format_func=lambda x: x["name"],
                                key="edit_crop_select"
                            )
                            if selected_crop_to_edit:
                                with st.form("edit_crop_form"):
                                    edit_c_name = st.text_input("New Crop/Plant Name", value=selected_crop_to_edit["name"])
                                    
                                    edit_submit = st.form_submit_button("Save Changes", type="primary")
                                    if edit_submit:
                                        if not edit_c_name.strip():
                                            st.error("Crop name cannot be empty.")
                                        else:
                                            ok, m = db_update_plant(selected_crop_to_edit["id"], edit_c_name)
                                            if ok:
                                                st.success(m)
                                                st.rerun()
                                            else:
                                                st.error(m)
                    with col_del:
                        with st.expander("🗑️ Remove Crop"):
                            selected_crop_to_delete = st.selectbox(
                                "Select Crop to Remove",
                                options=plants,
                                format_func=lambda x: x["name"],
                                key="delete_crop_select"
                            )
                            if selected_crop_to_delete:
                                st.warning(f"⚠️ Warning: Removing '{selected_crop_to_delete['name']}' will delete this crop assignment from the plot. It will not delete monitoring records.")
                                confirm_delete = st.checkbox(f"I confirm that I want to remove '{selected_crop_to_delete['name']}'", key="confirm_delete_crop")
                                if st.button("Remove Crop", type="primary", disabled=not confirm_delete, key="del_crop_btn"):
                                    ok, m = db_delete_plant(selected_crop_to_delete["id"])
                                    if ok:
                                        st.success(m)
                                        st.rerun()
                                    else:
                                        st.error(m)
                else:
                    st.info("No crops registered in this plot yet.")

# Render View: PLOT MONITORING (Module 4)
elif view_mode == "📷 Plot Monitoring":
    if not selected_farm_obj or not selected_plot_obj:
        st.info("⚠️ Please select a Farm and Plot in the sidebar, or go to the **Registry & Management** view to register them.")
    else:
        tab_record, tab_monitor = st.tabs(["📷 Take / Upload Record", "📊 Plot History Log"])
        
        # --- TAB 1: RECORD NEW PLOT DATA ---
        with tab_record:
            st.markdown("### Capture Plot Health")
            st.caption(f"Snap a photo of the crops in **{selected_plot_obj['plot_name']}** using your phone camera, or choose a file.")

            # Display active crops in this plot
            crops_list = db_get_plants(selected_plot_obj["id"])
            if crops_list:
                crops_str = ", ".join([c["name"] for c in crops_list])
                st.markdown(f"🌾 **Active crops in this plot:** {crops_str}")
            else:
                st.markdown("⚠️ *No crops registered for this plot yet. You can add them in the Crop Registry.*")

            camera_img = st.camera_input("📷 Use Mobile Camera")
            uploaded_img = st.file_uploader("📂 Or Upload Image file", type=["jpg", "jpeg", "png"])

            image_file = camera_img or uploaded_img

            # Reset temp diagnosis results if a new image is captured or uploaded
            if image_file is not None:
                img_id = getattr(image_file, "name", "camera_snap") + "_" + str(image_file.size)
                if "last_image_id" not in st.session_state or st.session_state.last_image_id != img_id:
                    st.session_state.last_image_id = img_id
                    if "temp_diagnosis" in st.session_state:
                        del st.session_state.temp_diagnosis

            if image_file is not None:
                st.markdown("---")
                
                # Vision Model Selector
                available_models = ["🤖 Auto-Detect (Best Available Model)"]
                if os.path.exists("models/best_(tomato_leaf_model).onnx"):
                    available_models.append("🍅 Tomato & Solanaceae Disease (8 Classes)")
                if os.path.exists("models/best_(okra_model).onnx"):
                    available_models.append("🌿 Okra / Bendi Disease & Yellow Vein (3 Classes)")
                if os.path.exists("models/yolov8_plant_detector.onnx"):
                    available_models.append("🌱 General Plant Detector")
                
                selected_vision_model = st.selectbox(
                    "🤖 Select AI Vision Model" if language_choice != "🇲🇾 Bahasa Melayu" else "🤖 Pilih Model Penglihatan AI",
                    available_models,
                    index=0
                )
                
                # Display annotated image if available, else show raw image
                if "temp_diagnosis" in st.session_state and "annotated_image" in st.session_state.temp_diagnosis and st.session_state.temp_diagnosis["annotated_image"] is not None:
                    st.image(
                        st.session_state.temp_diagnosis["annotated_image"], 
                        caption="Annotated AI Detections (Multi-Class Disease, Stress & Healthy Leaves)", 
                        use_container_width=True
                    )
                else:
                    st.image(image_file, caption="Selected Canopy Frame", use_container_width=True)
                
                # Primary Action Button
                if st.button("Diagnose Crop Health", type="primary"):
                    with st.spinner("Processing plant count & detecting abnormalities..."):
                        bil_pokok, diagnosis, annotated_image = run_yolo_count_and_diagnosis(image_file, model_preference=selected_vision_model)
                        intervention = run_intervention_recommendation(diagnosis, language_choice=language_choice)
                    
                    st.success("Diagnosis Complete!")
                    
                    st.session_state.temp_diagnosis = {
                        "bil_pokok": bil_pokok,
                        "diagnosis": diagnosis,
                        "intervention": intervention,
                        "annotated_image": annotated_image
                    }
                    st.rerun()

                # Check if we have results ready to display
                if "temp_diagnosis" in st.session_state:
                    res = st.session_state.temp_diagnosis
                    
                    # 1. Custom Metric Card
                    st.markdown(
                        f"""
                        <div class="custom-metric">
                            <div class="custom-metric-val">{res['bil_pokok']}</div>
                            <div class="custom-metric-lbl">Total Plants Detected (Bil Pokok)</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 2. Custom Styled Diagnosis card
                    st.markdown(
                        f"""
                        <div class="result-card neutral">
                            <div class="card-title">🔍 Crop Condition Diagnosis</div>
                            <div class="card-text">{res['diagnosis']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 3. Custom Styled Intervention card
                    st.markdown(
                        f"""
                        <div class="result-card alert">
                            <div class="card-title">💡 Recommended Intervention</div>
                            <div class="card-text">{res['intervention']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Additional Notes field
                    notes = st.text_area("📝 Field Notes / Observations (Optional)", placeholder="Add any details about soil dampness, weather, or manual inspection details.")
                    
                    # Save Action Button
                    if st.button("💾 Save Record to Plot Log"):
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                        
                        # Save uploaded image to disk
                        saved_img_path = save_uploaded_image(image_file, selected_plot_obj["id"])
                        
                        success, msg = db_add_record(
                            selected_plot_obj["id"],
                            time_str,
                            res["bil_pokok"],
                            res["diagnosis"],
                            res["intervention"],
                            notes,
                            image_path=saved_img_path
                        )
                        if success:
                            # Clean temporary workspace
                            if "temp_diagnosis" in st.session_state:
                                del st.session_state.temp_diagnosis
                            
                            st.success("✅ Record successfully logged to history. View the log in the 'Plot History Log' tab.")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Error saving record: {msg}")
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 2rem 0; color: #78909c;">
                        <p style="font-size: 1.2rem; margin: 0;">📷 Ready for Inspection</p>
                        <p style="font-size: 0.85rem; margin-top: 4px;">Capture or upload a crop photo to generate detection metrics.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- TAB 2: HISTORY LOG & MONITORING ---
        with tab_monitor:
            st.markdown(f"### Historical Logs for {selected_plot_obj['plot_name']}")
            st.caption("Review past diagnosis, counts, and recommended actions logged for this plot.")

            plot_records = db_get_records(selected_plot_obj["id"])

            if not plot_records:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 2.5rem 0; border: 1px dashed #cfd8dc; border-radius: 12px; background: #fafafa;">
                        <span style="font-size: 1.5rem; color: #b0bec5;">📊</span>
                        <p style="margin: 0.5rem 0 0; color: #78909c; font-size: 0.95rem; font-weight: 500;">No history records found</p>
                        <p style="margin: 0; color: #90a4ae; font-size: 0.85rem;">Take a crop reading in the record tab to create the first log entry.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Display history
                for r in plot_records:
                    expander_label = f"📅 {r['time']}  |  🌱 Count: {r['bil_pokok']} Plants"
                    if r.get("canopy_cover_pct") is not None:
                        expander_label += f"  |  🌿 CC: {r['canopy_cover_pct']}%"
                    
                    with st.expander(expander_label):
                        if r.get("image_path") and os.path.exists(r["image_path"]):
                            st.image(r["image_path"], caption="Logged Canopy Photo", use_container_width=True)
                        st.markdown(f"**🔍 Diagnosis:** {r['diagnosis']}")
                        st.markdown(f"**💡 Recommended Intervention:**\n{r['intervention']}")
                        
                        if r.get("notes"):
                            st.markdown(f"📝 **Field Notes:** *\"{r['notes']}\"*")
                        
                        # Delete Record action
                        st.markdown("---")
                        col_space, col_delete_btn = st.columns([5, 2])
                        with col_delete_btn:
                            if st.button("🗑️ Delete Log", key=f"del_rec_{r['id']}", type="secondary", use_container_width=True):
                                ok, m = db_delete_record(r['id'])
                                if ok:
                                    st.success("Log deleted!")
                                    st.rerun()
                                else:
                                    st.error(m)

