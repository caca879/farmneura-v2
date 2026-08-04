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

# Optional imports with graceful fallback to prevent app crashing
try:
    import onnxruntime as ort
    import numpy as np
    from PIL import Image, ImageDraw
    HAS_YOLO_DEPS = True
except ImportError:
    HAS_YOLO_DEPS = False

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
        FOREIGN KEY (plot_id) REFERENCES plots (id) ON DELETE CASCADE
    )
    """)
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

def db_add_record(plot_id, time, bil_pokok, diagnosis, intervention, notes):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO monitoring_records (plot_id, time, bil_pokok, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                     (plot_id, time, bil_pokok, diagnosis, intervention, notes.strip()))
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

def run_yolo_count_and_diagnosis(image_file):
    """
    Runs a YOLOv8 ONNX model to count plants and diagnose their condition.
    
    Integration details:
    1. Preprocesses the user image (resizes to 640x640, scaling to float [0, 1]).
    2. Runs inference using ONNX runtime session.
    3. Triggers simulated fallback if model file or dependencies are absent.
    
    Returns:
        tuple: (bil_pokok: int, diagnosis: str, annotated_image: PIL.Image)
    """
    model_path = "models/yolov8_plant_detector.onnx"
    if not os.path.exists(model_path) and os.path.exists("models/best_epoch50.onnx"):
        model_path = "models/best_epoch50.onnx"
    
    # Verify environment has necessary libraries and the model file exists
    if not HAS_YOLO_DEPS or not os.path.exists(model_path):
        # Developer guidelines if they want to wire it up
        st.markdown(
            f"""
            <div class="dev-banner">
                <strong>🔧 YOLOv8 Integration Info:</strong> Running in simulated mode.<br/>
                To connect your real model:<br/>
                1. Save your YOLOv8 ONNX model to: <code>{os.path.abspath(model_path)}</code><br/>
                2. Install required dependencies: <code>pip install onnxruntime numpy Pillow</code>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Open image to draw mock boxes
        image = Image.open(image_file).convert("RGB")
        draw = ImageDraw.Draw(image)
        orig_w, orig_h = image.size
        
        simulated_count = random.randint(35, 48)
        
        # Draw some mock boxes for demonstration
        # We will draw a few green borders to show how the bounding boxes would look
        num_mock_boxes = min(simulated_count, 15) # cap at 15 so it's not overcrowded
        for _ in range(num_mock_boxes):
            box_w = random.randint(int(orig_w * 0.08), int(orig_w * 0.15))
            box_h = random.randint(int(orig_h * 0.08), int(orig_h * 0.15))
            x1 = random.randint(0, orig_w - box_w)
            y1 = random.randint(0, orig_h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            # 85% healthy green, 15% diseased red
            is_healthy = random.random() > 0.15
            color = "#00e676" if is_healthy else "#ff1744"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
        diagnoses = [
            "Healthy growth. Strong chlorophyll signature across all rows.",
            "Mild leaf chlorosis detected (suspected nitrogen deficiency) in 15% of plants.",
            "Early stage leaf spot disease detected on a few lower-canopy crop clusters.",
            "Dehydration stress noticed; minor wilting signs on crop borders."
        ]
        return simulated_count, random.choice(diagnoses), image
    
    try:
        # Load ONNX session (cached for performance)
        @st.cache_resource
        def get_onnx_session(path):
            return ort.InferenceSession(path)
            
        session = get_onnx_session(model_path)
        
        # 1. Image Preprocessing
        image = Image.open(image_file).convert("RGB")
        orig_w, orig_h = image.size
        
        # YOLOv8 standard dimensions (width=640, height=640)
        resized_img = image.resize((640, 640))
        img_data = np.array(resized_img).astype(np.float32) / 255.0
        # Transpose HWC to CHW
        img_data = np.transpose(img_data, (2, 0, 1))
        # Add batch dimension -> Shape: (1, 3, 640, 640)
        img_data = np.expand_dims(img_data, axis=0)
        
        # 2. Run Inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_data})
        
        # 3. Postprocess Output
        # YOLOv8 model outputs are typically shape (1, 4 + num_classes, 8400)
        output_tensor = np.squeeze(outputs[0])  # Shape: (4 + num_classes, 8400)
        output_tensor = np.transpose(output_tensor)  # Shape: (8400, 4 + num_classes)
        
        # Coordinates (cx, cy, w, h)
        boxes = output_tensor[:, :4]
        # Class probabilities
        scores = output_tensor[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter detections by confidence threshold (lowered to 0.15 for early epochs)
        confidence_threshold = 0.15
        mask = confidences > confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = confidences[mask]
        filtered_class_ids = class_ids[mask]
        
        # Apply NMS
        keep_indices = simple_numpy_nms(filtered_boxes, filtered_scores, iou_threshold=0.45)
        
        # Count plants (total detections)
        plant_count = len(keep_indices)
        
        # Draw bounding boxes on the original image using PIL
        draw = ImageDraw.Draw(image)
        
        # Auto-detect if ONNX output coordinates are normalized [0, 1] or absolute [0, 640]
        # If the maximum coordinate in our detections is <= 1.01, we treat them as normalized
        is_normalized = (np.max(filtered_boxes) <= 1.01) if len(filtered_boxes) > 0 else False
        
        # Determine crop health based on class mappings
        # Class 0: Healthy, Class 1: Diseased/Stressed
        healthy_count = 0
        diseased_count = 0
        for idx in keep_indices:
            box = filtered_boxes[idx]
            cid = filtered_class_ids[idx]
            
            if cid == 0:
                healthy_count += 1
            else:
                diseased_count += 1
                
            # Scale coordinates back to original size
            x_center, y_center, w, h = box[0], box[1], box[2], box[3]
            
            if is_normalized:
                x1 = int((x_center - w / 2) * orig_w)
                y1 = int((y_center - h / 2) * orig_h)
                x2 = int((x_center + w / 2) * orig_w)
                y2 = int((y_center + h / 2) * orig_h)
            else:
                x1 = int((x_center - w / 2) * (orig_w / 640))
                y1 = int((y_center - h / 2) * (orig_h / 640))
                x2 = int((x_center + w / 2) * (orig_w / 640))
                y2 = int((y_center + h / 2) * (orig_h / 640))
            
            # Green border for healthy, Red border for diseased
            color = "#00e676" if cid == 0 else "#ff1744"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
        if plant_count == 0:
            diagnosis = "No plants detected in frame. Please adjust distance or camera angle."
        elif diseased_count > 0:
            percentage = int((diseased_count / plant_count) * 100)
            diagnosis = f"Detected {diseased_count} stressed/diseased crops (~{percentage}% of total detected)."
        else:
            diagnosis = "Healthy growth. All detected crop clusters appear healthy and robust."
            
        return plant_count, diagnosis, image

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")
        return 0, f"Error processing model: {str(e)}", None


def run_intervention_recommendation(diagnosis):
    """
    Calls Groq API (Llama-4-Scout or fallback model) to generate an actionable agronomist intervention.
    
    Integration details:
    1. Looks for the API key in st.secrets["GROQ_API_KEY"] or environment variables.
    2. Sends the diagnosis with an agronomy persona prompt to Groq.
    3. Handles lack of credentials gracefully by falling back to simulation and warning.
    
    Returns:
        str: Recommended crop action plan
    """
    # Attempt to load the API Key
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        # st.secrets raises StreamlitSecretNotFoundError if no secrets file exists
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    
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
        # Mock responses based on diagnosis keyword matches
        d_lower = diagnosis.lower()
        if "nitrogen" in d_lower or "chlorosis" in d_lower:
            return "- Apply a nitrogen-rich liquid fertilizer (e.g. urea or organic seaweed extract) at sunset.\n- Test soil moisture levels at root depth.\n- Inspect solar panels above to verify rain runoff isn't causing localized soil leaching."
        elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower:
            return "- Spray organic neem oil or recommended copper-based fungicide to affected canopy rows.\n- Prune diseased lower leaves to prevent spore transmission.\n- Ensure solar panel shading is not trapping excessive humidity overnight."
        elif "dehydration" in d_lower or "wilting" in d_lower:
            return "- Increase drip irrigation duration by 15% during morning hours.\n- Verify solar tracker positions; utilize panel shadows to shade distressed crops during peak midday heat."
        else:
            return "- Continue regular moisture tracking and weekly canopy inspections.\n- Ensure solar panels are clean (dust runoff can alter soil salinity near panel edges)."

    # Standard Prompt configuration (optimized for Pakchoy under solar panels)
    system_prompt = (
        "You are FarmNeura's precision agricultural agent, a professional agronomist specializing in agrivoltaics (crops grown under solar panel arrays) and leafy green brassicas, specifically Pakchoy.\n"
        "Provide a highly actionable, concise list of agronomist recommendations for the farmer based on the provided diagnosis.\n"
        "Take into consideration agrivoltaic environmental factors (e.g., panel rain runoff lines, altered soil shade humidity, and panel-washing runoff pH values).\n"
        "Format your answer as a clean bullet-point list in simple farmer-friendly terms. Keep it under 3-4 bullet points."
    )
    
    # Primary model requested: llama-4-scout. (Fallback model used if it is unavailable on the account)
    primary_model = "llama-4-scout"
    fallback_model = "llama-3.1-8b-instant"
    
    try:
        if HAS_GROQ_SDK:
            client = Groq(api_key=api_key)
            try:
                chat_completion = client.chat.completions.create(
                    model=primary_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Diagnosis: {diagnosis}"}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                return chat_completion.choices[0].message.content
            except Exception:
                # Fallback to standard public model if primary fails
                chat_completion = client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Diagnosis: {diagnosis}"}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                return chat_completion.choices[0].message.content
        else:
            # Fallback to direct requests if Groq Python SDK is not installed
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": primary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Diagnosis: {diagnosis}"}
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                # Try fallback model
                payload["model"] = fallback_model
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"API Error (Status {response.status_code}): {response.text}"
                    
    except Exception as e:
        return f"Intervention retrieval failed: {str(e)}"


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION & SELECTION
# ---------------------------------------------------------------------
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
                
                # Display annotated image if available, else show raw image
                if "temp_diagnosis" in st.session_state and "annotated_image" in st.session_state.temp_diagnosis and st.session_state.temp_diagnosis["annotated_image"] is not None:
                    st.image(
                        st.session_state.temp_diagnosis["annotated_image"], 
                        caption="Annotated Crop Detections (Green = Healthy, Red = Diseased)", 
                        use_container_width=True
                    )
                else:
                    st.image(image_file, caption="Selected Canopy Frame", use_container_width=True)
                
                # Primary Action Button
                if st.button("Diagnose Crop Health", type="primary"):
                    with st.spinner("Processing plant count & detecting abnormalities..."):
                        bil_pokok, diagnosis, annotated_image = run_yolo_count_and_diagnosis(image_file)
                        intervention = run_intervention_recommendation(diagnosis)
                    
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
                        success, msg = db_add_record(
                            selected_plot_obj["id"],
                            time_str,
                            res["bil_pokok"],
                            res["diagnosis"],
                            res["intervention"],
                            notes
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
                    
                    with st.expander(expander_label):
                        st.markdown(f"**🔍 Diagnosis:** {r['diagnosis']}")
                        st.markdown(f"**💡 Recommended Intervention:**\n{r['intervention']}")
                        
                        if r.get("notes"):
                            st.markdown(f"📝 **Field Notes:** *\"{r['notes']}\"*")
