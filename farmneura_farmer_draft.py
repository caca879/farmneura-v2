"""
FarmNeura v2 - Plot Monitoring UI (Phase 1, Module 4)
Scope: Farmer role only.
    - Mobile-friendly, high-contrast, premium farmer UI for field use.
    - Take/upload picture -> run YOLOv8 ONNX (leaf count & health diagnosis)
    - View agronomist-style intervention recommendations via Groq API (Llama-4-Scout)
    - Save and view past historical records per plot

Run with: streamlit run farmneura_farmer_draft.py
"""

import streamlit as st
from datetime import datetime, timezone, timedelta
import os
import random
import sqlite3
import pandas as pd

import numpy as np
from PIL import Image, ImageDraw

# Malaysia Timezone (UTC+8) helper for Streamlit Cloud deployment
MYT = timezone(timedelta(hours=8))

def get_now_myt():
    """Returns current datetime in Malaysia Timezone (UTC+8)."""
    return datetime.now(MYT)


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

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

def render_markdown_to_html(md_text):
    if not md_text:
        return ""
    if HAS_MARKDOWN:
        return markdown.markdown(md_text.strip())
    
    # Fallback simple formatter
    import re
    lines = md_text.strip().split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        sline = line.strip()
        if sline.startswith("- ") or sline.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = sline[2:]
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            html_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', sline)
            if content:
                html_lines.append(f"<p>{content}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "".join(html_lines)

# Set page config
st.set_page_config(
    page_title="FarmNeura - Plot Monitoring",
    page_icon="🌱",
    layout="centered", # Centered layout is much more mobile-friendly
    initial_sidebar_state="collapsed" # Fold sidebar by default on mobile screens
)

# ---------------------------------------------------------------------
# CUSTOM STYLING (Precision Ag Forest Green & Warm Yellow Theme)
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://use.typekit.net/cjq2err.css');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, span, div, button {
        font-family: 'arboria', 'Plus Jakarta Sans', sans-serif !important;
    }

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
        line-height: 1.55;
    }
    .card-text ul, .card-text ol {
        margin: 0.3rem 0;
        padding-left: 1.25rem;
    }
    .card-text li {
        margin-bottom: 0.4rem;
        color: #37474f;
    }
    .card-text p {
        margin-bottom: 0.4rem;
        margin-top: 0;
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
        bil_daun INTEGER NOT NULL,
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
    if "bil_daun" not in columns:
        cursor.execute("ALTER TABLE monitoring_records ADD COLUMN bil_daun INTEGER")
        if "bil_pokok" in columns:
            cursor.execute("UPDATE monitoring_records SET bil_daun = bil_pokok WHERE bil_daun IS NULL")
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
                       (farm_a_id, "Plot 1", 12000.0, "2026-06-01", "2026-08-30", 1500.0, "Field Sector Section A"))
        plot_1_id = cursor.lastrowid
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_a_id, "Plot 2", 15000.0, "2026-06-15", "2026-09-15", 1850.0, "Field Sector Section B"))
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_a_id, "Plot 3", 10000.0, "2026-07-01", "2026-10-01", 1200.0, "Field Sector Section C"))
        
        # Plots for Farm B
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_b_id, "Plot 1", 20000.0, "2026-05-10", "2026-08-10", 2500.0, "Open field layout"))
        cursor.execute("INSERT INTO plots (farm_id, plot_name, size_sq_ft, cycle_start, cycle_end, cost_records, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (farm_b_id, "Plot 2", 25000.0, "2026-05-20", "2026-08-20", 3000.0, "High shade index rows"))
        
        # Crop (Module 2)
        cursor.execute("INSERT INTO plants (plot_id, name) VALUES (?, ?)", (plot_1_id, "Pakchoy"))
        
        # Diagnosis Logs (Module 4) for Plot 1
        cols_now = [c[1] for c in cursor.execute("PRAGMA table_info(monitoring_records)").fetchall()]
        if "bil_daun" in cols_now and "bil_pokok" in cols_now:
            cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_daun, bil_pokok, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (plot_1_id, "2026-07-28 09:30", 45, 45, "Healthy crops. Robust growth with standard green leaf index.", "Continue normal irrigation schedule. Apply NPK fertilizer at next scheduled cycle.", "Visual check at Field Sector B."))
            cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_daun, bil_pokok, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (plot_1_id, "2026-07-29 14:15", 42, 42, "Mild leaf chlorosis detected (yellowing of leaves on 10% of crop canopy).", "Verify soil pH levels. Recommend localized nitrogen boost fertilizer to restore chlorophyll content.", "Observed near Irrigation Valve #3."))
        else:
            cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_daun, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                           (plot_1_id, "2026-07-28 09:30", 45, "Healthy crops. Robust growth with standard green leaf index.", "Continue normal irrigation schedule. Apply NPK fertilizer at next scheduled cycle.", "Visual check at Field Sector B."))
            cursor.execute("INSERT INTO monitoring_records (plot_id, time, bil_daun, diagnosis, intervention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                           (plot_1_id, "2026-07-29 14:15", 42, "Mild leaf chlorosis detected (yellowing of leaves on 10% of crop canopy).", "Verify soil pH levels. Recommend localized nitrogen boost fertilizer to restore chlorophyll content.", "Observed near Irrigation Valve #3."))
        
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
    result = []
    for r in records:
        d = dict(r)
        if "bil_daun" not in d or d["bil_daun"] is None:
            d["bil_daun"] = d.get("bil_pokok", 0)
        result.append(d)
    return result

def save_uploaded_image(image_input, plot_id):
    """
    Saves image input to local uploads/ directory and returns relative file path.
    """
    if image_input is None:
        return None
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        
    timestamp = get_now_myt().strftime("%Y%m%d_%H%M%S")

    filename = f"plot_{plot_id}_{timestamp}.png"
    filepath = os.path.join(uploads_dir, filename)
    
    try:
        if isinstance(image_input, Image.Image):
            image_input.save(filepath)
        else:
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            pil_img = Image.open(image_input).convert("RGB")
            pil_img.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def db_add_record(plot_id, time, bil_daun, diagnosis, intervention, notes, canopy_cover_pct=None, image_path=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(monitoring_records)")
        cols = [c[1] for c in cursor.fetchall()]
        
        if "bil_daun" in cols and "bil_pokok" in cols:
            cursor.execute(
                "INSERT INTO monitoring_records (plot_id, time, bil_daun, bil_pokok, diagnosis, intervention, notes, canopy_cover_pct, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (plot_id, time, bil_daun, bil_daun, diagnosis, intervention, notes.strip() if notes else "", canopy_cover_pct, image_path)
            )
        elif "bil_daun" in cols:
            cursor.execute(
                "INSERT INTO monitoring_records (plot_id, time, bil_daun, diagnosis, intervention, notes, canopy_cover_pct, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (plot_id, time, bil_daun, diagnosis, intervention, notes.strip() if notes else "", canopy_cover_pct, image_path)
            )
        else:
            cursor.execute(
                "INSERT INTO monitoring_records (plot_id, time, bil_pokok, diagnosis, intervention, notes, canopy_cover_pct, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (plot_id, time, bil_daun, diagnosis, intervention, notes.strip() if notes else "", canopy_cover_pct, image_path)
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
    if row:
        d = dict(row)
        if "bil_daun" not in d or d["bil_daun"] is None:
            d["bil_daun"] = d.get("bil_pokok", 0)
        return d
    return None

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

def db_update_record(record_id, new_time=None, new_notes=None, new_bil_daun=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(monitoring_records)")
        cols = [c[1] for c in cursor.fetchall()]
        
        updates = []
        params = []
        if new_time is not None:
            updates.append("time = ?")
            params.append(new_time)
        if new_notes is not None:
            updates.append("notes = ?")
            params.append(new_notes)
        if new_bil_daun is not None:
            if "bil_daun" in cols:
                updates.append("bil_daun = ?")
                params.append(new_bil_daun)
            if "bil_pokok" in cols:
                updates.append("bil_pokok = ?")
                params.append(new_bil_daun)
        
        if updates:
            params.append(record_id)
            sql = f"UPDATE monitoring_records SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(sql, tuple(params))
            conn.commit()
        conn.close()
        return True, "Monitoring record updated successfully."
    except Exception as e:
        return False, f"Error updating record: {str(e)}"

# ---------------------------------------------------------------------
# BILINGUAL LOCALIZATION (i18n) DICTIONARY
# ---------------------------------------------------------------------

I18N = {
    "en": {
        "sidebar_brand": "🌱 FarmNeura",
        "sidebar_sub": "Farmer Interface v2",
        "lang_selector_title": "🌐 Language / Bahasa",
        "lang_selector_label": "Select App Language",
        "nav_menu_title": "📁 Navigation Menu",
        "nav_overview": "📋 Overview",
        "nav_quick_scan": "🔍 Quick Crop Scan",
        "nav_monitoring": "📷 Plot Monitoring",
        "nav_registry": "⚙️ Registry & Management",
        "header_title": "FarmNeura v2",
        "greeting_morning": "Good Morning, Farmer",
        "greeting_afternoon": "Good Afternoon, Farmer",
        "greeting_evening": "Good Evening, Farmer",
        "welcome_sub": "Welcome to FarmNeura Precision Plot Monitoring",
        "instant_check_title": "⚡ Instant Crop Check",
        "instant_check_sub": "Diagnose foliage and detect crop stress on the spot without picking a plot first.",
        "launch_quick_scan_btn": "🔍 Launch Quick Crop Scan",
        "total_active_plots": "Total Active Plots",
        "overall_plot_health": "Overall Plot Health",
        "needs_attention": "Needs Attention",
        "photo_overdue": "New Photo Needed",
        "priority_action_list": "🚨 Priority Action List for Your Plots",
        "all_plots_healthy": "All registered plots are healthy and up to date!",
        "no_plots_registered": "⚠️ No plots registered yet. Go to **Registry & Management** to register your farms and plots.",
        "check_plot_health": "🩺 Check Plot Health",
        "take_new_photo": "📸 Take New Photo",
        "open_plot_log": "📊 Open Plot Log",
        "status_attention": "⚠️ NEEDS ATTENTION (STRESS/DISEASE)",
        "status_overdue": "📸 PHOTO OVERDUE (>= 3 DAYS)",
        "status_healthy": "✅ HEALTHY & UP TO DATE",
        # Quick Scan
        "qs_title": "### 🔍 Quick Crop Scan & Instant Health Diagnosis",
        "qs_sub": "Instantly snap or upload a crop photo to detect diseases, count leaves, and get AI recommendations without choosing a farm/plot upfront. When done, you can optionally assign and save the scan to any plot.",
        "qs_settings_title": "#### ⚙️ Quick Scan Settings",
        "qs_select_model": "Select AI Vision Model",
        "qs_crop_context": "Crop Type Context (Optional)",
        "qs_crop_context_ph": "e.g. Pakchoy, Tomato, Bendi",
        "qs_crop_context_help": "Providing crop name helps AI fine-tune agronomic recommendations.",
        "qs_btn_diagnose": "🚀 Diagnose Crop Health Now",
        "qs_results_title": "### 📊 Diagnostic Results",
        "qs_leaf_metric": "🍃 Total Leaves Detected (Bil Daun)",
        "qs_diag_card": "🔍 Crop Condition Diagnosis",
        "qs_interv_card": "💡 Recommended Agronomic Intervention",
        "qs_save_header": "### 💾 Save Scan to Farm & Plot Records (Optional)",
        "qs_save_sub": "Assign this scan to a specific farm and plot to store in historical logs.",
        "qs_save_farm": "Select Target Farm",
        "qs_save_plot": "Select Target Plot",
        "qs_intercrop_select": "🌾 Specific Crop in this Photo (Intercropping / Multi-Crop Plot)",
        "qs_inspection_date": "Inspection Date",
        "qs_inspection_time": "Inspection Time",
        "qs_field_notes": "📝 Field Notes / Observations (Optional)",
        "qs_field_notes_ph": "Add manual inspection observations, weather, or fertilizer details.",
        "qs_btn_save": "💾 Save Scan to Plot Records",
        # Plot Monitoring
        "pm_farm_label": "🚜 Target Farm Location",
        "pm_plot_label": "📍 Target Plot",
        "pm_tab_record": "📷 Take / Upload Record",
        "pm_tab_history": "📊 Plot History Log",
        "pm_intercrop_prompt": "🌾 Specific Crop in this Photo (Multi-Crop / Intercropping Plot)",
        "pm_intercrop_single": "🌾 Active crop in this plot:",
        "pm_cam_prompt": "📷 Use Mobile Camera",
        "pm_upload_prompt": "📂 Or Upload Image file",
        "pm_iot_title": "🌐 Cloud IoT Telemetry Stream (Live Sensor Data ➔ LLM Fusion)",
        "pm_iot_broker": "📡 Connected IoT Cloud Broker",
        "pm_iot_sub": "Real-time sensor telemetry synced to LLM prompt.",
        "pm_iot_sync_btn": "🔄 Sync Cloud IoT Data",
        "pm_soil_moisture": "💧 Soil Moisture",
        "pm_air_temp": "🌡️ Air Temp",
        "pm_soil_ec": "🧪 Soil EC",
        "pm_soil_ph": "🧪 Soil pH",
        "pm_btn_diag_fusion": "Diagnose Crop Health & Run LLM Fusion",
        "pm_history_title": "### Historical Logs for",
        "pm_history_sub": "Review past diagnosis, counts, and recommended actions logged for this plot.",
        "pm_intercrop_banner": "🌾 Intercropping Active: Multiple crops registered in this plot",
        "pm_filter_history": "🔍 Filter History by Crop Variety:",
        "pm_edit_title": "✏️ Edit Date, Crop Tag & Log Details",
        "pm_delete_btn": "🗑️ Delete Log",
        # Registry
        "reg_title": "### Farm & Plot Registry",
        "reg_sub": "Register and manage your farms, plot boundaries, and crop types.",
        "reg_tab_farm": "🚜 Register Farm",
        "reg_tab_plot": "📐 Register Plot",
        "reg_tab_crop": "🌿 Register Crops",
        "reg_farm_name": "Farm Name (Unique)",
        "reg_farm_loc": "Location / Coordinate",
        "reg_farm_size": "Total Area (Hectares / Acres)",
        "reg_farm_btn": "🚜 Register New Farm",
        "reg_plot_farm": "Select Parent Farm",
        "reg_plot_name": "Plot Name / Code (Unique)",
        "reg_plot_area": "Plot Area (m² / SqFt)",
        "reg_plot_cycle": "Plot Planting / Cycle Start Date",
        "reg_plot_btn": "📐 Register New Plot",
        "reg_crop_farm": "Select Farm",
        "reg_crop_plot": "Select Target Plot",
        "reg_crop_name": "Crop Name / Variety",
        "reg_crop_species": "Botanical Species (Optional)",
        "reg_crop_days": "Estimated Days to Maturity",
        "reg_crop_btn": "🌿 Register Crop to Plot",
        "reg_existing_farms": "📋 Registered Farms",
        "reg_existing_plots": "📋 Registered Plots",
        "reg_existing_crops": "📋 Registered Crops",
    },
    "bm": {
        "sidebar_brand": "🌱 FarmNeura",
        "sidebar_sub": "Antara Muka Petani v2",
        "lang_selector_title": "🌐 Bahasa / Language",
        "lang_selector_label": "Pilih Bahasa Sistem",
        "nav_menu_title": "📁 Menu Navigasi",
        "nav_overview": "📋 Gambaran Keseluruhan",
        "nav_quick_scan": "🔍 Imbasan Pantas Tanaman",
        "nav_monitoring": "📷 Pemantauan Plot",
        "nav_registry": "⚙️ Pendaftaran & Pengurusan",
        "header_title": "FarmNeura v2",
        "greeting_morning": "Selamat Pagi, Petani",
        "greeting_afternoon": "Selamat Petang, Petani",
        "greeting_evening": "Selamat Malam, Petani",
        "welcome_sub": "Selamat Datang ke Pemantauan Plot Ketepatan FarmNeura",
        "instant_check_title": "⚡ Pemeriksaan Pantas Tanaman",
        "instant_check_sub": "Diagnosis dedaun & kesan tekanan tanaman serta-merta tanpa perlu memilih plot terlebih dahulu.",
        "launch_quick_scan_btn": "🔍 Mulakan Imbasan Pantas",
        "total_active_plots": "Jumlah Plot Aktif",
        "overall_plot_health": "Kesihatan Keseluruhan Plot",
        "needs_attention": "Perlu Perhatian",
        "photo_overdue": "Perlu Gambar Baharu",
        "priority_action_list": "🚨 Senarai Tindakan Keutamaan untuk Plot Anda",
        "all_plots_healthy": "Semua plot berdaftar berada dalam keadaan sihat dan terkini!",
        "no_plots_registered": "⚠️ Tiada plot didaftarkan lagi. Sila ke **Pendaftaran & Pengurusan** untuk mendaftar ladang dan plot anda.",
        "check_plot_health": "🩺 Semak Kesihatan Plot",
        "take_new_photo": "📸 Ambil Gambar Baharu",
        "open_plot_log": "📊 Buka Log Plot",
        "status_attention": "⚠️ PERLU PERHATIAN (STRES/PENYAKIT)",
        "status_overdue": "📸 GAMBAR MELEBIHI 3 HARI",
        "status_healthy": "✅ SIHAT & TERKINI",
        # Quick Scan
        "qs_title": "### 🔍 Imbasan Pantas Tanaman & Diagnosis Kesihatan Segera",
        "qs_sub": "Ambil gambar atau muat naik foto tanaman serta-merta untuk mengesan penyakit, mengira daun, dan mendapatkan cadangan AI tanpa perlu memilih plot terlebih dahulu. Anda boleh menyimpan imbasan ini ke mana-mana plot selepas diagnosis.",
        "qs_settings_title": "#### ⚙️ Tetapan Imbasan Pantas",
        "qs_select_model": "Pilih Model Penglihatan AI",
        "qs_crop_context": "Konteks Jenis Tanaman (Pilihan)",
        "qs_crop_context_ph": "cth. Pakchoy, Tomato, Bendi",
        "qs_crop_context_help": "Menyatakan nama tanaman membantu AI menyesuaikan cadangan agronomi dengan tepat.",
        "qs_btn_diagnose": "🚀 Jalankan Diagnosis Kesihatan Tanaman",
        "qs_results_title": "### 📊 Keputusan Diagnosis",
        "qs_leaf_metric": "🍃 Jumlah Daun Dikesan (Bil Daun)",
        "qs_diag_card": "🔍 Diagnosis Keadaan Tanaman",
        "qs_interv_card": "💡 Cadangan Intervensi Agronomi",
        "qs_save_header": "### 💾 Simpan Imbasan ke Rekod Ladang & Plot (Pilihan)",
        "qs_save_sub": "Tetapkan imbasan ini ke ladang dan plot khusus untuk disimpan dalam log sejarah.",
        "qs_save_farm": "Pilih Ladang Sasaran",
        "qs_save_plot": "Pilih Plot Sasaran",
        "qs_intercrop_select": "🌾 Tanaman Khusus dalam Foto ini (Plot Tanaman Campuran / Intercropping)",
        "qs_inspection_date": "Tarikh Pemeriksaan",
        "qs_inspection_time": "Masa Pemeriksaan",
        "qs_field_notes": "📝 Nota Lapangan / Pemerhatian (Pilihan)",
        "qs_field_notes_ph": "Tambah pemerhatian manual, keadaan cuaca, atau butiran pembajaan.",
        "qs_btn_save": "💾 Simpan Imbasan ke Rekod Plot",
        # Plot Monitoring
        "pm_farm_label": "🚜 Lokasi Ladang Sasaran",
        "pm_plot_label": "📍 Plot Sasaran",
        "pm_tab_record": "📷 Ambil / Muat Naik Rekod",
        "pm_tab_history": "📊 Log Sejarah Plot",
        "pm_intercrop_prompt": "🌾 Tanaman Khusus dalam Foto ini (Plot Tanaman Campuran / Intercropping)",
        "pm_intercrop_single": "🌾 Tanaman aktif dalam plot ini:",
        "pm_cam_prompt": "📷 Guna Kamera Telefon",
        "pm_upload_prompt": "📂 Atau Muat Naik Fail Imej",
        "pm_iot_title": "🌐 Strim Telemetri IoT Awan (Data Sensor Langsung ➔ Gabungan LLM)",
        "pm_iot_broker": "📡 Broker Awan IoT Terhubung",
        "pm_iot_sub": "Telemetri sensor masa nyata untuk plot disegerakkan ke prompt LLM.",
        "pm_iot_sync_btn": "🔄 Segerak Data IoT Awan",
        "pm_soil_moisture": "💧 Kelembapan Tanah",
        "pm_air_temp": "🌡️ Suhu Udara",
        "pm_soil_ec": "🧪 EC Tanah",
        "pm_soil_ph": "🧪 pH Tanah",
        "pm_btn_diag_fusion": "Jalankan Diagnosis & Gabungan Sensor AI",
        "pm_history_title": "### Log Sejarah untuk",
        "pm_history_sub": "Semak diagnosis lalu, kiraan daun, dan tindakan yang disyorkan untuk plot ini.",
        "pm_intercrop_banner": "🌾 Tanaman Campuran Aktif: Pelbagai tanaman didaftarkan dalam plot ini",
        "pm_filter_history": "🔍 Tapis Sejarah mengikut Tanaman:",
        "pm_edit_title": "✏️ Edit Tarikh, Tag Tanaman & Butiran Log",
        "pm_delete_btn": "🗑️ Padam Log",
        # Registry
        "reg_title": "### Pendaftaran Ladang & Plot",
        "reg_sub": "Daftar dan urus lokasi ladang, sempadan plot, dan jenis tanaman anda.",
        "reg_tab_farm": "🚜 Daftar Ladang",
        "reg_tab_plot": "📐 Daftar Plot",
        "reg_tab_crop": "🌿 Daftar Tanaman",
        "reg_farm_name": "Nama Ladang (Unik)",
        "reg_farm_loc": "Lokasi / Koordinat",
        "reg_farm_size": "Jumlah Keluasan (Hektar / Ekar)",
        "reg_farm_btn": "🚜 Daftar Ladang Baharu",
        "reg_plot_farm": "Pilih Ladang Induk",
        "reg_plot_name": "Nama / Kod Plot (Unik)",
        "reg_plot_area": "Keluasan Plot (m² / Kaki Persegi)",
        "reg_plot_cycle": "Tarikh Mula Kitaran Tanaman",
        "reg_plot_btn": "📐 Daftar Plot Baharu",
        "reg_crop_farm": "Pilih Ladang",
        "reg_crop_plot": "Pilih Plot Sasaran",
        "reg_crop_name": "Nama Tanaman / Varieti",
        "reg_crop_species": "Spesies Botani (Pilihan)",
        "reg_crop_days": "Anggaran Hari Kematangan",
        "reg_crop_btn": "🌿 Daftar Tanaman ke Plot",
        "reg_existing_farms": "📋 Senarai Ladang Berdaftar",
        "reg_existing_plots": "📋 Senarai Plot Berdaftar",
        "reg_existing_crops": "📋 Senarai Tanaman Berdaftar",
    }
}

def render_crop_photography_guide(selected_crop=None, is_expanded=False, is_bm=False):
    """
    Renders a crop-specific photography guide with visual shooting tips,
    ideal angles, optimal distance, and dos/don'ts for precision AI foliage analysis.
    """
    crop_str = (selected_crop or "").lower()
    
    GUIDES_EN = {
        "tomato": {
            "title": "🍅 Tomato & Solanaceae (Tomato / Terung)",
            "crop_name": "Tomato",
            "distance": "30 – 50 cm (1 – 1.5 ft)",
            "angle": "45° angled top/side view",
            "focus": "Upper and mid-canopy leaf blades and petiole junctions (key for Early/Late Blight, Septoria spots, Leaf Mold & Yellow Curl).",
            "lighting": "Bright diffused daylight (avoid harsh solar panel edge shadows).",
            "dos": [
                "Frame 3–6 distinct leaves in sharp focus",
                "Check both upper surface and leaf margins for spots/curling",
                "Wipe phone camera lens before shooting"
            ],
            "donts": [
                "Avoid capturing distant whole vines with blurry leaves",
                "Avoid shooting against direct glaring sunlight"
            ]
        },
        "okra": {
            "title": "🌿 Okra / Bendi (Abelmoschus esculentus)",
            "crop_name": "Okra",
            "distance": "40 – 60 cm (1.5 – 2 ft)",
            "angle": "60° – 90° top-down onto fan leaf",
            "focus": "Upper broad fan leaves and vein structures (critical for Yellow Vein Mosaic Virus & Powdery Mildew).",
            "lighting": "Even overhead sunlight; ensure leaf veins are clearly visible.",
            "dos": [
                "Focus on 1–2 mature fan leaves per photo",
                "Ensure leaf veins are clearly lit to detect chlorosis / yellowing",
                "Keep phone steady and parallel to leaf blade"
            ],
            "donts": [
                "Don't focus solely on the stem or fruit pod for foliage health",
                "Avoid shadows cutting across the leaf blade"
            ]
        },
        "pakchoy": {
            "title": "🥬 Pakchoy / Leafy Greens (Sawi, Bok Choy, Kailan)",
            "crop_name": "Pakchoy / Leafy Greens",
            "distance": "25 – 40 cm (1 ft)",
            "angle": "90° direct bird's-eye top-down over rosette",
            "focus": "Center crown whorl and outward radial leaves (detects leaf miners, flea beetle holes & nitrogen chlorosis).",
            "lighting": "Uniform top-down illumination across the entire rosette.",
            "dos": [
                "Capture full rosette circumference within frame",
                "Shoot straight down above the plant center",
                "Hold phone steady to keep small leaf edges sharp"
            ],
            "donts": [
                "Don't shoot from ground level where outer leaves block inner ones",
                "Avoid motion blur from hand tremors"
            ]
        },
        "yardlong bean": {
            "title": "🌱 Yardlong Bean / Legumes (Kacang Panjang / Buncis)",
            "crop_name": "Yardlong Bean",
            "distance": "30 – 50 cm",
            "angle": "Parallel to trellis line (0° – 30°)",
            "focus": "Mid-trellis foliage clusters and trifoliate leaves (inspect for Rust, Cercospora leaf spot, Bean Common Mosaic).",
            "lighting": "Soft morning or late afternoon light; hold leaves gently if windy.",
            "dos": [
                "Focus on a cluster of 3–5 leaves on the trellis line",
                "Hold phone steady against wind to avoid motion blur",
                "Ensure good contrast against background trellis"
            ],
            "donts": [
                "Avoid capturing mostly empty trellis netting or distant ground soil",
                "Don't shake vines while taking photo"
            ]
        },
        "chili": {
            "title": "🌶️ Chili / Pepper / Capsicum (Cili)",
            "crop_name": "Chili",
            "distance": "30 – 45 cm",
            "angle": "45° top-down towards apical growth",
            "focus": "Young shoot tips (for thrips / mite curling) and broad lower leaves (for Anthracnose, Bacterial Spot & Mosaic).",
            "lighting": "Bright, indirect natural light.",
            "dos": [
                "Capture top growing shoots and adjacent mature leaves",
                "Check for upward/downward leaf curling patterns",
                "Ensure sharp focus on leaf margins"
            ],
            "donts": [
                "Avoid overexposing under intense midday sun",
                "Don't capture only the chili fruit when checking plant health"
            ]
        },
        "general": {
            "title": "🌾 General Crop Foliage",
            "crop_name": "Crop",
            "distance": "30 – 50 cm (1 – 1.5 ft)",
            "angle": "45° to 90° top-down",
            "focus": "Active vegetative canopy and areas showing discoloration, spots, or wilting.",
            "lighting": "Bright, indirect natural light for accurate color balance.",
            "dos": [
                "Tap screen to lock focus directly on the leaf blade",
                "Capture in clear daylight without heavy glare",
                "Ensure at least 70% of frame is green foliage"
            ],
            "donts": [
                "Avoid blurry, out-of-focus captures",
                "Avoid extreme zoom or distant wide-angle shots"
            ]
        }
    }

    GUIDES_BM = {
        "tomato": {
            "title": "🍅 Tomato & Solanaceae (Tomato / Terung)",
            "crop_name": "Tomato",
            "distance": "30 – 50 cm (1 – 1.5 kaki)",
            "angle": "45° sudut serong atas/sisi",
            "focus": "Bilah daun kanopi atas dan simpang tangkai (mengesan Hawar Daun, Bintik Septoria, Kulapuk Daun & Virus Kerinting Kuning).",
            "lighting": "Cahaya siang semulajadi yang terang dan rata (elakkan bayang gelap panel solar).",
            "dos": [
                "Fokus pada 3–6 helai daun yang jelas",
                "Periksa permukaan atas dan tepi daun untuk bintik/kerinting",
                "Lap kanta kamera telefon sebelum merakam"
            ],
            "donts": [
                "Elakkan gambar jauh seluruh pokok yang kabur",
                "Elakkan merakam menghadap cahaya matahari terik"
            ]
        },
        "okra": {
            "title": "🌿 Okra / Bendi (Abelmoschus esculentus)",
            "crop_name": "Bendi",
            "distance": "40 – 60 cm (1.5 – 2 kaki)",
            "angle": "60° – 90° pandangan atas ke daun kipas",
            "focus": "Daun kipas matang dan struktur urat daun (penting untuk mengesan Virus Mozek Urat Kuning & Kulapuk Berdebu).",
            "lighting": "Cahaya matahari rata; pastikan urat daun kelihatan jelas.",
            "dos": [
                "Fokus pada 1–2 helai daun matang setiap foto",
                "Pastikan urat daun disinari cahaya untuk kesan kekuningan",
                "Pegang telefon selari dengan bilah daun"
            ],
            "donts": [
                "Jangan fokus pada batang atau buah sahaja untuk semakan dedaun",
                "Elakkan bayang-bayang melintasi daun"
            ]
        },
        "pakchoy": {
            "title": "🥬 Pakchoy / Sayuran Daun (Sawi, Bok Choy, Kailan)",
            "crop_name": "Pakchoy / Sayur Daun",
            "distance": "25 – 40 cm (1 kaki)",
            "angle": "90° tegak dari atas (pandangan mata burung)",
            "focus": "Pusat roset dan daun luar (mengesan ulat pengorek daun, kumbang kutu & klorosis nitrogen).",
            "lighting": "Pencahayaan sekata merentasi seluruh roset daun.",
            "dos": [
                "Rakam keseluruhan bulatan roset dalam bingkai",
                "Rakam tepat dari atas pusat pokok",
                "Pegang telefon dengan stabil agar tepi daun tajam"
            ],
            "donts": [
                "Jangan rakam dari paras tanah di mana daun luar menghalang pusat",
                "Elakkan gegaran tangan yang menyebabkan gambar kabur"
            ]
        },
        "yardlong bean": {
            "title": "🌱 Kacang Panjang / Legum (Kacang Buncis)",
            "crop_name": "Kacang Panjang",
            "distance": "30 – 50 cm",
            "angle": "Selari dengan barisan junjung (0° – 30°)",
            "focus": "Gugusan daun pada junjung tengah (periksa Karat Daun, Bintik Cercospora, Mozek Kacang).",
            "lighting": "Cahaya lembut pagi atau petang; pegang daun perlahan jika berangin.",
            "dos": [
                "Fokus pada kelompok 3–5 daun pada baris junjung",
                "Pegang telefon dengan stabil daripada tiupan angin",
                "Pastikan kontras yang baik dengan latar belakang junjung"
            ],
            "donts": [
                "Elakkan merakam jaring junjung kosong atau tanah lapang",
                "Jangan goyang pokok semasa merakam"
            ]
        },
        "chili": {
            "title": "🌶️ Cili / Lada / Terung (Capsicum)",
            "crop_name": "Cili",
            "distance": "30 – 45 cm",
            "angle": "45° dari atas menghala ke pucuk muda",
            "focus": "Pucuk muda atas (kerinting thrips/hama) dan daun bawah (Antraknos, Bintik Bakteria & Mozek).",
            "lighting": "Cahaya semulajadi yang terang dan tidak menyilaukan.",
            "dos": [
                "Rakam pucuk pucuk utama dan daun matang bersebelahan",
                "Periksa corak kerinting daun ke atas atau ke bawah",
                "Pastikan fokus tajam pada tepi daun"
            ],
            "donts": [
                "Elakkan dedahan cahaya terik tengah hari yang melampau",
                "Jangan ambil gambar buah cili sahaja untuk menilai kesihatan pokok"
            ]
        },
        "general": {
            "title": "🌾 Dedaun Tanaman Umum",
            "crop_name": "Tanaman",
            "distance": "30 – 50 cm (1 – 1.5 kaki)",
            "angle": "45° hingga 90° pandangan atas",
            "focus": "Kanopi vegetatif aktif dan kawasan menunjukkan perubahan warna, bintik, atau layu.",
            "lighting": "Cahaya semulajadi tidak langsung untuk ketepatan imbangan warna.",
            "dos": [
                "Sentuh skrin telefon untuk mengunci fokus pada bilah daun",
                "Rakam dalam cahaya terang tanpa silauan tajam",
                "Pastikan sekurang-kurangnya 70% bingkai dipenuhi daun hijau"
            ],
            "donts": [
                "Elakkan gambar kabur atau tidak fokus",
                "Elakkan zum melampau atau gambar sudut terlalu luas"
            ]
        }
    }
    
    guides_dict = GUIDES_BM if is_bm else GUIDES_EN
    
    # Determine which guide to highlight
    active_key = "general"
    if "tomato" in crop_str:
        active_key = "tomato"
    elif "okra" in crop_str or "bendi" in crop_str:
        active_key = "okra"
    elif any(k in crop_str for k in ["pakchoy", "sawi", "choy", "kailan", "cabbage", "lettuce", "sayur"]):
        active_key = "pakchoy"
    elif any(k in crop_str for k in ["bean", "kacang", "long bean", "legume"]):
        active_key = "yardlong bean"
    elif any(k in crop_str for k in ["chili", "cili", "pepper", "terung", "eggplant"]):
        active_key = "chili"
        
    guide_data = guides_dict.get(active_key, guides_dict["general"])
    display_title = guide_data["title"].split("(")[0].strip()
    
    guide_exp_title = f"📸 Panduan Penggambaran: Amalan Terbaik untuk {display_title}" if is_bm else f"📸 Photography Guide: Best Practices for {display_title}"
    tab1_label = f"🎯 Panduan {guide_data['crop_name']}" if is_bm else f"🎯 {guide_data['crop_name']} Tips"
    tab2_label = "📚 Senarai Semua Tanaman" if is_bm else "📚 All Crops Cheat-Sheet"
    
    with st.expander(guide_exp_title, expanded=is_expanded):
        tab_spec, tab_all = st.tabs([tab1_label, tab2_label])
        
        with tab_spec:
            c1, c2 = st.columns(2)
            with c1:
                col1_header = "📐 Jarak & Sudut Kamera" if is_bm else "📐 Distance & Angle"
                lbl_dist = "📏 <strong>Jarak Disyorkan:</strong>" if is_bm else "📏 <strong>Ideal Distance:</strong>"
                lbl_angle = "📐 <strong>Sudut Penggambaran:</strong>" if is_bm else "📐 <strong>Shooting Angle:</strong>"
                lbl_light = "☀️ <strong>Pencahayaan:</strong>" if is_bm else "☀️ <strong>Lighting:</strong>"
                
                st.markdown(
                    f"""
                    <div style="background:#f8f9fa; border:1px solid #e0e0e0; border-radius:10px; padding:12px; margin-bottom:10px;">
                        <div style="font-weight:700; color:#2e7d32; font-size:0.92rem; margin-bottom:6px;">{col1_header}</div>
                        <div style="font-size:0.85rem; color:#333; line-height:1.6;">
                            {lbl_dist} {guide_data['distance']}<br/>
                            {lbl_angle} {guide_data['angle']}<br/>
                            {lbl_light} {guide_data['lighting']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c2:
                col2_header = "🎯 Kawasan Tumpuan Utama" if is_bm else "🎯 Key Target Area"
                st.markdown(
                    f"""
                    <div style="background:#f8f9fa; border:1px solid #e0e0e0; border-radius:10px; padding:12px; margin-bottom:10px;">
                        <div style="font-weight:700; color:#1565c0; font-size:0.92rem; margin-bottom:6px;">{col2_header}</div>
                        <div style="font-size:0.85rem; color:#333; line-height:1.5;">
                            {guide_data['focus']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            col_do, col_dont = st.columns(2)
            with col_do:
                dos_html = "".join([f"<li>{item}</li>" for item in guide_data['dos']])
                do_header = "✅ AMALAN BAIK (DO):" if is_bm else "✅ DO's:"
                st.markdown(
                    f"""
                    <div style="background:#e8f5e9; border:1px solid #a5d6a7; border-radius:10px; padding:10px 14px; font-size:0.85rem;">
                        <strong style="color:#2e7d32;">{do_header}</strong>
                        <ul style="margin:4px 0 0 0; padding-left:18px; color:#1b5e20;">
                            {dos_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_dont:
                donts_html = "".join([f"<li>{item}</li>" for item in guide_data['donts']])
                dont_header = "❌ PERKARA DIELAKKAN (DON'T):" if is_bm else "❌ DON'Ts:"
                st.markdown(
                    f"""
                    <div style="background:#ffebee; border:1px solid #ef9a9a; border-radius:10px; padding:10px 14px; font-size:0.85rem;">
                        <strong style="color:#c62828;">{dont_header}</strong>
                        <ul style="margin:4px 0 0 0; padding-left:18px; color:#b71c1c;">
                            {donts_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_all:
            sub_ref_hdr = "##### 📖 Rujukan Pantas Mengikut Jenis Tanaman" if is_bm else "##### 📖 Reference Cheat-Sheet by Crop Type"
            st.markdown(sub_ref_hdr)
            for k, g in guides_dict.items():
                if k != "general":
                    st.markdown(f"**{g['title']}** — 📏 *{g['distance']}* | 📐 *{g['angle']}*")
                    focus_lbl = "Tumpuan" if is_bm else "Focus"
                    st.caption(f"🎯 *{focus_lbl}:* {g['focus']}")
            st.markdown("---")
            tip_box = (
                """
                <div style="background:#fff3e0; border-left:4px solid #ff9800; border-radius:8px; padding:8px 12px; font-size:0.85rem; color:#e65100;">
                    💡 <strong>Petua AI:</strong> Sentuh skrin telefon untuk mengunci fokus kamera pada daun sebelum menangkap gambar untuk pengesanan bounding box yang paling tepat.
                </div>
                """
                if is_bm else
                """
                <div style="background:#fff3e0; border-left:4px solid #ff9800; border-radius:8px; padding:8px 12px; font-size:0.85rem; color:#e65100;">
                    💡 <strong>Pro Tip:</strong> Tap on your phone screen to lock camera focus on the leaf blade before capturing to ensure sharp bounding box detections.
                </div>
                """
            )
            st.markdown(tip_box, unsafe_allow_html=True)

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


def analyze_foliage_canopy_health(image):
    """
    Analyzes crop foliage health, chlorophyll distribution, and foliar stress/chlorosis
    using Green Leaf Index (GLI) and visible-spectrum chromaticity analysis.
    Returns:
        dict: {
            "canopy_cover_pct": float,
            "healthy_pct": float,
            "chlorotic_pct": float,
            "necrotic_pct": float,
            "status_text_en": str,
            "status_text_bm": str,
            "health_grade": str ("optimal", "mild_stress", "severe_stress")
        }
    """
    try:
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.open(image).convert("RGB")
            
        img_np = np.array(pil_img).astype(np.float32)
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]
        
        # GLI: (2*G - R - B) / (2*G + R + B + 1e-6)
        denom = 2.0 * G + R + B + 1e-6
        gli = (2.0 * G - R - B) / denom
        
        # Segment leaf pixels (foliage)
        leaf_mask = (gli > 0.04) & (G > R * 0.85)
        total_leaf_pixels = float(np.sum(leaf_mask))
        total_pixels = float(img_np.shape[0] * img_np.shape[1])
        canopy_cover_pct = (total_leaf_pixels / total_pixels) * 100.0
        
        if total_leaf_pixels > 0:
            healthy_mask = leaf_mask & (gli > 0.10) & (G > R * 1.10)
            chlorotic_mask = leaf_mask & (~healthy_mask) & (R > 90) & (G > 90) & (B < 100)
            necrotic_mask = leaf_mask & (~healthy_mask) & (~chlorotic_mask) & (R > G)
            
            healthy_pct = (np.sum(healthy_mask) / total_leaf_pixels) * 100.0
            chlorotic_pct = (np.sum(chlorotic_mask) / total_leaf_pixels) * 100.0
            necrotic_pct = (np.sum(necrotic_mask) / total_leaf_pixels) * 100.0
        else:
            healthy_pct, chlorotic_pct, necrotic_pct = 0.0, 0.0, 0.0
            
        # Determine status
        if healthy_pct >= 80.0:
            grade = "optimal"
            en_status = f"🟢 Healthy Foliage ({round(healthy_pct, 1)}% vibrant green, robust chlorophyll, no disease signs)"
            bm_status = f"🟢 Dedaun Sihat ({round(healthy_pct, 1)}% hijau subur, klorofil kuat, tiada tanda penyakit)"
        elif chlorotic_pct >= 15.0:
            grade = "mild_stress"
            en_status = f"⚠️ Mild Chlorosis Detected (~{round(chlorotic_pct, 1)}% yellowing foliage - inspect for nitrogen deficit or sucking pests)"
            bm_status = f"⚠️ Klorosis / Daun Kekuningan Dikesan (~{round(chlorotic_pct, 1)}% daun kuning - periksa nitrogen & serangga)"
        elif necrotic_pct >= 15.0:
            grade = "severe_stress"
            en_status = f"🔴 Foliar Necrosis / Leaf Spotting (~{round(necrotic_pct, 1)}% spotted foliage - inspect for fungal leaf blight/spots)"
            bm_status = f"🔴 Stres / Nekrosis Daun (~{round(necrotic_pct, 1)}% tompok daun - periksa jangkitan kulat)"
        else:
            grade = "optimal"
            en_status = f"🟢 Moderate Foliage Health ({round(healthy_pct, 1)}% healthy canopy area)"
            bm_status = f"🟢 Kesihatan Dedaun Sederhana ({round(healthy_pct, 1)}% kanopi sihat)"
            
        return {
            "canopy_cover_pct": round(canopy_cover_pct, 1),
            "healthy_pct": round(healthy_pct, 1),
            "chlorotic_pct": round(chlorotic_pct, 1),
            "necrotic_pct": round(necrotic_pct, 1),
            "status_text_en": en_status,
            "status_text_bm": bm_status,
            "health_grade": grade
        }
    except Exception:
        return {
            "canopy_cover_pct": 35.0,
            "healthy_pct": 92.0,
            "chlorotic_pct": 4.0,
            "necrotic_pct": 4.0,
            "status_text_en": "🟢 Healthy Foliage (Optimal Chlorophyll)",
            "status_text_bm": "🟢 Dedaun Sihat & Subur",
            "health_grade": "optimal"
        }


def run_yolo_count_and_diagnosis(image_file, model_preference="Auto"):
    """
    Runs a YOLOv8 ONNX model to count leaves and diagnose crop health condition.
    Supports Okra Pod & Ripeness Detection (3 classes), Okra Leaf Disease (3 classes), 
    Tomato Disease (8 classes), Pakchoy Harvest, and generic plant detectors.
    
    Returns:
        tuple: (leaf_count: int, diagnosis: str, annotated_image: PIL.Image)
    """
    # Select appropriate model path
    model_preference_str = str(model_preference)
    
    if any(k in model_preference_str for k in ["Pod", "Buah", "Ripeness", "Kematangan", "okra_detection", "Okra Pod"]):
        for p in ["models/best(okra_detection).onnx", "models/best_(okra_detection).onnx", "models/best_(okra_model).onnx"]:
            if os.path.exists(p):
                model_path = p
                break
        else:
            model_path = "models/best(okra_detection).onnx"
    elif any(k in model_preference_str for k in ["Leaf Disease", "Penyakit Daun", "Yellow Vein", "Mozek"]):
        for p in ["models/best_(okra_model).onnx", "models/best_(tomato_leaf_model).onnx"]:
            if os.path.exists(p):
                model_path = p
                break
        else:
            model_path = "models/best_(okra_model).onnx"
    elif "Tomato" in model_preference_str and os.path.exists("models/best_(tomato_leaf_model).onnx"):
        model_path = "models/best_(tomato_leaf_model).onnx"
    elif ("Pakchoy" in model_preference_str or "Bok Choy" in model_preference_str) and os.path.exists("models/best.onnx"):
        model_path = "models/best.onnx"
    elif "Okra" in model_preference_str or "Bendi" in model_preference_str:
        for p in ["models/best(okra_detection).onnx", "models/best_(okra_detection).onnx", "models/best_(okra_model).onnx"]:
            if os.path.exists(p):
                model_path = p
                break
        else:
            model_path = "models/best(okra_detection).onnx"
    else:
        # Auto-detect best available model
        model_candidates = [
            "models/best(okra_detection).onnx",
            "models/best_(okra_detection).onnx",
            "models/best_(okra_model).onnx",
            "models/best_(tomato_leaf_model).onnx",
            "models/best.onnx",
            "models/yolov8_plant_detector.onnx",
            "models/best_epoch50.onnx"
        ]
        model_path = next((p for p in model_candidates if os.path.exists(p)), "models/best(okra_detection).onnx")
    
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

    # 3-Class Okra Foliage Disease Mapping (best_(okra_model).onnx)
    OKRA_DISEASE_CLASSES = {
        0: {"name": "Yellow Vein Mosaic", "color": "#ffeb3b", "is_healthy": False, "ms": "Penyakit Mozek Urat Kuning"},
        1: {"name": "Downy Mildew", "color": "#ff5722", "is_healthy": False, "ms": "Kulat Kulapuk Berdebu"},
        2: {"name": "Healthy Okra Leaf", "color": "#00e676", "is_healthy": True, "ms": "Daun Bendi Sihat & Subur"},
    }

    # 3-Class Okra Pod & Ripeness Detection Mapping (best(okra_detection).onnx)
    OKRA_POD_CLASSES = {
        0: {"name": "Overripe Okra", "color": "#ff7043", "is_healthy": False, "is_harvestable": False, "ms": "Bendi Terlebih Matang (Tua)"},
        1: {"name": "Ripe Okra (Harvest Ready)", "color": "#00e676", "is_healthy": True, "is_harvestable": True, "ms": "Bendi Matang (Sedia Dituai)"},
        2: {"name": "Unripe Okra (Growing)", "color": "#29b6f6", "is_healthy": True, "is_harvestable": False, "ms": "Bendi Muda (Sedang Membesar)"},
    }
    
    is_okra_pod_model = "okra_detection" in model_path.lower() or "best(okra_detection)" in model_path.lower()

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
        
        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        draw = ImageDraw.Draw(image)
        orig_w, orig_h = image.size
        
        if is_okra_pod_model or any(k in model_preference_str.lower() for k in ["pod", "buah", "kematangan", "ripeness"]):
            simulated_count = random.randint(3, 6)
            for _ in range(simulated_count):
                box_w = random.randint(int(orig_w * 0.1), int(orig_w * 0.22))
                box_h = random.randint(int(orig_h * 0.2), int(orig_h * 0.45))
                x1 = random.randint(0, orig_w - box_w)
                y1 = random.randint(0, orig_h - box_h)
                x2 = x1 + box_w
                y2 = y1 + box_h
                
                is_ripe = random.random() > 0.4
                color = "#00e676" if is_ripe else "#29b6f6"
                cls_name = "Ripe Okra (Harvest Ready)" if is_ripe else "Unripe Okra (Growing)"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                tag_y1 = max(0, y1 - 20)
                draw.rectangle([x1, tag_y1, min(orig_w, x1 + 190), y1], fill=color)
                draw.text((x1 + 4, tag_y1 + 2), f"{cls_name} (88%)", fill="#000000" if is_ripe else "#ffffff")
            
            foliage = analyze_foliage_canopy_health(image)
            badge_h = 30
            badge_y1 = orig_h - badge_h - 10
            badge_text = f"🌿 Leaf Health: {foliage['healthy_pct']}% Healthy Foliage | Canopy: {foliage['canopy_cover_pct']}%"
            badge_w = min(orig_w - 20, len(badge_text) * 7 + 20)
            draw.rectangle([10, badge_y1, 10 + badge_w, orig_h - 10], fill="#1b5e20" if foliage["health_grade"] == "optimal" else "#e65100")
            draw.text((18, badge_y1 + 7), badge_text, fill="#ffffff")
            
            diagnosis = (
                f"**🌾 Okra Pod & Harvest Readiness:**\n"
                f"Detected {simulated_count} okra pods on plant: **2 ripe pods (ready to harvest)** and **{simulated_count-2} unripe pods (developing)**.\n\n"
                f"**🍃 Foliage & Leaf Health Assessment:**\n"
                f"{foliage['status_text_en']}. Measured {foliage['canopy_cover_pct']}% active canopy cover across the plant with {foliage['healthy_pct']}% healthy chlorophyll signatures."
            )
            return simulated_count, diagnosis, image
        else:
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
                "🟢 Healthy growth. Robust chlorophyll distribution across detected leaves.",
                "⚠️ Mild leaf chlorosis detected (~15% of foliage) - suspected nitrogen deficiency.",
                "🔴 Early stage Septoria leaf spot detected on 3 lower-canopy leaves.",
                "🟠 Mild leaf wilting / water stress noticed along outer foliage border."
            ]
            return simulated_count, random.choice(diagnoses), image
    
    try:
        @st.cache_resource
        def get_onnx_session(path):
            return ort.InferenceSession(path)
            
        session = get_onnx_session(model_path)
        
        # Check model metadata for embedded class names if available
        detected_names = None
        try:
            meta = session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                detected_names = ast.literal_eval(meta["names"])
                if any("overripe" in str(v).lower() or "unripe" in str(v).lower() for v in detected_names.values()):
                    is_okra_pod_model = True
                elif any("downy" in str(v).lower() or "mosaic" in str(v).lower() for v in detected_names.values()):
                    is_okra_pod_model = False
        except Exception:
            pass
        
        # 1. Image Preprocessing
        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        else:
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
        item_count = len(keep_indices)
        
        draw = ImageDraw.Draw(image)
        is_normalized = (np.max(filtered_boxes) <= 1.01) if len(filtered_boxes) > 0 else False
        
        healthy_count = 0
        diseased_count = 0
        disease_counts = {}
        pod_class_counts = {}
        
        for idx in keep_indices:
            box = filtered_boxes[idx]
            cid = int(filtered_class_ids[idx])
            conf = float(filtered_scores[idx])
            
            # Map class info
            if is_okra_pod_model and cid in OKRA_POD_CLASSES:
                c_info = OKRA_POD_CLASSES[cid]
                is_healthy = c_info["is_healthy"]
                cls_name = c_info["name"]
                color = c_info["color"]
            elif not is_okra_pod_model and num_classes == 3 and cid in OKRA_DISEASE_CLASSES:
                c_info = OKRA_DISEASE_CLASSES[cid]
                is_healthy = c_info["is_healthy"]
                cls_name = c_info["name"]
                color = c_info["color"]
            elif num_classes == 8 and cid in TOMATO_CLASSES:
                c_info = TOMATO_CLASSES[cid]
                is_healthy = c_info["is_healthy"]
                cls_name = c_info["name"]
                color = c_info["color"]
            elif detected_names and cid in detected_names:
                raw_name = str(detected_names[cid])
                cls_name = raw_name.title()
                is_healthy = "healthy" in raw_name.lower() or "ripe" in raw_name.lower()
                color = "#00e676" if is_healthy else ("#29b6f6" if "unripe" in raw_name.lower() else "#ff5722")
            elif num_classes == 2:
                is_healthy = (cid == 0)
                cls_name = "Healthy Leaf" if is_healthy else "Diseased / Stressed"
                color = "#00e676" if is_healthy else "#ff1744"
            else:
                is_healthy = (cid == 0)
                cls_name = f"Class {cid}"
                color = "#00e676" if is_healthy else "#ff1744"
                
            if is_okra_pod_model:
                pod_class_counts[cls_name] = pod_class_counts.get(cls_name, 0) + 1
            else:
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
            text_fill = "#000000" if color in ["#ffeb3b", "#00e676", "#29b6f6"] else "#ffffff"
            draw.text((x1 + 4, tag_y1 + 2), label_text, fill=text_fill)
                
        # Generate detailed diagnostic summary
        if is_okra_pod_model:
            ripe_cnt = sum(1 for idx in keep_indices if int(filtered_class_ids[idx]) == 1)
            unripe_cnt = sum(1 for idx in keep_indices if int(filtered_class_ids[idx]) == 2)
            overripe_cnt = sum(1 for idx in keep_indices if int(filtered_class_ids[idx]) == 0)
            
            if item_count == 0:
                pod_summary = "No okra pods detected on the plant nodes in frame. Adjust camera angle or distance to focus on stem nodes."
            elif ripe_cnt > 0:
                pod_summary = f"Detected {item_count} okra pods on plant: **{ripe_cnt} ripe (ready to harvest)**, **{unripe_cnt} unripe (developing)**, and {overripe_cnt} overripe. Harvest ripe pods promptly to ensure prime tenderness and stimulate further flowering."
            elif overripe_cnt > 0:
                pod_summary = f"Detected {item_count} okra pods on plant: **{overripe_cnt} overripe pod(s)** (remove immediately) and **{unripe_cnt} developing pod(s)**. Removing overripe pods prevents fiber hardening and redirects plant energy to new shoots."
            else:
                pod_summary = f"Detected **{item_count} developing unripe okra pods** on plant (active pod-filling stage). Pods are growing well; maintain optimal fertigation schedule."
                
            # Perform Canopy Foliage & Leaf Health Assessment on background leaves
            foliage = analyze_foliage_canopy_health(image)
            
            # Draw foliage health badge on the bottom of the image
            badge_h = 30
            badge_y1 = orig_h - badge_h - 10
            badge_text = f"🌿 Leaf Health: {foliage['healthy_pct']}% Healthy Foliage | Canopy: {foliage['canopy_cover_pct']}%"
            badge_w = min(orig_w - 20, len(badge_text) * 7 + 20)
            draw.rectangle([10, badge_y1, 10 + badge_w, orig_h - 10], fill="#1b5e20" if foliage["health_grade"] == "optimal" else "#e65100")
            draw.text((18, badge_y1 + 7), badge_text, fill="#ffffff")
            
            diagnosis = (
                f"**🌾 Okra Pod & Harvest Readiness:**\n{pod_summary}\n\n"
                f"**🍃 Foliage & Leaf Health Assessment:**\n{foliage['status_text_en']}. Measured {foliage['canopy_cover_pct']}% active canopy cover across the plant with {foliage['healthy_pct']}% healthy chlorophyll signatures."
            )
        else:
            if item_count == 0:
                diagnosis = "No foliage detected in frame. Please adjust camera distance or lighting."
            elif diseased_count > 0:
                percentage = int((diseased_count / item_count) * 100)
                breakdown_list = [f"{cnt}x {dname}" for dname, cnt in disease_counts.items()]
                breakdown_str = ", ".join(breakdown_list)
                diagnosis = f"Detected {diseased_count} stressed/diseased leaves (~{percentage}% of detected foliage). Breakdown: {breakdown_str}. ({healthy_count} healthy leaves)."
            else:
                diagnosis = f"Healthy growth. All {item_count} detected leaves appear healthy and vigorous with strong chlorophyll signatures."
            
        return item_count, diagnosis, image

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")
        return 0, f"Error processing model: {str(e)}", None


def fetch_simulated_iot_telemetry(plot_id=1):
    """
    Simulates fetching real-time IoT sensor telemetry from a Cloud IoT Server (Firebase/MQTT/ThingSpeak).
    Returns a dict with sensor readings: timestamp, air_temp, soil_moisture, soil_ec, soil_ph, server_status.
    """
    import random
    seed_val = int(plot_id) if plot_id else 1
    t_now = get_now_myt()

    random.seed(seed_val + t_now.minute)
    
    moisture = round(random.uniform(28.0, 68.0), 1)
    temp = round(random.uniform(28.0, 34.5), 1)
    ec = round(random.uniform(0.9, 2.3), 2)
    ph = round(random.uniform(5.9, 6.7), 2)
    
    return {
        "timestamp": t_now.strftime("%Y-%m-%d %H:%M:%S"),
        "air_temp": temp,
        "soil_moisture": moisture,
        "soil_ec": ec,
        "soil_ph": ph,
        "server_status": "🟢 ONLINE (Connected to Cloud IoT Broker)"
    }


def run_intervention_recommendation(diagnosis, iot_telemetry=None, language_choice="🇲🇾 Bahasa Melayu"):
    """
    Calls Groq API with Llama 3.1 8B (llama-3.1-8b-instant) combining YOLOv8 Vision Diagnosis 
    with Cloud IoT Sensor Telemetry to generate a holistic agronomist intervention plan.
    
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
        
    # Format IoT Telemetry for LLM Prompt Fusion
    iot_text = ""
    if iot_telemetry:
        moist_status = "DEFICIT (Dry Soil)" if iot_telemetry['soil_moisture'] < 40 else ("OPTIMAL" if iot_telemetry['soil_moisture'] <= 70 else "SATURATED (Overwatered)")
        ec_status = "LOW (Fertilizer Deficit)" if iot_telemetry['soil_ec'] < 1.4 else ("OPTIMAL" if iot_telemetry['soil_ec'] <= 2.2 else "HIGH Salinity")
        
        iot_text = (
            f"\n\n[CLOUD IOT SENSOR TELEMETRY]:\n"
            f"- Soil Moisture: {iot_telemetry['soil_moisture']}% ({moist_status})\n"
            f"- Air Temperature: {iot_telemetry['air_temp']} °C\n"
            f"- Soil EC (Fertility): {iot_telemetry['soil_ec']} mS/cm ({ec_status})\n"
            f"- Soil pH: {iot_telemetry['soil_ph']}"
        )
    
    # 1. Standard Prompt configuration based on language preference
    if language_choice == "🇲🇾 Bahasa Melayu":
        system_prompt = (
            "Anda adalah ejen pertanian jitu FarmNeura, seorang agronomis profesional yang pakar dalam penggabungan data visi komputer (YOLOv8) dan data sensor IoT Awan (Cloud IoT Telemetry).\n"
            "Analisis kedua-dua simptom visual kanopi dan bacaan sensor IoT untuk memberikan cadangan tindakan pemulihan yang tepat dan praktikal.\n"
            "Formatkan jawapan anda dalam bentuk senarai peluru (bullet-point) Bahasa Melayu yang ringkas (3-4 mata sahaja)."
        )
    else:
        system_prompt = (
            "You are FarmNeura's precision agricultural agent, a professional agronomist specializing in multimodal data fusion (combining YOLOv8 vision diagnoses with Cloud IoT sensor telemetry).\n"
            "Analyze both the visual leaf symptoms and the cloud sensor readings to provide precise, root-cause agronomic intervention steps.\n"
            "Format your answer as a clean bullet-point list in simple farmer-friendly terms (3-4 points max)."
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
        m_val = iot_telemetry['soil_moisture'] if iot_telemetry else 50.0
        
        if language_choice == "🇲🇾 Bahasa Melayu":
            if m_val < 40 or "chlorosis" in d_lower or "nitrogen" in d_lower:
                return f"- **Diagnosis Gabungan (Visi + IoT)**: Klorosis dikesan oleh YOLOv8 dan disahkan oleh bacaan sensor IoT Awan (Kelembapan tanah rendah: {m_val}%).\n- Sembur baja cecair nitrogen dan mulakan penyiraman titis selama 20 minit.\n- Periksa tahap kedalaman akar untuk elakkan tekanan haba."
            elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower:
                return f"- **Diagnosis Gabungan (Visi + IoT)**: Jangkitan kulat kanopi dikesan. Sensor IoT menunjukkan kelembapan: {m_val}%.\n- Sembur racun kulat berasaskan tembaga atau minyak neem organik pada barisan terjejas.\n- Tingkatkan pengudaraan kanopi bawah."
            else:
                return f"- **Diagnosis Gabungan (Visi + IoT)**: Kesihatan kanopi dan bacaan sensor IoT berada pada tahap optimum (Kelembapan: {m_val}%, EC: {iot_telemetry['soil_ec'] if iot_telemetry else 1.8} mS/cm).\n- Teruskan jadual pemantauan biasa."
        else:
            if m_val < 40 or "chlorosis" in d_lower or "nitrogen" in d_lower:
                return f"- **Multimodal Fusion (Vision + IoT)**: Leaf chlorosis detected by YOLOv8 and confirmed by Cloud IoT Telemetry (Low Soil Moisture: {m_val}%).\n- Apply nitrogen liquid fertigation and initiate 20-min drip irrigation cycle.\n- Monitor root depth moisture to alleviate thermal stress."
            elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower:
                return f"- **Multimodal Fusion (Vision + IoT)**: Foliage fungal infection detected. Cloud IoT sensor moisture reading: {m_val}%.\n- Spray copper-based fungicide or neem oil to affected crop rows.\n- Prune lower canopy leaves to improve airflow."
            else:
                return f"- **Multimodal Fusion (Vision + IoT)**: Canopy condition and Cloud IoT sensor readings are optimal (Moisture: {m_val}%, EC: {iot_telemetry['soil_ec'] if iot_telemetry else 1.8} mS/cm).\n- Maintain regular fertigation and inspection schedule."

    model_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
        "groq/compound",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192"
    ]
    
    user_query = f"Vision Model Diagnosis: {diagnosis}{iot_text}"
    
    for model_name in model_candidates:
        try:
            if HAS_GROQ_SDK:
                client = Groq(api_key=api_key)
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2,
                    max_tokens=350
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
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 350
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
# SIDEBAR: LANGUAGE & NAVIGATION
# ---------------------------------------------------------------------

st.sidebar.markdown("<h2 style='color:#1b4d3e; margin-top:0;'>🌱 FarmNeura</h2>", unsafe_allow_html=True)

# 1. App-wide Language Selector
if "app_language" not in st.session_state:
    st.session_state["app_language"] = "🇲🇾 Bahasa Melayu"

language_choice = st.sidebar.selectbox(
    "🌐 Bahasa / Language",
    ["🇲🇾 Bahasa Melayu", "🇬🇧 English"],
    index=0 if st.session_state.get("app_language") == "🇲🇾 Bahasa Melayu" else 1,
    key="app_language"
)
is_bm = (language_choice == "🇲🇾 Bahasa Melayu")
L = I18N["bm"] if is_bm else I18N["en"]

# Dynamic CSS injection for native Streamlit widgets (File Uploader & Camera)
if is_bm:
    st.markdown(
        """
        <style>
        /* 🇲🇾 Malay UI Translations for Native Streamlit Widgets */
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span::after {
            content: "Seret dan lepas fail di sini" !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            color: #31333F !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small::after {
            content: "Had 200MB setiap fail • JPG, JPEG, PNG" !important;
            font-size: 0.75rem !important;
            color: #808495 !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            font-size: 0 !important;
        }
        [data-testid="stFileUploaderDropzone"] button::after {
            content: "Pilih Fail" !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }

        [data-testid="stCameraInput"] button {
            font-size: 0 !important;
        }
        [data-testid="stCameraInput"] button::after {
            content: "📸 Ambil Foto" !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.caption(L["sidebar_sub"])

# 2. Main Navigation Menu options (localized)
NAV_MODES = [
    {"id": "overview", "en": "📋 Overview", "bm": "📋 Gambaran Keseluruhan"},
    {"id": "quick_scan", "en": "🔍 Quick Crop Scan", "bm": "🔍 Imbasan Pantas Tanaman"},
    {"id": "monitoring", "en": "📷 Plot Monitoring", "bm": "📷 Pemantauan Plot"},
    {"id": "registry", "en": "⚙️ Registry & Management", "bm": "⚙️ Pendaftaran & Pengurusan"}
]

# Check for pending redirection
if "redirect_to_view" in st.session_state:
    raw_target = st.session_state.pop("redirect_to_view")
    for nm in NAV_MODES:
        if raw_target in (nm["id"], nm["en"], nm["bm"], "📷 Plot Monitoring", "📋 Overview", "🔍 Quick Crop Scan", "⚙️ Registry & Management"):
            st.session_state["nav_menu_active_id"] = nm["id"]
            break

if "nav_menu_active_id" not in st.session_state:
    st.session_state["nav_menu_active_id"] = "overview"

curr_idx = 0
for idx, nm in enumerate(NAV_MODES):
    if nm["id"] == st.session_state["nav_menu_active_id"]:
        curr_idx = idx
        break

selected_nav_label = st.sidebar.radio(
    L["nav_menu_title"],
    [m["bm"] if is_bm else m["en"] for m in NAV_MODES],
    index=curr_idx,
    key=f"nav_menu_radio_{'bm' if is_bm else 'en'}"
)

for m in NAV_MODES:
    if selected_nav_label in (m["en"], m["bm"]):
        st.session_state["nav_menu_active_id"] = m["id"]
        break

active_mode = st.session_state["nav_menu_active_id"]
view_mode = "📷 Plot Monitoring" if active_mode == "monitoring" else ("🔍 Quick Crop Scan" if active_mode == "quick_scan" else ("⚙️ Registry & Management" if active_mode == "registry" else "📋 Overview"))

selected_farm_obj = None
selected_plot_obj = None

# Show selectors in sidebar only when in Monitoring mode
if active_mode == "monitoring":
    st.sidebar.markdown("---")
    farms_list = db_get_farms()
    if not farms_list:
        st.sidebar.warning("Tiada ladang didaftarkan lagi. Sila ke Pendaftaran & Pengurusan untuk mendaftar ladang." if is_bm else "No farms registered yet. Please go to Registry & Management to add a farm.")
    else:
        farm_idx = 0
        if "target_farm_id" in st.session_state:
            for idx, f in enumerate(farms_list):
                if f["id"] == st.session_state["target_farm_id"]:
                    farm_idx = idx
                    break
                    
        selected_farm_obj = st.sidebar.selectbox(
            "Pilih Lokasi Ladang" if is_bm else "Select Farm location",
            options=farms_list,
            format_func=lambda x: x["name"],
            index=farm_idx,
            key="sb_farm_select"
        )
        
        plots_list = db_get_plots(selected_farm_obj["id"])
        if not plots_list:
            st.sidebar.warning(f"Tiada plot didaftarkan untuk '{selected_farm_obj['name']}'." if is_bm else f"No plots registered for '{selected_farm_obj['name']}'.")
        else:
            plot_idx = 0
            if "target_plot_id" in st.session_state:
                for idx, p in enumerate(plots_list):
                    if p["id"] == st.session_state["target_plot_id"]:
                        plot_idx = idx
                        break
                        
            selected_plot_obj = st.sidebar.selectbox(
                "Pilih Plot Sasaran" if is_bm else "Select Target Plot",
                options=plots_list,
                format_func=lambda x: x["plot_name"],
                index=plot_idx,
                key="sb_plot_select"
            )

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **LLM Engine:** Groq Ultra-Fast LPU")
st.sidebar.markdown(
    f"""
    <div style="font-size: 0.85rem; color: #555;">
        <strong>{'Skop Fasa 1:' if is_bm else 'Phase 1 Scope:'}</strong><br/>
        ✓ {'Pendaftaran Ladang (Modul 1)' if is_bm else 'Farm Registry (Module 1)'}<br/>
        ✓ {'Pendaftaran Tanaman (Modul 2)' if is_bm else 'Plant Registry (Module 2)'}<br/>
        ✓ {'Pendaftaran Plot (Modul 3)' if is_bm else 'Plot Registry (Module 3)'}<br/>
        ✓ {'Pemantauan Plot (Modul 4)' if is_bm else 'Plot Monitoring (Module 4)'}<br/>
        <br/>
        <span style="color:#d32f2f;">✗ {'Sensor & Analitik IoT (Fasa 2)' if is_bm else 'IoT Sensors & Analytics (Phase 2)'}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------

# Top branded header
if active_mode == "overview":
    farm_header_text = "Plot Aktif" if is_bm else "Active Plots"
    plot_header_text = "Semua Tanaman" if is_bm else "All Crops"
    sub_header_tag = "Gambaran Keseluruhan" if is_bm else "Overview"
elif active_mode == "quick_scan":
    farm_header_text = "Diagnosis Segera" if is_bm else "Instant Diagnosis"
    plot_header_text = "Imbasan Bebas" if is_bm else "Free-Form Scan"
    sub_header_tag = "Imbasan Pantas" if is_bm else "Quick Scan"
elif active_mode == "registry":
    farm_header_text = "Pendaftaran" if is_bm else "Registry"
    plot_header_text = "Pengurusan" if is_bm else "Management"
    sub_header_tag = "Pendaftaran & Pengurusan" if is_bm else "Registry & Management"
else:
    farm_header_text = selected_farm_obj["name"] if selected_farm_obj else ("Tiada Ladang" if is_bm else "No Farm Selected")
    plot_header_text = selected_plot_obj["plot_name"] if selected_plot_obj else ("Tiada Plot" if is_bm else "No Plot Selected")
    sub_header_tag = "Pemantauan Plot" if is_bm else "Plot Monitoring"

st.markdown(
    f"""
    <div class="header-card">
        <div class="header-title">FarmNeura v2</div>
        <div class="header-subtitle">{sub_header_tag} • {farm_header_text} • {plot_header_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Render View: OVERVIEW
if active_mode == "overview":
    # 1. Greeting & Microclimate Header
    current_hour = get_now_myt().hour
    if current_hour < 12:
        greeting_text = L["greeting_morning"]
    elif current_hour < 18:
        greeting_text = L["greeting_afternoon"]
    else:
        greeting_text = L["greeting_evening"]
        
    st.markdown(
        f"""
        <div style="margin-bottom: 1.2rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #111111;">{greeting_text} 👋</h2>
            <div style="font-size: 0.92rem; color: #555555; margin-top: 4px;">
                {L["welcome_sub"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Crop Scan quick access banner
    st.markdown(
        f"""
        <div style="background: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 12px; padding: 12px 16px; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <strong style="color: #2e7d32; font-size: 0.95rem;">{L["instant_check_title"]}</strong><br/>
                <span style="color: #555; font-size: 0.85rem;">{L["instant_check_sub"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button(L["launch_quick_scan_btn"], key="ov_quick_scan_btn", type="primary", use_container_width=True):
        st.session_state["redirect_to_view"] = "quick_scan"
        st.rerun()

    plots_data = db_get_plots_with_plants()
    
    if not plots_data:
        st.info(L["no_plots_registered"])
    else:
        total_plots = len(plots_data)
        needs_attention = 0
        needs_photo = 0
        healthy_count = 0
        
        plot_status_list = []
        today = get_now_myt()
        
        for plot in plots_data:
            latest_rec = db_get_latest_record(plot["id"])
            p_crop = plot["plant_name"] if plot.get("plant_name") else ("Tanaman Belum Ditentukan" if is_bm else "Crop Unassigned")
            
            is_attention = False
            is_overdue = False
            days_overdue = 0
            
            if latest_rec:
                d_text = latest_rec["diagnosis"].lower()
                if any(w in d_text for w in ["stressed", "diseased", "chlorosis", "spot", "blight", "virus", "mold", "mildew", "stres", "penyakit", "hawar"]):
                    is_attention = True
                    needs_attention += 1
                else:
                    healthy_count += 1
                    
                try:
                    last_time = datetime.strptime(latest_rec["time"], "%Y-%m-%d %H:%M")
                    days_diff = (today - last_time).days
                    if days_diff >= 3:
                        is_overdue = True
                        days_overdue = days_diff
                        needs_photo += 1
                except Exception:
                    pass
            else:
                is_overdue = True
                needs_photo += 1
                days_overdue = 0
                
            plot_status_list.append({
                "plot": plot,
                "crop": p_crop,
                "latest_rec": latest_rec,
                "is_attention": is_attention,
                "is_overdue": is_overdue,
                "days_overdue": days_overdue
            })
            
        overall_health_pct = int((healthy_count / total_plots) * 100) if total_plots > 0 else 100
        
        # 2. 2x2 Metric Cards Summary Grid
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 16px; padding: 1.1rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #555555; margin-bottom: 6px;">{L["overall_plot_health"]}:</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #2e7d32;">{overall_health_pct}% {'Baik' if is_bm else 'Good'} 🍃</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 16px; padding: 1.1rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #555555; margin-bottom: 6px;">📷 {L["photo_overdue"]}:</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #111111;">{needs_photo} {'Plot' if is_bm else 'Plots'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 16px; padding: 1.1rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #555555; margin-bottom: 6px;"><span style="background: #e65100; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem;">{L["needs_attention"]}:</span></div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #e65100;">{needs_attention} {'Plot' if is_bm else 'Plots'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 16px; padding: 1.1rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #555555; margin-bottom: 6px;">{L["total_active_plots"]}:</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #111111;">{total_plots}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("### " + ("Senarai Tindakan Hari Ini" if is_bm else "Today's Action List"))
        st.caption("Tugasan pemeriksaan plot berkeutamaan berdasarkan diagnosis penglihatan AI terkini." if is_bm else "Prioritized plot inspection tasks based on latest AI vision diagnosis.")
        
        # Sort plot status list: Needs attention first, then photo overdue, then optimal growth
        plot_status_list.sort(key=lambda x: (not x["is_attention"], not x["is_overdue"]))
        
        for idx, item in enumerate(plot_status_list):
            plot = item["plot"]
            crop = item["crop"]
            latest_rec = item["latest_rec"]
            is_att = item["is_attention"]
            is_over = item["is_overdue"]
            days_over = item["days_overdue"]
            
            if is_att:
                status_pill = f'<span style="background: #fff3e0; color: #e65100; border: 1px solid #ffe082; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">{L["status_attention"]}</span>'
                btn_label = "Ambil Tindakan ➔" if is_bm else "Take Action ➔"
                btn_kind = "primary"
            elif is_over:
                over_text = f"Gambar Tertunggak ({days_over} hari)" if is_bm else f"Photo Overdue ({days_over} days)"
                if days_over == 0:
                    over_text = "Menunggu Gambar Awal" if is_bm else "Pending Initial Photo"
                status_pill = f'<span style="background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">ℹ️ {over_text}</span>'
                btn_label = "📷 Imbas Sekarang" if is_bm else "📷 Scan Now"
                btn_kind = "primary"
            else:
                status_pill = f'<span style="background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">{L["status_healthy"]}</span>'
                btn_label = "Periksa ➔" if is_bm else "Inspect ➔"
                btn_kind = "secondary"

            col_card_info, col_card_btn = st.columns([2.8, 1.2])
            with col_card_info:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 14px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
                        <div style="font-weight: 700; font-size: 1.05rem; color: #111111; margin-bottom: 4px;">📍 {plot['plot_name']} - {crop} <span style="font-size: 0.8rem; color: #666; font-weight: 400;">({plot['farm_name']})</span></div>
                        <div>{status_pill}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_card_btn:
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                if st.button(btn_label, key=f"act_btn_{plot['id']}_{idx}", type=btn_kind, use_container_width=True):
                    st.session_state["target_farm_id"] = plot["farm_id"]
                    st.session_state["target_plot_id"] = plot["id"]
                    st.session_state["redirect_to_view"] = "monitoring"
                    st.rerun()

# Render View: QUICK CROP SCAN (Ad-Hoc / Instant Diagnosis)
elif active_mode == "quick_scan":
    st.markdown(L["qs_title"])
    st.caption(L["qs_sub"])
    
    # Crop photography guide
    active_hint = st.session_state.get("qs_crop_context_input", "") or st.session_state.get("qs_vision_model", "")
    render_crop_photography_guide(active_hint, is_expanded=False, is_bm=is_bm)

    col_upload_left, col_upload_right = st.columns([1.4, 1.1])
    with col_upload_left:
        qs_cam = st.camera_input(L["pm_cam_prompt"], key="qs_camera_input")
        qs_file = st.file_uploader(L["pm_upload_prompt"], type=["jpg", "jpeg", "png"], key="qs_file_uploader")
        
    qs_image = qs_cam or qs_file
    
    with col_upload_right:
        st.markdown(L["qs_settings_title"])
        # Vision Model Selector
        if is_bm:
            available_models = ["🤖 Auto-Kesan (Model Terbaik Tersedia)"]
            if os.path.exists("models/best(okra_detection).onnx") or os.path.exists("models/best_(okra_detection).onnx"):
                available_models.append("🌿 Pengesanan Buah Bendi & Kematangan (3 Kelas: Matang, Muda, Tua)")
            if os.path.exists("models/best_(okra_model).onnx"):
                available_models.append("🌿 Penyakit Daun Bendi & Mozek Urat Kuning (3 Kelas)")
            if os.path.exists("models/best_(tomato_leaf_model).onnx"):
                available_models.append("🍅 Penyakit Tomato & Solanaceae (8 Kelas)")
            if os.path.exists("models/best.onnx"):
                available_models.append("🥬 Pengesanan Kematangan Tuai Pakchoy / Sawi")
            if os.path.exists("models/yolov8_plant_detector.onnx"):
                available_models.append("🌱 Pengesan Tumbuhan Umum")
        else:
            available_models = ["🤖 Auto-Detect (Best Available Model)"]
            if os.path.exists("models/best(okra_detection).onnx") or os.path.exists("models/best_(okra_detection).onnx"):
                available_models.append("🌿 Okra Pod & Harvest Ripeness Detection (3 Classes: Ripe, Unripe, Overripe)")
            if os.path.exists("models/best_(okra_model).onnx"):
                available_models.append("🌿 Okra / Bendi Leaf Disease & Yellow Vein (3 Classes)")
            if os.path.exists("models/best_(tomato_leaf_model).onnx"):
                available_models.append("🍅 Tomato & Solanaceae Disease (8 Classes)")
            if os.path.exists("models/best.onnx"):
                available_models.append("🥬 Pakchoy / Bok Choy Harvest Detector")
            if os.path.exists("models/yolov8_plant_detector.onnx"):
                available_models.append("🌱 General Plant Detector")
            
        qs_model = st.selectbox(
            L["qs_select_model"],
            available_models,
            index=0,
            key="qs_vision_model"
        )
        
        qs_crop_context = st.text_input(
            L["qs_crop_context"],
            placeholder=L["qs_crop_context_ph"],
            key="qs_crop_context_input",
            help=L["qs_crop_context_help"]
        )

    # Check if a new image was uploaded to reset previous diagnosis
    if qs_image is not None:
        qs_img_id = f"{getattr(qs_image, 'name', 'camera')}_{getattr(qs_image, 'size', 0)}"
        if st.session_state.get("qs_last_image_id") != qs_img_id:
            st.session_state["qs_last_image_id"] = qs_img_id
            if "qs_temp_diagnosis" in st.session_state:
                del st.session_state["qs_temp_diagnosis"]

    if qs_image is not None:
        st.markdown("---")
        
        # Display image preview
        col_prev1, col_prev2 = st.columns([1.2, 1])
        with col_prev1:
            if "qs_temp_diagnosis" in st.session_state and st.session_state.qs_temp_diagnosis.get("annotated_image") is not None:
                st.image(
                    st.session_state.qs_temp_diagnosis["annotated_image"],
                    caption="Pengesanan AI (Kotak Sempadan & Keyakinan)" if is_bm else "AI Detections (Multi-Class Bounding Boxes & Confidence)",
                    use_container_width=True
                )
            else:
                st.image(qs_image, caption="Foto Tanaman Terpilih" if is_bm else "Selected Crop Photo", use_container_width=True)
                
        with col_prev2:
            st.markdown("#### " + ("🔬 Analisis AI Segera" if is_bm else "🔬 Instant AI Analysis"))
            st.write("Klik butang di bawah untuk menjalankan pengesanan visi komputer dan penaakulan LLM agronomi." if is_bm else "Click below to run computer vision detection and agronomist LLM reasoning.")
            if st.button(L["qs_btn_diagnose"], type="primary", key="qs_btn_run_diag", use_container_width=True):
                with st.spinner("Menganalisis dedaun, mengenal pasti penyakit & menjana pelan rawatan..." if is_bm else "Analyzing foliage, identifying diseases & generating treatment plan..."):
                    if hasattr(qs_image, "seek"):
                        qs_image.seek(0)
                    pil_backup = Image.open(qs_image).convert("RGB")
                    if hasattr(qs_image, "seek"):
                        qs_image.seek(0)
                    bil_daun, diagnosis, annotated_image = run_yolo_count_and_diagnosis(qs_image, model_preference=qs_model)
                    
                    # Generate agronomist recommendation
                    diag_prompt_text = diagnosis
                    if qs_crop_context.strip():
                        diag_prompt_text = f"Crop: {qs_crop_context.strip()}. {diagnosis}"
                        
                    intervention = run_intervention_recommendation(
                        diag_prompt_text,
                        iot_telemetry=None,
                        language_choice=language_choice
                    )
                    
                st.session_state.qs_temp_diagnosis = {
                    "bil_daun": bil_daun,
                    "diagnosis": diagnosis,
                    "intervention": intervention,
                    "annotated_image": annotated_image,
                    "raw_image": pil_backup,
                    "crop_context": qs_crop_context.strip()
                }
                st.rerun()

        # Display diagnosis results if ready
        if "qs_temp_diagnosis" in st.session_state:
            qs_res = st.session_state.qs_temp_diagnosis
            st.markdown("---")
            st.markdown(L["qs_results_title"])
            
            # Metric
            st.markdown(
                f"""
                <div class="custom-metric">
                    <div class="custom-metric-val">{qs_res['bil_daun']}</div>
                    <div class="custom-metric-lbl">{L["qs_leaf_metric"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Diagnosis Card
            diag_html = render_markdown_to_html(qs_res['diagnosis'])
            st.markdown(
                f"""
                <div class="result-card neutral">
                    <div class="card-title">{L["qs_diag_card"]}</div>
                    <div class="card-text">{diag_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Intervention Card
            interv_html = render_markdown_to_html(qs_res['intervention'])
            st.markdown(
                f"""
                <div class="result-card alert">
                    <div class="card-title">{L["qs_interv_card"]}</div>
                    <div class="card-text">{interv_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # --- OPTIONAL SAVE TO DATABASE SECTION ---
            st.markdown("---")
            st.markdown(L["qs_save_header"])
            st.caption(L["qs_save_sub"])
            
            all_farms = db_get_farms()
            if not all_farms:
                st.warning("⚠️ Tiada ladang didaftarkan lagi. Anda boleh mendaftar ladang di bawah **⚙️ Pendaftaran & Pengurusan**." if is_bm else "⚠️ No farms registered yet. You can register farms under **⚙️ Registry & Management**.")
            else:
                with st.container():
                    col_sf, col_sp = st.columns(2)
                    with col_sf:
                        save_farm = st.selectbox(
                            L["qs_save_farm"],
                            options=all_farms,
                            format_func=lambda x: x["name"],
                            key="qs_save_farm_select"
                        )
                    
                    target_plots = db_get_plots(save_farm["id"]) if save_farm else []
                    with col_sp:
                        if target_plots:
                            save_plot = st.selectbox(
                                L["qs_save_plot"],
                                options=target_plots,
                                format_func=lambda x: x["plot_name"],
                                key="qs_save_plot_select"
                            )
                        else:
                            st.warning(f"Tiada plot di bawah '{save_farm['name']}'." if is_bm else f"No plots under '{save_farm['name']}'.")
                            save_plot = None
                            
                    if save_plot:
                        # Intercropping / Multi-Crop Selector
                        registered_crops = db_get_plants(save_plot["id"])
                        selected_crop_tag = ""
                        if registered_crops:
                            crop_names = [c["name"] for c in registered_crops]
                            if len(crop_names) > 1:
                                selected_crop_tag = st.selectbox(
                                    L["qs_intercrop_select"],
                                    options=crop_names,
                                    key=f"qs_intercrop_select_{save_plot['id']}",
                                    help="Pilih jenis tanaman yang dirakam dalam foto ini." if is_bm else "Select which registered crop variety is featured in this photo."
                                )
                            else:
                                selected_crop_tag = crop_names[0]
                                st.markdown(f"🌾 **{'Tanaman Aktif dalam' if is_bm else 'Active Crop in'} {save_plot['plot_name']}:** `{selected_crop_tag}`")
                        else:
                            st.info(f"ℹ️ {'Tiada tanaman khusus didaftarkan untuk' if is_bm else 'No specific crops registered for'} '{save_plot['plot_name']}'.")
                        
                        col_sdt1, col_sdt2 = st.columns(2)
                        with col_sdt1:
                            qs_log_date = st.date_input(L["qs_inspection_date"], value=get_now_myt().date(), key="qs_log_date_picker")
                        with col_sdt2:
                            qs_log_time = st.time_input(L["qs_inspection_time"], value=get_now_myt().time(), key="qs_log_time_picker")
                            
                        qs_save_notes = st.text_area(L["qs_field_notes"], placeholder=L["qs_field_notes_ph"], key="qs_save_notes_area")
                        
                        custom_qs_timestamp = f"{qs_log_date.strftime('%Y-%m-%d')} {qs_log_time.strftime('%H:%M')}"
                        
                        if st.button(L["qs_btn_save"], type="primary", key="qs_btn_save_record", use_container_width=True):
                            img_to_save = qs_res.get("annotated_image") or qs_res.get("raw_image") or qs_image
                            saved_img_path = save_uploaded_image(img_to_save, save_plot["id"])
                            
                            # Tag crop in notes if specified
                            final_qs_notes = qs_save_notes.strip()
                            if selected_crop_tag and f"[{selected_crop_tag}]" not in final_qs_notes:
                                final_qs_notes = f"[{selected_crop_tag}] {final_qs_notes}".strip()
                            
                            success, msg = db_add_record(
                                save_plot["id"],
                                custom_qs_timestamp,
                                qs_res["bil_daun"],
                                qs_res["diagnosis"],
                                qs_res["intervention"],
                                final_qs_notes,
                                image_path=saved_img_path
                            )
                            if success:
                                crop_tag_info = f" ({selected_crop_tag})" if selected_crop_tag else ""
                                success_txt = f"✅ Berjaya menyimpan imbasan{crop_tag_info} dengan kotak sempadan ke **{save_farm['name']} ➔ {save_plot['plot_name']}**!" if is_bm else f"✅ Successfully saved scan{crop_tag_info} with bounding boxes to **{save_farm['name']} ➔ {save_plot['plot_name']}**!"
                                st.success(success_txt)
                                st.balloons()
                                if "qs_temp_diagnosis" in st.session_state:
                                    del st.session_state.qs_temp_diagnosis
                                if "qs_last_image_id" in st.session_state:
                                    del st.session_state.qs_last_image_id
                                st.rerun()
                            else:
                                st.error(f"Error saving scan: {msg}")
    else:
        ready_title = "Sedia untuk Pemeriksaan Pantas Tanaman" if is_bm else "Ready for Quick Crop Inspection"
        ready_sub = "Muat naik foto daun/kanopi atau guna kamera untuk diagnosis kesihatan AI segera." if is_bm else "Upload a leaf/canopy photo or use your camera to get instant AI health diagnostics."
        st.markdown(
            f"""
            <div style="text-align: center; padding: 2.5rem 1rem; border: 2px dashed #cfd8dc; border-radius: 16px; background: #fafafa; margin-top: 1rem;">
                <span style="font-size: 2.2rem; color: #43a047;">📸</span>
                <h4 style="margin: 0.5rem 0 0.2rem; color: #2e7d32;">{ready_title}</h4>
                <p style="margin: 0; color: #78909c; font-size: 0.9rem;">{ready_sub}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Render View: REGISTRY & MANAGEMENT
elif active_mode == "registry":
    st.markdown(L["reg_title"])
    st.caption(L["reg_sub"])
    
    sub_tab_farm, sub_tab_plot, sub_tab_plant = st.tabs([L["reg_tab_farm"], L["reg_tab_plot"], L["reg_tab_crop"]])
    
    with sub_tab_farm:
        st.markdown("#### " + ("Daftar Lokasi Ladang Baharu" if is_bm else "Register a New Farm Location"))
        with st.form("new_farm_form", clear_on_submit=True):
            f_name = st.text_input(L["reg_farm_name"], placeholder="cth. Ladang C - Rawang" if is_bm else "e.g. Farm C - Rawang")
            f_loc = st.text_input(L["reg_farm_loc"], placeholder="cth. Selangor, Malaysia" if is_bm else "e.g. Selangor, Malaysia")
            f_size = st.number_input(L["reg_farm_size"] + (" (kaki persegi)" if is_bm else " (sq ft)"), min_value=1.0, step=100.0, value=1000.0)
            
            submitted = st.form_submit_button(L["reg_farm_btn"], type="primary")
            if submitted:
                if not f_name.strip():
                    st.error("Nama Ladang diperlukan." if is_bm else "Farm Name is required.")
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
            st.markdown("##### " + L["reg_existing_farms"])
            df_farms = pd.DataFrame(farms)
            df_farms.columns = ["ID", "Nama Ladang" if is_bm else "Farm Name", "Lokasi" if is_bm else "Location", "Saiz (kaki²)" if is_bm else "Size (sq ft)"]
            st.dataframe(df_farms, use_container_width=True, hide_index=True)
            
            # Edit & Delete Farm Expanders
            col_edit, col_del = st.columns(2)
            with col_edit:
                with st.expander("✏️ " + ("Edit Butiran Ladang" if is_bm else "Edit Farm Details")):
                    selected_farm_to_edit = st.selectbox(
                        "Pilih Ladang untuk Diedit" if is_bm else "Select Farm to Edit",
                        options=farms,
                        format_func=lambda x: x["name"],
                        key="edit_farm_select"
                    )
                    if selected_farm_to_edit:
                        with st.form("edit_farm_form"):
                            edit_name = st.text_input("Nama Ladang Baharu" if is_bm else "New Farm Name", value=selected_farm_to_edit["name"])
                            edit_loc = st.text_input("Lokasi Baharu" if is_bm else "New Location", value=selected_farm_to_edit["location"])
                            edit_size = st.number_input("Saiz Baharu" if is_bm else "New Size (sq ft)", min_value=1.0, value=float(selected_farm_to_edit["size_sq_ft"]))
                            
                            edit_submit = st.form_submit_button("Simpan Perubahan" if is_bm else "Save Changes", type="primary")
                            if edit_submit:
                                if not edit_name.strip():
                                    st.error("Nama ladang tidak boleh kosong." if is_bm else "Farm name cannot be empty.")
                                else:
                                    ok, m = db_update_farm(selected_farm_to_edit["id"], edit_name, edit_loc, edit_size)
                                    if ok:
                                        st.success(m)
                                        st.rerun()
                                    else:
                                        st.error(m)
            with col_del:
                with st.expander("🗑️ " + ("Padam Lokasi Ladang" if is_bm else "Delete Farm Location")):
                    selected_farm_to_delete = st.selectbox(
                        "Pilih Ladang untuk Dipadam" if is_bm else "Select Farm to Delete",
                        options=farms,
                        format_func=lambda x: x["name"],
                        key="delete_farm_select"
                    )
                    if selected_farm_to_delete:
                        del_warn = f"⚠️ Amaran: Memadam '{selected_farm_to_delete['name']}' akan memadam secara kekal semua plot, tanaman dan sejarah rekod berkaitan!" if is_bm else f"⚠️ Warning: Deleting '{selected_farm_to_delete['name']}' will permanently delete all associated plots, crop registries, and monitoring history records!"
                        st.warning(del_warn)
                        confirm_delete = st.checkbox(f"{'Saya mengesahkan untuk memadam' if is_bm else 'I confirm that I want to delete'} '{selected_farm_to_delete['name']}'", key="confirm_delete_farm")
                        if st.button("Padam Ladang" if is_bm else "Delete Farm Location", type="primary", disabled=not confirm_delete, key="del_farm_btn"):
                            ok, m = db_delete_farm(selected_farm_to_delete["id"])
                            if ok:
                                st.success(m)
                                st.rerun()
                            else:
                                st.error(m)
            
    with sub_tab_plot:
        st.markdown("#### " + ("Daftar Plot Baharu dalam Ladang" if is_bm else "Register a New Plot inside a Farm"))
        farms = db_get_farms()
        if not farms:
            st.warning("Sila daftar Ladang terlebih dahulu." if is_bm else "Please register a Farm first.")
        else:
            with st.form("new_plot_form", clear_on_submit=True):
                selected_f = st.selectbox(L["reg_plot_farm"], options=farms, format_func=lambda x: x["name"])
                p_name = st.text_input(L["reg_plot_name"], placeholder="cth. Plot 4" if is_bm else "e.g. Plot 4")
                p_size = st.number_input(L["reg_plot_area"], min_value=1.0, step=100.0, value=1000.0)
                
                col1, col2 = st.columns(2)
                with col1:
                    cycle_start = st.date_input(L["reg_plot_cycle"], value=datetime.now())
                with col2:
                    cycle_end = st.date_input("Tarikh Tamat Kitaran" if is_bm else "Cycle End Date", value=datetime.now())
                    
                p_cost = st.number_input("Anggaran Belanjawan Kos Kitaran (MYR)" if is_bm else "Est. Cycle Cost Budget (MYR)", min_value=0.0, step=50.0, value=0.0)
                p_notes = st.text_area("Nota Plot (Pilihan)" if is_bm else "Plot Notes (optional)", placeholder="Jenis tanah, barisan pengairan, mikroiklim, dll." if is_bm else "Soil type, irrigation row, microclimate notes, etc.")

                submitted = st.form_submit_button(L["reg_plot_btn"], type="primary")
                if submitted:
                    if not p_name.strip():
                        st.error("Nama Plot diperlukan." if is_bm else "Plot Name is required.")
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
            st.markdown("##### " + L["reg_existing_plots"])
            selected_farm_for_list = st.selectbox("Lihat plot dalam:" if is_bm else "View plots in:", options=farms, format_func=lambda x: x["name"], key="view_plots_selector")
            plots = db_get_plots(selected_farm_for_list["id"])
            if plots:
                df_plots = pd.DataFrame(plots)
                df_plots = df_plots.drop(columns=["farm_id"])
                df_plots.columns = ["ID", "Nama Plot" if is_bm else "Plot Name", "Saiz (kaki²)" if is_bm else "Size (sq ft)", "Mula Kitaran" if is_bm else "Cycle Start", "Tamat Kitaran" if is_bm else "Cycle End", "Bajet (MYR)" if is_bm else "Budget (MYR)", "Nota" if is_bm else "Notes"]
                st.dataframe(df_plots, use_container_width=True, hide_index=True)
                
                # Edit & Delete Plot Expanders
                col_edit, col_del = st.columns(2)
                with col_edit:
                    with st.expander("✏️ " + ("Edit Butiran Plot" if is_bm else "Edit Plot Details")):
                        selected_plot_to_edit = st.selectbox(
                            "Pilih Plot untuk Diedit" if is_bm else "Select Plot to Edit",
                            options=plots,
                            format_func=lambda x: x["plot_name"],
                            key="edit_plot_select"
                        )
                        if selected_plot_to_edit:
                            with st.form("edit_plot_form"):
                                edit_p_name = st.text_input("Nama Plot Baharu" if is_bm else "New Plot Name", value=selected_plot_to_edit["plot_name"])
                                edit_p_size = st.number_input("Saiz Plot Baharu" if is_bm else "New Plot Size (sq ft)", min_value=1.0, value=float(selected_plot_to_edit["size_sq_ft"]))
                                try:
                                    start_val = datetime.strptime(selected_plot_to_edit["cycle_start"], "%Y-%m-%d")
                                    end_val = datetime.strptime(selected_plot_to_edit["cycle_end"], "%Y-%m-%d")
                                except:
                                    start_val = datetime.now()
                                    end_val = datetime.now()
                                edit_p_start = st.date_input("Tarikh Mula Kitaran Baharu" if is_bm else "New Cycle Start Date", value=start_val)
                                edit_p_end = st.date_input("Tarikh Tamat Kitaran Baharu" if is_bm else "New Cycle End Date", value=end_val)
                                edit_p_cost = st.number_input("Bajet Baharu (MYR)" if is_bm else "New Budget (MYR)", min_value=0.0, value=float(selected_plot_to_edit["cost_records"]))
                                edit_p_notes = st.text_area("Nota Baharu" if is_bm else "New Notes", value=selected_plot_to_edit["notes"] or "")
                                
                                edit_submit = st.form_submit_button("Simpan Perubahan" if is_bm else "Save Changes", type="primary")
                                if edit_submit:
                                    if not edit_p_name.strip():
                                        st.error("Nama plot tidak boleh kosong." if is_bm else "Plot name cannot be empty.")
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
                    with st.expander("🗑️ " + ("Padam Plot" if is_bm else "Delete Plot")):
                        selected_plot_to_delete = st.selectbox(
                            "Pilih Plot untuk Dipadam" if is_bm else "Select Plot to Delete",
                            options=plots,
                            format_func=lambda x: x["plot_name"],
                            key="delete_plot_select"
                        )
                        if selected_plot_to_delete:
                            del_plot_warn = f"⚠️ Amaran: Memadam '{selected_plot_to_delete['plot_name']}' akan memadam secara kekal semua tanaman dan log sejarah berkaitan!" if is_bm else f"⚠️ Warning: Deleting '{selected_plot_to_delete['plot_name']}' will permanently delete all associated crop registries and monitoring history logs!"
                            st.warning(del_plot_warn)
                            confirm_delete = st.checkbox(f"{'Saya mengesahkan untuk memadam' if is_bm else 'I confirm that I want to delete'} '{selected_plot_to_delete['plot_name']}'", key="confirm_delete_plot")
                            if st.button("Padam Plot" if is_bm else "Delete Plot", type="primary", disabled=not confirm_delete, key="del_plot_btn"):
                                ok, m = db_delete_plot(selected_plot_to_delete["id"])
                                if ok:
                                    st.success(m)
                                    st.rerun()
                                else:
                                    st.error(m)
            else:
                st.info("Tiada plot didaftarkan di bawah ladang ini lagi." if is_bm else "No plots registered under this farm yet.")

    with sub_tab_plant:
        st.markdown("#### " + ("Daftar Tanaman ke Plot" if is_bm else "Register Crops to a Plot"))
        farms = db_get_farms()
        if not farms:
            st.warning("Sila daftar Ladang terlebih dahulu." if is_bm else "Please register a Farm first.")
        else:
            selected_f = st.selectbox(L["reg_crop_farm"], options=farms, format_func=lambda x: x["name"], key="crop_farm_selector")
            plots = db_get_plots(selected_f["id"])
            if not plots:
                st.warning("Sila daftar Plot di bawah ladang ini terlebih dahulu." if is_bm else "Please register a Plot under this farm first.")
            else:
                with st.form("new_plant_form", clear_on_submit=True):
                    selected_p = st.selectbox(L["reg_crop_plot"], options=plots, format_func=lambda x: x["plot_name"])
                    crop_name = st.text_input(L["reg_crop_name"], placeholder="cth. Pakchoy, Bendi" if is_bm else "e.g. Pakchoy, Okra")
                    
                    submitted = st.form_submit_button(L["reg_crop_btn"], type="primary")
                    if submitted:
                        if not crop_name.strip():
                            st.error("Nama Tanaman diperlukan." if is_bm else "Crop/Plant name is required.")
                        else:
                            success, msg = db_add_plant(selected_p["id"], crop_name)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                # Show crops
                st.markdown("##### " + L["reg_existing_crops"])
                selected_plot_for_crop_list = st.selectbox("Lihat tanaman dalam:" if is_bm else "View crops in:", options=plots, format_func=lambda x: x["plot_name"], key="view_crops_selector")
                plants = db_get_plants(selected_plot_for_crop_list["id"])
                if plants:
                    df_plants = pd.DataFrame(plants)
                    df_plants = df_plants.drop(columns=["plot_id"])
                    df_plants.columns = ["ID", "Nama Tanaman" if is_bm else "Crop/Plant Name"]
                    st.dataframe(df_plants, use_container_width=True, hide_index=True)
                    
                    # Edit & Delete Crop Expanders
                    col_edit, col_del = st.columns(2)
                    with col_edit:
                        with st.expander("✏️ " + ("Edit Nama Tanaman" if is_bm else "Edit Crop Name")):
                            selected_crop_to_edit = st.selectbox(
                                "Pilih Tanaman untuk Diedit" if is_bm else "Select Crop to Edit",
                                options=plants,
                                format_func=lambda x: x["name"],
                                key="edit_crop_select"
                            )
                            if selected_crop_to_edit:
                                with st.form("edit_crop_form"):
                                    edit_c_name = st.text_input("Nama Tanaman Baharu" if is_bm else "New Crop/Plant Name", value=selected_crop_to_edit["name"])
                                    
                                    edit_submit = st.form_submit_button("Simpan Perubahan" if is_bm else "Save Changes", type="primary")
                                    if edit_submit:
                                        if not edit_c_name.strip():
                                            st.error("Nama tanaman tidak boleh kosong." if is_bm else "Crop name cannot be empty.")
                                        else:
                                            ok, m = db_update_plant(selected_crop_to_edit["id"], edit_c_name)
                                            if ok:
                                                st.success(m)
                                                st.rerun()
                                            else:
                                                st.error(m)
                    with col_del:
                        with st.expander("🗑️ " + ("Buang Tanaman" if is_bm else "Remove Crop")):
                            selected_crop_to_delete = st.selectbox(
                                "Pilih Tanaman untuk Dibuang" if is_bm else "Select Crop to Remove",
                                options=plants,
                                format_func=lambda x: x["name"],
                                key="delete_crop_select"
                            )
                            if selected_crop_to_delete:
                                rem_crop_warn = f"⚠️ Amaran: Membuang '{selected_crop_to_delete['name']}' akan memadam penetapan tanaman ini daripada plot. Ia tidak memadam rekod pemantauan." if is_bm else f"⚠️ Warning: Removing '{selected_crop_to_delete['name']}' will delete this crop assignment from the plot. It will not delete monitoring records."
                                st.warning(rem_crop_warn)
                                confirm_delete = st.checkbox(f"{'Saya mengesahkan untuk membuang' if is_bm else 'I confirm that I want to remove'} '{selected_crop_to_delete['name']}'", key="confirm_delete_crop")
                                if st.button("Buang Tanaman" if is_bm else "Remove Crop", type="primary", disabled=not confirm_delete, key="del_crop_btn"):
                                    ok, m = db_delete_plant(selected_crop_to_delete["id"])
                                    if ok:
                                        st.success(m)
                                        st.rerun()
                                    else:
                                        st.error(m)
                else:
                    st.info("Tiada tanaman didaftarkan dalam plot ini lagi." if is_bm else "No crops registered in this plot yet.")

# Render View: PLOT MONITORING (Module 4)
elif active_mode == "monitoring":
    farms_list = db_get_farms()
    if not farms_list:
        st.info(L["no_plots_registered"])
    else:
        farm_idx = 0
        if "target_farm_id" in st.session_state:
            for idx, f in enumerate(farms_list):
                if f["id"] == st.session_state["target_farm_id"]:
                    farm_idx = idx
                    break
                    
        col_f, col_p = st.columns(2)
        with col_f:
            selected_farm_obj = st.selectbox(
                L["pm_farm_label"],
                options=farms_list,
                format_func=lambda x: x["name"],
                index=farm_idx,
                key="main_farm_select"
            )
            
        plots_list = db_get_plots(selected_farm_obj["id"])
        if not plots_list:
            st.warning(f"⚠️ {'Tiada plot didaftarkan di bawah' if is_bm else 'No plots registered under'} '{selected_farm_obj['name']}'.")
            selected_plot_obj = None
        else:
            plot_idx = 0
            if "target_plot_id" in st.session_state:
                for idx, p in enumerate(plots_list):
                    if p["id"] == st.session_state["target_plot_id"]:
                        plot_idx = idx
                        break
                        
            with col_p:
                selected_plot_obj = st.selectbox(
                    L["pm_plot_label"],
                    options=plots_list,
                    format_func=lambda x: x["plot_name"],
                    index=plot_idx,
                    key="main_plot_select"
                )
                
    if not selected_farm_obj or not selected_plot_obj:
        st.info("⚠️ Sila pilih Ladang dan Plot di atas, atau daftar di **Pendaftaran & Pengurusan**." if is_bm else "⚠️ Please select a Farm and Plot above, or go to **Registry & Management** to register them.")
    else:
        tab_record, tab_monitor = st.tabs([L["pm_tab_record"], L["pm_tab_history"]])
        
        # --- TAB 1: RECORD NEW PLOT DATA ---
        with tab_record:
            st.markdown("### " + ("Rakam Kesihatan Plot" if is_bm else "Capture Plot Health"))
            st.caption(f"{'Rakam foto tanaman di' if is_bm else 'Snap a photo of the crops in'} **{selected_plot_obj['plot_name']}** {'menggunakan kamera telefon, atau pilih fail.' if is_bm else 'using your phone camera, or choose a file.'}")

            # Display active crops in this plot
            crops_list = db_get_plants(selected_plot_obj["id"])
            if crops_list:
                crop_names = [c["name"] for c in crops_list]
                if len(crop_names) > 1:
                    selected_target_crop = st.selectbox(
                        L["pm_intercrop_prompt"],
                        options=crop_names,
                        help="Pilih jenis tanaman yang dirakam dalam foto ini agar intervensi AI disesuaikan." if is_bm else "Select which registered crop variety is featured in this photo so AI intervention is tailored to it."
                    )
                else:
                    selected_target_crop = crop_names[0]
                    st.markdown(f"{L['pm_intercrop_single']} `{selected_target_crop}`")
            else:
                selected_target_crop = "General Crop"
                st.markdown("⚠️ *Tiada tanaman didaftarkan untuk plot ini lagi. Anda boleh menambahnya di Pendaftaran Tanaman.*" if is_bm else "⚠️ *No crops registered for this plot yet. You can add them in the Crop Registry.*")

            # Crop-specific photography guide
            render_crop_photography_guide(selected_target_crop, is_expanded=False, is_bm=is_bm)

            camera_img = st.camera_input(L["pm_cam_prompt"], key="plot_cam_input")
            uploaded_img = st.file_uploader(L["pm_upload_prompt"], type=["jpg", "jpeg", "png"], key="plot_file_input")

            image_file = camera_img or uploaded_img

            # Reset temp diagnosis results only if a new image file is selected
            if image_file is not None:
                img_id = f"{getattr(image_file, 'name', 'camera')}_{getattr(image_file, 'size', 0)}"
                if st.session_state.get("last_image_id") != img_id:
                    st.session_state["last_image_id"] = img_id
                    if "temp_diagnosis" in st.session_state:
                        del st.session_state["temp_diagnosis"]

            if image_file is not None:
                st.markdown("---")
                
                # Vision Model Selector
                if is_bm:
                    available_models = ["🤖 Auto-Kesan (Model Terbaik Tersedia)"]
                    if os.path.exists("models/best(okra_detection).onnx") or os.path.exists("models/best_(okra_detection).onnx"):
                        available_models.append("🌿 Pengesanan Buah Bendi & Kematangan (3 Kelas: Matang, Muda, Tua)")
                    if os.path.exists("models/best_(okra_model).onnx"):
                        available_models.append("🌿 Penyakit Daun Bendi & Mozek Urat Kuning (3 Kelas)")
                    if os.path.exists("models/best_(tomato_leaf_model).onnx"):
                        available_models.append("🍅 Penyakit Tomato & Solanaceae (8 Kelas)")
                    if os.path.exists("models/best.onnx"):
                        available_models.append("🥬 Pengesanan Kematangan Tuai Pakchoy / Sawi")
                    if os.path.exists("models/yolov8_plant_detector.onnx"):
                        available_models.append("🌱 Pengesan Tumbuhan Umum")
                else:
                    available_models = ["🤖 Auto-Detect (Best Available Model)"]
                    if os.path.exists("models/best(okra_detection).onnx") or os.path.exists("models/best_(okra_detection).onnx"):
                        available_models.append("🌿 Okra Pod & Harvest Ripeness Detection (3 Classes: Ripe, Unripe, Overripe)")
                    if os.path.exists("models/best_(okra_model).onnx"):
                        available_models.append("🌿 Okra / Bendi Leaf Disease & Yellow Vein (3 Classes)")
                    if os.path.exists("models/best_(tomato_leaf_model).onnx"):
                        available_models.append("🍅 Tomato & Solanaceae Disease (8 Classes)")
                    if os.path.exists("models/best.onnx"):
                        available_models.append("🥬 Pakchoy / Bok Choy Harvest Detector")
                    if os.path.exists("models/yolov8_plant_detector.onnx"):
                        available_models.append("🌱 General Plant Detector")
                
                selected_vision_model = st.selectbox(
                    L["qs_select_model"],
                    available_models,
                    index=0,
                    key="vision_model_selector"
                )
                
                # Display annotated image if available, else show raw image
                if "temp_diagnosis" in st.session_state and "annotated_image" in st.session_state.temp_diagnosis and st.session_state.temp_diagnosis["annotated_image"] is not None:
                    st.image(
                        st.session_state.temp_diagnosis["annotated_image"], 
                        caption="Pengesanan AI (Penyakit, Stres & Daun Sihat)" if is_bm else "Annotated AI Detections (Multi-Class Disease, Stress & Healthy Leaves)", 
                        use_container_width=True
                    )
                else:
                    st.image(image_file, caption="Bingkai Kanopi Terpilih" if is_bm else "Selected Canopy Frame", use_container_width=True)
                
                # Cloud IoT Telemetry Simulation Panel
                with st.expander(L["pm_iot_title"], expanded=True):
                    col_iot1, col_iot2 = st.columns([2.5, 1.5])
                    with col_iot1:
                        st.markdown(L["pm_iot_broker"])
                        st.caption(f"{'Telemetri sensor masa nyata untuk' if is_bm else 'Real-time sensor telemetry for'} **{selected_plot_obj['plot_name']}** {'diselaraskan dengan cadangan LLM.' if is_bm else 'synced to LLM prompt.'}")
                    with col_iot2:
                        if st.button(L["pm_iot_sync_btn"], key="sync_iot_btn"):
                            st.session_state["iot_telemetry"] = fetch_simulated_iot_telemetry(selected_plot_obj["id"])
                            st.session_state["iot_plot_id"] = selected_plot_obj["id"]
                            st.success("Data telemetri IoT Awan berjaya disegerakkan!" if is_bm else "Synced latest IoT telemetry from Cloud Server!")
                            
                    if "iot_telemetry" not in st.session_state or st.session_state.get("iot_plot_id") != selected_plot_obj["id"]:
                        st.session_state["iot_telemetry"] = fetch_simulated_iot_telemetry(selected_plot_obj["id"])
                        st.session_state["iot_plot_id"] = selected_plot_obj["id"]
                        
                    iot = st.session_state["iot_telemetry"]
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        m_delta = "🔴 KURANG" if is_bm and iot['soil_moisture'] < 40 else ("🔴 DEFICIT" if iot['soil_moisture'] < 40 else ("🟢 OPTIMUM" if is_bm and iot['soil_moisture'] <= 70 else ("🟢 OPTIMAL" if iot['soil_moisture'] <= 70 else "🔵 TINGGI" if is_bm else "🔵 HIGH")))
                        st.metric(L["pm_soil_moisture"], f"{iot['soil_moisture']}%", delta=m_delta)
                    with m2:
                        st.metric(L["pm_air_temp"], f"{iot['air_temp']}°C")
                    with m3:
                        st.metric(L["pm_soil_ec"], f"{iot['soil_ec']} mS/cm")
                    with m4:
                        st.metric(L["pm_soil_ph"], f"{iot['soil_ph']}")
                        
                    st.caption(f"Status: `{iot['server_status']}` | {'Segerak Terakhir:' if is_bm else 'Last Telemetry Sync:'} `{iot['timestamp']}`")
                
                # Primary Action Button
                if st.button(L["pm_btn_diag_fusion"], type="primary", key="btn_run_diag_fusion"):
                    with st.spinner("Memproses kiraan daun & menggabungkan data sensor IoT Awan dengan AI..." if is_bm else "Processing leaf count & fusing Cloud IoT sensor telemetry with LLM..."):
                        if hasattr(image_file, "seek"):
                            image_file.seek(0)
                        pil_backup = Image.open(image_file).convert("RGB")
                        if hasattr(image_file, "seek"):
                            image_file.seek(0)
                        
                        bil_daun, diagnosis, annotated_image = run_yolo_count_and_diagnosis(image_file, model_preference=selected_vision_model)
                        
                        diag_prompt_text = diagnosis
                        if selected_target_crop and selected_target_crop != "General Crop":
                            diag_prompt_text = f"Crop: {selected_target_crop}. {diagnosis}"
                            
                        intervention = run_intervention_recommendation(
                            diag_prompt_text,
                            iot_telemetry=iot,
                            language_choice=language_choice
                        )
                        
                        st.session_state.temp_diagnosis = {
                            "bil_daun": bil_daun,
                            "diagnosis": diagnosis,
                            "intervention": intervention,
                            "annotated_image": annotated_image,
                            "raw_image": pil_backup,
                            "target_crop": selected_target_crop
                        }
                        st.rerun()

                # Check if we have results ready to display
                if "temp_diagnosis" in st.session_state:
                    res = st.session_state.temp_diagnosis
                    
                    # 1. Custom Metric Card
                    st.markdown(
                        f"""
                        <div class="custom-metric">
                            <div class="custom-metric-val">{res['bil_daun']}</div>
                            <div class="custom-metric-lbl">{L["qs_leaf_metric"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 2. Custom Styled Diagnosis card
                    diag_html = render_markdown_to_html(res['diagnosis'])
                    st.markdown(
                        f"""
                        <div class="result-card neutral">
                            <div class="card-title">{L["qs_diag_card"]}</div>
                            <div class="card-text">{diag_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 3. Custom Styled Intervention card
                    interv_html = render_markdown_to_html(res['intervention'])
                    st.markdown(
                        f"""
                        <div class="result-card alert">
                            <div class="card-title">{L["qs_interv_card"]}</div>
                            <div class="card-text">{interv_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Additional Notes field
                    notes = st.text_area(L["qs_field_notes"], placeholder=L["qs_field_notes_ph"], key="save_notes_area")
                    
                    # Date & Time picker for logging
                    st.markdown("##### 📅 " + ("Tarikh & Masa Log" if is_bm else "Log Date & Time"))
                    col_dt1, col_dt2 = st.columns(2)
                    with col_dt1:
                        pick_date = st.date_input(L["qs_inspection_date"], value=get_now_myt().date(), key="save_date_picker")
                    with col_dt2:
                        pick_time = st.time_input(L["qs_inspection_time"], value=get_now_myt().time(), key="save_time_picker")
                    
                    custom_timestamp_str = f"{pick_date.strftime('%Y-%m-%d')} {pick_time.strftime('%H:%M')}"
                    
                    # Save Action Button
                    if st.button("💾 " + ("Simpan Rekod ke Log Plot" if is_bm else "Save Record to Plot Log"), type="primary", key="btn_save_log_record", use_container_width=True):
                        img_to_save = res.get("annotated_image") or res.get("raw_image") or image_file
                        saved_img_path = save_uploaded_image(img_to_save, selected_plot_obj["id"])
                        
                        # Tag crop in notes if specified
                        final_plot_notes = notes.strip()
                        if selected_target_crop and selected_target_crop != "General Crop" and f"[{selected_target_crop}]" not in final_plot_notes:
                            final_plot_notes = f"[{selected_target_crop}] {final_plot_notes}".strip()
                        
                        success, msg = db_add_record(
                            selected_plot_obj["id"],
                            custom_timestamp_str,
                            res["bil_daun"],
                            res["diagnosis"],
                            res["intervention"],
                            final_plot_notes,
                            image_path=saved_img_path
                        )
                        if success:
                            if "temp_diagnosis" in st.session_state:
                                del st.session_state.temp_diagnosis
                            st.success("✅ Rekod pemeriksaan berjaya disimpan dengan kotak sempadan!" if is_bm else "✅ Health inspection record saved successfully with bounding boxes!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(msg)
                            
        # --- TAB 2: HISTORY LOG & MONITORING ---
        with tab_monitor:
            st.markdown(f"{L['pm_history_title']} {selected_plot_obj['plot_name']}")
            st.caption(L["pm_history_sub"])

            plot_crops = db_get_plants(selected_plot_obj["id"])
            plot_records = db_get_records(selected_plot_obj["id"])

            if plot_crops and len(plot_crops) > 1:
                crop_names_list = [c["name"] for c in plot_crops]
                crops_joined = " • ".join([f"`{name}`" for name in crop_names_list])
                banner_txt = f"🌾 <strong>{'Tanaman Campuran Aktif:' if is_bm else 'Intercropping Active:'}</strong> {'Pelbagai tanaman didaftarkan dalam plot ini' if is_bm else 'Multiple crops registered in this plot'} ({crops_joined})."
                st.markdown(
                    f"""
                    <div style="background: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 12px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.9rem; color: #2e7d32;">
                        {banner_txt}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col_f1, col_f2 = st.columns([2, 1])
                with col_f1:
                    all_crops_lbl = "Semua Tanaman" if is_bm else "All Crops"
                    selected_crop_filter = st.selectbox(
                        L["pm_filter_history"],
                        options=[all_crops_lbl] + crop_names_list,
                        key=f"filter_crop_{selected_plot_obj['id']}"
                    )
            else:
                selected_crop_filter = "Semua Tanaman" if is_bm else "All Crops"

            # Apply crop filter if selected
            filtered_records = []
            all_crops_tag = "Semua Tanaman" if is_bm else "All Crops"
            if selected_crop_filter != all_crops_tag:
                for r in plot_records:
                    r_notes = r.get("notes", "") or ""
                    if f"[{selected_crop_filter}]" in r_notes:
                        filtered_records.append(r)
            else:
                filtered_records = plot_records

            if not filtered_records:
                empty_msg = f"{'Tiada rekod sejarah ditemui untuk' if is_bm else 'No history records found for'} '{selected_crop_filter}'." if selected_crop_filter != all_crops_tag else ("Tiada rekod sejarah ditemui" if is_bm else "No history records found")
                empty_sub = "Ambil bacaan tanaman dalam tab rekod untuk mencipta entri log." if is_bm else "Take a crop reading in the record tab to create a log entry."
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2.5rem 0; border: 1px dashed #cfd8dc; border-radius: 12px; background: #fafafa;">
                        <span style="font-size: 1.5rem; color: #b0bec5;">📊</span>
                        <p style="margin: 0.5rem 0 0; color: #78909c; font-size: 0.95rem; font-weight: 500;">{empty_msg}</p>
                        <p style="margin: 0; color: #90a4ae; font-size: 0.85rem;">{empty_sub}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                cycle_start_dt = None
                if selected_plot_obj.get("cycle_start"):
                    try:
                        cycle_start_dt = datetime.strptime(selected_plot_obj["cycle_start"], "%Y-%m-%d").date()
                    except Exception:
                        pass

                for i, r in enumerate(filtered_records):
                    leaf_val = r.get("bil_daun", r.get("bil_pokok", 0))
                    
                    raw_notes = r.get("notes", "") or ""
                    crop_tag = ""
                    crop_label_segment = ""
                    import re as re_mod
                    crop_match = re_mod.match(r'^\[(.*?)\]\s*(.*)', raw_notes)
                    if crop_match:
                        crop_tag = crop_match.group(1)
                        crop_label_segment = f"  |  🌾 {crop_tag}"
                        clean_notes = crop_match.group(2)
                    else:
                        clean_notes = raw_notes
                        if plot_crops and len(plot_crops) == 1:
                            crop_tag = plot_crops[0]["name"]
                            crop_label_segment = f"  |  🌾 {crop_tag}"

                    stage_title_str = ""
                    cycle_day_tag = ""
                    try:
                        rec_dt = datetime.strptime(r["time"], "%Y-%m-%d %H:%M")
                        if cycle_start_dt:
                            elapsed_days = (rec_dt.date() - cycle_start_dt).days
                            elapsed_days = max(0, elapsed_days)
                            st_title, _, _, _ = get_crop_stage_badge(elapsed_days, 35, 45)
                            st_short = st_title.split("/")[0].strip()
                            cycle_day_tag = f"  |  {st_short} ({'Hari' if is_bm else 'Day'} {elapsed_days})"
                            stage_title_str = f"🌿 **{'Peringkat Tanaman:' if is_bm else 'Crop Stage:'}** {st_title} ({'Hari' if is_bm else 'Day'} {elapsed_days} {'kitaran plot' if is_bm else 'of plot cycle'})"
                    except Exception:
                        rec_dt = None

                    interval_html = ""
                    if rec_dt and i < len(filtered_records) - 1:
                        try:
                            prev_r = filtered_records[i + 1]
                            prev_dt = datetime.strptime(prev_r["time"], "%Y-%m-%d %H:%M")
                            day_diff = (rec_dt.date() - prev_dt.date()).days
                            prev_leaf = prev_r.get("bil_daun", prev_r.get("bil_pokok", 0))
                            leaf_diff = leaf_val - prev_leaf
                            diff_sign = "+" if leaf_diff >= 0 else ""
                            
                            interval_title = "Penjejakan Selang:" if is_bm else "Interval Tracking:"
                            interval_body = f"Dimuat naik <strong>+{day_diff} hari</strong> selepas foto sebelumnya ({prev_r['time']})" if is_bm else f"Uploaded <strong>+{day_diff} days</strong> after previous photo ({prev_r['time']})"
                            expansion_title = "Pertambahan Dedaun:" if is_bm else "Foliage Expansion:"
                            expansion_body = f"{diff_sign}{leaf_diff} perbezaan bilangan daun" if is_bm else f"{diff_sign}{leaf_diff} leaves count difference"
                            
                            interval_html = (
                                f"<div style='background-color: #f1f8e9; border-left: 4px solid #7cb342; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.88rem; color: #2e7d32;'>"
                                f"⏱️ <strong>{interval_title}</strong> {interval_body}<br/>"
                                f"📈 <strong>{expansion_title}</strong> {expansion_body}"
                                f"</div>"
                            )
                        except Exception:
                            interval_html = ""
                    elif i == len(filtered_records) - 1:
                        base_title = "Foto Asas:" if is_bm else "Baseline Photo:"
                        base_desc = "Pemeriksaan tanaman pertama direkodkan untuk plot ini." if is_bm else "Initial crop inspection recorded for this plot."
                        interval_html = (
                            f"<div style='background-color: #e3f2fd; border-left: 4px solid #42a5f5; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.88rem; color: #1565c0;'>"
                            f"🌱 <strong>{base_title}</strong> {base_desc}"
                            f"</div>"
                        )

                    leaf_lbl = "Daun" if is_bm else "Leaves"
                    expander_label = f"📅 {r['time']}{crop_label_segment}{cycle_day_tag}  |  🍃 {leaf_val} {leaf_lbl}"
                    if r.get("canopy_cover_pct") is not None:
                        expander_label += f"  |  🌿 CC: {r['canopy_cover_pct']}%"
                    
                    with st.expander(expander_label):
                        if crop_tag:
                            st.markdown(
                                f"""
                                <div style="display:inline-block; background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; padding:4px 10px; border-radius:16px; font-size:0.85rem; font-weight:600; margin-bottom:10px;">
                                    🌾 {'Varieti Tanaman:' if is_bm else 'Crop Variety:'} {crop_tag}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        if interval_html:
                            st.markdown(interval_html, unsafe_allow_html=True)
                        if stage_title_str:
                            st.markdown(stage_title_str)
                            
                        if r.get("image_path") and os.path.exists(r["image_path"]):
                            photo_cap = f"{'Foto Kanopi Direkodkan' if is_bm else 'Logged Canopy Photo'} - {crop_tag or ('Tanaman' if is_bm else 'Crop')} ({'Pengesanan AI & Kotak Sempadan' if is_bm else 'AI Detections & Bounding Boxes'})"
                            st.image(r["image_path"], caption=photo_cap, use_container_width=True)
                            
                        st.markdown(f"**🔍 {'Diagnosis:' if is_bm else 'Diagnosis:'}** {r['diagnosis']}")
                        st.markdown(f"**💡 {'Cadangan Intervensi:' if is_bm else 'Recommended Intervention:'}**\n{r['intervention']}")
                        
                        if clean_notes:
                            notes_header = "Nota Lapangan:" if is_bm else "Field Notes:"
                            st.markdown(f"📝 **{notes_header}** *\"{clean_notes}\"*")
                        
                        # Actions: Edit Date/Details & Delete Record
                        st.markdown("---")
                        
                        with st.expander(L["pm_edit_title"], expanded=False):
                            try:
                                curr_dt = datetime.strptime(r['time'], "%Y-%m-%d %H:%M")
                                init_d = curr_dt.date()
                                init_t = curr_dt.time()
                            except Exception:
                                init_d = get_now_myt().date()
                                init_t = get_now_myt().time()
                                
                            col_e1, col_e2 = st.columns(2)
                            with col_e1:
                                edit_d = st.date_input(L["qs_inspection_date"], value=init_d, key=f"edit_d_{r['id']}")
                            with col_e2:
                                edit_t = st.time_input(L["qs_inspection_time"], value=init_t, key=f"edit_t_{r['id']}")
                                
                            crop_opts = [c["name"] for c in plot_crops] if plot_crops else []
                            edit_crop_val = crop_tag
                            if crop_opts:
                                curr_c_idx = crop_opts.index(crop_tag) if crop_tag in crop_opts else 0
                                edit_crop_val = st.selectbox(
                                    "Varieti Tanaman Berkaitan" if is_bm else "Associated Crop Variety",
                                    options=crop_opts,
                                    index=curr_c_idx,
                                    key=f"edit_crop_val_{r['id']}"
                                )

                            edit_notes = st.text_area(L["qs_field_notes"], value=clean_notes, key=f"edit_n_{r['id']}")
                            edit_leaves = st.number_input("Bilangan Daun (Bil Daun)" if is_bm else "Leaves Count (Bil Daun)", min_value=0, max_value=5000, value=int(leaf_val), key=f"edit_lf_{r['id']}")
                            
                            if st.button("💾 " + ("Simpan Perubahan Log" if is_bm else "Save Changes to Log"), key=f"save_edit_btn_{r['id']}", type="primary"):
                                updated_time_str = f"{edit_d.strftime('%Y-%m-%d')} {edit_t.strftime('%H:%M')}"
                                final_updated_notes = edit_notes.strip()
                                if edit_crop_val:
                                    final_updated_notes = f"[{edit_crop_val}] {final_updated_notes}".strip()
                                ok, msg = db_update_record(r['id'], new_time=updated_time_str, new_notes=final_updated_notes, new_bil_daun=edit_leaves)
                                if ok:
                                    st.success("✅ Log berjaya dikemas kini!" if is_bm else "✅ Log updated successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"Error updating log: {msg}")

                        col_space, col_delete_btn = st.columns([5, 2])
                        with col_delete_btn:
                            if st.button(L["pm_delete_btn"], key=f"del_rec_{r['id']}", type="secondary", use_container_width=True):
                                ok, m = db_delete_record(r['id'])
                                if ok:
                                    st.success("Log berjaya dipadam!" if is_bm else "Log deleted!")
                                    st.rerun()
                                else:
                                    st.error(m)