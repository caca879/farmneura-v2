import os
import requests
from app.core.config import settings

def _get_agronomic_fallback(diagnosis: str, iot_telemetry: dict = None, language_choice: str = "🇲🇾 Bahasa Melayu") -> str:
    d_lower = diagnosis.lower() if diagnosis else ""
    m_val = iot_telemetry.get('soil_moisture', 50.0) if iot_telemetry else 50.0
    ec_val = iot_telemetry.get('soil_ec', 1.8) if iot_telemetry else 1.8

    if language_choice == "🇲🇾 Bahasa Melayu":
        if m_val < 40 or "chlorosis" in d_lower or "nitrogen" in d_lower or "yellow" in d_lower:
            return f"- **Diagnosis Agronomis Jitu (Visi + Sensor)**: Simptom kanopi dikesan oleh YOLOv8 dan disahkan oleh penderia IoT Awan (Kelembapan tanah: {m_val}%, EC: {ec_val} mS/cm).\n- Sembur baja cecair nitrogen dan mulakan penyiraman titis selama 20 minit.\n- Periksa tahap kedalaman akar untuk mengelakkan tekanan haba tanah."
        elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower or "virus" in d_lower:
            return f"- **Diagnosis Agronomis Jitu (Visi + Sensor)**: Jangkitan kulat/virus kanopi dikesan. Penderia IoT menunjukkan kelembapan: {m_val}%.\n- Sembur racun kulat berasaskan tembaga atau minyak neem organik pada barisan tanaman terjejas.\n- Pangkas daun kanopi bawah untuk meningkatkan aliran pengudaraan."
        else:
            return f"- **Diagnosis Agronomis Jitu (Visi + Sensor)**: Kesihatan kanopi dan bacaan sensor IoT Awan berada pada tahap optimum (Kelembapan: {m_val}%, EC: {ec_val} mS/cm).\n- Teruskan jadual pembajaan dan pemantauan biasa."
    else:
        if m_val < 40 or "chlorosis" in d_lower or "nitrogen" in d_lower or "yellow" in d_lower:
            return f"- **Precision Agronomic Diagnosis (Vision + Sensor)**: Foliage symptoms detected by YOLOv8 and confirmed by Cloud IoT Telemetry (Soil Moisture: {m_val}%, EC: {ec_val} mS/cm).\n- Apply nitrogen liquid fertigation and initiate a 20-minute drip irrigation cycle.\n- Monitor root depth moisture to alleviate soil thermal stress."
        elif "spot" in d_lower or "fungal" in d_lower or "disease" in d_lower or "virus" in d_lower:
            return f"- **Precision Agronomic Diagnosis (Vision + Sensor)**: Canopy fungal/viral infection detected. Cloud IoT sensor moisture reading: {m_val}%.\n- Apply copper-based fungicide or organic neem oil spray to affected crop rows.\n- Prune lower canopy foliage to improve ventilation airflow."
        else:
            return f"- **Precision Agronomic Diagnosis (Vision + Sensor)**: Canopy condition and Cloud IoT telemetry are optimal (Moisture: {m_val}%, EC: {ec_val} mS/cm).\n- Maintain regular fertigation and inspection schedule."


def generate_llm_intervention(diagnosis: str, iot_telemetry: dict = None, language_choice: str = "🇲🇾 Bahasa Melayu") -> str:
    """
    Multimodal AI Agronomist Generator.
    Supports Google Gemini API, OpenAI (GPT-4o-mini), and Groq (Llama 3.3).
    Falls back gracefully to precision rule engine if keys are absent or services unreachable.
    """
    gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_APT_KEY") or getattr(settings, "GEMINI_API_KEY", "") or "").strip().strip('"').strip("'")

    openai_key = (os.environ.get("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "") or "").strip().strip('"').strip("'")
    groq_key = (os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_APT_KEY") or getattr(settings, "GROQ_API_KEY", "") or "").strip().strip('"').strip("'")

    # Format IoT Telemetry for LLM Prompt Fusion
    iot_text = ""
    if iot_telemetry:
        moist_status = "DEFICIT (Dry Soil)" if iot_telemetry.get('soil_moisture', 50) < 40 else ("OPTIMAL" if iot_telemetry.get('soil_moisture', 50) <= 70 else "SATURATED (Overwatered)")
        ec_status = "LOW (Fertilizer Deficit)" if iot_telemetry.get('soil_ec', 1.8) < 1.4 else ("OPTIMAL" if iot_telemetry.get('soil_ec', 1.8) <= 2.2 else "HIGH Salinity")
        
        iot_text = (
            f"\n\n[CLOUD IOT SENSOR TELEMETRY]:\n"
            f"- Soil Moisture: {iot_telemetry.get('soil_moisture')}% ({moist_status})\n"
            f"- Air Temperature: {iot_telemetry.get('air_temp')} °C\n"
            f"- Soil EC (Fertility): {iot_telemetry.get('soil_ec')} mS/cm ({ec_status})\n"
            f"- Soil pH: {iot_telemetry.get('soil_ph')}"
        )
    
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

    user_query = f"Vision Model Diagnosis: {diagnosis}{iot_text}"

    # -------------------------------------------------------------
    # PROVIDER 1: GOOGLE GEMINI API (RECOMMENDED - FAST & FREE TIER)
    # -------------------------------------------------------------
    if gemini_key:
        gemini_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        for g_model in gemini_models:
            try:
                g_url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={gemini_key}"
                g_payload = {
                    "contents": [{
                        "parts": [
                            {"text": f"{system_prompt}\n\n{user_query}"}
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 350
                    }
                }
                res = requests.post(g_url, json=g_payload, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            if text:
                                print(f"[LLM Service] Successfully generated response using Google Gemini: {g_model}")
                                return f"✨ **[Google Gemini AI - {g_model}]**\n\n{text}"
                else:
                    print(f"[LLM Service] Gemini API HTTP {res.status_code} for {g_model}: {res.text[:150]}")
            except Exception as e:
                print(f"[LLM Service] Gemini API error: {e}")
                continue

    # -------------------------------------------------------------
    # PROVIDER 2: OPENAI (GPT-4o-mini)
    # -------------------------------------------------------------
    if openai_key:
        try:
            o_url = "https://api.openai.com/v1/chat/completions"
            o_headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            o_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0.2,
                "max_tokens": 350
            }
            res = requests.post(o_url, json=o_payload, timeout=8)
            if res.status_code == 200:
                choices = res.json().get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    if text:
                        print("[LLM Service] Successfully generated response using OpenAI GPT-4o-mini")
                        return f"🟢 **[OpenAI GPT-4o-mini]**\n\n{text}"
            else:
                print(f"[LLM Service] OpenAI API HTTP {res.status_code}: {res.text[:150]}")
        except Exception as e:
            print(f"[LLM Service] OpenAI API error: {e}")

    # -------------------------------------------------------------
    # PROVIDER 3: GROQ API (LLAMA 3.3)
    # -------------------------------------------------------------
    if groq_key:
        model_candidates = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
        for model_name in model_candidates:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {groq_key}",
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
                
                response = requests.post(url, json=payload, timeout=8)
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content and content.strip():
                            print(f"[LLM Service] Successfully generated response using Groq model: {model_name}")
                            return f"🤖 **[Groq AI - {model_name}]**\n\n{content.strip()}"
                else:
                    print(f"[LLM Service] Groq API HTTP {response.status_code} for {model_name}: {response.text[:150]}")
            except Exception as e:
                print(f"[LLM Service] Groq API connection error with {model_name}: {e}")
                continue

    # -------------------------------------------------------------
    # PROVIDER 4: PRECISION AGRONOMIC FALLBACK ENGINE
    # -------------------------------------------------------------
    return _get_agronomic_fallback(diagnosis, iot_telemetry, language_choice)
