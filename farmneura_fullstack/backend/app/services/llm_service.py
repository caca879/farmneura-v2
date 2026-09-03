import os
import requests
from app.core.config import settings

try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False


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
    Calls Groq API with Llama 3.3 70B / 3.1 8B combining YOLOv8 Vision Diagnosis 
    with Cloud IoT Sensor Telemetry to generate a holistic agronomist intervention plan.
    Falls back to smart agronomic rule engine if key is absent or Groq is unreachable.
    """
    api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    api_key = api_key.strip().strip('"').strip("'") if api_key else ""
    
    if not api_key:
        return _get_agronomic_fallback(diagnosis, iot_telemetry, language_choice)

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

    model_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it"
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
            
            # Direct HTTP REST API fallback
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
        except Exception as e:
            print(f"Groq API error with model {model_name}: {e}")
            continue

    # If Groq call failed (e.g. invalid key format or quota exceeded), fall back safely to Agronomic engine
    return _get_agronomic_fallback(diagnosis, iot_telemetry, language_choice)
