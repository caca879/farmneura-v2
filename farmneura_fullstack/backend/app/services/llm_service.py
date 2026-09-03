import os
import requests
from app.core.config import settings


def _get_gemini_models(api_key: str) -> list[str]:
    configured_model = getattr(settings, "GEMINI_MODEL", "").strip()
    models = [configured_model] if configured_model else []

    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=8,
        )
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            discovered_models = [
                model.get("name", "").removeprefix("models/")
                for model in available_models
                if "generateContent" in model.get("supportedGenerationMethods", [])
            ]
            models.extend(model for model in preferred_models if model in discovered_models)
            models.extend(model for model in discovered_models if model not in models)
        else:
            print(f"[LLM Service] Gemini model discovery HTTP {response.status_code}: {response.text[:150]}")
            models.extend(preferred_models)
    except requests.RequestException as error:
        print(f"[LLM Service] Gemini model discovery error: {error}")
        models.extend(preferred_models)

    if not models:
        models = preferred_models

    return list(dict.fromkeys(models))


def _clean_llm_response(text: str, language_choice: str) -> str:
    if not text:
        return ""

    metadata_prefixes = (
        "role:", "expertise:", "input:", "input data:", "goal:", "constraint:",
        "vision diagnosis:", "diagnosis:", "peranan:", "kepakaran:", "analisis:",
        "cadangan:", "tindakan:", "recommendation:", "recommendations:",
        "action plan:", "model:", "prompt:", "vision model diagnosis:"
    )

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = line.lstrip("-*•0123456789.) ").strip().lower().replace("**", "")

        # Skip lines that start with metadata tags or prompt echo
        if any(normalized.startswith(prefix) for prefix in metadata_prefixes):
            continue

        # Skip short header lines ending with colon (e.g. "Cadangan Tindakan:")
        if normalized.endswith(":") and len(normalized.split()) <= 5:
            continue

        bullet_text = line.lstrip("-*•0123456789.) ").replace("**", "").strip()
        if bullet_text:
            cleaned_lines.append(f"- {bullet_text}")

    if not cleaned_lines:
        return ""

    # Prevent returning prompt echoes in Malay mode
    is_english = "english" in (language_choice or "").lower()
    if not is_english:
        sample_lower = " ".join(cleaned_lines).lower()
        if any(bad in sample_lower for bad in ("vision diagnosis", "precision agricultural agent", "cloud iot sensor telemetry")):
            return ""

    return "\n".join(cleaned_lines[:4]).strip()

def _get_agronomic_fallback(diagnosis: str, iot_telemetry: dict = None, language_choice: str = "🇲🇾 Bahasa Melayu") -> str:
    d_lower = diagnosis.lower() if diagnosis else ""
    m_val = iot_telemetry.get('soil_moisture', 50.0) if iot_telemetry else 50.0
    ec_val = iot_telemetry.get('soil_ec', 1.8) if iot_telemetry else 1.8

    is_english = "english" in (language_choice or "").lower()
    if not is_english:
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
        
        if "english" in (language_choice or "").lower():
            iot_text = (
                f"\nSoil moisture: {iot_telemetry.get('soil_moisture')}% ({moist_status})\n"
                f"Air temperature: {iot_telemetry.get('air_temp')} C\n"
                f"Soil EC: {iot_telemetry.get('soil_ec')} mS/cm ({ec_status})\n"
                f"Soil pH: {iot_telemetry.get('soil_ph')}"
            )
        else:
            iot_text = (
                f"\nKelembapan tanah: {iot_telemetry.get('soil_moisture')}% ({'Kering' if moist_status.startswith('DEFICIT') else 'Optimum' if moist_status == 'OPTIMAL' else 'Terlalu basah'})\n"
                f"Suhu udara: {iot_telemetry.get('air_temp')} C\n"
                f"EC tanah: {iot_telemetry.get('soil_ec')} mS/cm ({'Kurang baja' if ec_status.startswith('LOW') else 'Optimum' if ec_status == 'OPTIMAL' else 'Kemasinan tinggi'})\n"
                f"pH tanah: {iot_telemetry.get('soil_ph')}"
            )
    
    is_english = "english" in (language_choice or "").lower()
    if not is_english:
        system_prompt = (
            "Anda adalah ejen pertanian jitu FarmNeura, seorang agronomis profesional. "
            "Berdasarkan diagnosis visi tanaman dan data penderia IoT, berikan TEPAT 3 tindakan agronomis praktikal untuk petani. "
            "WAJIB dijawab dalam Bahasa Melayu sahaja. "
            "Terus mulakan setiap baris dengan simbol '-'. "
            "DILARANG menulis sebarang pengenalan, tajuk, peranan, label input, atau proses analisis."
        )
        user_query = f"Diagnosis Tanaman: {diagnosis}{iot_text}\n\nBerikan 3 tindakan agronomis segera:"
    else:
        system_prompt = (
            "You are FarmNeura's precision agricultural agent, a professional agronomist. "
            "Based on crop vision diagnosis and IoT sensor telemetry, provide EXACTLY 3 practical agronomic recovery actions for the farmer. "
            "MUST be answered in English only. "
            "Start each line directly with '-'. "
            "DO NOT include any introduction, title, role explanation, input labels, or analysis steps."
        )
        user_query = f"Crop Diagnosis: {diagnosis}{iot_text}\n\nProvide 3 immediate agronomic actions:"

    # -------------------------------------------------------------
    # PROVIDER 1: GOOGLE GEMINI API (RECOMMENDED - FAST & FREE TIER)
    # -------------------------------------------------------------
    if gemini_key:
        gemini_models = _get_gemini_models(gemini_key)
        for g_model in gemini_models:

            try:
                g_url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={gemini_key}"
                g_payload = {
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "contents": [{
                        "parts": [
                            {"text": user_query}
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 500
                    }
                }
                res = requests.post(g_url, json=g_payload, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "").strip()
                            clean_text = _clean_llm_response(raw_text, language_choice)
                            if clean_text:
                                print(f"[LLM Service] Successfully generated response using Google Gemini: {g_model}")
                                return clean_text
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
                "max_tokens": 400
            }
            res = requests.post(o_url, json=o_payload, timeout=8)
            if res.status_code == 200:
                choices = res.json().get("choices", [])
                if choices:
                    raw_text = choices[0].get("message", {}).get("content", "").strip()
                    clean_text = _clean_llm_response(raw_text, language_choice)
                    if clean_text:
                        print("[LLM Service] Successfully generated response using OpenAI GPT-4o-mini")
                        return clean_text
            else:
                print(f"[LLM Service] OpenAI API HTTP {res.status_code}: {res.text[:150]}")
        except Exception as e:
            print(f"[LLM Service] OpenAI API error: {e}")

    # -------------------------------------------------------------
    # PROVIDER 3: GROQ API (LLAMA / GPT-OSS)
    # -------------------------------------------------------------
    if groq_key:
        model_candidates = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound-mini",
            "groq/compound",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
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
                    "max_tokens": 400
                }
                
                response = requests.post(url, json=payload, timeout=8)
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        clean_content = _clean_llm_response(content.strip(), language_choice)
                        if clean_content:
                            print(f"[LLM Service] Successfully generated response using Groq model: {model_name}")
                            return clean_content
                else:
                    print(f"[LLM Service] Groq API HTTP {response.status_code} for {model_name}: {response.text[:150]}")
            except Exception as e:
                print(f"[LLM Service] Groq API connection error with {model_name}: {e}")
                continue

    # -------------------------------------------------------------
    # PROVIDER 4: PRECISION AGRONOMIC FALLBACK ENGINE
    # -------------------------------------------------------------
    return _get_agronomic_fallback(diagnosis, iot_telemetry, language_choice)
