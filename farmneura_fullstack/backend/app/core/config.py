import os

class Settings:
    PROJECT_NAME: str = "FarmNeura v2 API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database Settings (PostgreSQL with SQLite fallback)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./farmneura_fullstack.db"
    )
    
    # LLM & AI Settings (Supports Gemini, Groq, and OpenAI)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_APT_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


    
    # Storage Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOADS_DIR: str = os.path.join(BASE_DIR, "uploads")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models_onnx")

settings = Settings()

os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
