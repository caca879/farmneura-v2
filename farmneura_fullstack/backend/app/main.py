import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine, Base
from app.models import db_models
from app.api.v1.api_router import api_router


# Create database tables
Base.metadata.create_all(bind=engine)

# Auto-migrate missing columns & tables for existing databases (PostgreSQL & SQLite)
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE farms ADD COLUMN user_id VARCHAR(36);"))
        conn.commit()
except Exception:
    pass

try:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS harvest_records (
                id VARCHAR(36) PRIMARY KEY,
                plot_id VARCHAR(36) NOT NULL,
                user_id VARCHAR(36),
                yield_weight_kg FLOAT NOT NULL DEFAULT 0.0,
                price_per_kg_myr FLOAT NOT NULL DEFAULT 0.0,
                total_revenue_myr FLOAT NOT NULL DEFAULT 0.0,
                harvest_date VARCHAR(20) NOT NULL,
                notes VARCHAR(255),
                created_at TIMESTAMP
            );
        """))
        conn.commit()
except Exception as e:
    print("Auto-migration harvest_records note:", e)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploaded image storage directory
app.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
