from fastapi import APIRouter
from app.api.v1.endpoints import farms, plots, crops, telemetry, inspections, overview, auth, harvests

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(farms.router)
api_router.include_router(plots.router)
api_router.include_router(crops.router)
api_router.include_router(telemetry.router)
api_router.include_router(inspections.router)
api_router.include_router(overview.router)
api_router.include_router(harvests.router)



