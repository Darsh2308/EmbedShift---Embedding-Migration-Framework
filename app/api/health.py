"""Health check endpoint."""

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
