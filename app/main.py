"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import health, migration
from app.config import get_settings
from app.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    logger = get_logger(__name__)
    logger.info("Starting %s v%s (%s)", settings.app_name, __version__, settings.environment)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Migrate to a new embedding model without re-embedding the whole corpus.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(migration.router)
    return app


app = create_app()
