"""FastAPI application for labeling service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cat_watcher.collection.storage import FrameStorage
from cat_watcher.config import Settings, get_settings
from cat_watcher.labeling.api import router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings: Settings = app.state.settings
    data_dir = Path(settings.data_dir)

    # Initialize storage
    app.state.storage = FrameStorage(data_dir)

    logger.info(
        "Labeling service started",
        data_dir=str(data_dir),
        port=settings.labeling.port,
    )

    yield

    logger.info("Labeling service stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: Application settings (uses global if not provided)

    Returns:
        Configured FastAPI application
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Cat Watcher Labeling",
        description="Web UI for labeling cat behavior training data",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(router, prefix="/api")

    # Mount static files for serving images
    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        app.mount(
            "/data",
            StaticFiles(directory=str(data_dir)),
            name="data",
        )

    # Mount static UI files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    # Redirect root to UI
    from fastapi.responses import RedirectResponse

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    return app


def main() -> None:
    """CLI entry point for labeling service."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Cat Watcher Labeling Service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    uvicorn.run(
        "cat_watcher.labeling.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
