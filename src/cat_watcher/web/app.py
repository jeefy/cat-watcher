"""Unified FastAPI application for Cat Watcher Web UI."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from cat_watcher.collection.storage import FrameStorage
from cat_watcher.config import Settings, get_settings
from cat_watcher.logging_config import configure_logging

logger = structlog.get_logger(__name__)


def _get_detection_state_path(data_dir: Path) -> Path:
    """Get path to detection state file."""
    return data_dir / ".detection_state.json"


def _save_detection_state(data_dir: Path, running: bool, cameras: list[str] | None = None) -> None:
    """Save detection service state to persistent storage."""
    state_file = _get_detection_state_path(data_dir)
    state = {"running": running, "cameras": cameras}
    try:
        state_file.write_text(json.dumps(state))
        logger.debug("Saved detection state", running=running, cameras=cameras)
    except Exception as e:
        logger.warning("Failed to save detection state", error=str(e))


def _load_detection_state(data_dir: Path) -> dict:
    """Load detection service state from persistent storage."""
    state_file = _get_detection_state_path(data_dir)
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            logger.debug("Loaded detection state", state=state)
            return state
        except Exception as e:
            logger.warning("Failed to load detection state", error=str(e))
    return {"running": False, "cameras": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings: Settings = app.state.settings
    data_dir = Path(settings.data_dir)

    # Initialize storage
    app.state.storage = FrameStorage(data_dir)
    
    # Initialize service states
    app.state.training_running = False
    app.state.training_progress = {}
    app.state.detection_service = None
    
    # Store helper functions in app state for routes to use
    app.state.save_detection_state = lambda running, cameras=None: _save_detection_state(data_dir, running, cameras)

    logger.info(
        "Cat Watcher Web UI started",
        data_dir=str(data_dir),
    )

    # Auto-start detection service if it was previously running
    prev_state = _load_detection_state(data_dir)
    if prev_state.get("running"):
        logger.info("Auto-starting detection service (was running before restart)")
        # Delay startup slightly to allow app to fully initialize
        asyncio.create_task(_auto_start_detection(app, prev_state.get("cameras")))

    yield

    # Save state as not running on shutdown (graceful shutdown)
    # Note: We don't clear the state file here because we want to preserve
    # the "should be running" intent for crash recovery
    
    # Stop detection service if running
    if app.state.detection_service and app.state.detection_service.is_running:
        logger.info("Stopping detection service on shutdown")
        try:
            await app.state.detection_service.stop()
        except Exception as e:
            logger.error("Error stopping detection service", error=str(e))

    logger.info("Cat Watcher Web UI stopped")


async def _auto_start_detection(app: FastAPI, cameras: list[str] | None) -> None:
    """Auto-start detection service after a brief delay."""
    await asyncio.sleep(2)  # Wait for app to be fully ready
    
    try:
        from cat_watcher.detection.service import DetectionService, ServiceSettings
        
        settings = app.state.settings
        det_config = settings.detection
        db_path = settings.data_dir / "samples.db"
        
        service_settings = ServiceSettings(
            cat_model=det_config.cat_model,
            cat_confidence=det_config.cat_confidence,
            device=det_config.device,
            target_fps=det_config.frame_rate,
            min_event_duration=det_config.min_event_duration,
            max_event_duration=det_config.max_event_duration,
            event_cooldown=det_config.event_cooldown,
            disappeared_timeout=det_config.disappeared_timeout,
            output_dir=det_config.output_dir,
            save_frames=det_config.save_frames,
            db_path=db_path,
            behavior_model=det_config.behavior_model,
            behavior_confidence=det_config.behavior_confidence,
            behavior_min_detection_conf=det_config.behavior_min_detection_conf,
            ha_enabled=True,
            ha_topic_prefix=settings.mqtt.publish_topic_prefix,
            mqtt_broker=settings.mqtt.broker,
            mqtt_port=settings.mqtt.port,
            mqtt_username=settings.mqtt.username,
            mqtt_password=settings.mqtt.password,
        )
        
        service = DetectionService(
            frigate_url=settings.frigate.url,
            settings=service_settings,
            rtsp_username=settings.frigate.rtsp_username,
            rtsp_password=settings.frigate.rtsp_password,
        )
        
        # Use saved cameras or default from config
        start_cameras = cameras if cameras else (det_config.cameras if det_config.cameras else None)
        
        await service.start(cameras=start_cameras)
        app.state.detection_service = service
        logger.info("Detection service auto-started successfully", cameras=start_cameras)
        
    except Exception as e:
        logger.error("Failed to auto-start detection service", error=str(e))
        # Clear the state so we don't keep trying on every restart
        _save_detection_state(Path(app.state.settings.data_dir), False)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the unified FastAPI application.

    Args:
        settings: Application settings (uses global if not provided)

    Returns:
        Configured FastAPI application
    """
    settings = settings or get_settings()

    # Configure logging based on settings
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Cat Watcher",
        description="ML-powered cat behavior detection - Web UI",
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

    # Import and include routers
    from cat_watcher.web.routes import dashboard, labeling, prepare, training, detection

    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(labeling.router, prefix="/api", tags=["labeling"])
    app.include_router(prepare.router, prefix="/api/prepare", tags=["prepare"])
    app.include_router(training.router, prefix="/api/train", tags=["training"])
    app.include_router(detection.router, prefix="/api/detection", tags=["detection"])

    # Mount static files for serving images (data directory)
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

    # Root redirects to main UI
    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    return app


def main() -> None:
    """CLI entry point for web UI."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Cat Watcher Web UI")
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
        "cat_watcher.web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
