"""FastAPI application for inference service."""

from __future__ import annotations

import argparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cat_watcher.inference.api import router, set_pipeline
from cat_watcher.inference.pipeline import InferencePipeline, PipelineConfig

logger = structlog.get_logger()

# Global config for lifespan
_app_config: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    logger.info("Starting inference API server")

    # Initialize pipeline
    config = PipelineConfig(
        behavior_model_path=_app_config.get("behavior_model", ""),
        catid_model_path=_app_config.get("catid_model", ""),
        use_onnx=_app_config.get("use_onnx", True),
        device=_app_config.get("device", ""),
    )

    pipeline = InferencePipeline(config)

    if config.behavior_model_path or config.catid_model_path:
        logger.info("Loading models...")
        try:
            pipeline.load()
            logger.info(
                "Models loaded",
                has_detector=pipeline.has_detector,
                has_identifier=pipeline.has_identifier,
            )
        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            # Continue anyway - endpoints will return 503

    set_pipeline(pipeline)

    yield

    # Shutdown
    logger.info("Shutting down inference API server")


def create_app(
    behavior_model: str | None = None,
    catid_model: str | None = None,
    use_onnx: bool = True,
    device: str = "",
) -> FastAPI:
    """Create FastAPI application.

    Args:
        behavior_model: Path to behavior detection model
        catid_model: Path to cat identification model
        use_onnx: Use ONNX runtime
        device: Inference device

    Returns:
        FastAPI application
    """
    global _app_config
    _app_config = {
        "behavior_model": behavior_model or "",
        "catid_model": catid_model or "",
        "use_onnx": use_onnx,
        "device": device,
    }

    app = FastAPI(
        title="Cat Watcher Inference API",
        description="API for cat behavior detection and identification",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router)

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "service": "Cat Watcher Inference API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cat Watcher Inference API Server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--behavior-model",
        type=str,
        help="Path to behavior detection model",
    )
    parser.add_argument(
        "--catid-model",
        type=str,
        help="Path to cat identification model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Use PyTorch instead of ONNX",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    import uvicorn

    # Create app
    app = create_app(
        behavior_model=args.behavior_model,
        catid_model=args.catid_model,
        use_onnx=not args.no_onnx,
        device=args.device,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
