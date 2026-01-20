"""FastAPI inference API endpoints."""

from __future__ import annotations

import io
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from cat_watcher.inference.pipeline import (
    InferencePipeline,
    InferenceResult,
)
from cat_watcher.schemas import BehaviorType, CatName

router = APIRouter(prefix="/api/v1", tags=["inference"])

# Global pipeline instance (initialized in app.py)
_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    """Get the inference pipeline."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialized",
        )
    return _pipeline


def set_pipeline(pipeline: InferencePipeline) -> None:
    """Set the inference pipeline."""
    global _pipeline
    _pipeline = pipeline


class DetectionResponse(BaseModel):
    """Detection result model."""

    behavior: str
    confidence: float
    bbox: dict[str, int]


class IdentificationResponse(BaseModel):
    """Identification result model."""

    cat: str
    confidence: float
    probabilities: dict[str, float]


class InferenceResponse(BaseModel):
    """Full inference response model."""

    timestamp: str
    source: str
    processing_time_ms: float
    detections: list[DetectionResponse]
    identifications: list[IdentificationResponse]
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    pipeline_loaded: bool
    has_detector: bool
    has_identifier: bool


class StatsResponse(BaseModel):
    """Service statistics response."""

    events_processed: int
    detections: int
    alerts_sent: int
    errors: int
    uptime_seconds: float | None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health."""
    try:
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            pipeline_loaded=pipeline.is_loaded,
            has_detector=pipeline.has_detector,
            has_identifier=pipeline.has_identifier,
        )
    except HTTPException:
        return HealthResponse(
            status="unhealthy",
            pipeline_loaded=False,
            has_detector=False,
            has_identifier=False,
        )


@router.post("/infer", response_model=InferenceResponse)
async def run_inference(
    file: Annotated[UploadFile, File(description="Image to analyze")],
    source: Annotated[str, Query(description="Source identifier")] = "api",
) -> InferenceResponse:
    """Run full inference on an uploaded image.

    Args:
        file: Uploaded image file
        source: Source identifier for tracking

    Returns:
        Complete inference results
    """
    pipeline = get_pipeline()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    # Read image
    try:
        from PIL import Image

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}",
        ) from e

    # Run inference
    try:
        result = await pipeline.process_async(image, source=source)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        ) from e

    return _result_to_response(result)


@router.post("/detect", response_model=list[DetectionResponse])
async def detect_behaviors(
    file: Annotated[UploadFile, File(description="Image to analyze")],
) -> list[DetectionResponse]:
    """Run behavior detection only.

    Args:
        file: Uploaded image file

    Returns:
        List of detected behaviors
    """
    pipeline = get_pipeline()

    if not pipeline.has_detector:
        raise HTTPException(
            status_code=503,
            detail="Behavior detector not configured",
        )

    # Read image
    try:
        from PIL import Image

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}",
        ) from e

    # Run detection
    try:
        detections = pipeline.detect_only(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        ) from e

    return [
        DetectionResponse(
            behavior=d.behavior.value,
            confidence=d.confidence,
            bbox={
                "x_min": d.bbox.x_min,
                "y_min": d.bbox.y_min,
                "x_max": d.bbox.x_max,
                "y_max": d.bbox.y_max,
            },
        )
        for d in detections
    ]


@router.post("/identify", response_model=IdentificationResponse)
async def identify_cat(
    file: Annotated[UploadFile, File(description="Image to analyze")],
    x_min: Annotated[float | None, Query(description="Bbox X min (normalized 0-1)")] = None,
    y_min: Annotated[float | None, Query(description="Bbox Y min (normalized 0-1)")] = None,
    x_max: Annotated[float | None, Query(description="Bbox X max (normalized 0-1)")] = None,
    y_max: Annotated[float | None, Query(description="Bbox Y max (normalized 0-1)")] = None,
) -> IdentificationResponse:
    """Run cat identification only.

    Args:
        file: Uploaded image file
        x_min, y_min, x_max, y_max: Optional bounding box coordinates (normalized 0-1)

    Returns:
        Cat identification result
    """
    pipeline = get_pipeline()

    if not pipeline.has_identifier:
        raise HTTPException(
            status_code=503,
            detail="Cat identifier not configured",
        )

    # Read image
    try:
        from PIL import Image

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}",
        ) from e

    # Create bbox if provided
    from cat_watcher.schemas import BoundingBox

    bbox = None
    if (
        x_min is not None
        and y_min is not None
        and x_max is not None
        and y_max is not None
    ):
        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    # Run identification
    try:
        identification = pipeline.identify_only(image, bbox=bbox)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Identification failed: {str(e)}",
        ) from e

    return IdentificationResponse(
        cat=identification.cat.value,
        confidence=identification.confidence,
        probabilities=identification.probabilities,
    )


@router.get("/behaviors", response_model=list[str])
async def list_behaviors() -> list[str]:
    """List all supported behavior types."""
    return [b.value for b in BehaviorType]


@router.get("/cats", response_model=list[str])
async def list_cats() -> list[str]:
    """List all supported cat names."""
    return [c.value for c in CatName]


def _result_to_response(result: InferenceResult) -> InferenceResponse:
    """Convert InferenceResult to API response."""
    return InferenceResponse(
        timestamp=result.timestamp.isoformat(),
        source=result.source,
        processing_time_ms=result.processing_time_ms,
        detections=[
            DetectionResponse(
                behavior=d.behavior.value,
                confidence=d.confidence,
                bbox={
                    "x_min": d.bbox.x_min,
                    "y_min": d.bbox.y_min,
                    "x_max": d.bbox.x_max,
                    "y_max": d.bbox.y_max,
                },
            )
            for d in result.detections
        ],
        identifications=[
            IdentificationResponse(
                cat=i.cat.value,
                confidence=i.confidence,
                probabilities=i.probabilities,
            )
            for i in result.identifications
        ],
        summary=result.summary,
    )
