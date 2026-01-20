"""API routes for labeling service."""

from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from cat_watcher.collection.storage import CollectedSample, FrameStorage
from cat_watcher.schemas import BehaviorType, CatName

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["labeling"])


def get_storage(request: Request) -> FrameStorage:
    """Get storage from app state."""
    storage: FrameStorage = request.app.state.storage
    return storage


# Request/Response models


class LabelRequest(BaseModel):
    """Request to label a sample."""

    behavior: BehaviorType | None = Field(default=None, description="Behavior label")
    cat: CatName | None = Field(default=None, description="Cat identity")
    notes: str | None = Field(default=None, description="Annotator notes")
    is_blurry: bool = Field(default=False, description="Mark as blurry")
    is_occluded: bool = Field(default=False, description="Mark as occluded")
    is_ambiguous: bool = Field(default=False, description="Mark as ambiguous")
    skip_training: bool = Field(default=False, description="Exclude from training")


class SampleResponse(BaseModel):
    """Response containing sample data."""

    id: str
    timestamp: datetime
    camera: str
    frame_url: str
    crop_url: str | None
    bounding_box: dict[str, float]
    frigate_score: float
    frame_width: int
    frame_height: int
    behavior_label: str | None
    cat_label: str | None
    is_labeled: bool
    labeled_at: datetime | None
    notes: str | None
    is_blurry: bool
    is_occluded: bool
    is_ambiguous: bool
    skip_training: bool
    source: str = "frigate"  # "frigate" or "detection"
    detection_confidence: float | None = None
    track_id: int | None = None
    event_id: str | None = None

    @classmethod
    def from_sample(cls, sample: CollectedSample, base_url: str = "/data") -> "SampleResponse":
        """Create from CollectedSample."""
        # Convert absolute paths to URLs
        frame_url = f"{base_url}/frames/{sample.timestamp.strftime('%Y-%m-%d')}/{sample.id}.jpg"
        crop_url = (
            f"{base_url}/crops/{sample.timestamp.strftime('%Y-%m-%d')}/{sample.id}_crop.jpg"
            if sample.crop_path
            else None
        )

        return cls(
            id=sample.id,
            timestamp=sample.timestamp,
            camera=sample.camera,
            frame_url=frame_url,
            crop_url=crop_url,
            bounding_box={
                "x_min": sample.bounding_box.x_min,
                "y_min": sample.bounding_box.y_min,
                "x_max": sample.bounding_box.x_max,
                "y_max": sample.bounding_box.y_max,
            },
            frigate_score=sample.frigate_score,
            frame_width=sample.frame_width,
            frame_height=sample.frame_height,
            behavior_label=sample.behavior_label.value if sample.behavior_label else None,
            cat_label=sample.cat_label.value if sample.cat_label else None,
            is_labeled=sample.is_labeled,
            labeled_at=sample.labeled_at,
            notes=sample.notes,
            is_blurry=sample.is_blurry,
            is_occluded=sample.is_occluded,
            is_ambiguous=sample.is_ambiguous,
            skip_training=sample.skip_training,
            source=sample.source,
            detection_confidence=sample.detection_confidence,
            track_id=sample.track_id,
            event_id=sample.event_id,
        )


class PaginatedSamples(BaseModel):
    """Paginated list of samples."""

    items: list[SampleResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class StatsResponse(BaseModel):
    """Statistics response."""

    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    skipped_samples: int
    trainable_samples: int
    behavior_distribution: dict[str, int]
    cat_distribution: dict[str, int]


class ExportRequest(BaseModel):
    """Request to export data."""

    format: str = Field(default="coco", description="Export format (coco, yolo, csv)")
    output_dir: str = Field(default="exports", description="Output directory name")


class ExportResponse(BaseModel):
    """Export result."""

    format: str
    path: str
    samples: int
    annotations: int | None = None


# API Endpoints


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request) -> StatsResponse:
    """Get labeling statistics."""
    storage = get_storage(request)
    stats = storage.get_stats()
    return StatsResponse(**stats)


@router.get("/behaviors")
async def get_behaviors() -> list[dict[str, str]]:
    """Get available behavior types."""
    return [
        {"value": b.value, "name": b.name}
        for b in BehaviorType
    ]


@router.get("/cats")
async def get_cats() -> list[dict[str, str]]:
    """Get available cat identities."""
    return [
        {"value": c.value, "name": c.name}
        for c in CatName
    ]


@router.get("/samples/unlabeled", response_model=PaginatedSamples)
async def get_unlabeled_samples(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    source: str | None = Query(default=None, description="Filter by source: frigate or detection"),
) -> PaginatedSamples:
    """Get unlabeled samples for annotation."""
    storage = get_storage(request)
    offset = (page - 1) * page_size

    samples = storage.get_unlabeled_samples(limit=page_size + 1, offset=offset, source=source)
    has_more = len(samples) > page_size
    samples = samples[:page_size]

    stats = storage.get_stats()

    return PaginatedSamples(
        items=[SampleResponse.from_sample(s) for s in samples],
        total=stats["unlabeled_samples"],
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


@router.get("/samples/labeled", response_model=PaginatedSamples)
async def get_labeled_samples(
    request: Request,
    behavior: BehaviorType | None = None,
    cat: CatName | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    source: str | None = Query(default=None, description="Filter by source: frigate or detection"),
) -> PaginatedSamples:
    """Get labeled samples, optionally filtered."""
    storage = get_storage(request)
    offset = (page - 1) * page_size

    samples = storage.get_labeled_samples(
        behavior=behavior,
        cat=cat,
        limit=page_size + 1,
        offset=offset,
        source=source,
    )
    has_more = len(samples) > page_size
    samples = samples[:page_size]

    stats = storage.get_stats()

    return PaginatedSamples(
        items=[SampleResponse.from_sample(s) for s in samples],
        total=stats["labeled_samples"],
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


@router.get("/samples/{sample_id}", response_model=SampleResponse)
async def get_sample(request: Request, sample_id: str) -> SampleResponse:
    """Get a specific sample by ID."""
    storage = get_storage(request)
    sample = storage.get_sample(sample_id)

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    return SampleResponse.from_sample(sample)


@router.post("/samples/{sample_id}/label", response_model=SampleResponse)
async def label_sample(
    request: Request, sample_id: str, label: LabelRequest
) -> SampleResponse:
    """Add or update labels for a sample."""
    storage = get_storage(request)

    # Check sample exists
    existing = storage.get_sample(sample_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sample not found")

    # Update labels
    updated = storage.update_labels(
        sample_id=sample_id,
        behavior=label.behavior,
        cat=label.cat,
        notes=label.notes,
        is_blurry=label.is_blurry,
        is_occluded=label.is_occluded,
        is_ambiguous=label.is_ambiguous,
        skip_training=label.skip_training,
    )

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update labels")

    logger.info(
        "Labeled sample",
        sample_id=sample_id,
        behavior=label.behavior.value if label.behavior else None,
        cat=label.cat.value if label.cat else None,
    )

    return SampleResponse.from_sample(updated)


@router.post("/samples/{sample_id}/skip", response_model=SampleResponse)
async def skip_sample(request: Request, sample_id: str) -> SampleResponse:
    """Skip a sample (mark as skip_training)."""
    storage = get_storage(request)

    existing = storage.get_sample(sample_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sample not found")

    updated = storage.update_labels(sample_id=sample_id, skip_training=True)

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to skip sample")

    logger.info("Skipped sample", sample_id=sample_id)

    return SampleResponse.from_sample(updated)


@router.post("/export", response_model=ExportResponse)
async def export_data(request: Request, export_req: ExportRequest) -> ExportResponse:
    """Export labeled data for training."""
    storage = get_storage(request)
    settings = request.app.state.settings

    output_path = Path(settings.data_dir) / export_req.output_dir

    result = storage.export_for_training(output_path, format=export_req.format)

    logger.info(
        "Exported data",
        format=export_req.format,
        samples=result.get("samples", 0),
        path=result.get("path"),
    )

    return ExportResponse(
        format=result["format"],
        path=result.get("path", str(output_path)),
        samples=result.get("samples", 0),
        annotations=result.get("annotations"),
    )


@router.get("/next-unlabeled", response_model=SampleResponse | None)
async def get_next_unlabeled(request: Request) -> SampleResponse | None:
    """Get the next unlabeled sample for quick labeling workflow."""
    storage = get_storage(request)
    samples = storage.get_unlabeled_samples(limit=1)

    if not samples:
        return None

    return SampleResponse.from_sample(samples[0])
