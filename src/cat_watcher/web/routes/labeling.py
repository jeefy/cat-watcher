"""Labeling routes for the unified web UI.

Re-exports the existing labeling API with minimal changes.
"""

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from cat_watcher.collection.storage import CollectedSample, FrameStorage
from cat_watcher.schemas import BehaviorType, CatName

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["labeling"])


def get_storage(request: Request) -> FrameStorage:
    """Get FrameStorage from app state."""
    return request.app.state.storage


# Request/Response Models


class LabelRequest(BaseModel):
    """Request to label a sample."""

    behavior: BehaviorType | None = None
    cat: CatName | None = None
    notes: str | None = None
    is_blurry: bool = False
    is_occluded: bool = False
    is_ambiguous: bool = False
    skip_training: bool = False


class SampleResponse(BaseModel):
    """Response containing sample data."""

    id: str
    frame_path: str
    camera: str
    timestamp: str
    event_id: str | None = None
    behavior: str | None = None
    cat: str | None = None
    notes: str | None = None
    is_labeled: bool = False
    is_blurry: bool = False
    is_occluded: bool = False
    is_ambiguous: bool = False
    skip_training: bool = False
    bounding_box: dict | None = None
    source: str | None = None  # "frigate" or "detection"

    @classmethod
    def from_sample(cls, sample: CollectedSample) -> "SampleResponse":
        """Create response from CollectedSample object."""
        # frame_path in DB can be:
        # - Old format: "data/training/frames/..." (relative)
        # - New format: "/data/frames/..." (absolute, from detection service)
        # Static files are served at /data, so we need a URL path like /data/frames/...
        frame_path = str(sample.frame_path)
        if frame_path.startswith("data/training/"):
            # Old format: strip prefix and add /data/
            frame_path = f"/data/{frame_path[len('data/training/'):]}"
        elif frame_path.startswith("/data/"):
            # New format: already an absolute path, use as-is for URL
            pass  # frame_path is already correct
        else:
            # Unknown format: prepend /data/
            frame_path = f"/data/{frame_path}"
        
        return cls(
            id=sample.id,
            frame_path=frame_path,
            camera=sample.camera,
            timestamp=sample.timestamp.isoformat(),
            event_id=sample.id,  # event ID is the sample ID
            behavior=sample.behavior_label.value if sample.behavior_label else None,
            cat=sample.cat_label.value if sample.cat_label else None,
            notes=sample.notes,
            is_labeled=sample.is_labeled,
            is_blurry=sample.is_blurry,
            is_occluded=sample.is_occluded,
            is_ambiguous=sample.is_ambiguous,
            skip_training=sample.skip_training,
            bounding_box=sample.bounding_box.model_dump() if sample.bounding_box else None,
            source=sample.source,
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
    return [{"value": b.value, "name": b.name} for b in BehaviorType]


@router.get("/cats")
async def get_cats(request: Request) -> list[dict[str, Any]]:
    """Get available cat identities from database.
    
    Returns cats defined by users, plus 'unknown' option.
    """
    storage = get_storage(request)
    cats = storage.get_cats()
    
    # Convert to labeling format and add 'unknown' option
    result = [{"value": cat["name"], "name": cat["name"].title(), "id": cat["id"]} for cat in cats]
    result.append({"value": "unknown", "name": "Unknown", "id": None})
    
    return result


# Cat management endpoints

class CatCreate(BaseModel):
    """Request to create a cat."""
    name: str
    age: int | None = None
    notes: str | None = None


class CatUpdate(BaseModel):
    """Request to update a cat."""
    name: str | None = None
    age: int | None = None
    notes: str | None = None


class CatResponse(BaseModel):
    """Cat response."""
    id: int
    name: str
    age: int | None
    notes: str | None
    created_at: str
    updated_at: str


@router.get("/cats/all", response_model=list[CatResponse])
async def get_all_cats(request: Request) -> list[CatResponse]:
    """Get all cats with full details for management UI."""
    storage = get_storage(request)
    cats = storage.get_cats()
    return [CatResponse(**cat) for cat in cats]


@router.post("/cats", response_model=CatResponse)
async def create_cat(request: Request, body: CatCreate) -> CatResponse:
    """Create a new cat."""
    storage = get_storage(request)
    try:
        cat = storage.add_cat(name=body.name, age=body.age, notes=body.notes)
        return CatResponse(**cat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/cats/{cat_id}", response_model=CatResponse)
async def update_cat(request: Request, cat_id: int, body: CatUpdate) -> CatResponse:
    """Update an existing cat."""
    storage = get_storage(request)
    try:
        cat = storage.update_cat(cat_id, name=body.name, age=body.age, notes=body.notes)
        if not cat:
            raise HTTPException(status_code=404, detail="Cat not found")
        return CatResponse(**cat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/cats/{cat_id}")
async def delete_cat(request: Request, cat_id: int) -> dict[str, str]:
    """Delete a cat."""
    storage = get_storage(request)
    if storage.delete_cat(cat_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Cat not found")


# Settings management endpoints

class SettingValue(BaseModel):
    """Single setting value."""
    value: str
    description: str | None = None


class SettingsUpdate(BaseModel):
    """Request to update multiple settings."""
    settings: dict[str, str]


class SettingsCategoryResponse(BaseModel):
    """Settings for a single category."""
    category: str
    settings: dict[str, SettingValue]


@router.get("/settings")
async def get_all_settings(request: Request) -> dict[str, dict[str, Any]]:
    """Get all settings grouped by category."""
    storage = get_storage(request)
    return storage.get_all_settings()


@router.get("/settings/{category}")
async def get_settings_by_category(request: Request, category: str) -> dict[str, Any]:
    """Get settings for a specific category."""
    storage = get_storage(request)
    return storage.get_settings_by_category(category)


@router.put("/settings")
async def update_settings(request: Request, body: SettingsUpdate) -> dict[str, str]:
    """Update multiple settings at once."""
    storage = get_storage(request)
    storage.set_settings_bulk(body.settings)
    return {"status": "updated", "count": str(len(body.settings))}


@router.put("/settings/{key:path}")
async def update_single_setting(request: Request, key: str, body: SettingValue) -> dict[str, str]:
    """Update a single setting."""
    storage = get_storage(request)
    storage.set_setting(key, body.value, description=body.description)
    return {"status": "updated", "key": key}


@router.delete("/settings/{key:path}")
async def delete_setting(request: Request, key: str) -> dict[str, str]:
    """Delete a setting (revert to default)."""
    storage = get_storage(request)
    if storage.delete_setting(key):
        return {"status": "deleted", "key": key}
    return {"status": "not_found", "key": key}


@router.get("/samples/unlabeled", response_model=PaginatedSamples)
async def get_unlabeled_samples(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=500),
) -> PaginatedSamples:
    """Get unlabeled samples for annotation."""
    storage = get_storage(request)
    offset = (page - 1) * page_size

    samples = storage.get_unlabeled_samples(limit=page_size + 1, offset=offset)
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
    page_size: int = Query(default=20, ge=1, le=500),
) -> PaginatedSamples:
    """Get labeled samples, optionally filtered."""
    storage = get_storage(request)
    offset = (page - 1) * page_size

    samples = storage.get_labeled_samples(
        behavior=behavior,
        cat=cat,
        limit=page_size + 1,
        offset=offset,
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
