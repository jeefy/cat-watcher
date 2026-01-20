"""Dataset preparation routes for the unified web UI.

Provides endpoints to prepare datasets for training.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["prepare"])


class PrepareStatus(str, Enum):
    """Status of a preparation job."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PrepareRequest(BaseModel):
    """Request to prepare a dataset."""

    output_name: str = Field(
        default="",
        description="Output directory name. Empty = auto-generated timestamp.",
    )
    val_split: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Validation split ratio",
    )
    min_samples_per_class: int = Field(
        default=5,
        ge=1,
        description="Minimum samples required per class",
    )
    include_augmentation: bool = Field(
        default=True,
        description="Include data augmentation config",
    )


class PrepareProgress(BaseModel):
    """Progress of a preparation job."""

    status: PrepareStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_samples: int = 0
    processed_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    output_path: str | None = None
    errors: list[str] = Field(default_factory=list)
    message: str = ""


class DatasetInfo(BaseModel):
    """Information about a prepared dataset."""

    name: str
    path: str
    created_at: datetime
    train_samples: int
    val_samples: int
    classes: list[str]
    has_yaml: bool


# In-memory state
_prepare_state: PrepareProgress = PrepareProgress(status=PrepareStatus.IDLE)


def get_prepare_state() -> PrepareProgress:
    """Get current preparation state."""
    return _prepare_state


async def run_prepare(
    request: PrepareRequest,
    data_dir: str,
) -> None:
    """Run dataset preparation in background."""
    global _prepare_state

    import shutil
    import random
    import yaml

    try:
        _prepare_state.status = PrepareStatus.RUNNING
        _prepare_state.started_at = datetime.now(timezone.utc)
        _prepare_state.message = "Preparing dataset..."
        _prepare_state.errors = []

        data_path = Path(data_dir)

        # Generate output name if not provided
        output_name = request.output_name
        if not output_name:
            output_name = datetime.now(timezone.utc).strftime("dataset_%Y%m%d_%H%M%S")

        output_path = data_path / "datasets" / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Get trainable samples from storage
        from cat_watcher.collection.storage import FrameStorage
        from cat_watcher.schemas import BehaviorType
        
        storage = FrameStorage(data_path)
        samples = storage.get_labeled_samples(limit=100000)
        trainable = [s for s in samples if not s.skip_training and s.behavior_label]
        
        _prepare_state.total_samples = len(trainable)
        
        if len(trainable) < request.min_samples_per_class:
            raise ValueError(f"Not enough samples: {len(trainable)} < {request.min_samples_per_class}")
        
        # Shuffle and split
        random.shuffle(trainable)
        val_count = int(len(trainable) * request.val_split)
        val_samples = trainable[:val_count]
        train_samples = trainable[val_count:]
        
        # Create directory structure
        (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Get class list
        classes = [b.value for b in BehaviorType]
        
        # Process samples
        for i, sample in enumerate(train_samples):
            _copy_sample(sample, output_path / "images" / "train", output_path / "labels" / "train", classes, data_path)
            _prepare_state.processed_samples = i + 1
        
        for i, sample in enumerate(val_samples):
            _copy_sample(sample, output_path / "images" / "val", output_path / "labels" / "val", classes, data_path)
            _prepare_state.processed_samples = len(train_samples) + i + 1
        
        # Create dataset.yaml
        yaml_content = {
            "path": str(output_path),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)},
        }
        
        with open(output_path / "dataset.yaml", "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        _prepare_state.train_samples = len(train_samples)
        _prepare_state.val_samples = len(val_samples)
        _prepare_state.output_path = str(output_path)
        _prepare_state.status = PrepareStatus.COMPLETED
        _prepare_state.completed_at = datetime.now(timezone.utc)
        _prepare_state.message = (
            f"Dataset ready: {len(train_samples)} train, "
            f"{len(val_samples)} val samples"
        )

        logger.info(
            "Dataset preparation completed",
            output_path=str(output_path),
            train_samples=len(train_samples),
            val_samples=len(val_samples),
        )

    except Exception as e:
        logger.exception("Dataset preparation failed", error=str(e))
        _prepare_state.status = PrepareStatus.FAILED
        _prepare_state.completed_at = datetime.now(timezone.utc)
        _prepare_state.errors.append(str(e))
        _prepare_state.message = f"Failed: {e}"


def _copy_sample(sample, images_dir: Path, labels_dir: Path, classes: list[str], data_path: Path) -> None:
    """Copy a sample to the dataset directory."""
    import shutil
    
    # Copy image
    # frame_path in DB is like "data/training/frames/..." but data_path is already "data/training"
    # so we need to strip the "data/training/" prefix
    frame_path = str(sample.frame_path)
    if frame_path.startswith("data/training/"):
        frame_path = frame_path[len("data/training/"):]
    
    src_img = data_path / frame_path
    dst_img = images_dir / f"{sample.id}.jpg"
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
    
    # Create YOLO label
    if sample.behavior_label:
        class_id = classes.index(sample.behavior_label.value)
        bbox = sample.bounding_box
        
        # YOLO format: class x_center y_center width height (normalized)
        x_center = (bbox.x_min + bbox.x_max) / 2
        y_center = (bbox.y_min + bbox.y_max) / 2
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        
        label_path = labels_dir / f"{sample.id}.txt"
        label_path.write_text(f"{class_id} {x_center} {y_center} {width} {height}")


@router.get("/status", response_model=PrepareProgress)
async def get_status() -> PrepareProgress:
    """Get current preparation status."""
    return _prepare_state


@router.post("/start", response_model=PrepareProgress)
async def start_prepare(
    request: Request,
    prepare: PrepareRequest,
    background_tasks: BackgroundTasks,
) -> PrepareProgress:
    """Start dataset preparation."""
    global _prepare_state

    if _prepare_state.status == PrepareStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Preparation already in progress",
        )

    settings = request.app.state.settings

    # Reset state
    _prepare_state = PrepareProgress(
        status=PrepareStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        message="Starting preparation...",
    )

    # Start background task
    background_tasks.add_task(
        run_prepare,
        prepare,
        settings.data_dir,
    )

    return _prepare_state


@router.get("/datasets", response_model=list[DatasetInfo])
async def list_datasets(request: Request) -> list[DatasetInfo]:
    """List available prepared datasets."""
    settings = request.app.state.settings
    datasets_dir = Path(settings.data_dir) / "datasets"

    if not datasets_dir.exists():
        return []

    datasets = []
    for dataset_path in sorted(datasets_dir.iterdir(), reverse=True):
        if not dataset_path.is_dir():
            continue

        yaml_path = dataset_path / "dataset.yaml"
        has_yaml = yaml_path.exists()

        # Count samples
        train_samples = 0
        val_samples = 0
        classes: list[str] = []

        train_images = dataset_path / "images" / "train"
        val_images = dataset_path / "images" / "val"

        if train_images.exists():
            train_samples = len(list(train_images.glob("*.jpg")))
        if val_images.exists():
            val_samples = len(list(val_images.glob("*.jpg")))

        # Read classes from yaml if exists
        if has_yaml:
            import yaml
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
                names = config.get("names", [])
                # names can be a dict {0: 'class0', 1: 'class1'} or a list
                if isinstance(names, dict):
                    # Convert dict to list, sorted by key
                    classes = [names[k] for k in sorted(names.keys())]
                else:
                    classes = names

        datasets.append(
            DatasetInfo(
                name=dataset_path.name,
                path=str(dataset_path),
                created_at=datetime.fromtimestamp(
                    dataset_path.stat().st_mtime, tz=timezone.utc
                ),
                train_samples=train_samples,
                val_samples=val_samples,
                classes=classes,
                has_yaml=has_yaml,
            )
        )

    return datasets


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(request: Request, dataset_name: str) -> dict[str, str]:
    """Delete a prepared dataset."""
    import shutil

    settings = request.app.state.settings
    dataset_path = Path(settings.data_dir) / "datasets" / dataset_name

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset_path.is_dir():
        raise HTTPException(status_code=400, detail="Not a valid dataset")

    try:
        shutil.rmtree(dataset_path)
        logger.info("Deleted dataset", dataset_name=dataset_name)
        return {"status": "deleted", "dataset": dataset_name}
    except Exception as e:
        logger.error("Failed to delete dataset", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@router.get("/stats")
async def get_labeling_stats(request: Request) -> dict:
    """Get stats about labeled data available for preparation."""
    from cat_watcher.collection.storage import FrameStorage

    storage: FrameStorage = request.app.state.storage
    stats = storage.get_stats()

    return {
        "total_samples": stats.get("total_samples", 0),
        "labeled_samples": stats.get("labeled_samples", 0),
        "trainable_samples": stats.get("trainable_samples", 0),
        "skipped_samples": stats.get("skipped_samples", 0),
        "behavior_distribution": stats.get("behavior_distribution", {}),
        "cat_distribution": stats.get("cat_distribution", {}),
        "ready_for_training": stats.get("trainable_samples", 0) >= 10,
    }
