"""Training routes for the unified web UI.

Provides endpoints to manage model training.
"""

import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["training"])


class TrainingStatus(str, Enum):
    """Status of a training job."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRequest(BaseModel):
    """Request to start training."""

    dataset: str = Field(description="Dataset directory name to train on")
    epochs: int = Field(default=100, ge=1, le=1000, description="Number of epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size")
    image_size: int = Field(default=640, ge=320, le=1280, description="Image size")
    model_name: str = Field(
        default="",
        description="Output model name. Empty = auto-generated.",
    )
    base_model: str = Field(
        default="yolov8n.pt",
        description="Base model to fine-tune from",
    )
    patience: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Early stopping patience",
    )


class TrainingProgress(BaseModel):
    """Progress of a training job."""

    status: TrainingStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    current_loss: float = 0.0
    output_path: str | None = None
    errors: list[str] = Field(default_factory=list)
    message: str = ""


class ModelInfo(BaseModel):
    """Information about a trained model."""

    name: str
    path: str
    created_at: datetime
    size_mb: float
    dataset: str | None = None
    epochs: int | None = None
    best_map50: float | None = None
    best_map50_95: float | None = None


# In-memory state
_training_state: TrainingProgress = TrainingProgress(status=TrainingStatus.IDLE)
_cancel_requested: bool = False  # Flag to signal cancellation


def get_training_state() -> TrainingProgress:
    """Get current training state."""
    return _training_state


def _run_training_sync(
    request: TrainingRequest,
    data_dir: str,
) -> None:
    """Run model training synchronously (called from thread pool)."""
    global _training_state, _cancel_requested

    try:
        _training_state.status = TrainingStatus.RUNNING
        _training_state.started_at = datetime.now(timezone.utc)
        _training_state.message = "Initializing training..."
        _training_state.current_epoch = 0
        _training_state.total_epochs = request.epochs
        _training_state.errors = []

        data_path = Path(data_dir)
        dataset_path = data_path / "datasets" / request.dataset

        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {request.dataset}")

        yaml_path = dataset_path / "dataset.yaml"
        if not yaml_path.exists():
            raise ValueError(f"dataset.yaml not found in {request.dataset}")

        # Generate model name if not provided
        model_name = request.model_name
        if not model_name:
            model_name = datetime.now(timezone.utc).strftime("model_%Y%m%d_%H%M%S")

        output_path = data_path / "models" / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        _training_state.output_path = str(output_path)
        _training_state.message = f"Training on {request.dataset}..."

        # Import and run training
        from cat_watcher.training.behavior import BehaviorTrainer, BehaviorTrainerConfig

        config = BehaviorTrainerConfig(
            model_name=request.base_model,
            epochs=request.epochs,
            batch_size=request.batch_size,
            img_size=request.image_size,
            patience=request.patience,
            # Use absolute path to ensure YOLO saves to the correct location
            project=str(output_path.parent.absolute()),
            name=output_path.name,
        )

        trainer = BehaviorTrainer(
            data_dir=dataset_path,
            config=config,
        )

        # Define callback to update progress during training
        def on_train_epoch_end(trainer_instance):
            """Callback called at end of each training epoch."""
            global _training_state, _cancel_requested
            _training_state.current_epoch = trainer_instance.epoch + 1
            _training_state.message = f"Epoch {_training_state.current_epoch}/{_training_state.total_epochs}"
            
            # Check for cancellation request
            if _cancel_requested:
                _training_state.status = TrainingStatus.CANCELLED
                _training_state.message = "Cancelled by user"
                _training_state.completed_at = datetime.now(timezone.utc)
                # Stop training by raising KeyboardInterrupt (YOLO handles this gracefully)
                raise KeyboardInterrupt("Training cancelled by user")
            
            # Update metrics if available
            if hasattr(trainer_instance, 'metrics') and trainer_instance.metrics:
                metrics = trainer_instance.metrics
                if hasattr(metrics, 'box'):
                    _training_state.best_map50 = max(
                        _training_state.best_map50,
                        float(getattr(metrics.box, 'map50', 0) or 0)
                    )
                    _training_state.best_map50_95 = max(
                        _training_state.best_map50_95,
                        float(getattr(metrics.box, 'map', 0) or 0)
                    )
            
            # Also try to get loss
            if hasattr(trainer_instance, 'loss') and trainer_instance.loss is not None:
                try:
                    _training_state.current_loss = float(trainer_instance.loss.item())
                except:
                    pass

        _training_state.message = f"Training on {request.dataset}..."
        
        result = trainer.train(on_epoch_end=on_train_epoch_end)
        
        # Check if cancelled during training
        if _cancel_requested:
            return  # Exit early, state already updated

        # Extract final metrics from result
        if result.get("metrics"):
            metrics = result["metrics"]
            _training_state.best_map50 = metrics.get("mAP50", 0.0)
            _training_state.best_map50_95 = metrics.get("mAP50-95", 0.0)

        _training_state.current_epoch = request.epochs
        _training_state.status = TrainingStatus.COMPLETED
        _training_state.completed_at = datetime.now(timezone.utc)
        _training_state.message = (
            f"Training complete! Best mAP50: {_training_state.best_map50:.3f}"
        )

        logger.info(
            "Training completed",
            model_path=str(output_path),
            best_map50=_training_state.best_map50,
        )

    except KeyboardInterrupt:
        # Graceful cancellation - state already updated by callback
        logger.info("Training cancelled by user")
    except Exception as e:
        logger.exception("Training failed", error=str(e))
        _training_state.status = TrainingStatus.FAILED
        _training_state.completed_at = datetime.now(timezone.utc)
        _training_state.errors.append(str(e))
        _training_state.message = f"Failed: {e}"


@router.get("/status", response_model=TrainingProgress)
async def get_status() -> TrainingProgress:
    """Get current training status."""
    return _training_state


@router.post("/start", response_model=TrainingProgress)
async def start_training(
    request: Request,
    training: TrainingRequest,
) -> TrainingProgress:
    """Start model training."""
    global _training_state, _cancel_requested

    if _training_state.status == TrainingStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Training already in progress",
        )

    settings = request.app.state.settings

    # Verify dataset exists
    dataset_path = Path(settings.data_dir) / "datasets" / training.dataset
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {training.dataset}",
        )

    # Reset cancellation flag
    _cancel_requested = False

    # Reset state
    _training_state = TrainingProgress(
        status=TrainingStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        total_epochs=training.epochs,
        message="Starting training...",
    )

    # Run training in a separate thread so it doesn't block the event loop
    # This allows /api/train/status to respond during training
    thread = threading.Thread(
        target=_run_training_sync,
        args=(training, settings.data_dir),
        daemon=True,
    )
    thread.start()

    return _training_state


@router.post("/cancel")
async def cancel_training() -> dict[str, str]:
    """Cancel the current training job."""
    global _training_state, _cancel_requested

    if _training_state.status != TrainingStatus.RUNNING:
        raise HTTPException(status_code=400, detail="No training in progress")

    # Set cancellation flag - will be checked in epoch callback
    _cancel_requested = True
    _training_state.message = "Cancelling... (will stop after current epoch)"

    return {"status": "cancelling"}


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    """List available trained models."""
    settings = request.app.state.settings
    models_dir = Path(settings.data_dir) / "models"

    if not models_dir.exists():
        return []

    models = []
    for model_path in sorted(models_dir.iterdir(), reverse=True):
        if not model_path.is_dir():
            continue

        # Look for best.pt or last.pt
        best_pt = model_path / "weights" / "best.pt"
        last_pt = model_path / "weights" / "last.pt"
        model_file = best_pt if best_pt.exists() else last_pt

        if not model_file.exists():
            # Maybe direct .pt file
            pt_files = list(model_path.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0]
            else:
                continue

        size_mb = model_file.stat().st_size / (1024 * 1024)

        # Try to read training args
        args_yaml = model_path / "args.yaml"
        dataset = None
        epochs = None
        if args_yaml.exists():
            import yaml
            with open(args_yaml) as f:
                args = yaml.safe_load(f)
                dataset = args.get("data", "").split("/")[-2] if args.get("data") else None
                epochs = args.get("epochs")

        # Try to read results
        best_map50 = None
        best_map50_95 = None
        results_csv = model_path / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    map50 = float(row.get("metrics/mAP50(B)", 0) or 0)
                    map50_95 = float(row.get("metrics/mAP50-95(B)", 0) or 0)
                    if best_map50 is None or map50 > best_map50:
                        best_map50 = map50
                    if best_map50_95 is None or map50_95 > best_map50_95:
                        best_map50_95 = map50_95

        models.append(
            ModelInfo(
                name=model_path.name,
                path=str(model_file),
                created_at=datetime.fromtimestamp(
                    model_path.stat().st_mtime, tz=timezone.utc
                ),
                size_mb=round(size_mb, 2),
                dataset=dataset,
                epochs=epochs,
                best_map50=best_map50,
                best_map50_95=best_map50_95,
            )
        )

    return models


@router.delete("/models/{model_name}")
async def delete_model(request: Request, model_name: str) -> dict[str, str]:
    """Delete a trained model."""
    import shutil

    settings = request.app.state.settings
    model_path = Path(settings.data_dir) / "models" / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        shutil.rmtree(model_path)
        logger.info("Deleted model", model_name=model_name)
        return {"status": "deleted", "model": model_name}
    except Exception as e:
        logger.error("Failed to delete model", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@router.get("/base-models")
async def list_base_models() -> list[dict[str, str]]:
    """List available base models for training."""
    return [
        {"name": "yolov8n.pt", "description": "YOLOv8 Nano - Fastest, smallest"},
        {"name": "yolov8s.pt", "description": "YOLOv8 Small - Good balance"},
        {"name": "yolov8m.pt", "description": "YOLOv8 Medium - Better accuracy"},
        {"name": "yolov8l.pt", "description": "YOLOv8 Large - High accuracy"},
        {"name": "yolov8x.pt", "description": "YOLOv8 XLarge - Best accuracy"},
    ]
