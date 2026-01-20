"""Dashboard API routes."""

from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter()


class SystemStatus(BaseModel):
    """System component status."""
    
    frigate_connected: bool
    frigate_url: str
    mqtt_connected: bool
    mqtt_broker: str
    data_dir: str
    data_dir_exists: bool
    models_available: dict[str, bool]


class DatasetStats(BaseModel):
    """Dataset statistics."""
    
    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    skipped_samples: int
    trainable_samples: int
    samples_by_behavior: dict[str, int]
    samples_by_cat: dict[str, int]
    samples_by_camera: dict[str, int]


class ModelInfo(BaseModel):
    """Trained model information."""
    
    name: str
    path: str
    trained_at: datetime | None
    metrics: dict[str, float]


class DashboardStats(BaseModel):
    """Complete dashboard statistics."""
    
    system: SystemStatus
    dataset: DatasetStats
    models: list[ModelInfo]
    training_running: bool


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(request: Request) -> DashboardStats:
    """Get comprehensive dashboard statistics."""
    settings = request.app.state.settings
    storage = request.app.state.storage
    
    # Check Frigate connection
    frigate_connected = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.frigate.url}/api/version")
            frigate_connected = response.status_code == 200
    except Exception:
        pass
    
    # Check for trained models
    models_dir = Path(settings.models_dir)
    behavior_model_exists = (models_dir / "behavior" / "best.pt").exists()
    catid_model_exists = (models_dir / "catid" / "best.pt").exists()
    
    # Also check runs directory
    runs_dir = Path("runs/detect/models")
    if not behavior_model_exists and (runs_dir / "behavior" / "behavior" / "weights" / "best.pt").exists():
        behavior_model_exists = True
    
    # Get dataset stats from storage
    samples_by_behavior: dict[str, int] = {}
    samples_by_cat: dict[str, int] = {}
    samples_by_camera: dict[str, int] = {}
    total_samples = 0
    labeled_samples = 0
    skipped_samples = 0
    unlabeled_samples = 0
    trainable_samples = 0
    
    try:
        with storage._get_db() as conn:
            # Total and labeled counts
            cursor = conn.execute("SELECT COUNT(*) FROM samples")
            total_samples = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM samples WHERE is_labeled = 1")
            labeled_samples = cursor.fetchone()[0]
            
            # Skipped samples (marked as skip_training)
            cursor = conn.execute("SELECT COUNT(*) FROM samples WHERE skip_training = 1")
            skipped_samples = cursor.fetchone()[0]
            
            # Unlabeled samples that are available for labeling (not skipped)
            cursor = conn.execute("SELECT COUNT(*) FROM samples WHERE is_labeled = 0 AND skip_training = 0")
            unlabeled_samples = cursor.fetchone()[0]
            
            # Trainable samples (labeled and not skipped)
            cursor = conn.execute("SELECT COUNT(*) FROM samples WHERE is_labeled = 1 AND skip_training = 0")
            trainable_samples = cursor.fetchone()[0]
            
            # By behavior
            cursor = conn.execute("""
                SELECT behavior_label, COUNT(*) 
                FROM samples 
                WHERE behavior_label IS NOT NULL 
                GROUP BY behavior_label
            """)
            for row in cursor.fetchall():
                samples_by_behavior[row[0]] = row[1]
            
            # By cat
            cursor = conn.execute("""
                SELECT cat_label, COUNT(*) 
                FROM samples 
                WHERE cat_label IS NOT NULL 
                GROUP BY cat_label
            """)
            for row in cursor.fetchall():
                samples_by_cat[row[0]] = row[1]
            
            # By camera
            cursor = conn.execute("""
                SELECT camera, COUNT(*) 
                FROM samples 
                GROUP BY camera
            """)
            for row in cursor.fetchall():
                samples_by_camera[row[0]] = row[1]
    except Exception as e:
        logger.warning("Failed to get dataset stats", error=str(e))
    
    # Build models list
    models: list[ModelInfo] = []
    
    # Check for behavior model
    behavior_model_path = runs_dir / "behavior" / "behavior" / "weights" / "best.pt"
    if behavior_model_path.exists():
        # Try to read metrics from results.csv
        metrics: dict[str, float] = {}
        results_csv = behavior_model_path.parent.parent / "results.csv"
        if results_csv.exists():
            try:
                import csv
                with open(results_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        # Clean up column names (they have spaces)
                        for key, value in last_row.items():
                            clean_key = key.strip()
                            if "mAP" in clean_key or "loss" in clean_key.lower():
                                try:
                                    metrics[clean_key] = float(value.strip())
                                except ValueError:
                                    pass
            except Exception:
                pass
        
        models.append(ModelInfo(
            name="Behavior Detection (YOLOv8)",
            path=str(behavior_model_path),
            trained_at=datetime.fromtimestamp(behavior_model_path.stat().st_mtime),
            metrics=metrics,
        ))
    
    return DashboardStats(
        system=SystemStatus(
            frigate_connected=frigate_connected,
            frigate_url=settings.frigate.url,
            mqtt_connected=False,  # TODO: Check MQTT
            mqtt_broker=f"{settings.mqtt.broker}:{settings.mqtt.port}",
            data_dir=str(settings.data_dir),
            data_dir_exists=Path(settings.data_dir).exists(),
            models_available={
                "behavior": behavior_model_exists,
                "catid": catid_model_exists,
            },
        ),
        dataset=DatasetStats(
            total_samples=total_samples,
            labeled_samples=labeled_samples,
            unlabeled_samples=unlabeled_samples,
            skipped_samples=skipped_samples,
            trainable_samples=trainable_samples,
            samples_by_behavior=samples_by_behavior,
            samples_by_cat=samples_by_cat,
            samples_by_camera=samples_by_camera,
        ),
        models=models,
        training_running=getattr(request.app.state, 'training_running', False),
    )


@router.get("/status")
async def get_system_status(request: Request) -> dict:
    """Quick health check for system components."""
    settings = request.app.state.settings
    
    # Check Frigate
    frigate_ok = False
    frigate_version = None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.frigate.url}/api/version")
            if response.status_code == 200:
                frigate_ok = True
                frigate_version = response.text.strip()
    except Exception:
        pass
    
    return {
        "frigate": {
            "connected": frigate_ok,
            "url": settings.frigate.url,
            "version": frigate_version,
        },
        "mqtt": {
            "connected": False,
            "broker": settings.mqtt.broker,
        },
        "storage": {
            "data_dir": str(settings.data_dir),
            "exists": Path(settings.data_dir).exists(),
        },
    }
