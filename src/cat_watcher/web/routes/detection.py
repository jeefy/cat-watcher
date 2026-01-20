"""Detection service API routes."""

import json
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cat_watcher.collection.storage import FrameStorage

logger = structlog.get_logger(__name__)

router = APIRouter()


def get_db_setting(storage: FrameStorage, key: str, default: Any = None) -> Any:
    """Get a setting from the database with type conversion.
    
    Args:
        storage: FrameStorage instance
        key: Setting key (e.g., 'detection.frame_rate')
        default: Default value if not set
        
    Returns:
        Setting value with appropriate type conversion
    """
    value = storage.get_setting(key)
    if value is None:
        return default
    
    # Type conversion based on value content
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.startswith('[') or value.startswith('{'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    else:
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


# Request/Response models
class StartRequest(BaseModel):
    """Request to start detection service."""
    cameras: list[str] | None = None


class CameraStatus(BaseModel):
    """Status of a single camera."""
    name: str
    enabled: bool
    width: int
    height: int
    fps: int
    pipeline_running: bool = False
    stats: dict[str, Any] | None = None


class ServiceStatus(BaseModel):
    """Detection service status."""
    state: str
    runtime: float
    cameras: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    settings: dict[str, Any]
    behavior_service: dict[str, Any] | None = None


class BehaviorServiceStatus(BaseModel):
    """Behavior inference service status."""
    state: str
    model_path: str | None
    model_loaded: bool
    events_processed: int
    behaviors_detected: int
    ha_events_published: int
    last_detection: dict[str, Any] | None
    errors: int


class BehaviorStartRequest(BaseModel):
    """Request to start behavior inference service."""
    model_path: str | None = None
    confidence_threshold: float | None = None


@router.get("/debug-cv2")
async def debug_cv2(request: Request) -> dict[str, Any]:
    """Debug endpoint to test cv2 VideoCapture from within uvicorn process."""
    import cv2
    import threading
    import os
    import sys
    
    settings = request.app.state.settings
    url = f"rtsp://{settings.frigate.rtsp_username}:{settings.frigate.rtsp_password}@192.168.1.233/stream1"
    masked_url = f"rtsp://***:***@192.168.1.233/stream1"
    
    result = {
        "url": masked_url,
        "cv2_version": cv2.__version__,
        "python_version": sys.version,
        "env_vars": {
            k: v for k, v in os.environ.items() 
            if any(x in k.lower() for x in ['opencv', 'ffmpeg', 'cuda', 'nvidia', 'ld_', 'path'])
        },
        "cv2_modules_loaded": [m for m in sys.modules if 'cv2' in m],
        "tests": {}
    }
    
    # Check cv2 build info
    build_info = cv2.getBuildInformation()
    ffmpeg_section = ""
    in_video = False
    for line in build_info.split('\n'):
        if 'Video I/O:' in line:
            in_video = True
        if in_video:
            ffmpeg_section += line + '\n'
            if line.strip() == '' and in_video and 'FFMPEG' in ffmpeg_section:
                break
    result["cv2_ffmpeg_info"] = ffmpeg_section.strip()[:500]
    
    # Test 1: Direct capture in main thread (asyncio)
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        opened = cap.isOpened()
        result["tests"]["direct_main_thread"] = {
            "isOpened": opened,
            "frame": None
        }
        if opened:
            ret, frame = cap.read()
            result["tests"]["direct_main_thread"]["frame"] = f"read={ret}, shape={frame.shape if ret and frame is not None else None}"
        cap.release()
    except Exception as e:
        result["tests"]["direct_main_thread"] = {"error": str(e)}
    
    # Test 2: Try with CAP_ANY
    try:
        cap2 = cv2.VideoCapture(url, cv2.CAP_ANY)
        result["tests"]["cap_any"] = {"isOpened": cap2.isOpened()}
        cap2.release()
    except Exception as e:
        result["tests"]["cap_any"] = {"error": str(e)}
    
    # Test 3: Try without any backend specified
    try:
        cap3 = cv2.VideoCapture(url)
        result["tests"]["no_backend"] = {"isOpened": cap3.isOpened()}
        cap3.release()
    except Exception as e:
        result["tests"]["no_backend"] = {"error": str(e)}
    
    # Test 4: Check if we can open a file vs RTSP
    try:
        # Try a test pattern (will fail but shows if VideoCapture works at all)
        cap4 = cv2.VideoCapture(0)  # Try device 0
        result["tests"]["device_0"] = {"isOpened": cap4.isOpened()}
        cap4.release()
    except Exception as e:
        result["tests"]["device_0"] = {"error": str(e)}
    
    return result


@router.get("/status")
async def get_status(request: Request) -> dict[str, Any]:
    """Get detection service status.
    
    Returns current state, stats, and camera information.
    """
    service = getattr(request.app.state, "detection_service", None)
    
    if service is None:
        return {
            "running": False,
            "state": "stopped",
            "uptime": 0,
            "active_cameras": 0,
            "total_detections": 0,
            "total_events": 0,
        }
    
    full_status = service.get_status()
    summary = full_status.get("summary", {})
    
    return {
        "running": full_status.get("state") == "running",
        "state": full_status.get("state", "unknown"),
        "uptime": full_status.get("runtime", 0),
        "active_cameras": summary.get("active_cameras", 0),
        "total_detections": summary.get("total_detections", 0),
        "total_events": summary.get("total_events", 0),
    }


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Get detection configuration."""
    settings = request.app.state.settings
    det = settings.detection
    
    return {
        "cat_model": det.cat_model,
        "confidence": det.cat_confidence,
        "behavior_model": det.behavior_model,
        "behavior_confidence": det.behavior_confidence,
        "device": det.device,
        "frame_rate": det.frame_rate,
        "min_event_duration": det.min_event_duration,
        "max_event_duration": det.max_event_duration,
        "output_dir": str(det.output_dir),
        "save_frames": det.save_frames,
        "cameras": det.cameras or [],
    }


@router.get("/behavior/status")
async def get_behavior_status(request: Request) -> dict[str, Any]:
    """Get behavior inference service status.
    
    Returns current state, stats, and last detection information
    for the custom-trained behavior model.
    """
    settings = request.app.state.settings
    det_config = settings.detection
    
    # Check for standalone behavior service first
    behavior_service = getattr(request.app.state, "behavior_service", None)
    if behavior_service:
        status = behavior_service.get_status()
        return {
            "enabled": True,
            "standalone": True,
            **status,
        }
    
    # Check for behavior service within detection service
    detection_service = getattr(request.app.state, "detection_service", None)
    
    if detection_service is None:
        return {
            "enabled": False,
            "standalone": False,
            "state": "stopped",
            "model_path": det_config.behavior_model,
            "model_loaded": False,
            "events_processed": 0,
            "behaviors_detected": {},
            "ha_events_published": 0,
            "last_detection": None,
            "errors": 0,
        }
    
    # Check if behavior service is configured within detection service
    full_status = detection_service.get_status()
    behavior_status = full_status.get("behavior_service")
    
    if behavior_status is None:
        return {
            "enabled": False,
            "standalone": False,
            "state": "not_configured",
            "model_path": det_config.behavior_model,
            "model_loaded": False,
            "events_processed": 0,
            "behaviors_detected": {},
            "ha_events_published": 0,
            "last_detection": None,
            "errors": 0,
            "message": "No behavior model configured. Set detection.behavior_model in config.",
        }
    
    return {
        "enabled": True,
        "standalone": False,
        **behavior_status,
    }


@router.get("/behavior/models")
async def list_behavior_models(request: Request) -> list[dict[str, Any]]:
    """List available trained behavior models.
    
    Returns list of model files from the models directory.
    """
    from pathlib import Path
    
    models_dir = Path("models")
    models = []
    
    if models_dir.exists():
        # Find .pt and .onnx files
        for ext in ["*.pt", "*.onnx"]:
            for model_path in models_dir.glob(f"**/{ext}"):
                # Get file stats
                stat = model_path.stat()
                models.append({
                    "name": model_path.name,
                    "path": str(model_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x["modified"], reverse=True)
    return models


@router.post("/behavior/start")
async def start_behavior_service(
    request: Request, 
    body: BehaviorStartRequest | None = None
) -> dict[str, Any]:
    """Start the behavior inference service standalone.
    
    This starts the behavior model independently of the detection service,
    useful for testing the model or running inference on demand.
    
    Args:
        body: Optional configuration overrides
        
    Returns:
        Service status after starting
    """
    from pathlib import Path
    
    from cat_watcher.detection.behavior_inference import (
        BehaviorInferenceService,
        BehaviorServiceSettings,
    )
    
    settings = request.app.state.settings
    det_config = settings.detection
    
    # Check if already running
    existing_service = getattr(request.app.state, "behavior_service", None)
    if existing_service and existing_service.is_running:
        raise HTTPException(status_code=400, detail="Behavior service is already running")
    
    # Determine model path
    model_path = None
    if body and body.model_path:
        model_path = body.model_path
    elif det_config.behavior_model:
        model_path = det_config.behavior_model
    
    if not model_path:
        raise HTTPException(
            status_code=400, 
            detail="No behavior model specified. Provide model_path or configure detection.behavior_model."
        )
    
    # Verify model exists
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    
    # Create service settings
    confidence = det_config.behavior_confidence
    if body and body.confidence_threshold is not None:
        confidence = body.confidence_threshold
    
    behavior_settings = BehaviorServiceSettings(
        model_path=Path(model_path),
        confidence_threshold=confidence,
        min_detection_confidence=det_config.behavior_min_detection_conf,
        device=det_config.device,
        ha_enabled=True,
        ha_topic_prefix=settings.mqtt.publish_topic_prefix,
        mqtt_broker=settings.mqtt.broker,
        mqtt_port=settings.mqtt.port,
        mqtt_username=settings.mqtt.username,
        mqtt_password=settings.mqtt.password,
    )
    
    try:
        service = BehaviorInferenceService(behavior_settings)
        await service.start()
        request.app.state.behavior_service = service
        logger.info("Behavior inference service started via API", model=model_path)
        return service.get_status()
    except Exception as e:
        logger.error("Failed to start behavior service", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/behavior/stop")
async def stop_behavior_service(request: Request) -> dict[str, Any]:
    """Stop the standalone behavior inference service.
    
    Returns:
        Final service status before stopping
    """
    service = getattr(request.app.state, "behavior_service", None)
    
    if service is None or not service.is_running:
        raise HTTPException(status_code=400, detail="Behavior service is not running")
    
    try:
        final_status = service.get_status()
        await service.stop()
        request.app.state.behavior_service = None
        logger.info("Behavior inference service stopped via API")
        return final_status
    except Exception as e:
        logger.error("Failed to stop behavior service", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_service(request: Request, body: StartRequest | None = None) -> dict[str, Any]:
    """Start the detection service.
    
    Args:
        body: Optional request body with camera list
        
    Returns:
        Service status after starting
    """
    settings = request.app.state.settings
    det_config = settings.detection
    storage: FrameStorage = request.app.state.storage
    
    # Check if already running
    existing_service = getattr(request.app.state, "detection_service", None)
    if existing_service and existing_service.is_running:
        raise HTTPException(status_code=400, detail="Detection service is already running")
    
    # Import here to avoid circular imports
    from cat_watcher.detection.service import DetectionService, ServiceSettings
    
    # Get settings from database (with fallback to config defaults)
    # Detection settings
    frame_rate = get_db_setting(storage, 'detection.frame_rate', det_config.frame_rate)
    cat_confidence = get_db_setting(storage, 'detection.cat_confidence', det_config.cat_confidence)
    behavior_confidence = get_db_setting(storage, 'detection.behavior_confidence', det_config.behavior_confidence)
    save_frames = get_db_setting(storage, 'detection.save_frames', det_config.save_frames)
    min_event_duration = get_db_setting(storage, 'detection.min_event_duration', det_config.min_event_duration)
    max_event_duration = get_db_setting(storage, 'detection.max_event_duration', det_config.max_event_duration)
    
    # Frigate settings
    frigate_url = get_db_setting(storage, 'frigate.url', settings.frigate.url)
    frigate_cameras = get_db_setting(storage, 'frigate.cameras', det_config.cameras or [])
    
    # MQTT settings  
    mqtt_enabled = get_db_setting(storage, 'mqtt.enabled', True)
    mqtt_broker = get_db_setting(storage, 'mqtt.broker', settings.mqtt.broker)
    mqtt_port = get_db_setting(storage, 'mqtt.port', settings.mqtt.port)
    
    # Create service settings
    # Use the main data_dir for db_path to ensure detection events go to the same
    # database as the labeling UI reads from (not the default relative path)
    db_path = settings.data_dir / "samples.db"
    
    service_settings = ServiceSettings(
        cat_model=det_config.cat_model,
        cat_confidence=cat_confidence,
        device=det_config.device,
        target_fps=frame_rate,
        min_event_duration=min_event_duration,
        max_event_duration=max_event_duration,
        event_cooldown=det_config.event_cooldown,
        disappeared_timeout=det_config.disappeared_timeout,
        output_dir=det_config.output_dir,
        save_frames=save_frames,
        db_path=db_path,
        # Behavior model settings
        behavior_model=det_config.behavior_model,
        behavior_confidence=behavior_confidence,
        behavior_min_detection_conf=det_config.behavior_min_detection_conf,
        # Home Assistant / MQTT settings
        ha_enabled=mqtt_enabled,
        ha_topic_prefix=settings.mqtt.publish_topic_prefix,
        mqtt_broker=mqtt_broker,
        mqtt_port=mqtt_port,
        mqtt_username=settings.mqtt.username,
        mqtt_password=settings.mqtt.password,
    )
    
    # Create service
    service = DetectionService(
        frigate_url=frigate_url,
        settings=service_settings,
        rtsp_username=settings.frigate.rtsp_username,
        rtsp_password=settings.frigate.rtsp_password,
    )
    
    # Determine cameras (priority: request body > database > config)
    cameras = None
    if body and body.cameras:
        cameras = body.cameras
    elif frigate_cameras:
        cameras = frigate_cameras if isinstance(frigate_cameras, list) else [frigate_cameras]
    
    try:
        await service.start(cameras=cameras)
        request.app.state.detection_service = service
        logger.info("Detection service started via API", cameras=cameras)
        
        # Save state for auto-restart on container restart
        if hasattr(request.app.state, 'save_detection_state'):
            request.app.state.save_detection_state(True, cameras)
        
        return service.get_status()
    except Exception as e:
        logger.error("Failed to start detection service", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_service(request: Request) -> dict[str, Any]:
    """Stop the detection service.
    
    Returns:
        Final service status before stopping
    """
    service = getattr(request.app.state, "detection_service", None)
    
    if service is None or not service.is_running:
        raise HTTPException(status_code=400, detail="Detection service is not running")
    
    try:
        # Get final status before stopping
        final_status = service.get_status()
        await service.stop()
        request.app.state.detection_service = None
        logger.info("Detection service stopped via API")
        
        # Save state so it doesn't auto-restart on container restart
        if hasattr(request.app.state, 'save_detection_state'):
            request.app.state.save_detection_state(False)
        
        return final_status
    except Exception as e:
        logger.error("Failed to stop detection service", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras")
async def get_cameras(request: Request) -> list[CameraStatus]:
    """Get available cameras with their status.
    
    Returns list of all cameras from Frigate with pipeline status.
    """
    settings = request.app.state.settings
    service = getattr(request.app.state, "detection_service", None)
    
    # Fetch cameras from Frigate
    from cat_watcher.detection.stream import list_cameras
    
    try:
        frigate_cameras = await list_cameras(settings.frigate.url)
    except Exception as e:
        logger.error("Failed to fetch cameras from Frigate", error=str(e))
        raise HTTPException(status_code=502, detail=f"Failed to connect to Frigate: {e}")
    
    # Get pipeline status if service is running
    pipeline_status = {}
    if service and service.is_running:
        status = service.get_status()
        pipeline_status = status.get("cameras", {})
    
    result = []
    for cam in frigate_cameras:
        name = cam["name"]
        pipeline_info = pipeline_status.get(name, {})
        
        stats = None
        if pipeline_info.get("state") == "running":
            stats = {
                "fps": pipeline_info.get("fps", 0.0),
                "frames": pipeline_info.get("frames", 0),
                "detections": pipeline_info.get("detections", 0),
                "events": pipeline_info.get("events", 0),
                "active_events": pipeline_info.get("active_events", 0),
            }
        
        result.append(CameraStatus(
            name=name,
            enabled=cam["enabled"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
            pipeline_running=pipeline_info.get("state") == "running",
            stats=stats,
        ))
    
    return result


@router.post("/cameras/{camera}/enable")
async def enable_camera(request: Request, camera: str) -> dict[str, Any]:
    """Enable detection for a specific camera.
    
    Args:
        camera: Camera name to enable
        
    Returns:
        Updated camera status
    """
    service = getattr(request.app.state, "detection_service", None)
    
    if service is None or not service.is_running:
        raise HTTPException(status_code=400, detail="Detection service is not running")
    
    try:
        await service.enable_camera(camera)
        return {"status": "enabled", "camera": camera}
    except Exception as e:
        logger.error("Failed to enable camera", camera=camera, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera}/disable")
async def disable_camera(request: Request, camera: str) -> dict[str, Any]:
    """Disable detection for a specific camera.
    
    Args:
        camera: Camera name to disable
        
    Returns:
        Updated camera status
    """
    service = getattr(request.app.state, "detection_service", None)
    
    if service is None or not service.is_running:
        raise HTTPException(status_code=400, detail="Detection service is not running")
    
    try:
        await service.disable_camera(camera)
        return {"status": "disabled", "camera": camera}
    except Exception as e:
        logger.error("Failed to disable camera", camera=camera, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/recent")
async def get_recent_events(request: Request, limit: int = 20) -> list[dict[str, Any]]:
    """Get recent detection events.
    
    Args:
        limit: Maximum number of events to return
        
    Returns:
        List of recent events (most recent first)
    """
    settings = request.app.state.settings
    det_config = settings.detection
    
    events = []
    
    # Read from output directory
    output_dir = det_config.output_dir
    if output_dir.exists():
        # Get JPG files sorted by modification time
        files = sorted(
            output_dir.glob("*.jpg"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]
        
        for f in files:
            # Parse event ID from filename
            event_id = f.stem
            parts = event_id.split("-")
            
            # Extract info from event ID format: {timestamp}-{camera}-{track_id}-{uuid}
            # Camera name can contain dashes, so we need to parse carefully
            # UUID is 8 hex chars, track_id is a number
            timestamp = int(parts[0]) if parts and parts[0].isdigit() else 0
            
            # Work backwards: uuid is last, track_id is second to last
            # Everything in between timestamp and track_id is camera name
            if len(parts) >= 4:
                uuid = parts[-1]
                track_id = parts[-2]
                camera = "-".join(parts[1:-2])
            elif len(parts) >= 2:
                camera = parts[1]
            else:
                camera = "unknown"
            
            events.append({
                "id": event_id,
                "camera": camera,
                "timestamp": timestamp,
                "image_path": f"/data/detection/events/{f.name}",
                "file_size": f.stat().st_size,
            })
    
    return events


class ImportEventsResponse(BaseModel):
    """Response from importing detection events."""
    imported: int
    skipped: int
    errors: int
    message: str


@router.post("/events/import")
async def import_detection_events(request: Request) -> ImportEventsResponse:
    """Import saved detection events into the labeling database.
    
    This imports JPG files from the detection output directory that
    haven't been saved to the database yet. This is useful for
    recovering events that were created before database saving was enabled.
    
    Returns:
        Summary of import operation
    """
    import cv2
    
    from cat_watcher.collection.storage import BoundingBox, FrameStorage
    
    settings = request.app.state.settings
    det_config = settings.detection
    
    events_dir = det_config.output_dir
    db_path = det_config.db_path
    
    if not events_dir.exists():
        return ImportEventsResponse(
            imported=0,
            skipped=0,
            errors=0,
            message="No detection events directory found"
        )
    
    # Find all event JPG files
    event_files = list(events_dir.glob("*.jpg"))
    
    if not event_files:
        return ImportEventsResponse(
            imported=0,
            skipped=0,
            errors=0,
            message="No event files found"
        )
    
    # Initialize storage
    storage = FrameStorage(db_path.parent)
    
    imported = 0
    skipped = 0
    errors = 0
    
    for event_file in event_files:
        filename = event_file.stem
        
        try:
            parts = filename.split("-")
            
            if len(parts) >= 4 and parts[0].isdigit():
                # Format: {timestamp}-{camera}-{track_id}-{uuid}
                timestamp_str = parts[0]
                uuid_part = parts[-1]
                track_id_str = parts[-2]
                camera = "-".join(parts[1:-2])
                
                try:
                    track_id = int(track_id_str)
                except ValueError:
                    track_id = 0
                
                event_id = filename
            else:
                event_id = filename
                camera = "unknown"
                timestamp_str = str(int(event_file.stat().st_mtime))
                track_id = 0
            
            # Parse timestamp
            try:
                timestamp = float(timestamp_str)
            except ValueError:
                timestamp = event_file.stat().st_mtime
            
            # Skip if already in database
            existing = storage.get_sample(event_id)
            if existing:
                skipped += 1
                continue
            
            # Read frame
            frame = cv2.imread(str(event_file))
            if frame is None:
                errors += 1
                continue
            
            # Default bbox (full frame)
            bbox = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
            confidence = 0.5
            
            # Save to database
            storage.save_detection_sample(
                event_id=event_id,
                camera=camera,
                timestamp=timestamp,
                frame=frame,
                bbox=bbox,
                confidence=confidence,
                track_id=track_id,
                save_crop=True,
            )
            imported += 1
            
        except Exception as e:
            logger.error("Error importing event", file=str(event_file), error=str(e))
            errors += 1
            continue
    
    return ImportEventsResponse(
        imported=imported,
        skipped=skipped,
        errors=errors,
        message=f"Imported {imported} events, skipped {skipped} (already exist), {errors} errors"
    )


@router.get("/events/{filename}")
async def get_event_image(request: Request, filename: str) -> FileResponse:
    """Serve an event image file.
    
    Args:
        filename: Image filename (e.g., "1234567890-camera-1-uuid.jpg")
        
    Returns:
        Image file response
    """
    settings = request.app.state.settings
    det_config = settings.detection
    
    # Validate filename to prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = det_config.output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Event image not found")
    
    return FileResponse(file_path, media_type="image/jpeg")
