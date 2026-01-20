"""Detection service for managing multiple camera pipelines.

Provides a unified service that manages detection pipelines across
multiple cameras with shared resources (detector model, storage).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from cat_watcher.collection.storage import FrameStorage
from cat_watcher.detection.behavior_inference import (
    BehaviorInferenceService,
    BehaviorServiceSettings,
)
from cat_watcher.detection.cat_detector import CatDetector
from cat_watcher.detection.event_manager import CatEvent
from cat_watcher.detection.pipeline import (
    DetectionPipeline,
    PipelineSettings,
    PipelineState,
)
from cat_watcher.detection.stream import get_camera_rtsp_url, list_cameras

logger = structlog.get_logger(__name__)


class ServiceState(Enum):
    """Service operational state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class ServiceSettings:
    """Settings for the detection service."""

    # Model settings
    cat_model: str = "yolov8n.pt"
    cat_confidence: float = 0.2
    device: str = "auto"

    # Behavior model settings (Stage 2 - optional)
    behavior_model: str | None = None
    behavior_confidence: float = 0.5
    behavior_min_detection_conf: float = 0.3

    # Pipeline settings
    target_fps: float = 5.0
    min_event_duration: float = 0.5
    max_event_duration: float = 300.0
    event_cooldown: float = 5.0
    disappeared_timeout: float = 2.0

    # Storage settings
    output_dir: Path = field(default_factory=lambda: Path("data/detection/events"))
    save_frames: bool = True
    db_path: Path = field(default_factory=lambda: Path("data/training/samples.db"))

    # Home Assistant settings
    ha_enabled: bool = True
    ha_topic_prefix: str = "cat_watcher"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None


@dataclass
class CameraInfo:
    """Information about a camera."""

    name: str
    enabled: bool = True
    rtsp_url: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0


# Type alias for event callbacks
EventCallback = Callable[[CatEvent], Awaitable[None]]


class DetectionService:
    """Service that manages detection pipelines for multiple cameras.

    Provides a centralized service for running cat detection across
    multiple cameras with shared resources (detector model, storage).

    Example:
        ```python
        settings = ServiceSettings(
            cat_model="yolov8n.pt",
            cat_confidence=0.2,
        )

        service = DetectionService(
            frigate_url="http://192.168.50.36:5000",
            settings=settings,
        )

        # Start with specific cameras
        await service.start(cameras=["basement2", "apollo-dish"])

        # Or start with all available cameras
        await service.start()

        # Later...
        await service.stop()
        ```
    """

    def __init__(
        self,
        frigate_url: str,
        settings: ServiceSettings | None = None,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
        event_callback: EventCallback | None = None,
    ):
        """Initialize detection service.

        Args:
            frigate_url: Frigate API URL (e.g., http://192.168.50.36:5000)
            settings: Service settings
            rtsp_username: RTSP username for cameras
            rtsp_password: RTSP password for cameras
            event_callback: Optional callback for all events
        """
        self.frigate_url = frigate_url
        self.settings = settings or ServiceSettings()
        self.rtsp_username = rtsp_username
        self.rtsp_password = rtsp_password
        self.event_callback = event_callback

        # State
        self._state = ServiceState.STOPPED
        self._start_time: float = 0.0

        # Shared detector (loaded once)
        self._detector: CatDetector | None = None

        # Behavior inference service (Stage 2 - optional)
        self._behavior_service: BehaviorInferenceService | None = None

        # Frame storage for database samples (always save to DB)
        # FrameStorage expects the data directory, not the db file path
        self._storage = FrameStorage(self.settings.db_path.parent)

        # Camera info cache
        self._cameras: dict[str, CameraInfo] = {}

        # Active pipelines
        self._pipelines: dict[str, DetectionPipeline] = {}

        # Statistics
        self._total_events = 0

        self._log = logger.bind(service="detection")

    @property
    def state(self) -> ServiceState:
        """Current service state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._state == ServiceState.RUNNING

    @property
    def cameras(self) -> dict[str, CameraInfo]:
        """Get camera information."""
        return self._cameras.copy()

    @property
    def active_cameras(self) -> list[str]:
        """Get list of cameras with running pipelines."""
        return [
            name for name, pipeline in self._pipelines.items()
            if pipeline.is_running
        ]

    async def start(self, cameras: list[str] | None = None) -> None:
        """Start the detection service.

        Args:
            cameras: List of camera names to enable. If None, uses all
                    available cameras from Frigate.
        """
        if self._state != ServiceState.STOPPED:
            raise RuntimeError(f"Service already {self._state.value}")

        self._state = ServiceState.STARTING
        self._start_time = time.time()
        self._log.info("Starting detection service")

        try:
            # Fetch available cameras from Frigate
            self._log.info("Fetching cameras from Frigate")
            available = await list_cameras(self.frigate_url)

            for cam_info in available:
                self._cameras[cam_info["name"]] = CameraInfo(
                    name=cam_info["name"],
                    enabled=cam_info["enabled"],
                    width=cam_info["width"],
                    height=cam_info["height"],
                    fps=cam_info["fps"],
                )

            self._log.info(
                "Found cameras",
                count=len(self._cameras),
                cameras=list(self._cameras.keys()),
            )

            # Determine which cameras to run
            if cameras:
                cameras_to_run = cameras
            else:
                cameras_to_run = [
                    name for name, info in self._cameras.items()
                    if info.enabled
                ]

            if not cameras_to_run:
                raise ValueError("No cameras to run")

            # Load shared detector
            self._log.info(
                "Loading cat detector",
                model=self.settings.cat_model,
                device=self.settings.device,
            )
            self._detector = CatDetector(
                model_path=self.settings.cat_model,
                confidence_threshold=self.settings.cat_confidence,
                device=self.settings.device,
            )

            # Create output directory
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize behavior inference service if model is configured
            if self.settings.behavior_model:
                self._log.info(
                    "Initializing behavior inference service",
                    model=self.settings.behavior_model,
                )
                behavior_settings = BehaviorServiceSettings(
                    model_path=Path(self.settings.behavior_model),
                    confidence_threshold=self.settings.behavior_confidence,
                    min_detection_confidence=self.settings.behavior_min_detection_conf,
                    device=self.settings.device,
                    ha_enabled=self.settings.ha_enabled,
                    ha_topic_prefix=self.settings.ha_topic_prefix,
                    mqtt_broker=self.settings.mqtt_broker,
                    mqtt_port=self.settings.mqtt_port,
                    mqtt_username=self.settings.mqtt_username,
                    mqtt_password=self.settings.mqtt_password,
                )
                self._behavior_service = BehaviorInferenceService(behavior_settings)
                await self._behavior_service.start()
                self._log.info("Behavior inference service started")

            # Start pipelines for each camera (continue on individual failures)
            failed_cameras = []
            for camera in cameras_to_run:
                try:
                    await self._start_camera_pipeline(camera)
                except Exception as e:
                    self._log.warning(
                        "Failed to start camera - skipping",
                        camera=camera,
                        error=str(e),
                    )
                    failed_cameras.append(camera)

            # Check if at least one camera started
            if not self.active_cameras:
                raise RuntimeError(
                    f"No cameras could be started. Failed: {failed_cameras}"
                )

            if failed_cameras:
                self._log.warning(
                    "Some cameras failed to start",
                    failed=failed_cameras,
                    active=self.active_cameras,
                )

            self._state = ServiceState.RUNNING
            self._log.info(
                "Detection service started",
                cameras=self.active_cameras,
            )

        except Exception as e:
            self._state = ServiceState.STOPPED
            self._log.error("Failed to start service", error=str(e))
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the detection service gracefully."""
        if self._state not in (ServiceState.RUNNING, ServiceState.STARTING):
            return

        self._state = ServiceState.STOPPING
        self._log.info("Stopping detection service")

        # Stop all pipelines
        stop_tasks = []
        for name, pipeline in self._pipelines.items():
            self._log.info("Stopping pipeline", camera=name)
            stop_tasks.append(pipeline.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Stop behavior inference service
        if self._behavior_service:
            self._log.info("Stopping behavior inference service")
            await self._behavior_service.stop()

        await self._cleanup()

        self._state = ServiceState.STOPPED
        self._log.info(
            "Detection service stopped",
            runtime=f"{time.time() - self._start_time:.1f}s",
            total_events=self._total_events,
        )

    async def enable_camera(self, camera: str) -> None:
        """Enable detection for a camera.

        Args:
            camera: Camera name
        """
        if self._state != ServiceState.RUNNING:
            raise RuntimeError("Service not running")

        if camera in self._pipelines:
            self._log.warning("Camera already enabled", camera=camera)
            return

        await self._start_camera_pipeline(camera)

    async def disable_camera(self, camera: str) -> None:
        """Disable detection for a camera.

        Args:
            camera: Camera name
        """
        if camera not in self._pipelines:
            return

        pipeline = self._pipelines[camera]
        self._log.info("Disabling camera", camera=camera)
        await pipeline.stop()
        del self._pipelines[camera]

    def get_status(self) -> dict[str, Any]:
        """Get service status.

        Returns:
            Dictionary with service state, camera status, and statistics.
        """
        runtime = time.time() - self._start_time if self._start_time else 0

        cameras_status = {}
        total_frames = 0
        total_detections = 0

        for name, pipeline in self._pipelines.items():
            status = pipeline.get_status()
            cameras_status[name] = {
                "state": status["state"],
                "fps": status["stats"]["effective_fps"],
                "frames": status["stats"]["frames_processed"],
                "detections": status["stats"]["detections_total"],
                "events": status["stats"]["events_completed"],
                "active_events": status["active_events"],
            }
            total_frames += status["stats"]["frames_processed"]
            total_detections += status["stats"]["detections_total"]

        return {
            "state": self._state.value,
            "runtime": round(runtime, 1),
            "cameras": cameras_status,
            "summary": {
                "active_cameras": len(self.active_cameras),
                "total_cameras": len(self._cameras),
                "total_frames": total_frames,
                "total_detections": total_detections,
                "total_events": self._total_events,
            },
            "settings": {
                "cat_model": self.settings.cat_model,
                "cat_confidence": self.settings.cat_confidence,
                "target_fps": self.settings.target_fps,
                "device": self.settings.device,
                "behavior_model": self.settings.behavior_model,
                "behavior_confidence": self.settings.behavior_confidence,
            },
            "behavior_service": (
                self._behavior_service.get_status()
                if self._behavior_service
                else None
            ),
        }

    async def _start_camera_pipeline(self, camera: str) -> None:
        """Start pipeline for a camera.

        Args:
            camera: Camera name
        """
        if camera in self._pipelines:
            return

        # Get RTSP URL
        self._log.info("Getting RTSP URL", camera=camera)
        rtsp_url = await get_camera_rtsp_url(
            self.frigate_url,
            camera,
            rtsp_username=self.rtsp_username,
            rtsp_password=self.rtsp_password,
        )

        # Update camera info with URL
        if camera in self._cameras:
            self._cameras[camera].rtsp_url = rtsp_url

        # Create pipeline settings
        pipeline_settings = PipelineSettings(
            target_fps=self.settings.target_fps,
            confidence_threshold=self.settings.cat_confidence,
            min_event_duration=self.settings.min_event_duration,
            max_event_duration=self.settings.max_event_duration,
            event_cooldown=self.settings.event_cooldown,
            disappeared_timeout=self.settings.disappeared_timeout,
        )

        # Create pipeline
        pipeline = DetectionPipeline(
            camera=camera,
            stream_url=rtsp_url,
            cat_detector=self._detector,
            event_callback=self._create_event_handler(camera),
            settings=pipeline_settings,
        )

        # Start pipeline
        self._log.info("Starting pipeline", camera=camera)
        await pipeline.start()

        self._pipelines[camera] = pipeline

    def _create_event_handler(self, camera: str) -> EventCallback:
        """Create event handler for a camera.

        Args:
            camera: Camera name

        Returns:
            Event callback function
        """
        async def on_event(event: CatEvent) -> None:
            self._total_events += 1

            self._log.info(
                "Cat event",
                event_id=event.id,
                camera=camera,
                duration=f"{event.duration:.1f}s",
                confidence=event.best_confidence,
            )

            # Save frame
            if self.settings.save_frames and event.best_frame is not None:
                await self._save_event_frame(event)

            # Save to database (always enabled)
            if event.best_frame is not None:
                await self._save_event_to_db(event)

            # Process with behavior model (Stage 2 - triggers Home Assistant)
            if self._behavior_service and event.best_frame is not None:
                try:
                    result = await self._behavior_service.process_event(event)
                    if result and result.behavior_detected:
                        self._log.info(
                            "Behavior detected",
                            event_id=event.id,
                            behavior=result.behavior,
                            confidence=result.confidence,
                            ha_published=result.ha_published,
                        )
                except Exception as e:
                    self._log.error(
                        "Behavior inference error",
                        event_id=event.id,
                        error=str(e),
                    )

            # Call user callback
            if self.event_callback:
                try:
                    await self.event_callback(event)
                except Exception as e:
                    self._log.error(
                        "Event callback error",
                        event_id=event.id,
                        error=str(e),
                    )

        return on_event

    async def _save_event_frame(self, event: CatEvent) -> None:
        """Save event frame to disk.

        Args:
            event: Cat event
        """
        try:
            import cv2

            frame_path = self.settings.output_dir / f"{event.id}.jpg"

            # Draw bbox on frame
            annotated = event.best_frame.copy()
            h, w = annotated.shape[:2]
            if event.best_bbox:
                x1 = int(event.best_bbox.x_min * w)
                y1 = int(event.best_bbox.y_min * h)
                x2 = int(event.best_bbox.x_max * w)
                y2 = int(event.best_bbox.y_max * h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{event.camera} conf={event.best_confidence:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            cv2.imwrite(str(frame_path), annotated)
            self._log.debug("Saved event frame", path=str(frame_path))

        except Exception as e:
            self._log.error("Failed to save event frame", error=str(e))

    async def _save_event_to_db(self, event: CatEvent) -> None:
        """Save event to database.

        Args:
            event: Cat event
        """
        if event.best_frame is None or event.best_bbox is None:
            return

        try:
            # Use the proper save_detection_sample API
            self._storage.save_detection_sample(
                event_id=event.id,
                camera=event.camera,
                timestamp=event.best_timestamp,  # Already a Unix timestamp float
                frame=event.best_frame,
                bbox=event.best_bbox,
                confidence=event.best_confidence,
                track_id=event.track_id,
                save_crop=True,
                crop_padding=0.1,
            )
            self._log.debug("Saved event to database", event_id=event.id)

        except Exception as e:
            self._log.error("Failed to save to database", error=str(e))

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._pipelines.clear()
        self._detector = None
        self._behavior_service = None
