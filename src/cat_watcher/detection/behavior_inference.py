"""Behavior inference service for classifying detected cat events.

Runs a custom-trained behavior model on cat detection events and
publishes results to Home Assistant.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from PIL import Image

from cat_watcher.schemas import BehaviorType

if TYPE_CHECKING:
    from cat_watcher.detection.event_manager import CatEvent
    from cat_watcher.homeassistant.publisher import HAEventPublisher
    from cat_watcher.mqtt import MQTTPublisher

logger = structlog.get_logger(__name__)


class BehaviorServiceState(Enum):
    """Behavior inference service state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class BehaviorInferenceResult:
    """Result of behavior inference on a detection event."""

    event_id: str
    camera: str
    behavior: BehaviorType
    confidence: float
    timestamp: float
    processing_time_ms: float

    # Original detection info
    detection_confidence: float
    bbox: dict[str, float] | None = None

    # Publishing info
    ha_published: bool = False

    @property
    def behavior_detected(self) -> bool:
        """Check if a specific behavior (not just PRESENT) was detected."""
        return self.behavior != BehaviorType.PRESENT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "camera": self.camera,
            "behavior": self.behavior.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms,
            "detection_confidence": self.detection_confidence,
            "bbox": self.bbox,
            "behavior_detected": self.behavior_detected,
            "ha_published": self.ha_published,
        }


@dataclass
class BehaviorServiceSettings:
    """Settings for behavior inference service."""

    # Model settings
    model_path: Path | str = ""
    confidence_threshold: float = 0.5
    device: str = "auto"
    use_onnx: bool = True

    # Processing settings
    process_all_events: bool = True  # Process all events vs only high-confidence
    min_detection_confidence: float = 0.3  # Min detection confidence to process

    # Home Assistant / MQTT settings
    ha_enabled: bool = True
    ha_topic_prefix: str = "cat_watcher"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None


@dataclass
class BehaviorServiceStats:
    """Statistics for behavior inference service."""

    events_processed: int = 0
    events_skipped: int = 0
    behaviors_detected: dict[str, int] = field(default_factory=dict)
    ha_events_published: int = 0
    total_processing_time_ms: float = 0.0
    last_event_time: float = 0.0
    last_detection: dict[str, Any] | None = None
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        avg_time = (
            self.total_processing_time_ms / self.events_processed
            if self.events_processed > 0
            else 0
        )
        return {
            "events_processed": self.events_processed,
            "events_skipped": self.events_skipped,
            "behaviors_detected": self.behaviors_detected,
            "ha_events_published": self.ha_events_published,
            "avg_processing_time_ms": round(avg_time, 2),
            "last_event_time": self.last_event_time,
            "last_detection": self.last_detection,
            "errors": self.errors,
        }


class BehaviorInferenceService:
    """Service for running behavior inference on cat detection events.

    This service:
    1. Receives cat detection events from the detection pipeline
    2. Runs a custom-trained behavior model on the best frame
    3. Publishes behavior classifications to Home Assistant

    Example:
        ```python
        settings = BehaviorServiceSettings(
            model_path="models/behavior/best.pt",
            confidence_threshold=0.5,
        )

        service = BehaviorInferenceService(
            settings=settings,
            mqtt_publisher=mqtt,  # Optional, for HA integration
        )

        await service.start()

        # Process events from detection pipeline
        await service.process_event(cat_event)

        await service.stop()
        ```
    """

    def __init__(
        self,
        settings: BehaviorServiceSettings | None = None,
        mqtt_publisher: MQTTPublisher | None = None,
    ) -> None:
        """Initialize behavior inference service.

        Args:
            settings: Service settings
            mqtt_publisher: Optional MQTT publisher for Home Assistant
        """
        self.settings = settings or BehaviorServiceSettings()
        self._mqtt = mqtt_publisher
        self._mqtt_owned = False  # Track if we created the MQTT client
        self._ha_publisher: HAEventPublisher | None = None

        # State
        self._state = BehaviorServiceState.STOPPED
        self._start_time: float = 0.0
        self._stats = BehaviorServiceStats()

        # Model (lazy loaded)
        self._detector: Any = None

        self._log = logger.bind(service="behavior_inference")

    @property
    def state(self) -> BehaviorServiceState:
        """Current service state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._state == BehaviorServiceState.RUNNING

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._detector is not None

    @property
    def stats(self) -> BehaviorServiceStats:
        """Get service statistics."""
        return self._stats

    async def start(self) -> None:
        """Start the behavior inference service."""
        if self._state != BehaviorServiceState.STOPPED:
            raise RuntimeError(f"Service already {self._state.value}")

        if not self.settings.model_path:
            raise ValueError("No behavior model path configured")

        self._state = BehaviorServiceState.STARTING
        self._start_time = time.time()
        self._log.info(
            "Starting behavior inference service",
            model=str(self.settings.model_path),
        )

        try:
            # Load behavior model
            await self._load_model()

            # Initialize MQTT if HA is enabled and no external publisher
            if self.settings.ha_enabled and self._mqtt is None:
                await self._init_mqtt()

            # Initialize Home Assistant publisher if MQTT available
            if self._mqtt and self.settings.ha_enabled:
                await self._init_ha_publisher()

            self._state = BehaviorServiceState.RUNNING
            self._log.info("Behavior inference service started")

        except Exception as e:
            self._state = BehaviorServiceState.ERROR
            self._log.error("Failed to start service", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the behavior inference service."""
        if self._state not in (
            BehaviorServiceState.RUNNING,
            BehaviorServiceState.STARTING,
            BehaviorServiceState.ERROR,
        ):
            return

        self._state = BehaviorServiceState.STOPPING
        self._log.info("Stopping behavior inference service")

        # Cleanup MQTT if we created it
        if self._mqtt_owned and self._mqtt:
            try:
                await self._mqtt.disconnect()
            except Exception as e:
                self._log.warning("Error disconnecting MQTT", error=str(e))

        # Cleanup
        self._detector = None
        self._ha_publisher = None
        self._mqtt = None
        self._mqtt_owned = False

        self._state = BehaviorServiceState.STOPPED
        self._log.info(
            "Behavior inference service stopped",
            runtime=f"{time.time() - self._start_time:.1f}s",
            stats=self._stats.to_dict(),
        )

    async def process_event(self, event: CatEvent) -> BehaviorInferenceResult | None:
        """Process a cat detection event.

        Args:
            event: Cat detection event from pipeline

        Returns:
            Behavior inference result, or None if skipped
        """
        if not self.is_running:
            self._log.warning("Service not running, skipping event")
            return None

        if event.best_frame is None:
            self._log.debug("Event has no frame, skipping", event_id=event.id)
            self._stats.events_skipped += 1
            return None

        # Check minimum confidence
        if event.best_confidence < self.settings.min_detection_confidence:
            self._log.debug(
                "Event confidence too low, skipping",
                event_id=event.id,
                confidence=event.best_confidence,
                threshold=self.settings.min_detection_confidence,
            )
            self._stats.events_skipped += 1
            return None

        try:
            result = await self._run_inference(event)

            if result:
                self._stats.events_processed += 1
                self._stats.total_processing_time_ms += result.processing_time_ms
                self._stats.last_event_time = time.time()

                # Track behavior counts
                behavior_key = result.behavior.value
                self._stats.behaviors_detected[behavior_key] = (
                    self._stats.behaviors_detected.get(behavior_key, 0) + 1
                )

                # Publish to Home Assistant
                if self._ha_publisher and result.confidence >= self.settings.confidence_threshold:
                    published = await self._publish_to_ha(result)
                    result.ha_published = published
                    if published:
                        self._stats.ha_events_published += 1

                # Track last detection for monitoring
                if result.behavior_detected:
                    self._stats.last_detection = result.to_dict()

                self._log.info(
                    "Behavior classified",
                    event_id=event.id,
                    behavior=result.behavior.value,
                    confidence=f"{result.confidence:.2f}",
                    time_ms=f"{result.processing_time_ms:.1f}",
                )

            return result

        except Exception as e:
            self._stats.errors += 1
            self._log.error(
                "Failed to process event",
                event_id=event.id,
                error=str(e),
            )
            return None

    def get_status(self) -> dict[str, Any]:
        """Get service status.

        Returns:
            Status dictionary
        """
        runtime = time.time() - self._start_time if self._start_time else 0

        return {
            "state": self._state.value,
            "runtime": round(runtime, 1),
            "model_loaded": self.model_loaded,
            "model_path": str(self.settings.model_path),
            "ha_enabled": self.settings.ha_enabled and self._ha_publisher is not None,
            "settings": {
                "confidence_threshold": self.settings.confidence_threshold,
                "min_detection_confidence": self.settings.min_detection_confidence,
                "device": self.settings.device,
            },
            "stats": self._stats.to_dict(),
        }

    async def _load_model(self) -> None:
        """Load the behavior detection model."""
        from cat_watcher.inference.detector import BehaviorDetector

        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._log.info("Loading behavior model", path=str(model_path))

        # Determine if ONNX based on file extension or setting
        use_onnx = self.settings.use_onnx
        if model_path.suffix.lower() == ".onnx":
            use_onnx = True
        elif model_path.suffix.lower() == ".pt":
            use_onnx = False

        self._detector = BehaviorDetector(
            model_path=model_path,
            confidence_threshold=self.settings.confidence_threshold,
            device=self.settings.device,
            use_onnx=use_onnx,
        )

        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._detector.load)

        self._log.info("Behavior model loaded")

    async def _init_mqtt(self) -> None:
        """Initialize MQTT connection for Home Assistant publishing."""
        from cat_watcher.mqtt import MQTTPublisher

        self._log.info(
            "Connecting to MQTT broker",
            broker=self.settings.mqtt_broker,
            port=self.settings.mqtt_port,
        )

        try:
            self._mqtt = MQTTPublisher(
                broker=self.settings.mqtt_broker,
                port=self.settings.mqtt_port,
                username=self.settings.mqtt_username,
                password=self.settings.mqtt_password,
                topic_prefix=self.settings.ha_topic_prefix,
            )
            await self._mqtt.connect()
            self._mqtt_owned = True
            self._log.info("Connected to MQTT broker")
        except Exception as e:
            self._log.warning(
                "Failed to connect to MQTT broker, HA publishing disabled",
                error=str(e),
            )
            self._mqtt = None
            self._mqtt_owned = False

    async def _init_ha_publisher(self) -> None:
        """Initialize Home Assistant publisher."""
        from cat_watcher.homeassistant.publisher import HAEventPublisher

        self._ha_publisher = HAEventPublisher(
            mqtt_publisher=self._mqtt,
            topic_prefix=self.settings.ha_topic_prefix,
        )

        # Publish discovery messages
        try:
            await self._ha_publisher.publish_discovery()
            self._log.info("Published Home Assistant discovery")
        except Exception as e:
            self._log.warning("Failed to publish HA discovery", error=str(e))

    async def _run_inference(self, event: CatEvent) -> BehaviorInferenceResult | None:
        """Run behavior inference on an event.

        Args:
            event: Cat detection event

        Returns:
            Inference result or None
        """
        if self._detector is None:
            return None

        start_time = time.perf_counter()

        # Convert frame to PIL Image
        # Frame is BGR numpy array from OpenCV
        frame_rgb = event.best_frame[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(frame_rgb)

        # Crop to bounding box if available (with padding)
        if event.best_bbox:
            w, h = image.size
            padding = 0.1  # 10% padding

            x_min = event.best_bbox.x_min
            y_min = event.best_bbox.y_min
            x_max = event.best_bbox.x_max
            y_max = event.best_bbox.y_max

            # Add padding
            box_w = x_max - x_min
            box_h = y_max - y_min
            x_min = max(0, x_min - box_w * padding)
            y_min = max(0, y_min - box_h * padding)
            x_max = min(1, x_max + box_w * padding)
            y_max = min(1, y_max + box_h * padding)

            # Convert to pixels and crop
            crop_box = (
                int(x_min * w),
                int(y_min * h),
                int(x_max * w),
                int(y_max * h),
            )
            image = image.crop(crop_box)

        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            None,
            self._detector.detect,
            image,
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        # Get best detection
        if not detections:
            # Default to PRESENT if no behavior detected
            return BehaviorInferenceResult(
                event_id=event.id,
                camera=event.camera,
                behavior=BehaviorType.PRESENT,
                confidence=event.best_confidence,  # Use detection confidence
                timestamp=event.best_timestamp,
                processing_time_ms=processing_time,
                detection_confidence=event.best_confidence,
                bbox=event.best_bbox.to_dict() if event.best_bbox else None,
            )

        # Get highest confidence detection
        best_detection = max(detections, key=lambda d: d.confidence)

        return BehaviorInferenceResult(
            event_id=event.id,
            camera=event.camera,
            behavior=best_detection.behavior,
            confidence=best_detection.confidence,
            timestamp=event.best_timestamp,
            processing_time_ms=processing_time,
            detection_confidence=event.best_confidence,
            bbox=event.best_bbox.to_dict() if event.best_bbox else None,
        )

    async def _publish_to_ha(self, result: BehaviorInferenceResult) -> bool:
        """Publish inference result to Home Assistant.

        Args:
            result: Behavior inference result

        Returns:
            True if published successfully
        """
        if not self._ha_publisher:
            return False

        from cat_watcher.schemas import CatName

        try:
            await self._ha_publisher.publish_detection(
                behavior=result.behavior,
                cat=CatName.UNKNOWN,  # Cat ID not available yet
                confidence=result.confidence,
                camera=result.camera,
                event_id=result.event_id,
                bbox=result.bbox,
            )
            return True
        except Exception as e:
            self._log.error("Failed to publish to HA", error=str(e))
            return False
