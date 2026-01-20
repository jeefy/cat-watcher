"""Detection pipeline for a single camera.

Orchestrates stream reading, cat detection, tracking, and event management
for processing a camera's RTSP stream.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from cat_watcher.detection.cat_detector import CatDetector
from cat_watcher.detection.event_manager import CatEvent, EventManager
from cat_watcher.detection.stream import StreamReader
from cat_watcher.detection.tracker import CentroidTracker

logger = structlog.get_logger(__name__)


class PipelineState(Enum):
    """Pipeline operational state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PipelineSettings:
    """Settings for the detection pipeline."""

    # Stream settings
    target_fps: float = 5.0
    reconnect_delay: float = 5.0

    # Detection settings
    confidence_threshold: float = 0.2

    # Tracker settings
    max_disappeared_frames: int = 15  # At 5 FPS = 3 seconds
    max_distance: float = 0.2

    # Event settings
    min_event_duration: float = 0.5
    max_event_duration: float = 300.0
    event_cooldown: float = 5.0
    disappeared_timeout: float = 2.0


@dataclass
class PipelineStats:
    """Runtime statistics for the pipeline."""

    frames_processed: int = 0
    detections_total: int = 0
    events_started: int = 0
    events_completed: int = 0
    events_discarded: int = 0
    errors: int = 0
    start_time: float = 0.0
    last_frame_time: float = 0.0
    last_detection_time: float = 0.0

    @property
    def runtime(self) -> float:
        """Total runtime in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def effective_fps(self) -> float:
        """Effective frames per second."""
        if self.runtime == 0:
            return 0.0
        return self.frames_processed / self.runtime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "events_started": self.events_started,
            "events_completed": self.events_completed,
            "events_discarded": self.events_discarded,
            "errors": self.errors,
            "runtime": round(self.runtime, 1),
            "effective_fps": round(self.effective_fps, 2),
            "last_frame_time": self.last_frame_time,
            "last_detection_time": self.last_detection_time,
        }


# Type alias for event callbacks
EventCallback = Callable[[CatEvent], Awaitable[None]]


class DetectionPipeline:
    """Complete detection pipeline for a single camera.

    Orchestrates the flow: RTSP stream → cat detection → tracking → events.

    Example:
        ```python
        async def on_event(event: CatEvent):
            print(f"Event: {event.id}, duration: {event.duration:.1f}s")
            # Save to database, send alert, etc.

        pipeline = DetectionPipeline(
            camera="basement2",
            stream_url="rtsp://...",
            cat_detector=detector,
            event_callback=on_event,
        )

        await pipeline.start()
        # Pipeline runs until stopped
        await pipeline.stop()
        ```
    """

    def __init__(
        self,
        camera: str,
        stream_url: str,
        cat_detector: CatDetector,
        event_callback: EventCallback | None = None,
        settings: PipelineSettings | None = None,
    ):
        """Initialize detection pipeline.

        Args:
            camera: Camera name (for logging and events)
            stream_url: RTSP stream URL
            cat_detector: Cat detector instance (can be shared)
            event_callback: Async callback when event ends
            settings: Pipeline settings (uses defaults if None)
        """
        self.camera = camera
        self.stream_url = stream_url
        self.cat_detector = cat_detector
        self.event_callback = event_callback
        self.settings = settings or PipelineSettings()

        # Components (created on start)
        self._stream_reader: StreamReader | None = None
        self._tracker: CentroidTracker | None = None
        self._event_manager: EventManager | None = None

        # State
        self._state = PipelineState.STOPPED
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # Statistics
        self._stats = PipelineStats()

        self._log = logger.bind(camera=camera)

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._state == PipelineState.RUNNING

    @property
    def stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats

    @property
    def active_events(self) -> list[CatEvent]:
        """Get currently active events."""
        if self._event_manager:
            return self._event_manager.get_active_events(self.camera)
        return []

    async def start(self) -> None:
        """Start the detection pipeline.

        Creates components and starts the processing loop.
        """
        if self._state != PipelineState.STOPPED:
            raise RuntimeError(f"Pipeline already {self._state.value}")

        self._state = PipelineState.STARTING
        self._stop_event.clear()
        self._stats = PipelineStats()
        self._stats.start_time = time.time()

        self._log.info("Starting pipeline")

        try:
            # Create components
            self._stream_reader = StreamReader(
                url=self.stream_url,
                target_fps=self.settings.target_fps,
                reconnect_delay=self.settings.reconnect_delay,
            )

            self._tracker = CentroidTracker(
                max_disappeared=self.settings.max_disappeared_frames,
                max_distance=self.settings.max_distance,
            )

            self._event_manager = EventManager(
                min_event_duration=self.settings.min_event_duration,
                max_event_duration=self.settings.max_event_duration,
                event_cooldown=self.settings.event_cooldown,
                disappeared_timeout=self.settings.disappeared_timeout,
                on_event_start=self._on_event_start,
                on_event_end=self._on_event_end,
            )

            # Start stream reader
            await self._stream_reader.start()

            self._log.info(
                "Pipeline connected",
                resolution=f"{self._stream_reader.resolution[0]}x{self._stream_reader.resolution[1]}",
                source_fps=self._stream_reader.source_fps,
            )

            # Start processing loop
            self._state = PipelineState.RUNNING
            self._task = asyncio.create_task(self._run_loop())

        except Exception as e:
            self._state = PipelineState.ERROR
            self._log.error("Failed to start pipeline", error=str(e))
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the detection pipeline gracefully.

        Ends all active events and releases resources.
        """
        if self._state not in (PipelineState.RUNNING, PipelineState.ERROR):
            return

        self._state = PipelineState.STOPPING
        self._log.info("Stopping pipeline")

        # Signal stop
        self._stop_event.set()

        # Wait for task to complete
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._log.warning("Pipeline stop timed out, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # End all events
        if self._event_manager:
            await self._event_manager.end_all_events(reason="pipeline_stopped")

        # Cleanup
        await self._cleanup()

        self._state = PipelineState.STOPPED
        self._log.info(
            "Pipeline stopped",
            stats=self._stats.to_dict(),
        )

    async def _run_loop(self) -> None:
        """Main processing loop."""
        self._log.debug("Processing loop started")

        try:
            async for timestamp, frame in self._stream_reader.frames():
                # Check for stop signal
                if self._stop_event.is_set():
                    break

                try:
                    await self._process_frame(timestamp, frame)
                except Exception as e:
                    self._stats.errors += 1
                    self._log.error(
                        "Error processing frame",
                        error=str(e),
                        frame_count=self._stats.frames_processed,
                    )

        except asyncio.CancelledError:
            self._log.debug("Processing loop cancelled")
            raise
        except Exception as e:
            self._stats.errors += 1
            self._log.error("Processing loop error", error=str(e))
            self._state = PipelineState.ERROR

        self._log.debug("Processing loop ended")

    async def _process_frame(self, timestamp: float, frame) -> None:
        """Process a single frame through the pipeline."""
        self._stats.frames_processed += 1
        self._stats.last_frame_time = timestamp

        # Stage 1: Detect cats
        detections = self.cat_detector.detect(frame)

        if detections:
            self._stats.detections_total += len(detections)
            self._stats.last_detection_time = timestamp

        # Stage 2: Track objects
        tracked = self._tracker.update(detections)

        # Stage 3: Manage events
        await self._event_manager.process_frame(
            camera=self.camera,
            timestamp=timestamp,
            frame=frame,
            tracked_objects=tracked,
        )

    async def _on_event_start(self, event: CatEvent) -> None:
        """Handle event start."""
        self._stats.events_started += 1
        self._log.info(
            "Cat event started",
            event_id=event.id,
            track_id=event.track_id,
        )

    async def _on_event_end(self, event: CatEvent) -> None:
        """Handle event end."""
        self._stats.events_completed += 1
        self._log.info(
            "Cat event ended",
            event_id=event.id,
            track_id=event.track_id,
            duration=f"{event.duration:.1f}s",
            best_confidence=event.best_confidence,
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

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._stream_reader:
            await self._stream_reader.stop()
            self._stream_reader = None

        self._tracker = None
        self._event_manager = None
        self._task = None

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status for monitoring.

        Returns:
            Dictionary with state, stats, and component status.
        """
        status = {
            "camera": self.camera,
            "state": self._state.value,
            "stats": self._stats.to_dict(),
            "active_events": len(self.active_events),
        }

        if self._stream_reader and self._state == PipelineState.RUNNING:
            stream_stats = self._stream_reader.stats
            status["stream"] = {
                "effective_fps": round(stream_stats.effective_fps, 2),
                "frames_dropped": stream_stats.frames_dropped,
                "reconnects": stream_stats.reconnects,
            }

        if self._tracker:
            tracker_stats = self._tracker.stats
            status["tracker"] = {
                "active_tracks": tracker_stats["active_tracks"],
                "total_tracks": tracker_stats["total_tracks"],
            }

        if self._event_manager:
            event_stats = self._event_manager.stats
            status["events"] = {
                "total_created": event_stats["total_events_created"],
                "total_completed": event_stats["total_events_completed"],
                "total_discarded": event_stats["total_events_discarded"],
            }

        return status
