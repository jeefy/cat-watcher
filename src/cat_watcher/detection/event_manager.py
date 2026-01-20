"""Event manager for cat detection events.

Manages the lifecycle of cat events: creation when a cat is detected,
updates when better frames are captured, and completion when the cat leaves.
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from cat_watcher.detection.tracker import TrackedObject
from cat_watcher.schemas import BoundingBox

logger = structlog.get_logger(__name__)


@dataclass
class CatEvent:
    """A cat detection event spanning multiple frames.

    An event starts when a cat enters the frame (new track) and ends
    when the cat leaves or is occluded for too long.
    """

    id: str
    camera: str
    track_id: int
    start_time: float  # Unix timestamp
    end_time: float | None = None  # None if ongoing

    # Best detection during event (highest confidence)
    best_frame: np.ndarray | None = None
    best_bbox: BoundingBox | None = None
    best_confidence: float = 0.0
    best_timestamp: float = 0.0

    # Stage 2 results (optional, if behavior model deployed)
    behavior: str | None = None
    cat_id: str | None = None

    # Statistics
    frame_count: int = 0
    detection_count: int = 0

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_active(self) -> bool:
        """Check if event is still active (not ended)."""
        return self.end_time is None

    def update_best(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        confidence: float,
        timestamp: float,
    ) -> bool:
        """Update best detection if this one is better.

        Args:
            frame: The frame (BGR numpy array)
            bbox: Bounding box
            confidence: Detection confidence
            timestamp: Frame timestamp

        Returns:
            True if this was a new best detection
        """
        if confidence > self.best_confidence:
            self.best_frame = frame.copy()
            self.best_bbox = bbox
            self.best_confidence = confidence
            self.best_timestamp = timestamp
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without frame data)."""
        return {
            "id": self.id,
            "camera": self.camera,
            "track_id": self.track_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "is_active": self.is_active,
            "best_confidence": self.best_confidence,
            "best_timestamp": self.best_timestamp,
            "best_bbox": (
                {
                    "x_min": self.best_bbox.x_min,
                    "y_min": self.best_bbox.y_min,
                    "x_max": self.best_bbox.x_max,
                    "y_max": self.best_bbox.y_max,
                }
                if self.best_bbox
                else None
            ),
            "behavior": self.behavior,
            "cat_id": self.cat_id,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
        }

    def __repr__(self) -> str:
        status = "active" if self.is_active else f"ended({self.duration:.1f}s)"
        return (
            f"CatEvent(id={self.id[:12]}..., camera={self.camera}, "
            f"track={self.track_id}, {status}, "
            f"best_conf={self.best_confidence:.2f})"
        )


# Type alias for event callbacks
EventCallback = Callable[[CatEvent], Awaitable[None]]


@dataclass
class TrackState:
    """Internal state for tracking event lifecycle per track."""

    track_id: int
    event: CatEvent | None = None
    last_seen_time: float = 0.0
    cooldown_until: float = 0.0  # Don't create new event until this time


class EventManager:
    """Manages cat event lifecycle.

    Creates events when cats appear, updates them with better detections,
    and ends them when cats leave the frame.

    Example:
        ```python
        async def on_event_end(event: CatEvent):
            print(f"Event ended: {event.id}, duration: {event.duration:.1f}s")
            # Save best frame to database

        manager = EventManager(
            min_event_duration=0.5,
            on_event_end=on_event_end,
        )

        # In detection loop
        tracked = tracker.update(detections)
        await manager.process_frame(
            camera="basement2",
            timestamp=time.time(),
            frame=frame,
            tracked_objects=tracked,
        )
        ```
    """

    def __init__(
        self,
        min_event_duration: float = 0.5,
        max_event_duration: float = 300.0,
        event_cooldown: float = 5.0,
        disappeared_timeout: float = 2.0,
        on_event_start: EventCallback | None = None,
        on_event_end: EventCallback | None = None,
    ):
        """Initialize event manager.

        Args:
            min_event_duration: Minimum event duration before it's considered
                              valid. Events shorter than this may be discarded.
            max_event_duration: Maximum event duration in seconds. Events
                              longer than this will be force-ended.
            event_cooldown: Cooldown in seconds before a new event can be
                          created for the same track (prevents spam).
            disappeared_timeout: Seconds to wait after track disappears before
                               ending the event.
            on_event_start: Async callback when an event starts.
            on_event_end: Async callback when an event ends.
        """
        self.min_event_duration = min_event_duration
        self.max_event_duration = max_event_duration
        self.event_cooldown = event_cooldown
        self.disappeared_timeout = disappeared_timeout
        self.on_event_start = on_event_start
        self.on_event_end = on_event_end

        # Track state per camera/track combo
        self._track_states: dict[str, dict[int, TrackState]] = {}

        # Statistics
        self._total_events_created = 0
        self._total_events_completed = 0
        self._total_events_discarded = 0

    async def process_frame(
        self,
        camera: str,
        timestamp: float,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> list[CatEvent]:
        """Process tracked objects for a frame.

        Creates new events for new tracks, updates existing events with
        better detections, and ends events for disappeared tracks.

        Args:
            camera: Camera name
            timestamp: Frame timestamp (Unix time)
            frame: Frame data (BGR numpy array)
            tracked_objects: List of tracked objects from tracker

        Returns:
            List of currently active events for this camera
        """
        # Ensure camera state exists
        if camera not in self._track_states:
            self._track_states[camera] = {}

        track_states = self._track_states[camera]
        active_track_ids = {t.track_id for t in tracked_objects}

        # Process each tracked object
        for tracked in tracked_objects:
            track_id = tracked.track_id

            # Get or create track state
            if track_id not in track_states:
                track_states[track_id] = TrackState(track_id=track_id)

            state = track_states[track_id]
            state.last_seen_time = timestamp

            # Check if we should create a new event
            if state.event is None:
                if timestamp >= state.cooldown_until:
                    # Create new event
                    event = self._create_event(camera, track_id, timestamp)
                    event.update_best(frame, tracked.bbox, tracked.confidence, timestamp)
                    state.event = event

                    logger.info(
                        "Event started",
                        event_id=event.id,
                        camera=camera,
                        track_id=track_id,
                    )

                    if self.on_event_start:
                        await self.on_event_start(event)
            else:
                # Update existing event
                event = state.event
                event.frame_count += 1
                event.detection_count += 1

                # Update best detection if this one is better
                if event.update_best(frame, tracked.bbox, tracked.confidence, timestamp):
                    logger.debug(
                        "New best detection",
                        event_id=event.id,
                        confidence=tracked.confidence,
                    )

                # Check max duration
                if event.duration >= self.max_event_duration:
                    await self._end_event(camera, track_id, timestamp, reason="max_duration")

        # Check for disappeared tracks
        for track_id, state in list(track_states.items()):
            if track_id not in active_track_ids and state.event is not None:
                # Track disappeared, check if we should end the event
                time_since_seen = timestamp - state.last_seen_time
                if time_since_seen >= self.disappeared_timeout:
                    await self._end_event(camera, track_id, timestamp, reason="disappeared")

        return self.get_active_events(camera)

    async def end_all_events(self, reason: str = "shutdown") -> list[CatEvent]:
        """End all active events.

        Args:
            reason: Reason for ending events

        Returns:
            List of ended events
        """
        ended_events = []
        timestamp = time.time()

        for camera, track_states in self._track_states.items():
            for track_id, state in list(track_states.items()):
                if state.event is not None:
                    event = await self._end_event(camera, track_id, timestamp, reason)
                    if event:
                        ended_events.append(event)

        return ended_events

    def get_active_events(self, camera: str | None = None) -> list[CatEvent]:
        """Get currently active events.

        Args:
            camera: Filter by camera name (None for all cameras)

        Returns:
            List of active events
        """
        events = []

        cameras = [camera] if camera else self._track_states.keys()
        for cam in cameras:
            if cam in self._track_states:
                for state in self._track_states[cam].values():
                    if state.event is not None and state.event.is_active:
                        events.append(state.event)

        return events

    def get_event_by_id(self, event_id: str) -> CatEvent | None:
        """Get event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event if found, None otherwise
        """
        for track_states in self._track_states.values():
            for state in track_states.values():
                if state.event and state.event.id == event_id:
                    return state.event
        return None

    @property
    def stats(self) -> dict[str, Any]:
        """Get event manager statistics."""
        active_count = len(self.get_active_events())
        return {
            "active_events": active_count,
            "total_events_created": self._total_events_created,
            "total_events_completed": self._total_events_completed,
            "total_events_discarded": self._total_events_discarded,
            "cameras_tracked": list(self._track_states.keys()),
        }

    def _create_event(self, camera: str, track_id: int, timestamp: float) -> CatEvent:
        """Create a new event."""
        event_id = f"{int(timestamp)}-{camera}-{track_id}-{uuid.uuid4().hex[:8]}"

        event = CatEvent(
            id=event_id,
            camera=camera,
            track_id=track_id,
            start_time=timestamp,
        )

        self._total_events_created += 1
        return event

    async def _end_event(
        self,
        camera: str,
        track_id: int,
        timestamp: float,
        reason: str,
    ) -> CatEvent | None:
        """End an event for a track.

        Args:
            camera: Camera name
            track_id: Track ID
            timestamp: End timestamp
            reason: Reason for ending

        Returns:
            The ended event, or None if discarded
        """
        if camera not in self._track_states:
            return None

        state = self._track_states[camera].get(track_id)
        if state is None or state.event is None:
            return None

        event = state.event
        event.end_time = timestamp

        # Set cooldown for this track
        state.cooldown_until = timestamp + self.event_cooldown
        state.event = None

        # Check if event meets minimum duration
        if event.duration < self.min_event_duration:
            logger.debug(
                "Event discarded (too short)",
                event_id=event.id,
                duration=event.duration,
                min_duration=self.min_event_duration,
            )
            self._total_events_discarded += 1
            return None

        # Check if we have a valid best frame
        if event.best_frame is None:
            logger.warning(
                "Event discarded (no best frame)",
                event_id=event.id,
            )
            self._total_events_discarded += 1
            return None

        logger.info(
            "Event ended",
            event_id=event.id,
            camera=camera,
            track_id=track_id,
            duration=f"{event.duration:.1f}s",
            best_confidence=event.best_confidence,
            reason=reason,
        )

        self._total_events_completed += 1

        if self.on_event_end:
            await self.on_event_end(event)

        return event

    def reset(self) -> None:
        """Reset all state."""
        self._track_states.clear()
        self._total_events_created = 0
        self._total_events_completed = 0
        self._total_events_discarded = 0
