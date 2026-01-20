"""Shared data models and schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BehaviorType(str, Enum):
    """Cat behavior classification types."""

    EATING = "cat_eating"
    DRINKING = "cat_drinking"
    VOMITING = "cat_vomiting"
    WAITING = "cat_waiting"
    LITTERBOX = "cat_litterbox"
    YOWLING = "cat_yowling"
    SLEEPING = "cat_sleeping"
    PRESENT = "cat_present"


class CatName(str, Enum):
    """Known cat identifiers."""

    STARBUCK = "starbuck"
    APOLLO = "apollo"
    MIA = "mia"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0-1 range)."""

    x_min: float = Field(ge=0.0, le=1.0, description="Left edge (normalized)")
    y_min: float = Field(ge=0.0, le=1.0, description="Top edge (normalized)")
    x_max: float = Field(ge=0.0, le=1.0, description="Right edge (normalized)")
    y_max: float = Field(ge=0.0, le=1.0, description="Bottom edge (normalized)")

    @property
    def is_valid(self) -> bool:
        """Check if bounding box has valid coordinates."""
        return (
            0 <= self.x_min < self.x_max <= 1
            and 0 <= self.y_min < self.y_max <= 1
            and self.area > 0.001  # Minimum 0.1% of image
        )

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixels.
        """
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )

    @property
    def center(self) -> tuple[float, float]:
        """Get normalized center point."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        """Get normalized area (0-1)."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class CatWatcherEvent(BaseModel):
    """Enriched cat event with behavior and identification."""

    event_id: str = Field(description="Unique event identifier")
    camera: str = Field(description="Source camera")
    cat: str = Field(description="Identified cat name")
    cat_confidence: float = Field(ge=0.0, le=1.0, description="Cat identification confidence")
    behavior: str = Field(description="Detected behavior")
    behavior_confidence: float = Field(ge=0.0, le=1.0, description="Behavior detection confidence")
    timestamp: datetime = Field(description="Event timestamp")
    source_event_id: str | None = Field(default=None, description="Original detection event ID")
    snapshot_url: str | None = Field(default=None, description="URL to event snapshot")
    bounding_box: BoundingBox | None = Field(default=None, description="Detection bounding box")

    def to_mqtt_payload(self) -> dict[str, Any]:
        """Convert to MQTT-friendly JSON payload."""
        return {
            "event_id": self.event_id,
            "camera": self.camera,
            "cat": self.cat,
            "cat_confidence": round(self.cat_confidence, 3),
            "behavior": self.behavior,
            "behavior_confidence": round(self.behavior_confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "source_event_id": self.source_event_id,
            "snapshot_url": self.snapshot_url,
        }


class LabeledSample(BaseModel):
    """A labeled training sample."""

    id: str = Field(description="Sample identifier")
    image_path: str = Field(description="Path to image file")
    source_event_id: str | None = Field(default=None, description="Detection event ID if from detection")
    behavior: BehaviorType | None = Field(default=None, description="Labeled behavior")
    cat: CatName | None = Field(default=None, description="Labeled cat identity")
    bounding_box: BoundingBox | None = Field(default=None)
    labeled_at: datetime | None = Field(default=None)
    labeled_by: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    skip: bool = Field(default=False, description="Skip this sample (unclear/bad)")


class ReviewQueueItem(BaseModel):
    """Item in the active learning review queue."""

    id: str = Field(description="Queue item identifier")
    event_id: str = Field(description="Original event ID")
    image_path: str = Field(description="Path to snapshot image")
    predicted_behavior: str = Field(description="Model's behavior prediction")
    predicted_cat: str = Field(description="Model's cat prediction")
    behavior_confidence: float = Field(description="Behavior prediction confidence")
    cat_confidence: float = Field(description="Cat prediction confidence")
    queued_at: datetime = Field(description="When added to queue")
    reviewed: bool = Field(default=False)


class DatasetManifest(BaseModel):
    """Manifest for a collected dataset."""

    created_at: datetime = Field(default_factory=datetime.now)
    cameras: list[str] = Field(description="Cameras included")
    start_date: datetime = Field(description="Earliest event")
    end_date: datetime = Field(description="Latest event")
    total_events: int = Field(description="Total events collected")
    total_images: int = Field(description="Total images extracted")
    total_clips: int = Field(description="Total clips downloaded")
    events: list[dict[str, Any]] = Field(default_factory=list, description="Event metadata")
