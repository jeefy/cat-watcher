"""Object tracker for maintaining cat identities across frames.

Uses centroid-based tracking to assign stable IDs to detected cats
as they move through the camera's field of view.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from scipy.optimize import linear_sum_assignment

from cat_watcher.detection.cat_detector import CatDetection
from cat_watcher.schemas import BoundingBox

logger = structlog.get_logger(__name__)


@dataclass
class TrackedObject:
    """A tracked cat with stable ID across frames."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    centroid: tuple[float, float]  # Normalized (x, y) center
    frames_since_seen: int = 0
    total_frames: int = 1
    max_confidence: float = 0.0
    
    # Store the best detection (highest confidence)
    best_bbox: BoundingBox | None = None
    best_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.best_bbox is None:
            self.best_bbox = self.bbox
            self.best_confidence = self.confidence
        self.max_confidence = max(self.max_confidence, self.confidence)

    def update(self, detection: CatDetection) -> None:
        """Update track with new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.centroid = self._compute_centroid(detection.bbox)
        self.frames_since_seen = 0
        self.total_frames += 1
        
        # Update best detection if this one is better
        if detection.confidence > self.best_confidence:
            self.best_bbox = detection.bbox
            self.best_confidence = detection.confidence
        
        self.max_confidence = max(self.max_confidence, detection.confidence)

    @staticmethod
    def _compute_centroid(bbox: BoundingBox) -> tuple[float, float]:
        """Compute normalized centroid from bounding box."""
        cx = (bbox.x_min + bbox.x_max) / 2
        cy = (bbox.y_min + bbox.y_max) / 2
        return (cx, cy)

    @classmethod
    def from_detection(cls, track_id: int, detection: CatDetection) -> "TrackedObject":
        """Create a new tracked object from a detection."""
        centroid = cls._compute_centroid(detection.bbox)
        return cls(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            centroid=centroid,
            best_bbox=detection.bbox,
            best_confidence=detection.confidence,
        )

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.track_id}, "
            f"conf={self.confidence:.2f}, "
            f"seen={self.total_frames}, "
            f"missing={self.frames_since_seen})"
        )


class CentroidTracker:
    """Tracks objects across frames using centroid distance matching.

    This tracker assigns stable IDs to detected objects (cats) and maintains
    their identity across frames using the Hungarian algorithm for optimal
    assignment based on centroid distances.

    Example:
        ```python
        tracker = CentroidTracker(max_disappeared=30)

        # For each frame
        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        for obj in tracked:
            print(f"Track {obj.track_id} at {obj.centroid}")
        ```
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 0.2,
        min_confidence_to_track: float = 0.0,
    ):
        """Initialize tracker.

        Args:
            max_disappeared: Maximum frames an object can be missing before
                           being deregistered.
            max_distance: Maximum normalized distance (0-1) for centroid
                         matching. Objects further apart won't be matched.
            min_confidence_to_track: Minimum detection confidence to create
                                    new tracks.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_confidence_to_track = min_confidence_to_track

        # Track storage: track_id -> TrackedObject
        self._tracks: OrderedDict[int, TrackedObject] = OrderedDict()
        self._next_id = 0
        
        # Statistics
        self._total_tracks_created = 0

    def update(self, detections: list[CatDetection]) -> list[TrackedObject]:
        """Update tracks with new detections.

        Args:
            detections: List of detections from current frame.

        Returns:
            List of currently tracked objects (active tracks).
        """
        # Filter detections by confidence
        detections = [
            d for d in detections
            if d.confidence >= self.min_confidence_to_track
        ]

        # If no detections, mark all existing tracks as disappeared
        if not detections:
            self._mark_all_disappeared()
            return self.get_active_tracks()

        # If no existing tracks, register all detections as new tracks
        if not self._tracks:
            for detection in detections:
                self._register(detection)
            return self.get_active_tracks()

        # Compute distance matrix between existing tracks and new detections
        track_ids = list(self._tracks.keys())
        track_centroids = np.array([
            self._tracks[tid].centroid for tid in track_ids
        ])
        detection_centroids = np.array([
            self._compute_centroid(d.bbox) for d in detections
        ])

        # Compute pairwise distances
        distances = self._compute_distances(track_centroids, detection_centroids)

        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(distances)

        # Track which tracks and detections have been matched
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        # Process matches
        for row, col in zip(row_indices, col_indices):
            # Skip if distance exceeds threshold
            if distances[row, col] > self.max_distance:
                continue

            track_id = track_ids[row]
            detection = detections[col]

            # Update track with matched detection
            self._tracks[track_id].update(detection)
            matched_tracks.add(track_id)
            matched_detections.add(col)

        # Handle unmatched tracks (mark as disappeared)
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._tracks[track_id].frames_since_seen += 1

                # Deregister if disappeared too long
                if self._tracks[track_id].frames_since_seen > self.max_disappeared:
                    self._deregister(track_id)

        # Handle unmatched detections (register as new tracks)
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self._register(detection)

        return self.get_active_tracks()

    def get_active_tracks(self) -> list[TrackedObject]:
        """Get currently active (visible) tracks.

        Returns:
            List of tracks that were seen in the last frame.
        """
        return [
            track for track in self._tracks.values()
            if track.frames_since_seen == 0
        ]

    def get_all_tracks(self) -> list[TrackedObject]:
        """Get all tracks including disappeared ones.

        Returns:
            List of all tracks (active and disappeared).
        """
        return list(self._tracks.values())

    def get_disappeared_tracks(self) -> list[TrackedObject]:
        """Get tracks that have disappeared (not seen recently).

        Returns:
            List of tracks that are currently missing.
        """
        return [
            track for track in self._tracks.values()
            if track.frames_since_seen > 0
        ]

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_id = 0
        self._total_tracks_created = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "active_tracks": len(self.get_active_tracks()),
            "total_tracks": len(self._tracks),
            "total_tracks_created": self._total_tracks_created,
            "next_id": self._next_id,
        }

    def _register(self, detection: CatDetection) -> int:
        """Register a new track.

        Args:
            detection: Detection to create track from.

        Returns:
            New track ID.
        """
        track_id = self._next_id
        self._tracks[track_id] = TrackedObject.from_detection(track_id, detection)
        self._next_id += 1
        self._total_tracks_created += 1

        logger.debug(
            "Registered new track",
            track_id=track_id,
            confidence=detection.confidence,
        )

        return track_id

    def _deregister(self, track_id: int) -> None:
        """Deregister a track.

        Args:
            track_id: Track ID to remove.
        """
        if track_id in self._tracks:
            track = self._tracks[track_id]
            logger.debug(
                "Deregistered track",
                track_id=track_id,
                total_frames=track.total_frames,
                max_confidence=track.max_confidence,
            )
            del self._tracks[track_id]

    def _mark_all_disappeared(self) -> None:
        """Mark all tracks as disappeared and clean up old ones."""
        tracks_to_remove = []

        for track_id, track in self._tracks.items():
            track.frames_since_seen += 1
            if track.frames_since_seen > self.max_disappeared:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            self._deregister(track_id)

    @staticmethod
    def _compute_centroid(bbox: BoundingBox) -> tuple[float, float]:
        """Compute centroid from bounding box."""
        cx = (bbox.x_min + bbox.x_max) / 2
        cy = (bbox.y_min + bbox.y_max) / 2
        return (cx, cy)

    @staticmethod
    def _compute_distances(
        track_centroids: np.ndarray,
        detection_centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute distance matrix between track and detection centroids.

        Args:
            track_centroids: (N, 2) array of track centroids
            detection_centroids: (M, 2) array of detection centroids

        Returns:
            (N, M) distance matrix
        """
        # Euclidean distance
        # track_centroids: (N, 2), detection_centroids: (M, 2)
        # Result: (N, M) matrix where [i,j] is distance from track i to detection j
        
        # Expand dimensions for broadcasting
        # tracks: (N, 1, 2), detections: (1, M, 2)
        tracks_exp = track_centroids[:, np.newaxis, :]
        dets_exp = detection_centroids[np.newaxis, :, :]
        
        # Compute squared differences and sum
        diff = tracks_exp - dets_exp
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distances
