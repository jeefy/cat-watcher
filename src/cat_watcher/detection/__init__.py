"""Detection pipeline for processing video streams.

This module provides custom cat detection by processing camera RTSP streams
directly, ensuring accurate bounding boxes for training data collection.
"""

from cat_watcher.detection.behavior_inference import (
    BehaviorInferenceResult,
    BehaviorInferenceService,
    BehaviorServiceSettings,
    BehaviorServiceState,
    BehaviorServiceStats,
)
from cat_watcher.detection.cat_detector import CatDetection, CatDetector
from cat_watcher.detection.event_manager import CatEvent, EventManager
from cat_watcher.detection.pipeline import (
    DetectionPipeline,
    PipelineSettings,
    PipelineState,
    PipelineStats,
)
from cat_watcher.detection.service import (
    DetectionService,
    ServiceSettings,
    ServiceState,
)
from cat_watcher.detection.stream import StreamReader
from cat_watcher.detection.tracker import CentroidTracker, TrackedObject

__all__ = [
    "BehaviorInferenceResult",
    "BehaviorInferenceService",
    "BehaviorServiceSettings",
    "BehaviorServiceState",
    "BehaviorServiceStats",
    "CatDetection",
    "CatDetector",
    "CatEvent",
    "CentroidTracker",
    "DetectionPipeline",
    "DetectionService",
    "EventManager",
    "PipelineSettings",
    "PipelineState",
    "PipelineStats",
    "ServiceSettings",
    "ServiceState",
    "StreamReader",
    "TrackedObject",
]
