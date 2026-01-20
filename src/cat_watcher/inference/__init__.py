"""Inference module for cat behavior detection and identification."""

from cat_watcher.inference.detector import BehaviorDetector, Detection
from cat_watcher.inference.identifier import CatIdentifier, Identification
from cat_watcher.inference.pipeline import InferencePipeline, InferenceResult

__all__ = [
    "BehaviorDetector",
    "CatIdentifier",
    "Detection",
    "Identification",
    "InferencePipeline",
    "InferenceResult",
]
