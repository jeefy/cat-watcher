"""Combined inference pipeline for behavior detection and cat identification."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cat_watcher.inference.detector import BehaviorDetector, Detection
from cat_watcher.inference.identifier import CatIdentifier, Identification
from cat_watcher.schemas import BehaviorType, BoundingBox, CatName


@dataclass
class InferenceResult:
    """Complete inference result combining detection and identification."""

    timestamp: datetime
    detections: list[Detection]
    identifications: list[Identification]
    processing_time_ms: float
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "processing_time_ms": self.processing_time_ms,
            "detections": [d.to_dict() for d in self.detections],
            "identifications": [i.to_dict() for i in self.identifications],
            "summary": self.summary,
        }

    @property
    def summary(self) -> dict[str, Any]:
        """Get a summary of the inference result."""
        behaviors = [d.behavior.value for d in self.detections]
        cats = [i.cat.value for i in self.identifications if i.cat != CatName.UNKNOWN]

        return {
            "num_detections": len(self.detections),
            "behaviors": list(set(behaviors)),
            "cats": list(set(cats)),
            "has_alert": any(
                d.behavior in [BehaviorType.VOMITING, BehaviorType.YOWLING]
                for d in self.detections
            ),
        }

    def get_primary_detection(self) -> Detection | None:
        """Get the highest confidence detection."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    def get_primary_identification(self) -> Identification | None:
        """Get the highest confidence identification."""
        if not self.identifications:
            return None
        return max(self.identifications, key=lambda i: i.confidence)


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""

    # Model paths
    behavior_model_path: Path | str = ""
    catid_model_path: Path | str = ""

    # Thresholds
    behavior_confidence: float = 0.5
    catid_confidence: float = 0.5
    iou_threshold: float = 0.45

    # Hardware
    device: str = ""
    use_onnx: bool = True

    # Processing
    identify_all_detections: bool = True  # Run cat ID on each behavior detection
    max_detections: int = 10  # Limit detections to process


class InferencePipeline:
    """Combined pipeline for behavior detection and cat identification."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._detector: BehaviorDetector | None = None
        self._identifier: CatIdentifier | None = None
        self._loaded = False

    def load(self) -> None:
        """Load all models."""
        if self.config.behavior_model_path:
            self._detector = BehaviorDetector(
                model_path=self.config.behavior_model_path,
                confidence_threshold=self.config.behavior_confidence,
                iou_threshold=self.config.iou_threshold,
                device=self.config.device,
                use_onnx=self.config.use_onnx,
            )
            self._detector.load()

        if self.config.catid_model_path:
            self._identifier = CatIdentifier(
                model_path=self.config.catid_model_path,
                confidence_threshold=self.config.catid_confidence,
                device=self.config.device,
                use_onnx=self.config.use_onnx,
            )
            self._identifier.load()

        self._loaded = True

    def process(
        self,
        image: Image.Image | np.ndarray | Path | str,
        source: str = "",
    ) -> InferenceResult:
        """Run full inference pipeline on an image.

        Args:
            image: Input image
            source: Source identifier (e.g., camera name)

        Returns:
            Complete inference result
        """
        if not self._loaded:
            self.load()

        start_time = time.perf_counter()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Run behavior detection
        detections: list[Detection] = []
        if self._detector is not None:
            detections = self._detector.detect(image)
            # Limit detections
            detections = detections[: self.config.max_detections]

        # Run cat identification
        identifications: list[Identification] = []
        if self._identifier is not None:
            if self.config.identify_all_detections and detections:
                # Identify cat in each detection bbox
                for detection in detections:
                    identification = self._identifier.identify(
                        image, bbox=detection.bbox
                    )
                    identifications.append(identification)
            else:
                # Single identification on full image
                identification = self._identifier.identify(image)
                identifications.append(identification)

        processing_time = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            timestamp=datetime.now(UTC),
            detections=detections,
            identifications=identifications,
            processing_time_ms=processing_time,
            source=source,
        )

    async def process_async(
        self,
        image: Image.Image | np.ndarray | Path | str,
        source: str = "",
    ) -> InferenceResult:
        """Async wrapper for process.

        Args:
            image: Input image
            source: Source identifier

        Returns:
            Complete inference result
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process(image, source),
        )

    def detect_only(
        self,
        image: Image.Image | np.ndarray | Path | str,
    ) -> list[Detection]:
        """Run only behavior detection.

        Args:
            image: Input image

        Returns:
            List of detections
        """
        if self._detector is None:
            raise RuntimeError("Behavior detector not configured")

        if not self._detector.is_loaded:
            self._detector.load()

        return self._detector.detect(image)

    def identify_only(
        self,
        image: Image.Image | np.ndarray | Path | str,
        bbox: BoundingBox | None = None,
    ) -> Identification:
        """Run only cat identification.

        Args:
            image: Input image
            bbox: Optional bounding box to crop

        Returns:
            Identification result
        """
        if self._identifier is None:
            raise RuntimeError("Cat identifier not configured")

        if not self._identifier.is_loaded:
            self._identifier.load()

        return self._identifier.identify(image, bbox)

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._loaded

    @property
    def has_detector(self) -> bool:
        """Check if behavior detector is configured."""
        return self._detector is not None

    @property
    def has_identifier(self) -> bool:
        """Check if cat identifier is configured."""
        return self._identifier is not None


class AlertFilter:
    """Filter and deduplicate alerts based on cooldowns."""

    def __init__(
        self,
        cooldowns: dict[BehaviorType, float] | None = None,
    ) -> None:
        """Initialize filter.

        Args:
            cooldowns: Cooldown periods in seconds per behavior type
        """
        self.cooldowns = cooldowns or {
            BehaviorType.EATING: 300,  # 5 minutes
            BehaviorType.DRINKING: 300,
            BehaviorType.VOMITING: 60,  # 1 minute (always alert)
            BehaviorType.WAITING: 600,  # 10 minutes
            BehaviorType.LITTERBOX: 300,
            BehaviorType.YOWLING: 60,
            BehaviorType.PRESENT: 600,
        }
        self._last_alerts: dict[tuple[BehaviorType, CatName], float] = {}

    def should_alert(
        self,
        behavior: BehaviorType,
        cat: CatName,
    ) -> bool:
        """Check if an alert should be sent.

        Args:
            behavior: Detected behavior
            cat: Identified cat

        Returns:
            True if alert should be sent
        """
        key = (behavior, cat)
        now = time.time()

        last_alert = self._last_alerts.get(key)
        cooldown = self.cooldowns.get(behavior, 300)

        if last_alert is None or (now - last_alert) > cooldown:
            self._last_alerts[key] = now
            return True

        return False

    def filter_result(self, result: InferenceResult) -> list[tuple[Detection, Identification]]:
        """Filter inference result for alertable events.

        Args:
            result: Full inference result

        Returns:
            List of (detection, identification) pairs that should alert
        """
        alerts: list[tuple[Detection, Identification]] = []

        # Pair detections with identifications
        for i, detection in enumerate(result.detections):
            if i < len(result.identifications):
                identification = result.identifications[i]
            elif result.identifications:
                identification = result.identifications[0]
            else:
                identification = Identification(
                    cat=CatName.UNKNOWN,
                    confidence=0.0,
                    probabilities={},
                )

            if self.should_alert(detection.behavior, identification.cat):
                alerts.append((detection, identification))

        return alerts

    def reset(self) -> None:
        """Reset all cooldowns."""
        self._last_alerts.clear()
