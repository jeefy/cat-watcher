"""Tests for inference pipeline."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cat_watcher.schemas import BehaviorType, BoundingBox, CatName

# Skip all tests if required packages are not available
pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from cat_watcher.inference.detector import BehaviorDetector, Detection
from cat_watcher.inference.identifier import CatIdentifier, Identification
from cat_watcher.inference.pipeline import (
    AlertFilter,
    InferencePipeline,
    InferenceResult,
    PipelineConfig,
)


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self) -> None:
        """Test creating a Detection."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
        detection = Detection(
            behavior=BehaviorType.EATING,
            confidence=0.95,
            bbox=bbox,
        )

        assert detection.behavior == BehaviorType.EATING
        assert detection.confidence == 0.95
        assert detection.bbox.x_min == 0.1

    def test_detection_to_dict(self) -> None:
        """Test Detection to_dict method."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
        detection = Detection(
            behavior=BehaviorType.DRINKING,
            confidence=0.85,
            bbox=bbox,
        )

        result = detection.to_dict()

        assert result["behavior"] == "cat_drinking"
        assert result["confidence"] == 0.85
        assert result["bbox"]["x_min"] == 0.1
        assert result["bbox"]["x_max"] == 0.5


class TestIdentification:
    """Tests for Identification dataclass."""

    def test_identification_creation(self) -> None:
        """Test creating an Identification."""
        identification = Identification(
            cat=CatName.STARBUCK,
            confidence=0.92,
            probabilities={
                "starbuck": 0.92,
                "apollo": 0.05,
                "mia": 0.02,
                "unknown": 0.01,
            },
        )

        assert identification.cat == CatName.STARBUCK
        assert identification.confidence == 0.92

    def test_identification_to_dict(self) -> None:
        """Test Identification to_dict method."""
        identification = Identification(
            cat=CatName.APOLLO,
            confidence=0.88,
            probabilities={
                "starbuck": 0.08,
                "apollo": 0.88,
                "mia": 0.03,
                "unknown": 0.01,
            },
        )

        result = identification.to_dict()

        assert result["cat"] == "apollo"
        assert result["confidence"] == 0.88
        assert result["probabilities"]["apollo"] == 0.88


class TestBehaviorDetector:
    """Tests for BehaviorDetector."""

    def test_behavior_classes(self) -> None:
        """Test all behavior classes are defined."""
        assert len(BehaviorDetector.BEHAVIOR_CLASSES) == 7
        assert BehaviorType.EATING in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.DRINKING in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.VOMITING in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.WAITING in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.LITTERBOX in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.YOWLING in BehaviorDetector.BEHAVIOR_CLASSES
        assert BehaviorType.PRESENT in BehaviorDetector.BEHAVIOR_CLASSES

    def test_detector_init(self) -> None:
        """Test detector initialization."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            detector = BehaviorDetector(
                model_path=f.name,
                confidence_threshold=0.6,
            )

            assert detector.model_path == Path(f.name)
            assert detector.confidence_threshold == 0.6
            assert not detector.is_loaded

    def test_detector_auto_detect_onnx(self) -> None:
        """Test auto-detection of ONNX models."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            detector = BehaviorDetector(model_path=f.name)
            assert detector.use_onnx is True

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            detector = BehaviorDetector(model_path=f.name)
            assert detector.use_onnx is False

    def test_iou_calculation(self) -> None:
        """Test IoU calculation between overlapping boxes."""
        # Normalized coords: 50% overlap
        box1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
        box2 = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

        iou = BehaviorDetector._iou(box1, box2)

        # Intersection: 0.25 * 0.25 = 0.0625
        # Union: 0.25 + 0.25 - 0.0625 = 0.4375
        # IoU = 0.0625 / 0.4375 ≈ 0.143
        assert 0.14 < iou < 0.15

    def test_iou_no_overlap(self) -> None:
        """Test IoU calculation between non-overlapping boxes."""
        box1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2)
        box2 = BoundingBox(x_min=0.5, y_min=0.5, x_max=0.7, y_max=0.7)

        iou = BehaviorDetector._iou(box1, box2)

        assert iou == 0.0


class TestCatIdentifier:
    """Tests for CatIdentifier."""

    def test_cat_classes(self) -> None:
        """Test all cat classes are defined."""
        assert len(CatIdentifier.CAT_CLASSES) == 4
        assert CatName.STARBUCK in CatIdentifier.CAT_CLASSES
        assert CatName.APOLLO in CatIdentifier.CAT_CLASSES
        assert CatName.MIA in CatIdentifier.CAT_CLASSES
        assert CatName.UNKNOWN in CatIdentifier.CAT_CLASSES

    def test_identifier_init(self) -> None:
        """Test identifier initialization."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            identifier = CatIdentifier(
                model_path=f.name,
                confidence_threshold=0.7,
            )

            assert identifier.model_path == Path(f.name)
            assert identifier.confidence_threshold == 0.7
            assert not identifier.is_loaded

    def test_resize_and_crop_wide(self) -> None:
        """Test resize and crop for wide images."""
        from PIL import Image

        # Create a wide image
        img = Image.new("RGB", (640, 480))

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            identifier = CatIdentifier(model_path=f.name, img_size=224)
            result = identifier._resize_and_crop(img)

            assert result.size == (224, 224)

    def test_resize_and_crop_tall(self) -> None:
        """Test resize and crop for tall images."""
        from PIL import Image

        # Create a tall image
        img = Image.new("RGB", (480, 640))

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            identifier = CatIdentifier(model_path=f.name, img_size=224)
            result = identifier._resize_and_crop(img)

            assert result.size == (224, 224)


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_result_to_dict(self) -> None:
        """Test InferenceResult to_dict method."""
        result = InferenceResult(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            detections=[
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6),
                )
            ],
            identifications=[
                Identification(
                    cat=CatName.STARBUCK,
                    confidence=0.85,
                    probabilities={"starbuck": 0.85},
                )
            ],
            source="test_camera",
            processing_time_ms=150.5,
        )

        data = result.to_dict()

        assert data["source"] == "test_camera"
        assert data["processing_time_ms"] == 150.5
        assert len(data["detections"]) == 1
        assert len(data["identifications"]) == 1

    def test_result_summary(self) -> None:
        """Test InferenceResult summary generation."""
        result = InferenceResult(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            detections=[
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6),
                ),
                Detection(
                    behavior=BehaviorType.DRINKING,
                    confidence=0.8,
                    bbox=BoundingBox(x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7),
                ),
            ],
            identifications=[
                Identification(
                    cat=CatName.STARBUCK,
                    confidence=0.85,
                    probabilities={"starbuck": 0.85},
                )
            ],
            source="test_camera",
            processing_time_ms=100.0,
        )

        summary = result.summary

        assert summary["num_detections"] == 2
        assert "cat_eating" in summary["behaviors"]
        assert "cat_drinking" in summary["behaviors"]
        assert "starbuck" in summary["cats"]

    def test_primary_detection(self) -> None:
        """Test getting primary detection."""
        result = InferenceResult(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            detections=[
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.7,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
                ),
                Detection(
                    behavior=BehaviorType.VOMITING,
                    confidence=0.95,
                    bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.4),
                ),
            ],
            identifications=[],
            source="test",
            processing_time_ms=50.0,
        )

        primary = result.get_primary_detection()

        assert primary is not None
        assert primary.behavior == BehaviorType.VOMITING
        assert primary.confidence == 0.95


class TestAlertFilter:
    """Tests for AlertFilter."""

    def test_should_alert_first_time(self) -> None:
        """Test first detection should alert."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        should_alert = filter.should_alert(
            BehaviorType.EATING,
            CatName.STARBUCK,
        )

        assert should_alert is True

    def test_should_not_alert_within_cooldown(self) -> None:
        """Test detection within cooldown should not alert."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        # First alert
        filter.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Second attempt within cooldown
        should_alert = filter.should_alert(
            BehaviorType.EATING,
            CatName.STARBUCK,
        )

        assert should_alert is False

    def test_different_cats_different_cooldowns(self) -> None:
        """Test different cats have separate cooldowns."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        # Alert for first cat
        filter.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Alert for second cat should work
        should_alert = filter.should_alert(
            BehaviorType.EATING,
            CatName.APOLLO,
        )

        assert should_alert is True

    def test_different_behaviors_different_cooldowns(self) -> None:
        """Test different behaviors have separate cooldowns."""
        filter = AlertFilter(
            cooldowns={
                BehaviorType.EATING: 300,
                BehaviorType.DRINKING: 300,
            }
        )

        # Alert for eating
        filter.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Alert for drinking should work
        should_alert = filter.should_alert(
            BehaviorType.DRINKING,
            CatName.STARBUCK,
        )

        assert should_alert is True

    def test_reset(self) -> None:
        """Test resetting alert filter."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        # First alert
        filter.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Reset
        filter.reset()

        # Should alert again after reset
        should_alert = filter.should_alert(
            BehaviorType.EATING,
            CatName.STARBUCK,
        )

        assert should_alert is True


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = PipelineConfig()

        assert config.behavior_model_path == ""
        assert config.catid_model_path == ""
        assert config.use_onnx is True
        assert config.behavior_confidence == 0.5
        assert config.catid_confidence == 0.5

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PipelineConfig(
            behavior_model_path="/models/behavior.onnx",
            catid_model_path="/models/catid.onnx",
            use_onnx=True,
            device="cuda:0",
            behavior_confidence=0.7,
            catid_confidence=0.8,
        )

        assert config.behavior_model_path == "/models/behavior.onnx"
        assert config.device == "cuda:0"
        assert config.behavior_confidence == 0.7


class TestInferencePipeline:
    """Tests for InferencePipeline."""

    def test_pipeline_init(self) -> None:
        """Test pipeline initialization."""
        config = PipelineConfig(
            behavior_model_path="/models/behavior.onnx",
            catid_model_path="/models/catid.onnx",
        )
        pipeline = InferencePipeline(config)

        # Models are configured but not loaded yet
        assert pipeline.config.behavior_model_path == "/models/behavior.onnx"
        assert pipeline.config.catid_model_path == "/models/catid.onnx"
        assert not pipeline.is_loaded

    def test_pipeline_no_models(self) -> None:
        """Test pipeline with no models configured."""
        config = PipelineConfig()
        pipeline = InferencePipeline(config)

        assert pipeline.has_detector is False
        assert pipeline.has_identifier is False

    def test_detect_only_no_detector_raises(self) -> None:
        """Test detect_only raises when no detector configured."""
        from PIL import Image

        config = PipelineConfig()
        pipeline = InferencePipeline(config)
        img = Image.new("RGB", (640, 480))

        with pytest.raises(RuntimeError, match="not configured"):
            pipeline.detect_only(img)

    def test_identify_only_no_identifier_raises(self) -> None:
        """Test identify_only raises when no identifier configured."""
        from PIL import Image

        config = PipelineConfig()
        pipeline = InferencePipeline(config)
        img = Image.new("RGB", (640, 480))

        with pytest.raises(RuntimeError, match="not configured"):
            pipeline.identify_only(img)


class TestFilterResult:
    """Tests for AlertFilter.filter_result."""

    def test_filter_result_empty(self) -> None:
        """Test filtering empty result."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        result = InferenceResult(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            detections=[],
            identifications=[],
            source="test",
            processing_time_ms=10.0,
        )

        alerts = filter.filter_result(result)

        assert len(alerts) == 0

    def test_filter_result_with_alert(self) -> None:
        """Test filtering result with detection and identification."""
        filter = AlertFilter(cooldowns={BehaviorType.EATING: 300})

        result = InferenceResult(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            detections=[
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6),
                )
            ],
            identifications=[
                Identification(
                    cat=CatName.STARBUCK,
                    confidence=0.85,
                    probabilities={"starbuck": 0.85},
                )
            ],
            source="test",
            processing_time_ms=50.0,
        )

        alerts = filter.filter_result(result)

        assert len(alerts) == 1
        detection, identification = alerts[0]
        assert detection.behavior == BehaviorType.EATING
        assert identification.cat == CatName.STARBUCK
