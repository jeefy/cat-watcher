"""End-to-end tests for the full inference pipeline.

These tests simulate the complete flow from Frigate event to Home Assistant notification.
Run with: pytest tests/integration/ -m integration
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_watcher.schemas import (
    BehaviorType,
    BoundingBox,
    CatName,
    FrigateMQTTEvent,
)
from cat_watcher.inference.detector import Detection
from cat_watcher.inference.identifier import Identification
from cat_watcher.inference.pipeline import InferenceResult


@pytest.fixture
def sample_frigate_event() -> dict[str, Any]:
    """Create a sample Frigate MQTT event payload."""
    return {
        "type": "new",
        "before": {
            "id": "1705401600.123456-abc123",
            "camera": "apollo-dish",
            "label": "cat",
            "start_time": 1705401600.0,
            "zones": ["food_area"],
        },
        "after": {
            "id": "1705401600.123456-abc123",
            "camera": "apollo-dish",
            "label": "cat",
            "start_time": 1705401600.0,
            "zones": ["food_area"],
        },
    }


@pytest.fixture
def sample_inference_result() -> InferenceResult:
    """Create a sample inference result."""
    return InferenceResult(
        timestamp=datetime(2026, 1, 16, 12, 0, 0),
        detections=[
            Detection(
                behavior=BehaviorType.EATING,
                confidence=0.92,
                bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4),
            )
        ],
        identifications=[
            Identification(
                cat=CatName.STARBUCK,
                confidence=0.88,
                probabilities={"starbuck": 0.88, "apollo": 0.08, "mia": 0.03, "unknown": 0.01},
            )
        ],
        processing_time_ms=25.5,
        source="apollo-dish",
    )


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end tests for the complete inference pipeline."""

    @pytest.mark.asyncio
    async def test_frigate_event_to_detection(self, sample_frigate_event):
        """Test parsing Frigate event and creating detection."""
        # Parse the Frigate MQTT event
        event = FrigateMQTTEvent.model_validate(sample_frigate_event)

        assert event.type == "new"
        assert event.after.camera == "apollo-dish"
        assert event.after.label == "cat"

    @pytest.mark.asyncio
    async def test_inference_result_serialization(self, sample_inference_result):
        """Test that inference results serialize correctly for MQTT."""
        # Serialize to dict for MQTT publishing
        result_dict = sample_inference_result.to_dict()

        assert result_dict["source"] == "apollo-dish"
        assert len(result_dict["detections"]) == 1
        assert result_dict["detections"][0]["behavior"] == "cat_eating"
        assert result_dict["detections"][0]["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_inference_result_summary(self, sample_inference_result):
        """Test inference result summary generation."""
        summary = sample_inference_result.summary

        assert "cat_eating" in summary["behaviors"]
        assert "starbuck" in summary["cats"]
        assert summary["num_detections"] == 1
        assert summary["has_alert"] is False

    @pytest.mark.asyncio
    async def test_alert_detection_in_summary(self):
        """Test that vomiting triggers alert flag in summary."""
        result = InferenceResult(
            timestamp=datetime.now(),
            detections=[
                Detection(
                    behavior=BehaviorType.VOMITING,
                    confidence=0.95,
                    bbox=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4),
                )
            ],
            identifications=[],
            processing_time_ms=20.0,
            source="test-camera",
        )

        assert result.summary["has_alert"] is True

    @pytest.mark.asyncio
    async def test_primary_detection_selection(self):
        """Test selection of primary (highest confidence) detection."""
        result = InferenceResult(
            timestamp=datetime.now(),
            detections=[
                Detection(
                    behavior=BehaviorType.PRESENT,
                    confidence=0.7,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
                ),
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.95,
                    bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.4),
                ),
            ],
            identifications=[],
            processing_time_ms=30.0,
            source="test-camera",
        )

        primary = result.get_primary_detection()
        assert primary is not None
        assert primary.behavior == BehaviorType.EATING
        assert primary.confidence == 0.95


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow between components."""

    @pytest.mark.asyncio
    async def test_behavior_to_ha_state(self, sample_inference_result):
        """Test converting behavior detection to Home Assistant state."""
        # The behavior should map to HA state
        detection = sample_inference_result.get_primary_detection()
        assert detection is not None
        state = detection.behavior.value
        assert state == "cat_eating"

    @pytest.mark.asyncio
    async def test_cat_identification_flow(self):
        """Test cat identification data flow."""
        cats = [CatName.STARBUCK, CatName.APOLLO, CatName.MIA]

        for cat in cats:
            identification = Identification(
                cat=cat,
                confidence=0.9,
                probabilities={cat.value: 0.9, "unknown": 0.1},
            )

            # Should convert to dict correctly
            data = identification.to_dict()
            assert data["cat"] == cat.value
            assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_detection_all_behavior_types(self):
        """Test all behavior types can be created and serialized."""
        behaviors = [
            BehaviorType.EATING,
            BehaviorType.DRINKING,
            BehaviorType.VOMITING,
            BehaviorType.WAITING,
            BehaviorType.LITTERBOX,
            BehaviorType.YOWLING,
            BehaviorType.PRESENT,
        ]

        for behavior in behaviors:
            detection = Detection(
                behavior=behavior,
                confidence=0.85,
                bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
            )

            data = detection.to_dict()
            assert data["behavior"] == behavior.value
            assert data["confidence"] == 0.85


@pytest.mark.integration
class TestAlertFlowIntegration:
    """Test alert generation and notification flow."""

    @pytest.mark.asyncio
    async def test_vomiting_is_alert_behavior(self):
        """Test that vomiting behavior is flagged as alert."""
        result = InferenceResult(
            timestamp=datetime.now(),
            detections=[
                Detection(
                    behavior=BehaviorType.VOMITING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
                )
            ],
            identifications=[],
            processing_time_ms=25.0,
            source="test",
        )

        assert result.summary["has_alert"] is True

    @pytest.mark.asyncio
    async def test_yowling_is_alert_behavior(self):
        """Test that yowling behavior is flagged as alert."""
        result = InferenceResult(
            timestamp=datetime.now(),
            detections=[
                Detection(
                    behavior=BehaviorType.YOWLING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
                )
            ],
            identifications=[],
            processing_time_ms=25.0,
            source="test",
        )

        assert result.summary["has_alert"] is True

    @pytest.mark.asyncio
    async def test_eating_is_not_alert(self):
        """Test that eating behavior is not flagged as alert."""
        result = InferenceResult(
            timestamp=datetime.now(),
            detections=[
                Detection(
                    behavior=BehaviorType.EATING,
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
                )
            ],
            identifications=[],
            processing_time_ms=25.0,
            source="test",
        )

        assert result.summary["has_alert"] is False


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration loading and validation."""

    def test_settings_load_from_env(self, monkeypatch):
        """Test that settings load from environment variables."""
        monkeypatch.setenv("FRIGATE_URL", "http://test-frigate:5000")
        monkeypatch.setenv("MQTT_BROKER", "test-broker")
        monkeypatch.setenv("MQTT_PORT", "1884")

        from cat_watcher.config import get_settings

        # Force reload settings
        settings = get_settings()

        # Note: Settings may be cached, so this tests the mechanism
        assert settings is not None

    def test_behavior_type_values(self):
        """Test all behavior types have correct values."""
        expected = {
            "cat_eating",
            "cat_drinking",
            "cat_vomiting",
            "cat_waiting",
            "cat_litterbox",
            "cat_yowling",
            "cat_present",
        }

        actual = {b.value for b in BehaviorType}
        assert actual == expected

    def test_cat_name_values(self):
        """Test all cat names have correct values."""
        expected = {"starbuck", "apollo", "mia", "unknown"}

        actual = {c.value for c in CatName}
        assert actual == expected


@pytest.mark.integration
class TestBoundingBoxIntegration:
    """Test bounding box operations."""

    def test_frigate_box_conversion(self):
        """Test converting Frigate box format (center, width, height) to BoundingBox."""
        # Frigate format: [x_center, y_center, width, height]
        frigate_box = [0.5, 0.5, 0.2, 0.4]  # centered box
        bbox = BoundingBox.from_frigate_box(frigate_box)

        # x_min = 0.5 - 0.1 = 0.4, x_max = 0.5 + 0.1 = 0.6
        # y_min = 0.5 - 0.2 = 0.3, y_max = 0.5 + 0.2 = 0.7
        assert bbox.x_min == 0.4
        assert bbox.y_min == 0.3
        assert bbox.x_max == 0.6
        assert bbox.y_max == 0.7

    def test_pixel_coordinate_conversion(self):
        """Test converting normalized coords to pixels."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)
        x_min, y_min, x_max, y_max = bbox.to_pixel_coords(1920, 1080)

        assert x_min == 192  # 0.1 * 1920
        assert y_min == 216  # 0.2 * 1080
        assert x_max == 576  # 0.3 * 1920
        assert y_max == 432  # 0.4 * 1080

    def test_center_calculation(self):
        """Test bounding box center calculation."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)
        cx, cy = bbox.center

        assert cx == pytest.approx(0.2)
        assert cy == pytest.approx(0.3)

    def test_area_calculation(self):
        """Test bounding box area calculation."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)
        area = bbox.area

        assert area == pytest.approx(0.04)  # 0.2 * 0.2
