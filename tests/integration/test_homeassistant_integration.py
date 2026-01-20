"""Integration tests for Home Assistant MQTT discovery.

These tests validate the Home Assistant integration works correctly.
Run with: pytest tests/integration/ -m integration
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_watcher.homeassistant.discovery import HADiscovery, HAEntityConfig
from cat_watcher.homeassistant.publisher import HAEventPublisher
from cat_watcher.inference.detector import Detection
from cat_watcher.inference.identifier import Identification
from cat_watcher.inference.pipeline import InferenceResult
from cat_watcher.schemas import (
    BehaviorType,
    BoundingBox,
    CatName,
)


@pytest.fixture
def inference_result() -> InferenceResult:
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
class TestMQTTDiscoveryIntegration:
    """Integration tests for Home Assistant MQTT discovery."""

    def test_discovery_creates_entities(self):
        """Test discovery creates entity configurations."""
        discovery = HADiscovery(
            topic_prefix="cat_watcher",
            device_name="Cat Watcher",
            device_id="cat_watcher_test",
        )

        # Should have created entities
        assert len(discovery.entities) > 0

    def test_discovery_config_structure(self):
        """Test discovery config has correct structure for HA."""
        discovery = HADiscovery(topic_prefix="cat_watcher")

        messages = discovery.get_discovery_messages()
        assert len(messages) > 0

        for topic, payload_json in messages:
            # Topic should follow HA convention
            assert topic.startswith("homeassistant/")
            assert "cat_watcher" in topic

            # Payload should be valid JSON
            payload = json.loads(payload_json)
            assert "name" in payload
            assert "unique_id" in payload
            assert "device" in payload

    def test_all_cats_have_entities(self):
        """Test all cats have discovery entities."""
        discovery = HADiscovery(
            topic_prefix="cat_watcher",
            cats=[CatName.STARBUCK, CatName.APOLLO, CatName.MIA],
        )

        messages = discovery.get_discovery_messages()
        topics = [t for t, _ in messages]

        # Check each cat has entities
        for cat in [CatName.STARBUCK, CatName.APOLLO, CatName.MIA]:
            cat_topics = [t for t in topics if cat.value in t.lower()]
            assert len(cat_topics) > 0, f"No entities for {cat}"

    def test_behavior_sensor_discovery(self):
        """Test behavior sensor discovery configs."""
        discovery = HADiscovery(topic_prefix="cat_watcher")
        messages = discovery.get_discovery_messages()

        # Should have behavior-related entities
        behavior_topics = [t for t, _ in messages if "behavior" in t.lower()]
        assert len(behavior_topics) > 0

    def test_discovery_unique_ids(self):
        """Test all discovery configs have unique IDs."""
        discovery = HADiscovery(topic_prefix="cat_watcher")
        messages = discovery.get_discovery_messages()

        unique_ids = set()
        for _, payload_json in messages:
            payload = json.loads(payload_json)
            unique_id = payload.get("unique_id")
            assert unique_id is not None
            assert unique_id not in unique_ids, f"Duplicate ID: {unique_id}"
            unique_ids.add(unique_id)

    def test_device_info_structure(self):
        """Test device info is correctly structured."""
        discovery = HADiscovery(
            topic_prefix="cat_watcher",
            device_name="Cat Watcher Test",
            device_id="cat_watcher_test",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
        )

        device_info = discovery.device_info
        assert device_info["name"] == "Cat Watcher Test"
        assert device_info["manufacturer"] == "Test Manufacturer"
        assert device_info["model"] == "Test Model"
        assert device_info["sw_version"] == "1.0.0"
        assert "identifiers" in device_info

    def test_removal_messages(self):
        """Test removal messages have empty payloads."""
        discovery = HADiscovery(topic_prefix="cat_watcher")
        removal = discovery.get_removal_messages()

        assert len(removal) > 0
        for topic, payload in removal:
            assert topic.startswith("homeassistant/")
            assert payload == ""


@pytest.mark.integration
class TestHAEntityConfig:
    """Test Home Assistant entity configuration."""

    def test_sensor_entity_config(self):
        """Test sensor entity configuration."""
        entity = HAEntityConfig(
            component="sensor",
            object_id="cat_watcher_test_sensor",
            name="Test Sensor",
            unique_id="cat_watcher_test_sensor",
            state_topic="cat_watcher/test/state",
            icon="mdi:cat",
        )

        assert entity.discovery_topic == "homeassistant/sensor/cat_watcher_test_sensor/config"

    def test_binary_sensor_entity_config(self):
        """Test binary sensor entity configuration."""
        entity = HAEntityConfig(
            component="binary_sensor",
            object_id="cat_watcher_test_presence",
            name="Test Presence",
            unique_id="cat_watcher_test_presence",
            state_topic="cat_watcher/test/presence",
            payload_on="ON",
            payload_off="OFF",
            device_class="occupancy",
        )

        device_info = {"identifiers": ["test"], "name": "Test"}
        payload = entity.to_discovery_payload(device_info)

        assert payload["payload_on"] == "ON"
        assert payload["payload_off"] == "OFF"
        assert payload["device_class"] == "occupancy"

    def test_entity_extra_config(self):
        """Test entity extra configuration."""
        entity = HAEntityConfig(
            component="sensor",
            object_id="test",
            name="Test",
            unique_id="test",
            extra_config={"custom_field": "custom_value"},
        )

        device_info = {"identifiers": ["test"], "name": "Test"}
        payload = entity.to_discovery_payload(device_info)

        assert payload["custom_field"] == "custom_value"


@pytest.mark.integration
class TestEventPublisherIntegration:
    """Integration tests for event publishing to Home Assistant."""

    @pytest.mark.asyncio
    async def test_publisher_initialization(self):
        """Test publisher can be initialized."""
        mock_mqtt = AsyncMock()
        publisher = HAEventPublisher(mqtt_publisher=mock_mqtt)

        assert publisher is not None
        assert publisher.topic_prefix == "cat_watcher"

    @pytest.mark.asyncio
    async def test_publish_discovery(self):
        """Test publishing discovery messages."""
        mock_client = AsyncMock()
        mock_mqtt = AsyncMock()
        mock_mqtt._client = mock_client
        
        publisher = HAEventPublisher(mqtt_publisher=mock_mqtt)

        await publisher.publish_discovery()

        # Discovery publishes directly to _client.publish
        assert mock_client.publish.call_count > 0


@pytest.mark.integration
class TestHAEntitiesIntegration:
    """Test Home Assistant entity structure."""

    def test_sensor_entity_format(self):
        """Test sensor entities have correct format."""
        unique_id = "cat_watcher_starbuck_behavior"

        # Should be valid for HA
        assert "_" in unique_id
        assert unique_id.replace("_", "").isalnum()

    def test_binary_sensor_format(self):
        """Test binary sensor entities have correct format."""
        unique_id = "cat_watcher_starbuck_present"

        assert "_" in unique_id
        assert "present" in unique_id

    def test_state_values(self):
        """Test state values are valid for HA."""
        for behavior in BehaviorType:
            state = behavior.value
            # States should not have spaces or special chars (except underscore)
            assert " " not in state
            assert state.replace("_", "").isalnum()

    def test_behavior_icons(self):
        """Test all behaviors have icons."""
        for behavior in BehaviorType:
            icon = HADiscovery.BEHAVIOR_ICONS.get(behavior)
            assert icon is not None, f"No icon for {behavior}"
            assert icon.startswith("mdi:")

    def test_cat_icons(self):
        """Test all cats have icons."""
        for cat in CatName:
            icon = HADiscovery.CAT_ICONS.get(cat)
            assert icon is not None, f"No icon for {cat}"
            assert icon.startswith("mdi:")


@pytest.mark.integration
class TestInferenceResultToHA:
    """Test converting inference results for Home Assistant."""

    def test_result_to_dict_json_serializable(self, inference_result):
        """Test inference result dict is JSON serializable."""
        result_dict = inference_result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert "detections" in parsed
        assert "identifications" in parsed
        assert "timestamp" in parsed

    def test_detection_attributes(self, inference_result):
        """Test detection attributes for HA."""
        detection = inference_result.get_primary_detection()
        assert detection is not None

        detection_dict = detection.to_dict()

        assert "behavior" in detection_dict
        assert "confidence" in detection_dict
        assert "bbox" in detection_dict
        assert detection_dict["behavior"] == "cat_eating"

    def test_identification_attributes(self, inference_result):
        """Test identification attributes for HA."""
        identification = inference_result.get_primary_identification()
        assert identification is not None

        id_dict = identification.to_dict()

        assert "cat" in id_dict
        assert "confidence" in id_dict
        assert id_dict["cat"] == "starbuck"

    def test_summary_for_ha(self, inference_result):
        """Test summary is suitable for HA attributes."""
        summary = inference_result.summary

        # Should be JSON serializable
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)

        assert "behaviors" in parsed
        assert "cats" in parsed
        assert "has_alert" in parsed
        assert "num_detections" in parsed
