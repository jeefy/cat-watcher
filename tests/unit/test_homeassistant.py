"""Tests for Home Assistant integration module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_watcher.homeassistant.discovery import (
    HADiscovery,
    HAEntityConfig,
)
from cat_watcher.homeassistant.publisher import (
    HAAlertManager,
    HAEventPublisher,
)
from cat_watcher.schemas import BehaviorType, CatName


# =============================================================================
# Discovery Tests
# =============================================================================


class TestHAEntityConfig:
    """Tests for HAEntityConfig dataclass."""

    def test_basic_entity_config(self) -> None:
        """Test creating a basic entity config."""
        config = HAEntityConfig(
            component="binary_sensor",
            object_id="test_sensor",
            name="Test Sensor",
            unique_id="cat_watcher_test",
        )

        assert config.component == "binary_sensor"
        assert config.object_id == "test_sensor"
        assert config.name == "Test Sensor"
        assert config.unique_id == "cat_watcher_test"

    def test_to_discovery_payload(self) -> None:
        """Test conversion to discovery payload."""
        config = HAEntityConfig(
            component="binary_sensor",
            object_id="cat_eating",
            name="Cat Eating",
            unique_id="cat_watcher_eating",
            device_class="occupancy",
            icon="mdi:cat",
            state_topic="cat_watcher/behavior/cat_eating",
        )

        device_info = {
            "identifiers": ["cat_watcher"],
            "name": "Cat Watcher",
        }
        payload = config.to_discovery_payload(device_info)

        assert payload["name"] == "Cat Eating"
        assert payload["unique_id"] == "cat_watcher_eating"
        assert payload["device_class"] == "occupancy"
        assert payload["icon"] == "mdi:cat"
        assert payload["state_topic"] == "cat_watcher/behavior/cat_eating"
        assert "device" in payload
        assert payload["device"]["identifiers"] == ["cat_watcher"]

    def test_entity_with_json_attributes(self) -> None:
        """Test entity with JSON attributes topic."""
        config = HAEntityConfig(
            component="sensor",
            object_id="latest",
            name="Latest Detection",
            unique_id="cat_watcher_latest",
            state_topic="cat_watcher/latest",
            json_attributes_topic="cat_watcher/latest",
        )

        device_info = {"identifiers": ["cat_watcher"]}
        payload = config.to_discovery_payload(device_info)

        assert payload["json_attributes_topic"] == "cat_watcher/latest"

    def test_discovery_topic_property(self) -> None:
        """Test discovery topic generation."""
        config = HAEntityConfig(
            component="binary_sensor",
            object_id="test_sensor",
            name="Test",
            unique_id="cat_watcher_test",
        )

        assert config.discovery_topic == "homeassistant/binary_sensor/test_sensor/config"


class TestHADiscovery:
    """Tests for HADiscovery class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        discovery = HADiscovery()

        assert discovery.topic_prefix == "cat_watcher"
        assert discovery.device_name == "Cat Watcher"
        assert len(discovery.cats) > 0
        assert len(discovery.behaviors) > 0

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        discovery = HADiscovery(
            topic_prefix="my_cats",
            device_name="My Cat System",
            device_id="my_cats_v1",
            cats=[CatName.STARBUCK, CatName.APOLLO],
            behaviors=[BehaviorType.EATING, BehaviorType.DRINKING],
        )

        assert discovery.topic_prefix == "my_cats"
        assert discovery.device_name == "My Cat System"
        assert discovery.device_id == "my_cats_v1"
        assert len(discovery.cats) == 2
        assert len(discovery.behaviors) == 2

    def test_device_info_property(self) -> None:
        """Test device info property."""
        discovery = HADiscovery(
            device_name="Cat Watcher",
            device_id="cat_watcher_001",
        )

        info = discovery.device_info
        assert info["name"] == "Cat Watcher"
        assert info["identifiers"] == ["cat_watcher_001"]
        assert "manufacturer" in info
        assert "model" in info

    def test_get_discovery_messages(self) -> None:
        """Test getting discovery messages."""
        discovery = HADiscovery(
            cats=[CatName.STARBUCK],
            behaviors=[BehaviorType.EATING],
        )
        messages = discovery.get_discovery_messages()

        assert len(messages) > 0

        # All messages should be (topic, payload) tuples
        for topic, payload in messages:
            assert topic.startswith("homeassistant/")
            assert topic.endswith("/config")

            # Payload should be valid JSON
            payload_dict = json.loads(payload)
            assert "name" in payload_dict
            assert "unique_id" in payload_dict

    def test_get_removal_messages(self) -> None:
        """Test getting removal messages."""
        discovery = HADiscovery(
            cats=[CatName.STARBUCK],
            behaviors=[BehaviorType.EATING],
        )
        messages = discovery.get_removal_messages()

        assert len(messages) > 0

        # All removal messages should have empty payloads
        for topic, payload in messages:
            assert topic.startswith("homeassistant/")
            assert topic.endswith("/config")
            assert payload == ""

    def test_creates_status_entity(self) -> None:
        """Test that status entity is created in messages."""
        discovery = HADiscovery(
            cats=[CatName.STARBUCK],
            behaviors=[BehaviorType.EATING],
        )
        messages = discovery.get_discovery_messages()

        # Find status entity
        status_messages = [
            (t, p) for t, p in messages
            if "status" in t
        ]
        assert len(status_messages) >= 1

        # Check payload
        _, payload = status_messages[0]
        payload_dict = json.loads(payload)
        assert payload_dict["device_class"] == "connectivity"

    def test_creates_behavior_entities(self) -> None:
        """Test that behavior entities are created."""
        discovery = HADiscovery(
            cats=[CatName.STARBUCK],
            behaviors=[BehaviorType.EATING, BehaviorType.VOMITING],
        )
        messages = discovery.get_discovery_messages()

        # Find eating entity
        eating_messages = [
            (t, p) for t, p in messages
            if "cat_eating" in t and "binary_sensor" in t
        ]
        assert len(eating_messages) >= 1

    def test_creates_cat_entities(self) -> None:
        """Test that cat entities are created."""
        discovery = HADiscovery(
            cats=[CatName.STARBUCK],
            behaviors=[BehaviorType.EATING],
        )
        messages = discovery.get_discovery_messages()

        # Find starbuck presence entity
        starbuck_messages = [
            (t, p) for t, p in messages
            if "starbuck" in t and "present" in t
        ]
        assert len(starbuck_messages) >= 1


# =============================================================================
# Publisher Tests
# =============================================================================


class TestHAEventPublisher:
    """Tests for HAEventPublisher class."""

    @pytest.fixture
    def mock_mqtt(self) -> MagicMock:
        """Create mock MQTT publisher."""
        mock = MagicMock()
        mock.publish = AsyncMock()
        mock._client = MagicMock()
        mock._client.publish = AsyncMock()
        return mock

    @pytest.fixture
    def publisher(self, mock_mqtt: MagicMock) -> HAEventPublisher:
        """Create publisher instance."""
        return HAEventPublisher(
            mqtt_publisher=mock_mqtt,
            topic_prefix="cat_watcher",
            cats=[CatName.STARBUCK, CatName.APOLLO],
            behaviors=[BehaviorType.EATING, BehaviorType.DRINKING],
        )

    @pytest.mark.asyncio
    async def test_publish_discovery(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing discovery messages."""
        await publisher.publish_discovery()

        # Should publish multiple discovery messages
        assert mock_mqtt._client.publish.call_count > 0

        # All calls should be to homeassistant/ topic
        for call in mock_mqtt._client.publish.call_args_list:
            topic = call[0][0]
            assert topic.startswith("homeassistant/")

    @pytest.mark.asyncio
    async def test_remove_discovery(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test removing discovery messages."""
        await publisher.remove_discovery()

        # Should publish removal messages (empty payloads)
        assert mock_mqtt._client.publish.call_count > 0

        for call in mock_mqtt._client.publish.call_args_list:
            payload = call[0][1]
            assert payload == ""

    @pytest.mark.asyncio
    async def test_publish_status(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing status."""
        await publisher.publish_status(online=True)

        mock_mqtt.publish.assert_called_once_with("status", "online", retain=True)

    @pytest.mark.asyncio
    async def test_publish_status_offline(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing offline status."""
        await publisher.publish_status(online=False)

        mock_mqtt.publish.assert_called_once_with("status", "offline", retain=True)

    @pytest.mark.asyncio
    async def test_publish_detection(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing a detection event."""
        await publisher.publish_detection(
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
            confidence=0.95,
            cat_confidence=0.88,
            camera="kitchen",
            event_id="abc123",
        )

        # Should publish to multiple topics
        assert mock_mqtt.publish.call_count >= 5

        # Check for key topics
        call_topics = [call[0][0] for call in mock_mqtt.publish.call_args_list]

        assert "behavior/cat_eating" in call_topics
        assert "cat/starbuck/present" in call_topics
        assert "cat/starbuck/last_behavior" in call_topics
        assert "cat/starbuck/last_seen" in call_topics
        assert "latest" in call_topics
        assert "event" in call_topics

    @pytest.mark.asyncio
    async def test_publish_detection_unknown_cat(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing detection with unknown cat."""
        await publisher.publish_detection(
            behavior=BehaviorType.EATING,
            cat=CatName.UNKNOWN,
            confidence=0.9,
        )

        # Should NOT publish to cat-specific topics
        call_topics = [call[0][0] for call in mock_mqtt.publish.call_args_list]

        assert "behavior/cat_eating" in call_topics
        assert not any("cat/unknown/present" in t for t in call_topics)

    @pytest.mark.asyncio
    async def test_publish_off_states(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test auto-off state publishing."""
        # Simulate an old detection
        old_time = datetime(2020, 1, 1, tzinfo=UTC)
        publisher._active_states["behavior/cat_eating"] = old_time

        await publisher.publish_off_states()

        # Should publish OFF for expired state
        mock_mqtt.publish.assert_called_with("behavior/cat_eating", "OFF")
        assert "behavior/cat_eating" not in publisher._active_states

    @pytest.mark.asyncio
    async def test_clear_all_states(
        self,
        publisher: HAEventPublisher,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test clearing all states."""
        publisher._active_states["behavior/cat_eating"] = datetime.now(UTC)

        await publisher.clear_all_states()

        # Should have published OFF to multiple topics
        assert mock_mqtt.publish.call_count > 0
        assert len(publisher._active_states) == 0


# =============================================================================
# Alert Manager Tests
# =============================================================================


class TestHAAlertManager:
    """Tests for HAAlertManager class."""

    @pytest.fixture
    def mock_mqtt(self) -> MagicMock:
        """Create mock MQTT publisher."""
        mock = MagicMock()
        mock.publish = AsyncMock()
        return mock

    @pytest.fixture
    def alert_manager(self, mock_mqtt: MagicMock) -> HAAlertManager:
        """Create alert manager instance."""
        return HAAlertManager(
            mqtt_publisher=mock_mqtt,
            topic_prefix="cat_watcher",
            alert_cooldowns={
                BehaviorType.VOMITING: 30,
                BehaviorType.EATING: 60,
            },
        )

    def test_should_alert_first_time(
        self,
        alert_manager: HAAlertManager,
    ) -> None:
        """Test first alert always triggers."""
        result = alert_manager.should_alert(
            BehaviorType.EATING,
            CatName.STARBUCK,
        )

        assert result is True

    def test_should_not_alert_during_cooldown(
        self,
        alert_manager: HAAlertManager,
    ) -> None:
        """Test alert respects cooldown."""
        # First alert
        assert alert_manager.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Second alert immediately after
        assert not alert_manager.should_alert(BehaviorType.EATING, CatName.STARBUCK)

    def test_different_cats_independent_cooldowns(
        self,
        alert_manager: HAAlertManager,
    ) -> None:
        """Test different cats have independent cooldowns."""
        # First cat
        assert alert_manager.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Second cat should still alert
        assert alert_manager.should_alert(BehaviorType.EATING, CatName.APOLLO)

    def test_different_behaviors_independent_cooldowns(
        self,
        alert_manager: HAAlertManager,
    ) -> None:
        """Test different behaviors have independent cooldowns."""
        # First behavior
        assert alert_manager.should_alert(BehaviorType.EATING, CatName.STARBUCK)

        # Different behavior should still alert
        assert alert_manager.should_alert(BehaviorType.VOMITING, CatName.STARBUCK)

    @pytest.mark.asyncio
    async def test_publish_alert(
        self,
        alert_manager: HAAlertManager,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing an alert."""
        await alert_manager.publish_alert(
            behavior=BehaviorType.VOMITING,
            cat=CatName.STARBUCK,
            confidence=0.95,
            camera="kitchen",
        )

        # Should publish to alert topics
        assert mock_mqtt.publish.call_count == 2

        # Check main alert topic
        calls = mock_mqtt.publish.call_args_list
        alert_call = [c for c in calls if c[0][0] == "alert"][0]
        alert_payload = alert_call[0][1]

        assert alert_payload["behavior"] == "cat_vomiting"
        assert alert_payload["cat"] == "starbuck"
        assert alert_payload["confidence"] == 0.95
        assert alert_payload["priority"] == 1  # Vomiting is highest priority

    @pytest.mark.asyncio
    async def test_publish_alert_with_custom_message(
        self,
        alert_manager: HAAlertManager,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test publishing alert with custom message."""
        await alert_manager.publish_alert(
            behavior=BehaviorType.VOMITING,
            cat=CatName.STARBUCK,
            confidence=0.9,
            message="Starbuck might be sick!",
        )

        calls = mock_mqtt.publish.call_args_list
        alert_call = [c for c in calls if c[0][0] == "alert"][0]
        alert_payload = alert_call[0][1]

        assert alert_payload["message"] == "Starbuck might be sick!"

    @pytest.mark.asyncio
    async def test_publish_alert_respects_cooldown(
        self,
        alert_manager: HAAlertManager,
        mock_mqtt: MagicMock,
    ) -> None:
        """Test alert not published during cooldown."""
        # First alert
        await alert_manager.publish_alert(
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
            confidence=0.9,
        )

        # Reset mock
        mock_mqtt.publish.reset_mock()

        # Second alert should be blocked
        await alert_manager.publish_alert(
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
            confidence=0.9,
        )

        assert mock_mqtt.publish.call_count == 0

    def test_behavior_priorities(self) -> None:
        """Test behavior priority levels."""
        priorities = HAAlertManager.BEHAVIOR_PRIORITIES

        # Vomiting should have highest priority (lowest number)
        assert priorities[BehaviorType.VOMITING] < priorities[BehaviorType.EATING]
        assert priorities[BehaviorType.VOMITING] < priorities[BehaviorType.PRESENT]

        # Health issues should be higher priority than normal behaviors
        assert priorities[BehaviorType.YOWLING] < priorities[BehaviorType.EATING]
