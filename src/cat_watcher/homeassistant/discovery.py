"""Home Assistant MQTT Auto-Discovery.

Implements MQTT discovery protocol for automatic entity creation in Home Assistant.
See: https://www.home-assistant.io/integrations/mqtt/#mqtt-discovery
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from cat_watcher.schemas import BehaviorType, CatName


@dataclass
class HAEntityConfig:
    """Configuration for a Home Assistant entity."""

    component: Literal[
        "binary_sensor", "sensor", "camera", "device_tracker", "switch"
    ]
    object_id: str
    name: str
    unique_id: str
    device_class: str | None = None
    icon: str | None = None
    state_topic: str | None = None
    json_attributes_topic: str | None = None
    value_template: str | None = None
    payload_on: str | None = None
    payload_off: str | None = None
    unit_of_measurement: str | None = None
    expire_after: int | None = None  # Seconds before sensor becomes unavailable
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_discovery_payload(self, device_info: dict[str, Any]) -> dict[str, Any]:
        """Generate MQTT discovery payload.

        Args:
            device_info: Device information dict for grouping entities

        Returns:
            Discovery payload dict
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "unique_id": self.unique_id,
            "device": device_info,
        }

        if self.state_topic:
            payload["state_topic"] = self.state_topic
        if self.json_attributes_topic:
            payload["json_attributes_topic"] = self.json_attributes_topic
        if self.value_template:
            payload["value_template"] = self.value_template
        if self.device_class:
            payload["device_class"] = self.device_class
        if self.icon:
            payload["icon"] = self.icon
        if self.payload_on:
            payload["payload_on"] = self.payload_on
        if self.payload_off:
            payload["payload_off"] = self.payload_off
        if self.unit_of_measurement:
            payload["unit_of_measurement"] = self.unit_of_measurement
        if self.expire_after:
            payload["expire_after"] = self.expire_after

        # Add any extra configuration
        payload.update(self.extra_config)

        return payload

    @property
    def discovery_topic(self) -> str:
        """Get the MQTT discovery topic."""
        return f"homeassistant/{self.component}/{self.object_id}/config"


class HADiscovery:
    """Home Assistant MQTT Auto-Discovery manager.

    Creates and publishes discovery messages for Cat Watcher entities.

    Example:
        ```python
        discovery = HADiscovery(
            mqtt_publisher=publisher,
            device_name="Cat Watcher",
            device_id="cat_watcher_001",
        )
        await discovery.publish_all()
        ```
    """

    # Behavior icons for Home Assistant
    BEHAVIOR_ICONS = {
        BehaviorType.EATING: "mdi:food-drumstick",
        BehaviorType.DRINKING: "mdi:cup-water",
        BehaviorType.VOMITING: "mdi:emoticon-sick",
        BehaviorType.WAITING: "mdi:clock-outline",
        BehaviorType.LITTERBOX: "mdi:cat",
        BehaviorType.YOWLING: "mdi:bullhorn",
        BehaviorType.PRESENT: "mdi:eye",
    }

    # Cat icons
    CAT_ICONS = {
        CatName.STARBUCK: "mdi:cat",
        CatName.APOLLO: "mdi:cat",
        CatName.MIA: "mdi:cat",
        CatName.UNKNOWN: "mdi:help-circle",
    }

    def __init__(
        self,
        topic_prefix: str = "cat_watcher",
        device_name: str = "Cat Watcher",
        device_id: str = "cat_watcher",
        manufacturer: str = "Cat Watcher",
        model: str = "ML Cat Behavior Monitor",
        sw_version: str = "1.0.0",
        cats: list[CatName] | None = None,
        behaviors: list[BehaviorType] | None = None,
    ) -> None:
        """Initialize discovery manager.

        Args:
            topic_prefix: MQTT topic prefix for state topics
            device_name: Human-readable device name
            device_id: Unique device identifier
            manufacturer: Device manufacturer
            model: Device model
            sw_version: Software version
            cats: List of cats to create entities for (default: all)
            behaviors: List of behaviors to create entities for (default: all)
        """
        self.topic_prefix = topic_prefix
        self.device_name = device_name
        self.device_id = device_id
        self.manufacturer = manufacturer
        self.model = model
        self.sw_version = sw_version

        self.cats = cats or list(CatName)
        self.behaviors = behaviors or list(BehaviorType)

        self._entities: list[HAEntityConfig] = []
        self._build_entities()

    @property
    def device_info(self) -> dict[str, Any]:
        """Get device information for discovery payloads."""
        return {
            "identifiers": [self.device_id],
            "name": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "sw_version": self.sw_version,
        }

    def _build_entities(self) -> None:
        """Build all entity configurations."""
        self._entities = []

        # Service status sensor
        self._entities.append(
            HAEntityConfig(
                component="binary_sensor",
                object_id=f"{self.device_id}_status",
                name=f"{self.device_name} Status",
                unique_id=f"{self.device_id}_status",
                device_class="connectivity",
                state_topic=f"{self.topic_prefix}/status",
                payload_on="online",
                payload_off="offline",
            )
        )

        # Events processed counter
        self._entities.append(
            HAEntityConfig(
                component="sensor",
                object_id=f"{self.device_id}_events_processed",
                name=f"{self.device_name} Events Processed",
                unique_id=f"{self.device_id}_events_processed",
                icon="mdi:counter",
                state_topic=f"{self.topic_prefix}/stats",
                value_template="{{ value_json.events_processed }}",
            )
        )

        # Create per-behavior binary sensors
        for behavior in self.behaviors:
            behavior_name = behavior.value.replace("cat_", "").title()
            self._entities.append(
                HAEntityConfig(
                    component="binary_sensor",
                    object_id=f"{self.device_id}_{behavior.value}",
                    name=f"Cat {behavior_name}",
                    unique_id=f"{self.device_id}_{behavior.value}",
                    device_class="occupancy" if behavior == BehaviorType.PRESENT else None,
                    icon=self.BEHAVIOR_ICONS.get(behavior, "mdi:cat"),
                    state_topic=f"{self.topic_prefix}/behavior/{behavior.value}",
                    payload_on="ON",
                    payload_off="OFF",
                    expire_after=300,  # 5 minutes
                    json_attributes_topic=f"{self.topic_prefix}/behavior/{behavior.value}/attributes",
                )
            )

        # Create per-cat entities
        for cat in self.cats:
            if cat == CatName.UNKNOWN:
                continue

            cat_name = cat.value.title()

            # Cat presence binary sensor
            self._entities.append(
                HAEntityConfig(
                    component="binary_sensor",
                    object_id=f"{self.device_id}_{cat.value}_present",
                    name=f"{cat_name} Present",
                    unique_id=f"{self.device_id}_{cat.value}_present",
                    device_class="presence",
                    icon=self.CAT_ICONS.get(cat, "mdi:cat"),
                    state_topic=f"{self.topic_prefix}/cat/{cat.value}/present",
                    payload_on="ON",
                    payload_off="OFF",
                    expire_after=600,  # 10 minutes
                )
            )

            # Cat last behavior sensor
            self._entities.append(
                HAEntityConfig(
                    component="sensor",
                    object_id=f"{self.device_id}_{cat.value}_behavior",
                    name=f"{cat_name} Last Behavior",
                    unique_id=f"{self.device_id}_{cat.value}_behavior",
                    icon="mdi:cat",
                    state_topic=f"{self.topic_prefix}/cat/{cat.value}/last_behavior",
                )
            )

            # Cat last seen sensor
            self._entities.append(
                HAEntityConfig(
                    component="sensor",
                    object_id=f"{self.device_id}_{cat.value}_last_seen",
                    name=f"{cat_name} Last Seen",
                    unique_id=f"{self.device_id}_{cat.value}_last_seen",
                    device_class="timestamp",
                    icon="mdi:clock",
                    state_topic=f"{self.topic_prefix}/cat/{cat.value}/last_seen",
                )
            )

            # Per-cat behavior binary sensors (for automations)
            for behavior in self.behaviors:
                if behavior == BehaviorType.PRESENT:
                    continue  # Already have presence sensor

                behavior_name = behavior.value.replace("cat_", "").title()
                self._entities.append(
                    HAEntityConfig(
                        component="binary_sensor",
                        object_id=f"{self.device_id}_{cat.value}_{behavior.value}",
                        name=f"{cat_name} {behavior_name}",
                        unique_id=f"{self.device_id}_{cat.value}_{behavior.value}",
                        icon=self.BEHAVIOR_ICONS.get(behavior, "mdi:cat"),
                        state_topic=f"{self.topic_prefix}/cat/{cat.value}/{behavior.value}",
                        payload_on="ON",
                        payload_off="OFF",
                        expire_after=300,  # 5 minutes
                    )
                )

        # Latest detection sensor (shows most recent event)
        self._entities.append(
            HAEntityConfig(
                component="sensor",
                object_id=f"{self.device_id}_latest_detection",
                name=f"{self.device_name} Latest Detection",
                unique_id=f"{self.device_id}_latest_detection",
                icon="mdi:cat",
                state_topic=f"{self.topic_prefix}/latest",
                value_template="{{ value_json.summary }}",
                json_attributes_topic=f"{self.topic_prefix}/latest",
            )
        )

    @property
    def entities(self) -> list[HAEntityConfig]:
        """Get all entity configurations."""
        return self._entities

    def get_discovery_messages(self) -> list[tuple[str, str]]:
        """Get all discovery messages as (topic, payload) tuples.

        Returns:
            List of (topic, JSON payload) tuples
        """
        messages = []
        for entity in self._entities:
            topic = entity.discovery_topic
            payload = json.dumps(entity.to_discovery_payload(self.device_info))
            messages.append((topic, payload))
        return messages

    def get_removal_messages(self) -> list[tuple[str, str]]:
        """Get messages to remove all entities from Home Assistant.

        Returns:
            List of (topic, empty payload) tuples
        """
        return [(entity.discovery_topic, "") for entity in self._entities]


def create_event_entity(
    device_id: str,
    topic_prefix: str,
    event_types: list[str] | None = None,
) -> HAEntityConfig:
    """Create an event entity for Home Assistant 2023.8+ event platform.

    Args:
        device_id: Device identifier
        topic_prefix: MQTT topic prefix
        event_types: List of event type strings

    Returns:
        HAEntityConfig for event entity
    """
    event_types = event_types or [
        "cat_eating",
        "cat_drinking",
        "cat_vomiting",
        "cat_waiting",
        "cat_litterbox",
        "cat_yowling",
    ]

    return HAEntityConfig(
        component="sensor",  # Events use sensor platform with device_class: None
        object_id=f"{device_id}_event",
        name="Cat Watcher Event",
        unique_id=f"{device_id}_event",
        icon="mdi:cat",
        state_topic=f"{topic_prefix}/event",
        value_template="{{ value_json.event_type }}",
        json_attributes_topic=f"{topic_prefix}/event",
        extra_config={
            "event_types": event_types,
        },
    )
