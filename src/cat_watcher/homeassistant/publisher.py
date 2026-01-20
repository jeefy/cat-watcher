"""Home Assistant event publisher.

Publishes cat detection events to MQTT in Home Assistant compatible format.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from cat_watcher.homeassistant.discovery import HADiscovery
from cat_watcher.schemas import BehaviorType, CatName

if TYPE_CHECKING:
    from cat_watcher.mqtt import MQTTPublisher
    from cat_watcher.inference.pipeline import InferenceResult

logger = structlog.get_logger()


class HAEventPublisher:
    """Publisher for Home Assistant compatible MQTT events.

    Handles:
    - MQTT auto-discovery for entities
    - State updates for binary sensors
    - Event publishing with attributes
    - Per-cat and per-behavior topics

    Example:
        ```python
        publisher = HAEventPublisher(
            mqtt_publisher=mqtt,
            topic_prefix="cat_watcher",
        )

        # Publish discovery on startup
        await publisher.publish_discovery()

        # Publish detection event
        await publisher.publish_detection(
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
            confidence=0.95,
            camera="kitchen",
        )
        ```
    """

    def __init__(
        self,
        mqtt_publisher: MQTTPublisher,
        topic_prefix: str = "cat_watcher",
        device_name: str = "Cat Watcher",
        device_id: str = "cat_watcher",
        cats: list[CatName] | None = None,
        behaviors: list[BehaviorType] | None = None,
        auto_off_delay: float = 60.0,
    ) -> None:
        """Initialize publisher.

        Args:
            mqtt_publisher: MQTT publisher instance
            topic_prefix: Base topic prefix (published to {prefix}/...)
            device_name: Human-readable device name for HA
            device_id: Unique device identifier
            cats: List of cats (default: all except UNKNOWN)
            behaviors: List of behaviors (default: all)
            auto_off_delay: Seconds before auto-publishing OFF state
        """
        self.mqtt = mqtt_publisher
        self.topic_prefix = topic_prefix
        self.device_name = device_name
        self.device_id = device_id
        self.auto_off_delay = auto_off_delay

        # Default cats (exclude UNKNOWN from discovery)
        self.cats = cats or [c for c in CatName if c != CatName.UNKNOWN]
        self.behaviors = behaviors or list(BehaviorType)

        # Discovery manager
        self.discovery = HADiscovery(
            topic_prefix=topic_prefix,
            device_name=device_name,
            device_id=device_id,
            cats=self.cats,
            behaviors=self.behaviors,
        )

        # Track active states for auto-off
        self._active_states: dict[str, datetime] = {}

    async def publish_discovery(self) -> None:
        """Publish MQTT auto-discovery messages for all entities."""
        logger.info("Publishing Home Assistant discovery messages")

        messages = self.discovery.get_discovery_messages()
        for topic, payload in messages:
            # Discovery messages go directly to homeassistant/ prefix
            # Don't use the topic_prefix wrapper
            if self.mqtt._client is not None:
                await self.mqtt._client.publish(topic, payload, retain=True)

        logger.info(f"Published {len(messages)} discovery messages")

    async def remove_discovery(self) -> None:
        """Remove all entities from Home Assistant."""
        logger.info("Removing Home Assistant discovery messages")

        messages = self.discovery.get_removal_messages()
        for topic, payload in messages:
            if self.mqtt._client is not None:
                await self.mqtt._client.publish(topic, payload, retain=True)

        logger.info(f"Removed {len(messages)} discovery messages")

    async def publish_status(self, online: bool = True) -> None:
        """Publish service status.

        Args:
            online: Whether service is online
        """
        status = "online" if online else "offline"
        await self.mqtt.publish("status", status, retain=True)

    async def publish_stats(self, stats: dict[str, Any]) -> None:
        """Publish service statistics.

        Args:
            stats: Statistics dictionary
        """
        await self.mqtt.publish("stats", stats, retain=True)

    async def publish_detection(
        self,
        behavior: BehaviorType,
        cat: CatName,
        confidence: float,
        cat_confidence: float = 0.0,
        camera: str = "",
        event_id: str = "",
        bbox: dict[str, float] | None = None,
        snapshot_url: str | None = None,
    ) -> None:
        """Publish a cat behavior detection event.

        This publishes to multiple topics:
        - behavior/{behavior} - Binary ON state
        - behavior/{behavior}/attributes - Event details
        - cat/{cat}/present - Cat presence
        - cat/{cat}/{behavior} - Per-cat behavior
        - cat/{cat}/last_behavior - Last behavior string
        - cat/{cat}/last_seen - Timestamp
        - latest - Latest detection details
        - event - Event for automation triggers

        Args:
            behavior: Detected behavior type
            cat: Identified cat
            confidence: Behavior detection confidence
            cat_confidence: Cat identification confidence
            camera: Source camera name
            event_id: Frigate event ID
            bbox: Bounding box dict
            snapshot_url: URL to snapshot image
        """
        timestamp = datetime.now(UTC)
        timestamp_str = timestamp.isoformat()

        log = logger.bind(
            behavior=behavior.value,
            cat=cat.value,
            confidence=confidence,
        )
        log.info("Publishing detection to Home Assistant")

        # Build attributes payload
        attributes: dict[str, Any] = {
            "behavior": behavior.value,
            "cat": cat.value,
            "confidence": round(confidence, 3),
            "cat_confidence": round(cat_confidence, 3),
            "camera": camera,
            "event_id": event_id,
            "timestamp": timestamp_str,
        }
        if bbox:
            attributes["bbox"] = bbox
        if snapshot_url:
            attributes["snapshot_url"] = snapshot_url

        # 1. Publish behavior binary sensor ON
        await self.mqtt.publish(f"behavior/{behavior.value}", "ON")
        await self.mqtt.publish(
            f"behavior/{behavior.value}/attributes",
            attributes,
        )

        # 2. Publish cat presence (if not unknown)
        if cat != CatName.UNKNOWN:
            await self.mqtt.publish(f"cat/{cat.value}/present", "ON", retain=True)
            await self.mqtt.publish(
                f"cat/{cat.value}/last_behavior",
                behavior.value,
                retain=True,
            )
            await self.mqtt.publish(
                f"cat/{cat.value}/last_seen",
                timestamp_str,
                retain=True,
            )

            # 3. Publish per-cat behavior
            await self.mqtt.publish(f"cat/{cat.value}/{behavior.value}", "ON")

        # 4. Publish latest detection
        latest = {
            "summary": f"{cat.value.title()} {behavior.value.replace('cat_', '')}",
            **attributes,
        }
        await self.mqtt.publish("latest", latest, retain=True)

        # 5. Publish event for automation triggers
        event = {
            "event_type": behavior.value,
            "cat": cat.value,
            "confidence": confidence,
            "timestamp": timestamp_str,
            "camera": camera,
        }
        await self.mqtt.publish("event", event)

        # Track for auto-off
        self._active_states[f"behavior/{behavior.value}"] = timestamp
        if cat != CatName.UNKNOWN:
            self._active_states[f"cat/{cat.value}/{behavior.value}"] = timestamp

    async def publish_inference_result(
        self,
        result: InferenceResult,
        camera: str = "",
        event_id: str = "",
        snapshot_url: str | None = None,
    ) -> None:
        """Publish a full inference result.

        Args:
            result: Complete inference result
            camera: Source camera
            event_id: Frigate event ID
            snapshot_url: Snapshot URL
        """
        # Publish each detection paired with identification
        for i, detection in enumerate(result.detections):
            # Get corresponding identification if available
            identification = (
                result.identifications[i]
                if i < len(result.identifications)
                else None
            )

            cat = identification.cat if identification else CatName.UNKNOWN
            cat_confidence = identification.confidence if identification else 0.0

            await self.publish_detection(
                behavior=detection.behavior,
                cat=cat,
                confidence=detection.confidence,
                cat_confidence=cat_confidence,
                camera=camera,
                event_id=event_id,
                bbox={
                    "x_min": detection.bbox.x_min,
                    "y_min": detection.bbox.y_min,
                    "x_max": detection.bbox.x_max,
                    "y_max": detection.bbox.y_max,
                },
                snapshot_url=snapshot_url,
            )

    async def publish_off_states(self) -> None:
        """Publish OFF states for expired detections.

        Call this periodically to auto-clear binary sensors.
        """
        now = datetime.now(UTC)
        expired = []

        for topic, timestamp in self._active_states.items():
            age = (now - timestamp).total_seconds()
            if age > self.auto_off_delay:
                await self.mqtt.publish(topic, "OFF")
                expired.append(topic)

        for topic in expired:
            del self._active_states[topic]

        if expired:
            logger.debug(f"Published OFF for {len(expired)} expired states")

    async def clear_all_states(self) -> None:
        """Clear all active states (publish OFF to all)."""
        logger.info("Clearing all active states")

        # Clear behavior sensors
        for behavior in self.behaviors:
            await self.mqtt.publish(f"behavior/{behavior.value}", "OFF")

        # Clear cat states
        for cat in self.cats:
            await self.mqtt.publish(f"cat/{cat.value}/present", "OFF", retain=True)
            for behavior in self.behaviors:
                if behavior != BehaviorType.PRESENT:
                    await self.mqtt.publish(f"cat/{cat.value}/{behavior.value}", "OFF")

        self._active_states.clear()


class HAAlertManager:
    """Manages alert-worthy events with cooldowns and priorities.

    Integrates with Home Assistant's notification system.
    """

    # Priority levels for different behaviors
    BEHAVIOR_PRIORITIES = {
        BehaviorType.VOMITING: 1,  # Highest priority
        BehaviorType.YOWLING: 2,
        BehaviorType.WAITING: 3,
        BehaviorType.LITTERBOX: 4,
        BehaviorType.EATING: 5,
        BehaviorType.DRINKING: 5,
        BehaviorType.PRESENT: 6,  # Lowest priority
    }

    def __init__(
        self,
        mqtt_publisher: MQTTPublisher,
        topic_prefix: str = "cat_watcher",
        alert_cooldowns: dict[BehaviorType, float] | None = None,
    ) -> None:
        """Initialize alert manager.

        Args:
            mqtt_publisher: MQTT publisher
            topic_prefix: Topic prefix
            alert_cooldowns: Cooldown periods per behavior (seconds)
        """
        self.mqtt = mqtt_publisher
        self.topic_prefix = topic_prefix

        # Default cooldowns
        self.cooldowns = alert_cooldowns or {
            BehaviorType.VOMITING: 30,  # Alert quickly for health issues
            BehaviorType.YOWLING: 60,
            BehaviorType.WAITING: 120,
            BehaviorType.LITTERBOX: 300,
            BehaviorType.EATING: 300,
            BehaviorType.DRINKING: 300,
            BehaviorType.PRESENT: 600,
        }

        self._last_alerts: dict[tuple[BehaviorType, CatName], datetime] = {}

    def should_alert(
        self,
        behavior: BehaviorType,
        cat: CatName,
    ) -> bool:
        """Check if an alert should be sent.

        Args:
            behavior: Behavior type
            cat: Cat name

        Returns:
            True if alert should be sent
        """
        key = (behavior, cat)
        now = datetime.now(UTC)

        last_alert = self._last_alerts.get(key)
        cooldown = self.cooldowns.get(behavior, 300)

        if last_alert is None or (now - last_alert).total_seconds() > cooldown:
            self._last_alerts[key] = now
            return True

        return False

    async def publish_alert(
        self,
        behavior: BehaviorType,
        cat: CatName,
        confidence: float,
        camera: str = "",
        message: str | None = None,
    ) -> None:
        """Publish an alert notification.

        Args:
            behavior: Behavior type
            cat: Cat name
            confidence: Detection confidence
            camera: Source camera
            message: Custom message (auto-generated if None)
        """
        if not self.should_alert(behavior, cat):
            return

        priority = self.BEHAVIOR_PRIORITIES.get(behavior, 5)

        if message is None:
            behavior_name = behavior.value.replace("cat_", "").replace("_", " ").title()
            message = f"{cat.value.title()} is {behavior_name}"
            if camera:
                message += f" ({camera})"

        alert = {
            "title": "Cat Watcher Alert",
            "message": message,
            "priority": priority,
            "behavior": behavior.value,
            "cat": cat.value,
            "confidence": round(confidence, 3),
            "camera": camera,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Publish to alert topic for HA automations to pick up
        await self.mqtt.publish("alert", alert)

        # Also publish to behavior-specific alert topic
        await self.mqtt.publish(f"alert/{behavior.value}", alert)

        logger.info(
            "Published alert",
            behavior=behavior.value,
            cat=cat.value,
            priority=priority,
        )
