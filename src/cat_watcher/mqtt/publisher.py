"""Generic MQTT publisher for event publishing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

logger = structlog.get_logger()

# Try to import aiomqtt, but make it optional
try:
    import aiomqtt
    AIOMQTT_AVAILABLE = True
except ImportError:
    AIOMQTT_AVAILABLE = False


class MQTTPublisher:
    """Async MQTT publisher for event publishing.
    
    Handles connecting to an MQTT broker and publishing messages.
    
    Example:
        ```python
        async with MQTTPublisher("localhost", 1883) as mqtt:
            await mqtt.publish("topic", {"key": "value"})
        ```
    """

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        topic_prefix: str = "cat_watcher",
        client_id: str | None = None,
    ):
        """Initialize MQTT publisher.
        
        Args:
            broker: MQTT broker hostname
            port: MQTT broker port
            username: Optional username for authentication
            password: Optional password for authentication
            topic_prefix: Prefix for all topics
            client_id: Optional client ID
        """
        if not AIOMQTT_AVAILABLE:
            raise ImportError(
                "aiomqtt is required for MQTT publishing. "
                "Install with: pip install aiomqtt"
            )
        
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        self.client_id = client_id
        self._client: aiomqtt.Client | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to MQTT broker."""
        if self._connected:
            return
        
        self._client = aiomqtt.Client(
            hostname=self.broker,
            port=self.port,
            username=self.username,
            password=self.password,
            identifier=self.client_id,
        )
        await self._client.__aenter__()
        self._connected = True
        logger.info(
            "Connected to MQTT broker",
            broker=self.broker,
            port=self.port,
        )

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._client and self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            logger.info("Disconnected from MQTT broker")

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any] | str,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        """Publish a message to a topic.
        
        Args:
            topic: Topic to publish to (will be prefixed with topic_prefix)
            payload: Message payload (dict will be JSON encoded)
            retain: Whether to retain the message
            qos: Quality of service level
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to MQTT broker")
        
        full_topic = f"{self.topic_prefix}/{topic}"
        
        if isinstance(payload, dict):
            message = json.dumps(payload)
        else:
            message = payload
        
        await self._client.publish(
            full_topic,
            payload=message,
            retain=retain,
            qos=qos,
        )
        
        logger.debug(
            "Published MQTT message",
            topic=full_topic,
            retain=retain,
        )

    async def publish_raw(
        self,
        topic: str,
        payload: str | bytes,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        """Publish a raw message without topic prefix.
        
        Args:
            topic: Full topic path
            payload: Message payload
            retain: Whether to retain the message
            qos: Quality of service level
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to MQTT broker")
        
        await self._client.publish(
            topic,
            payload=payload,
            retain=retain,
            qos=qos,
        )
        
        logger.debug(
            "Published raw MQTT message",
            topic=topic,
            retain=retain,
        )

    async def __aenter__(self) -> "MQTTPublisher":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
