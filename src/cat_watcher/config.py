"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrigateSettings(BaseSettings):
    """Frigate NVR connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="CAT_WATCHER__FRIGATE__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="http://192.168.50.36:5000",
        description="Frigate API base URL",
    )
    cameras: list[str] = Field(
        default=["apollo-dish"],
        description="List of cameras to monitor for cat activity",
    )
    # RTSP credentials for cameras (Frigate masks these in the API)
    rtsp_username: str | None = Field(
        default=None,
        description="RTSP username for cameras",
    )
    rtsp_password: str | None = Field(
        default=None,
        description="RTSP password for cameras",
    )
    request_timeout: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds",
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds",
    )


class MQTTSettings(BaseSettings):
    """MQTT broker connection settings."""

    broker: str = Field(
        default="192.168.1.82",
        description="MQTT broker hostname or IP",
    )
    port: int = Field(
        default=1883,
        description="MQTT broker port",
    )
    username: str | None = Field(
        default=None,
        description="MQTT username (optional)",
    )
    password: str | None = Field(
        default=None,
        description="MQTT password (optional)",
    )
    client_id: str = Field(
        default="cat-watcher",
        description="MQTT client identifier",
    )
    frigate_topic_prefix: str = Field(
        default="frigate",
        description="Frigate MQTT topic prefix",
    )
    publish_topic_prefix: str = Field(
        default="cat_watcher",
        description="Cat Watcher publish topic prefix",
    )
    keepalive: int = Field(
        default=60,
        description="MQTT keepalive interval in seconds",
    )


class InferenceSettings(BaseSettings):
    """Model inference settings."""

    behavior_model_path: Path = Field(
        default=Path("/models/behavior_latest.onnx"),
        description="Path to behavior detection model",
    )
    cat_id_model_path: Path = Field(
        default=Path("/models/cat_id_latest.onnx"),
        description="Path to cat identification model",
    )
    reference_embeddings_path: Path = Field(
        default=Path("/models/cat_references.npz"),
        description="Path to cat reference embeddings",
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Inference device (cuda, cpu, or auto)",
    )
    behavior_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for behavior detection",
    )
    cat_id_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for cat identification",
    )
    model_check_interval: int = Field(
        default=60,
        description="Interval to check for model updates (seconds)",
    )


class CatConfig(BaseSettings):
    """Individual cat configuration."""

    name: str
    reference_images: str = Field(
        description="Glob pattern for reference images",
    )


class EventCooldowns(BaseSettings):
    """Cooldown periods for event deduplication."""

    cat_eating: int = Field(default=300, description="Cooldown for eating events (seconds)")
    cat_drinking: int = Field(default=300, description="Cooldown for drinking events (seconds)")
    cat_vomiting: int = Field(default=60, description="Cooldown for vomiting events (seconds)")
    cat_waiting: int = Field(default=120, description="Cooldown for waiting events (seconds)")
    cat_litterbox: int = Field(default=300, description="Cooldown for litterbox events (seconds)")
    cat_yowling: int = Field(default=60, description="Cooldown for yowling events (seconds)")
    cat_present: int = Field(default=600, description="Cooldown for presence events (seconds)")


class ActiveLearningSettings(BaseSettings):
    """Active learning configuration."""

    enabled: bool = Field(default=True, description="Enable active learning queue")
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Queue samples below this confidence",
    )
    max_queue_size: int = Field(
        default=500,
        description="Maximum samples in review queue",
    )


class LabelingSettings(BaseSettings):
    """Labeling service settings."""

    port: int = Field(default=8080, description="Labeling service port")
    db_path: Path = Field(
        default=Path("/data/labels.db"),
        description="Path to labels database",
    )


class DetectionCameraSettings(BaseSettings):
    """Configuration for a single camera's detection."""

    name: str = Field(description="Camera name (must match Frigate camera name)")
    enabled: bool = Field(default=True, description="Whether detection is enabled for this camera")
    confidence_override: float | None = Field(
        default=None,
        description="Override confidence threshold for this camera",
    )


class DetectionSettings(BaseSettings):
    """Detection pipeline settings."""

    # Stage 1: Cat Detection (YOLOv8)
    cat_model: str = Field(
        default="yolov8n.pt",
        description="Path to YOLOv8 model or model name (auto-downloads)",
    )
    cat_confidence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for cat detection",
    )

    # Stage 2: Behavior Model (optional)
    behavior_model: str | None = Field(
        default=None,
        description="Path to behavior classification model (optional)",
    )
    behavior_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for behavior classification",
    )
    behavior_min_detection_conf: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence to run behavior inference",
    )

    # Processing
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Inference device (auto, cuda, cpu)",
    )
    frame_rate: float = Field(
        default=5.0,
        gt=0.0,
        le=30.0,
        description="Target frames per second for processing",
    )

    # Tracker settings
    max_disappeared_seconds: float = Field(
        default=3.0,
        description="Seconds before track is considered lost",
    )
    max_distance: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum normalized distance for centroid matching",
    )

    # Event settings
    min_event_duration: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum event duration in seconds",
    )
    max_event_duration: float = Field(
        default=300.0,
        description="Maximum event duration in seconds (force end)",
    )
    event_cooldown: float = Field(
        default=5.0,
        ge=0.0,
        description="Cooldown between events for same track",
    )
    disappeared_timeout: float = Field(
        default=2.0,
        ge=0.0,
        description="Seconds to wait before ending event after track disappears",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("data/detection/events"),
        description="Directory for saving event frames",
    )
    save_frames: bool = Field(
        default=True,
        description="Save event frames to disk",
    )
    db_path: Path = Field(
        default=Path("data/training/samples.db"),
        description="Path to training database",
    )

    # Cameras (empty = all enabled cameras from Frigate)
    cameras: list[str] = Field(
        default=[],
        description="List of cameras to enable (empty = all enabled from Frigate)",
    )

    @field_validator("output_dir", "db_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Ensure value is a Path object."""
        return Path(v) if isinstance(v, str) else v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="CAT_WATCHER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Paths
    models_dir: Path = Field(
        default=Path("/models"),
        description="Directory containing model files",
    )
    data_dir: Path = Field(
        default=Path("/data"),
        description="Directory for data storage",
    )

    # Sub-configurations
    frigate: FrigateSettings = Field(default_factory=FrigateSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    cooldowns: EventCooldowns = Field(default_factory=EventCooldowns)
    active_learning: ActiveLearningSettings = Field(default_factory=ActiveLearningSettings)
    labeling: LabelingSettings = Field(default_factory=LabelingSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)

    # Cat definitions
    cats: list[CatConfig] = Field(
        default=[
            CatConfig(name="starbuck", reference_images="/data/references/starbuck/*.jpg"),
            CatConfig(name="apollo", reference_images="/data/references/apollo/*.jpg"),
            CatConfig(name="mia", reference_images="/data/references/mia/*.jpg"),
        ],
        description="List of cats to identify",
    )

    @field_validator("models_dir", "data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Ensure value is a Path object."""
        return Path(v) if isinstance(v, str) else v


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from environment and optional config file.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        Configured Settings instance.
    """
    import yaml

    settings_dict: dict[str, Any] = {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            settings_dict = yaml.safe_load(f) or {}

    return Settings(**settings_dict)


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        Global Settings instance, created on first call.
    """
    global _settings
    if _settings is None:
        config_path = Path("config.yaml")
        _settings = load_settings(config_path if config_path.exists() else None)
    return _settings
