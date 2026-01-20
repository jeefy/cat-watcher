# Architecture Overview

This document provides a detailed overview of the Cat Watcher system architecture.

## System Overview

Cat Watcher is a modular ML-powered system for detecting cat behaviors and identifying individual cats using video feeds from Frigate NVR. The system integrates with Home Assistant for notifications and automations.

## High-Level Architecture

```
                                    ┌─────────────────────────────────────────┐
                                    │            External Systems             │
                                    ├─────────────────────────────────────────┤
┌──────────────┐                    │  ┌──────────┐      ┌──────────────────┐ │
│   Cameras    │───RTSP────────────▶│  │ Frigate  │      │  Home Assistant  │ │
│  (IP/USB)    │                    │  │   NVR    │      │                  │ │
└──────────────┘                    │  └────┬─────┘      └────────▲─────────┘ │
                                    │       │                     │           │
                                    │       │ Events              │ Discovery │
                                    │       ▼                     │ & Events  │
                                    │  ┌──────────────────────────┴─────────┐ │
                                    │  │          MQTT Broker               │ │
                                    │  │   (Mosquitto / EMQX / HiveMQ)      │ │
                                    │  └──────────────────┬─────────────────┘ │
                                    └─────────────────────│─────────────────────┘
                                                          │
                          ┌───────────────────────────────┼───────────────────────────────┐
                          │                               │                               │
                          │                   Cat Watcher │System                         │
                          │                               │                               │
                          │  ┌────────────────────────────▼───────────────────────────┐  │
                          │  │                    MQTT Handler                         │  │
                          │  │  • Subscribe to frigate/events                         │  │
                          │  │  • Parse event payloads                                │  │
                          │  │  • Trigger inference pipeline                          │  │
                          │  └────────────────────────────┬───────────────────────────┘  │
                          │                               │                               │
                          │              ┌────────────────┴────────────────┐              │
                          │              │                                 │              │
                          │              ▼                                 ▼              │
                          │  ┌──────────────────────┐         ┌──────────────────────┐  │
                          │  │   Frigate Client     │         │  Inference Pipeline  │  │
                          │  │                      │         │                      │  │
                          │  │  • Fetch snapshots   │────────▶│  ┌────────────────┐  │  │
                          │  │  • Get event details │         │  │ Behavior Det.  │  │  │
                          │  │  • Download clips    │         │  │   (YOLOv8)     │  │  │
                          │  └──────────────────────┘         │  └───────┬────────┘  │  │
                          │                                   │          │           │  │
                          │                                   │          ▼           │  │
                          │                                   │  ┌────────────────┐  │  │
                          │                                   │  │   Cat ID       │  │  │
                          │                                   │  │ (EfficientNet) │  │  │
                          │                                   │  └───────┬────────┘  │  │
                          │                                   │          │           │  │
                          │                                   │          ▼           │  │
                          │                                   │  ┌────────────────┐  │  │
                          │                                   │  │ Alert Filter   │  │  │
                          │                                   │  │  (cooldowns)   │  │  │
                          │                                   │  └───────┬────────┘  │  │
                          │                                   └──────────│───────────┘  │
                          │                                              │               │
                          │                                              ▼               │
                          │                               ┌──────────────────────────┐  │
                          │                               │    HA Event Publisher    │  │
                          │                               │  • MQTT Discovery        │  │
                          │                               │  • State updates         │  │
                          │                               │  • Alert notifications   │  │
                          │                               └──────────────────────────┘  │
                          │                                                              │
                          └──────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frigate Client (`src/cat_watcher/frigate/client.py`)

The Frigate Client provides async HTTP communication with the Frigate NVR API.

**Responsibilities:**
- Fetch camera snapshots (JPEG/PNG)
- Retrieve event metadata
- Download video clips
- Query event history

**Key Classes:**
- `FrigateClient`: Main async HTTP client
- `FrigateEvent`: Event data model
- `FrigateCamera`: Camera configuration

**Example Usage:**
```python
async with FrigateClient(base_url="http://frigate:5000") as client:
    # Get latest snapshot
    snapshot = await client.get_snapshot("apollo-dish")
    
    # Get event details
    event = await client.get_event("1234567890.123456-abc123")
    
    # Get event thumbnail
    thumbnail = await client.get_event_thumbnail(event.id)
```

### 2. MQTT Handler (`src/cat_watcher/frigate/mqtt.py`)

Handles bidirectional MQTT communication.

**Subscribed Topics:**
| Topic | Description |
|-------|-------------|
| `frigate/events` | New detection events |
| `frigate/+/person` | Person detections (filtered) |
| `frigate/+/cat` | Cat detections |

**Published Topics:**
| Topic | Description |
|-------|-------------|
| `cat_watcher/status` | Service online/offline |
| `cat_watcher/event` | Detection results |
| `cat_watcher/behavior/{type}` | Per-behavior states |
| `cat_watcher/cat/{name}/*` | Per-cat states |

**Key Classes:**
- `FrigateMQTTHandler`: Event subscription and parsing
- `MQTTPublisher`: Publishing results
- `EventTracker`: Cooldown management

### 3. Inference Pipeline (`src/cat_watcher/inference/pipeline.py`)

Orchestrates the ML inference workflow.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Inference Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Image                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Preprocessing   │  • Resize to model input size                  │
│  │                 │  • Normalize pixel values                      │
│  │                 │  • Convert color space (BGR→RGB)               │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐     ┌─────────────────────────────────────┐   │
│  │ Behavior Detect │────▶│ Detections:                         │   │
│  │   (YOLOv8)      │     │  - bbox: [x_min, y_min, x_max, y_max]│   │
│  │                 │     │  - behavior: BehaviorType            │   │
│  └─────────────────┘     │  - confidence: float                 │   │
│           │              └─────────────────────────────────────┘   │
│           │                                                         │
│           ▼ (crop regions)                                          │
│  ┌─────────────────┐     ┌─────────────────────────────────────┐   │
│  │   Cat ID        │────▶│ Identifications:                    │   │
│  │ (EfficientNet)  │     │  - cat: CatName                     │   │
│  │                 │     │  - confidence: float                 │   │
│  └─────────────────┘     └─────────────────────────────────────┘   │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  Alert Filter   │  • Apply cooldowns                             │
│  │                 │  • Check confidence thresholds                 │
│  │                 │  • Deduplicate events                          │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  InferenceResult                                                    │
│    - detections: List[Detection]                                    │
│    - identifications: List[Identification]                          │
│    - processing_time: float                                         │
│    - alerts: List[Alert]                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Behavior Detector (`src/cat_watcher/inference/detector.py`)

YOLOv8-based object detection for cat behaviors.

**Model Architecture:**
- Base: YOLOv8n/s/m (configurable)
- Input: 640x640 RGB
- Output: Bounding boxes + class labels

**Detected Classes:**
| Class ID | Behavior | Description |
|----------|----------|-------------|
| 0 | `cat_eating` | Cat at food bowl |
| 1 | `cat_drinking` | Cat at water bowl |
| 2 | `cat_vomiting` | Vomiting posture |
| 3 | `cat_waiting` | Sitting at door |
| 4 | `cat_litterbox` | In/near litterbox |
| 5 | `cat_yowling` | Vocalizing (mouth open) |
| 6 | `cat_present` | General presence |

**ONNX Optimization:**
```python
# Export to ONNX for faster inference
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", opset=12, simplify=True)
```

### 5. Cat Identifier (`src/cat_watcher/inference/identifier.py`)

EfficientNet-based classification for individual cat identification.

**Model Architecture:**
- Base: EfficientNet-B0 (or B1-B7)
- Input: 224x224 RGB (cropped from detection bbox)
- Output: Softmax over cat classes

**Classes:**
| Class ID | Cat Name |
|----------|----------|
| 0 | starbuck |
| 1 | apollo |
| 2 | mia |
| 3 | unknown |

### 6. Home Assistant Integration (`src/cat_watcher/homeassistant/`)

MQTT-based integration with Home Assistant.

**Discovery Protocol:**
```json
// Published to: homeassistant/binary_sensor/cat_watcher_starbuck_present/config
{
  "name": "Starbuck Present",
  "unique_id": "cat_watcher_starbuck_present",
  "state_topic": "cat_watcher/cat/starbuck/present",
  "device_class": "presence",
  "device": {
    "identifiers": ["cat_watcher"],
    "name": "Cat Watcher",
    "manufacturer": "Cat Watcher",
    "model": "ML Cat Behavior Monitor"
  }
}
```

**Entity Hierarchy:**
```
Cat Watcher (Device)
├── binary_sensor.cat_watcher_status
├── sensor.cat_watcher_events_processed
├── Behaviors
│   ├── binary_sensor.cat_watcher_cat_eating
│   ├── binary_sensor.cat_watcher_cat_drinking
│   ├── binary_sensor.cat_watcher_cat_vomiting
│   └── ...
├── Starbuck
│   ├── binary_sensor.cat_watcher_starbuck_present
│   ├── sensor.cat_watcher_starbuck_behavior
│   ├── sensor.cat_watcher_starbuck_last_seen
│   └── Per-behavior sensors...
├── Apollo
│   └── ...
└── Mia
    └── ...
```

## Data Flow

### Real-Time Inference Flow

```
1. Camera captures frame
         │
         ▼
2. Frigate detects "cat" object
         │
         ▼
3. Frigate publishes to frigate/events
         │
         ▼
4. Cat Watcher MQTT handler receives event
         │
         ▼
5. Frigate Client fetches snapshot
         │
         ▼
6. Inference Pipeline processes image
   a. Behavior detection (YOLOv8)
   b. Cat identification (EfficientNet)
   c. Alert filtering (cooldowns)
         │
         ▼
7. HA Publisher sends results
   a. Update entity states
   b. Publish events
   c. Send alerts
         │
         ▼
8. Home Assistant receives updates
   a. Update UI
   b. Trigger automations
   c. Send notifications
```

### Training Data Flow

```
1. Frigate events trigger data collection
         │
         ▼
2. Collector saves frames + metadata
         │
         ▼
3. Labeling UI presents unlabeled frames
         │
         ▼
4. User annotates behaviors + cat IDs
         │
         ▼
5. Dataset preparer exports to:
   a. YOLO format (behavior detection)
   b. Classification format (cat ID)
         │
         ▼
6. Training scripts train models
         │
         ▼
7. Models exported to ONNX
         │
         ▼
8. Inference service loads new models
```

## Configuration Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Configuration Sources                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Priority (highest to lowest):                                    │
│                                                                   │
│  1. Environment Variables                                         │
│     CAT_WATCHER__FRIGATE__URL="http://..."                       │
│     CAT_WATCHER__MQTT__BROKER="192.168.1.82"                     │
│                        │                                          │
│                        ▼                                          │
│  2. config.yaml file                                              │
│     frigate:                                                      │
│       url: "http://..."                                           │
│                        │                                          │
│                        ▼                                          │
│  3. Default values (in Pydantic models)                          │
│     class Settings(BaseSettings):                                 │
│         frigate: FrigateSettings = FrigateSettings()             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Pydantic Settings                         │ │
│  │                                                              │ │
│  │  Settings                                                    │ │
│  │  ├── frigate: FrigateSettings                               │ │
│  │  │   ├── url: str                                           │ │
│  │  │   ├── camera: str                                        │ │
│  │  │   └── timeout: float                                     │ │
│  │  ├── mqtt: MQTTSettings                                     │ │
│  │  │   ├── broker: str                                        │ │
│  │  │   ├── port: int                                          │ │
│  │  │   ├── username: str | None                               │ │
│  │  │   └── password: str | None                               │ │
│  │  ├── inference: InferenceSettings                           │ │
│  │  │   ├── behavior_model: str                                │ │
│  │  │   ├── catid_model: str                                   │ │
│  │  │   ├── confidence_threshold: float                        │ │
│  │  │   └── use_onnx: bool                                     │ │
│  │  └── cooldowns: EventCooldowns                              │ │
│  │      ├── cat_eating: int                                    │ │
│  │      ├── cat_drinking: int                                  │ │
│  │      └── ...                                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Error Handling

### Retry Strategies

| Component | Strategy | Max Retries | Backoff |
|-----------|----------|-------------|---------|
| Frigate Client | Exponential | 3 | 1s, 2s, 4s |
| MQTT Connection | Infinite | ∞ | 5s fixed |
| Inference | None | 0 | - |
| HA Publisher | Linear | 3 | 1s |

### Circuit Breaker Pattern

```python
# Frigate client uses circuit breaker for resilience
class FrigateClient:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
        )
    
    async def get_snapshot(self, camera: str) -> bytes:
        async with self.circuit_breaker:
            return await self._fetch_snapshot(camera)
```

## Security Considerations

1. **MQTT Authentication**: Support for username/password
2. **TLS Support**: Optional TLS for MQTT connections
3. **API Authentication**: Optional API key for REST endpoints
4. **Network Isolation**: K8s NetworkPolicies for pod isolation
5. **Secrets Management**: K8s Secrets for credentials

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Inference latency (GPU) | ~50ms | YOLOv8n + EfficientNet-B0 |
| Inference latency (CPU) | ~500ms | ONNX optimized |
| MQTT message latency | <10ms | Local broker |
| Memory usage | ~512MB | Inference service |
| GPU memory | ~1GB | With ONNX-GPU |

## Scaling Considerations

### Horizontal Scaling

- Inference service can be replicated
- Use MQTT shared subscriptions for load balancing
- Stateless design allows easy scaling

### Vertical Scaling

- Larger YOLO models (s/m/l/x) for better accuracy
- EfficientNet-B1 to B7 for better cat ID
- GPU memory is the limiting factor

### Multi-Camera Support

- Each camera can have separate inference instance
- Or single instance processing multiple cameras
- Configurable per-camera settings
