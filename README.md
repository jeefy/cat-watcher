# Cat Watcher 🐱

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ML-powered cat behavior detection system that integrates with [Frigate NVR](https://frigate.video/) and publishes events via MQTT for [Home Assistant](https://www.home-assistant.io/).

## Features

- **Real-time Detection**: YOLOv8-based cat detection from Frigate camera streams
- **Behavior Classification**: Eating, drinking, sleeping, vomiting, waiting, litterbox, yowling
- **Web UI**: Dashboard for monitoring, labeling, training, and configuration
- **Database-driven Config**: All settings configurable via web UI
- **MQTT Publishing**: Events published for Home Assistant integration
- **GPU Accelerated**: CUDA support for fast inference

## Quick Start

### Docker (Recommended)

```bash
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/data:/data \
  -v $(pwd)/models:/models \
  --gpus all \
  ghcr.io/jeefy/cat-watcher:latest
```

Open http://localhost:8080 and configure via the **Settings** page.

### From Source

```bash
git clone https://github.com/jeefy/cat-watcher.git
cd cat-watcher
pip install -e .
cat-watcher web --port 8080
```

## Configuration

All configuration is managed through the **Settings** page in the web UI:

| Category | Settings |
|----------|----------|
| **Frigate** | URL, cameras to monitor, RTSP credentials |
| **Detection** | Confidence thresholds, frame rate, event duration |
| **MQTT** | Broker URL, credentials, topic prefix |

### Secrets via Environment

RTSP and MQTT credentials can be set via environment variables:

```bash
CAT_WATCHER__FRIGATE__RTSP_USERNAME=camera_user
CAT_WATCHER__FRIGATE__RTSP_PASSWORD=camera_pass
CAT_WATCHER__MQTT__USERNAME=mqtt_user
CAT_WATCHER__MQTT__PASSWORD=mqtt_pass
```

## Web UI

Access the web UI at http://localhost:8080 after starting the service.

| Page | Description |
|------|-------------|
| **Dashboard** | System status and detection statistics |
| **Live Detection** | Start/stop detection, view camera feeds |
| **Labeling** | Review and label captured frames |
| **Training** | Prepare datasets and train models |
| **Settings** | Configure cats, thresholds, connections |

## Kubernetes Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml    # Create from secrets.yaml.template first
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
```

GPU support requires the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin).

## Home Assistant Integration

Cat Watcher publishes events via MQTT. Configure your MQTT broker in Settings, then add automations like:

```yaml
automation:
  - alias: "Alert: Cat Vomiting"
    trigger:
      - platform: mqtt
        topic: "cat_watcher/events/cat_vomiting"
    action:
      - service: notify.mobile_app
        data:
          title: "🤢 Cat Alert"
          message: "{{ trigger.payload_json.cat_name }} may be vomiting!"
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format src tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.
