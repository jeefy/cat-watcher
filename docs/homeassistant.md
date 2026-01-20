# Home Assistant Integration Guide

This guide covers integrating Cat Watcher with Home Assistant.

## Overview

Cat Watcher integrates with Home Assistant via MQTT, using the auto-discovery protocol to automatically create entities.

## Prerequisites

- Home Assistant with MQTT integration configured
- MQTT broker (Mosquitto recommended)
- Cat Watcher inference service running

## Quick Setup

### 1. Configure MQTT in Home Assistant

Ensure MQTT is configured in Home Assistant:

```yaml
# configuration.yaml
mqtt:
  broker: 192.168.1.82
  port: 1883
  discovery: true
  discovery_prefix: homeassistant
```

### 2. Publish Discovery Messages

Run the discovery command to create entities:

```bash
cat-watcher homeassistant discover
```

Entities will appear in Home Assistant within seconds.

### 3. Verify Entities

Check Developer Tools → States for entities starting with `cat_watcher_`.

## Created Entities

### Service Entities

| Entity ID | Type | Description |
|-----------|------|-------------|
| `binary_sensor.cat_watcher_status` | Binary Sensor | Service online/offline |
| `sensor.cat_watcher_events_processed` | Sensor | Total events processed |
| `sensor.cat_watcher_latest_detection` | Sensor | Latest detection summary |

### Behavior Entities

| Entity ID | Type | Description |
|-----------|------|-------------|
| `binary_sensor.cat_watcher_cat_eating` | Binary Sensor | Any cat eating |
| `binary_sensor.cat_watcher_cat_drinking` | Binary Sensor | Any cat drinking |
| `binary_sensor.cat_watcher_cat_vomiting` | Binary Sensor | ⚠️ Vomiting detected |
| `binary_sensor.cat_watcher_cat_waiting` | Binary Sensor | Cat at door |
| `binary_sensor.cat_watcher_cat_litterbox` | Binary Sensor | Litterbox use |
| `binary_sensor.cat_watcher_cat_yowling` | Binary Sensor | Yowling detected |

### Per-Cat Entities

For each cat (Starbuck, Apollo, Mia):

| Entity ID | Type | Description |
|-----------|------|-------------|
| `binary_sensor.cat_watcher_{cat}_present` | Binary Sensor | Cat detected |
| `sensor.cat_watcher_{cat}_behavior` | Sensor | Last behavior |
| `sensor.cat_watcher_{cat}_last_seen` | Sensor | Last seen timestamp |
| `binary_sensor.cat_watcher_{cat}_cat_eating` | Binary Sensor | Cat eating |
| `binary_sensor.cat_watcher_{cat}_cat_drinking` | Binary Sensor | Cat drinking |
| ... | ... | (one per behavior) |

## Automation Blueprints

### Installing Blueprints

Copy blueprint files to Home Assistant:

```bash
# From Cat Watcher repository
cp homeassistant/blueprints/*.yaml \
   /config/blueprints/automation/cat_watcher/

# Restart Home Assistant or reload automations
```

Or import via URL in Home Assistant UI.

### Available Blueprints

#### Cat Behavior Alert

Sends notifications when specific behaviors are detected.

**Configuration:**
- Behavior to monitor
- Cat filter (optional)
- Notification device
- Cooldown period

**Example Automation:**
```yaml
alias: "Alert: Starbuck Eating"
use_blueprint:
  path: cat_watcher/cat_behavior_alert.yaml
  input:
    behavior: cat_eating
    cat: starbuck
    notify_device: device_tracker.phone
    cooldown_minutes: 30
```

#### Cat Health Monitor

Monitors for health issues (vomiting, excessive yowling).

**Configuration:**
- Enable/disable vomiting alerts
- Yowling threshold and timeframe
- Critical notification toggle

**Example Automation:**
```yaml
alias: "Health: Monitor Vomiting"
use_blueprint:
  path: cat_watcher/cat_health_monitor.yaml
  input:
    vomiting_enabled: true
    yowling_enabled: true
    yowling_threshold: 3
    yowling_timeframe: 30
    critical_alerts: true
    notify_device: device_tracker.phone
```

#### Cat Feeding Monitor

Tracks feeding schedules and alerts if a cat hasn't eaten.

**Configuration:**
- Cat to monitor
- Max hours without eating
- Check interval
- Quiet hours

**Example Automation:**
```yaml
alias: "Feeding: Monitor Starbuck"
use_blueprint:
  path: cat_watcher/cat_feeding_monitor.yaml
  input:
    cat: starbuck
    max_hours_without_eating: 12
    check_interval_hours: 2
    quiet_hours_start: "22:00:00"
    quiet_hours_end: "07:00:00"
    notify_device: device_tracker.phone
```

#### Cat Door Waiting

Alerts when a cat is waiting at a door.

**Configuration:**
- Wait duration before alert
- Voice announcement toggle
- Auto-action script (optional)

**Example Automation:**
```yaml
alias: "Door: Cat Waiting Alert"
use_blueprint:
  path: cat_watcher/cat_door_waiting.yaml
  input:
    wait_duration_seconds: 30
    notify_device: device_tracker.phone
    announce_enabled: true
    announce_device: media_player.kitchen_speaker
```

## Custom Automations

### Basic Notification

```yaml
automation:
  - alias: "Cat Watcher: Vomiting Alert"
    trigger:
      - platform: mqtt
        topic: "cat_watcher/alert/cat_vomiting"
    action:
      - service: notify.mobile_app_phone
        data:
          title: "⚠️ Cat Health Alert"
          message: "{{ trigger.payload_json.cat | title }} vomited"
          data:
            push:
              sound:
                name: default
                critical: 1
```

### Light Control

```yaml
automation:
  - alias: "Cat Watcher: Feeding Light"
    trigger:
      - platform: state
        entity_id: binary_sensor.cat_watcher_cat_eating
        to: "on"
    condition:
      - condition: sun
        before: sunrise
        after: sunset
    action:
      - service: light.turn_on
        target:
          entity_id: light.kitchen
        data:
          brightness_pct: 30
```

### TTS Announcement

```yaml
automation:
  - alias: "Cat Watcher: Door Announcement"
    trigger:
      - platform: state
        entity_id: binary_sensor.cat_watcher_cat_waiting
        to: "on"
        for:
          seconds: 30
    action:
      - service: tts.speak
        target:
          entity_id: media_player.living_room
        data:
          message: >
            {% set cat = states('sensor.cat_watcher_latest_detection') %}
            {{ cat }} is waiting at the door
```

## Lovelace Dashboard

### Import Dashboard

Import the pre-built dashboard from `homeassistant/lovelace/dashboard.yaml`.

### Manual Cards

#### Status Card

```yaml
type: entities
title: Cat Watcher
entities:
  - entity: binary_sensor.cat_watcher_status
  - entity: sensor.cat_watcher_events_processed
  - type: divider
  - entity: sensor.cat_watcher_latest_detection
```

#### Cat Status Cards

```yaml
type: horizontal-stack
cards:
  - type: custom:mushroom-template-card
    entity: binary_sensor.cat_watcher_starbuck_present
    primary: Starbuck
    secondary: >
      {% if is_state(entity, 'on') %}
        {{ states('sensor.cat_watcher_starbuck_behavior') }}
      {% else %}
        Not detected
      {% endif %}
    icon: mdi:cat
    icon_color: >
      {% if is_state(entity, 'on') %}green{% else %}grey{% endif %}
  # Repeat for other cats...
```

#### Activity Graph

```yaml
type: history-graph
title: Cat Activity (24h)
hours_to_show: 24
entities:
  - entity: binary_sensor.cat_watcher_starbuck_present
  - entity: binary_sensor.cat_watcher_apollo_present
  - entity: binary_sensor.cat_watcher_mia_present
```

## MQTT Topics Reference

### Subscribed by Home Assistant

| Topic | Payload | Description |
|-------|---------|-------------|
| `cat_watcher/status` | `online`/`offline` | Service status |
| `cat_watcher/behavior/{type}` | `ON`/`OFF` | Behavior state |
| `cat_watcher/cat/{name}/present` | `ON`/`OFF` | Cat presence |
| `cat_watcher/cat/{name}/last_behavior` | String | Last behavior |
| `cat_watcher/cat/{name}/last_seen` | ISO timestamp | Last seen time |
| `cat_watcher/latest` | JSON | Latest detection |
| `cat_watcher/event` | JSON | Event for triggers |
| `cat_watcher/alert` | JSON | Priority alerts |
| `cat_watcher/alert/{behavior}` | JSON | Per-behavior alerts |

### Event Payload Example

```json
{
  "event_type": "cat_eating",
  "cat": "starbuck",
  "confidence": 0.92,
  "cat_confidence": 0.88,
  "camera": "kitchen",
  "timestamp": "2026-01-16T10:30:00Z"
}
```

### Alert Payload Example

```json
{
  "title": "Cat Watcher Alert",
  "message": "Starbuck is Eating (kitchen)",
  "priority": 5,
  "behavior": "cat_eating",
  "cat": "starbuck",
  "confidence": 0.92,
  "camera": "kitchen",
  "timestamp": "2026-01-16T10:30:00Z"
}
```

## Helper Entities

Create these helpers for advanced automations:

### Counters (for yowling tracking)

```yaml
# configuration.yaml
counter:
  cat_watcher_yowling_starbuck:
    name: "Starbuck Yowling Counter"
    initial: 0
    step: 1
    restore: false
  cat_watcher_yowling_apollo:
    name: "Apollo Yowling Counter"
    initial: 0
  cat_watcher_yowling_mia:
    name: "Mia Yowling Counter"
    initial: 0
```

### Input Helpers

```yaml
# configuration.yaml
input_boolean:
  cat_watcher_alerts_enabled:
    name: "Cat Watcher Alerts Enabled"
    initial: true
    icon: mdi:bell

input_datetime:
  cat_watcher_starbuck_last_manual_feeding:
    name: "Starbuck Last Manual Feeding"
    has_date: true
    has_time: true
```

## Troubleshooting

### Entities Not Appearing

1. Check MQTT discovery is enabled
2. Verify Cat Watcher is publishing:
   ```bash
   mosquitto_sub -h broker -t "homeassistant/#" -v
   ```
3. Re-run discovery:
   ```bash
   cat-watcher ha discover
   ```

### Entities Show "Unavailable"

1. Check Cat Watcher service is running
2. Verify MQTT connection
3. Check status topic:
   ```bash
   mosquitto_sub -h broker -t "cat_watcher/status"
   ```

### Automations Not Triggering

1. Check entity states in Developer Tools
2. Verify MQTT messages are being received
3. Check automation trace in HA

### Remove Entities

To remove all Cat Watcher entities:

```bash
cat-watcher homeassistant discover --remove
```

Or delete discovery topics manually:
```bash
mosquitto_pub -h broker -t "homeassistant/binary_sensor/cat_watcher_status/config" -n -r
```
