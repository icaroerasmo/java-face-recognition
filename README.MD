# RTSP Face Recognition Configuration

This document describes the application properties supported by **RTSP Face Recognition**.

All examples below use **placeholder values only**. Do not commit real RTSP URLs, Telegram tokens, chat IDs, or other environment-specific values to the repository.

## Configuration root

All settings live under:

```yaml
face-recognition:
  streams:
    processing-fps: 5
    frame-queue-capacity: 2
    cameras:
      - name: "<camera-name>"
        url: "rtsp://<host>:<port>/<stream>"
        protocol: TCP
  training:
    dataset-path: "<path-to-trained-dataset.xml>"
    root-folder: "<training-folder>"
  acceleration:
    backend: OPENCV
    target: CPU
    face-detection-target: OPENCL
    person-detection-target: CPU
    enable-opencl: true
    fallback-to-cpu: true
  telegram:
    bot-token: "${TELEGRAM_BOT_TOKEN:}"
    chat-id: "${TELEGRAM_CHAT_ID:}"
    gif-fps: 10
```

## `face-recognition.streams`

Controls camera ingestion, frame freshness, and queueing behavior.

| Property | Type | Default | Description |
|---|---|---:|---|
| `face-recognition.streams.processing-fps` | integer | `5` | Maximum number of frames per second that the pipeline will process per camera. Lower values reduce latency and memory usage. Higher values can improve temporal smoothness but increase load. |
| `face-recognition.streams.frame-queue-capacity` | integer | `2` | Number of decoded frames buffered per camera. Small values prefer fresh frames over stale backlog. |
| `face-recognition.streams.cameras` | list | `[]` | List of camera definitions to ingest. |

### `face-recognition.streams.cameras[]`

Each entry defines one RTSP source.

| Property | Type | Default | Description |
|---|---|---:|---|
| `name` | string | none | Friendly name used in logs and notifications. |
| `url` | string | none | RTSP URL for the camera stream. Use placeholders in committed files and inject real values at deploy time. |
| `protocol` | enum | `TCP` | Transport mode for RTSP ingestion. Allowed values: `TCP`, `UDP`. |

## `face-recognition.training`

Controls where the application loads or stores training data.

| Property | Type | Default | Description |
|---|---|---:|---|
| `face-recognition.training.dataset-path` | string | `trained_dataset.xml` | Relative or absolute path to the trained dataset XML file used by the recognition pipeline. |
| `face-recognition.training.root-folder` | string | `train` | Folder containing the training images used to build or refresh the dataset. |

## `face-recognition.acceleration`

Controls the OpenCV DNN backend/target selection.

| Property | Type | Default | Description |
|---|---|---:|---|
| `face-recognition.acceleration.backend` | enum | `OPENCV` | Backend used by the DNN models. Allowed values: `AUTO`, `OPENCV`. |
| `face-recognition.acceleration.target` | enum | `CPU` | General fallback target for DNN execution. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `face-recognition.acceleration.face-detection-target` | enum | `OPENCL` | Preferred target for face detection. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `face-recognition.acceleration.person-detection-target` | enum | `CPU` | Preferred target for person detection. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `face-recognition.acceleration.enable-opencl` | boolean | `true` | Enables OpenCL-oriented target selection when supported by the runtime. |
| `face-recognition.acceleration.fallback-to-cpu` | boolean | `true` | Falls back to CPU if the requested acceleration backend or target cannot be initialized. |

## `face-recognition.telegram`

Controls Telegram notification delivery and animation rendering.

| Property | Type | Default | Description |
|---|---|---:|---|
| `face-recognition.telegram.bot-token` | string | none | Telegram bot token. Prefer environment-variable injection instead of committing the real token. |
| `face-recognition.telegram.chat-id` | string | none | Telegram chat ID that receives notifications. Prefer environment-variable injection instead of committing the real value. |
| `face-recognition.telegram.gif-fps` | integer | `10` | Frame rate used when building Telegram animation clips. Increase it for smoother clips; decrease it for smaller and lighter files. |

## Security guidance

1. Keep committed examples sanitized.
2. Inject secrets such as Telegram credentials through environment variables or private deployment files.
3. Avoid committing production RTSP endpoints, usernames, passwords, tokens, or chat identifiers.

## Notes

1. Spring Boot relaxed binding accepts kebab-case keys in YAML and also supports equivalent environment-variable mappings.
2. If you raise `processing-fps`, consider reviewing `frame-queue-capacity` and `gif-fps` too, because all three affect CPU usage, memory usage, and perceived latency.
