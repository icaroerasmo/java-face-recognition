# RTSP Object Detection Configuration

This document describes the application properties supported by **RTSP Object Detection**.

For architecture, operations, troubleshooting, and the release/deployment process, see [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md).

All examples below use **placeholder values only**. Do not commit real RTSP URLs, Telegram tokens, chat IDs, or other environment-specific values to the repository.

## Configuration root

All settings live under:

```yaml
object-detection:
  streams:
    processing-fps: 5
    frame-queue-capacity: 2
    tracking-max-frames: 30
    cameras:
      - name: "<camera-name>"
        url: "rtsp://<host>:<port>/<stream>"
        protocol: TCP
  training:
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
    gif-max-frames: 30
```

## `object-detection.streams`

Controls camera ingestion, frame freshness, and queueing behavior.

| Property | Type | Default | Description |
|---|---|---:|---|
| `object-detection.streams.processing-fps` | integer | `5` | Maximum number of frames per second that the pipeline will process per camera. Lower values reduce latency and memory usage. Higher values can improve temporal smoothness but increase load. |
| `object-detection.streams.frame-queue-capacity` | integer | `2` | Number of decoded frames buffered per camera. Small values prefer fresh frames over stale backlog. |
| `object-detection.streams.tracking-max-frames` | integer | `30` | Maximum number of captured frames kept in a single tracking session before the track is finalized and evaluated. Lower values cap memory use sooner; higher values keep longer motion context. |
| `object-detection.streams.identity-min-frames` | integer | `1` | Minimum number of frames a known identity must be observed before it is confirmed as that person. `1` = recognize on the first frame (fastest); raise this if identity flicker becomes a problem. |
| `object-detection.streams.cameras` | list | `[]` | List of camera definitions to ingest. |

### `object-detection.streams.cameras[]`

Each entry defines one RTSP source.

| Property | Type | Default | Description |
|---|---|---:|---|
| `name` | string | none | Friendly name used in logs and notifications. |
| `url` | string | none | RTSP URL for the camera stream. Use placeholders in committed files and inject real values at deploy time. |
| `protocol` | enum | `TCP` | Transport mode for RTSP ingestion. Allowed values: `TCP`, `UDP`. |

## `object-detection.training`

Controls where the application reads the training image folders used to build or refresh the stored face gallery.

| Property | Type | Default | Description |
|---|---|---:|---|
| `object-detection.training.root-folder` | string | `train` | Folder containing the training images used to build or refresh the dataset. |

The application persists the generated face-recognition gallery in SQLite; there is no longer a separate configurable dataset file path.

## `object-detection.acceleration`

Controls the OpenCV DNN backend/target selection.

| Property | Type | Default | Description |
|---|---|---:|---|
| `object-detection.acceleration.backend` | enum | `OPENCV` | Backend used by the DNN models. Allowed values: `AUTO`, `OPENCV`. |
| `object-detection.acceleration.target` | enum | `CPU` | General fallback target for DNN execution. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `object-detection.acceleration.face-detection-target` | enum | `OPENCL` | Preferred target for face detection. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `object-detection.acceleration.person-detection-target` | enum | `CPU` | Preferred target for person detection. Allowed values: `AUTO`, `CPU`, `OPENCL`, `OPENCL_FP16`. |
| `object-detection.acceleration.enable-opencl` | boolean | `true` | Enables OpenCL-oriented target selection when supported by the runtime. |
| `object-detection.acceleration.fallback-to-cpu` | boolean | `true` | Falls back to CPU if the requested acceleration backend or target cannot be initialized. |

## `object-detection.telegram`

Controls Telegram notification delivery and animation rendering.

| Property | Type | Default | Description |
|---|---|---:|---|
| `object-detection.telegram.bot-token` | string | none | Telegram bot token. Prefer environment-variable injection instead of committing the real token. |
| `object-detection.telegram.chat-id` | string | none | Telegram chat ID that receives notifications. Prefer environment-variable injection instead of committing the real value. |
| `object-detection.telegram.gif-fps` | integer | `10` | Frame rate used when building Telegram animation clips. Increase it for smoother clips; decrease it for smaller and lighter files. |
| `object-detection.telegram.gif-max-frames` | integer | `30` | Maximum number of frame images retained per tracked person for Telegram clips. Lower values reduce RAM and clip-encoding cost. |

## Security guidance

1. Keep committed examples sanitized.
2. Inject secrets such as Telegram credentials through environment variables or private deployment files.
3. Avoid committing production RTSP endpoints, usernames, passwords, tokens, or chat identifiers.

## Notes

1. Spring Boot relaxed binding accepts kebab-case keys in YAML and also supports equivalent environment-variable mappings.
2. If you raise `processing-fps`, consider reviewing `frame-queue-capacity`, `tracking-max-frames`, `gif-fps`, and `gif-max-frames` too, because all four affect CPU usage, memory usage, and perceived latency.
3. On the `feature/object-detection-refactor` branch, face recognition uses SCRFD face detection plus SFace (`FaceRecognizerSF`) embeddings stored in SQLite, not LBPH model files.
