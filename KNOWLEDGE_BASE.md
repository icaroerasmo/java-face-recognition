# RTSP Face Recognition Knowledge Base

This document is the operational knowledge base for **rtsp-face-recognition**. It describes the application architecture, runtime configuration, troubleshooting tips, and the release and deployment workflow used by this repository and the Frigate stack.

## What the service does

`rtsp-face-recognition` is a Spring Boot background application that:

1. Connects to one or more RTSP cameras.
2. Samples frames through a bounded producer/consumer pipeline.
3. Detects people first with SSD MobileNet.
4. Runs face detection and recognition only on frames that contain people.
5. Tracks detections across multiple frames.
6. Sends Telegram notifications with still images and optional clips.

It is optimized for long-running camera ingestion rather than request/response web traffic.

## Main runtime pipeline

### 1. RTSP ingestion

`RtspFrameExtractorService` connects to each configured camera with `FFmpegFrameGrabber`.

- `processing-fps` caps how many frames per second the downstream pipeline will process.
- `frame-queue-capacity` keeps the queue short so stale frames are dropped instead of building backlog.
- `max-consecutive-null-frames` controls when an unstable stream is treated as broken and reconnected.

For heavier cameras, the extractor now uses:

- a 50 MB RTSP buffer
- `grabImage()` instead of generic `grab()`
- channel-aware `Mat` reconstruction

These changes help reduce stalls on streams such as `garagem1`.

### 2. Person detection

`PersonDetector` runs SSD MobileNet as the first gate.

- only class `person` is kept
- `face-recognition.detection.person-confidence-threshold` controls the minimum confidence
- the current default is `0.8`

Raising this threshold reduces false positives like cars or static objects being treated as people.

### 3. Face detection and recognition

`FaceRecognitionService` uses:

- SCRFD for face detection with 5 landmarks
- SFace (`FaceRecognizerSF`) for embedding extraction and comparison

The trained gallery is stored in SQLite instead of only living on disk. The service reloads or retrains based on the training-folder hashes stored in the database.

### 4. Tracking and notification

`PeopleTrackingService` aggregates detections across frames before notifying Telegram.

This avoids one-frame noise and decides:

- whether the tracked subject is known or unknown
- which frame should be used in the notification
- whether enough evidence exists to send a message

## Important configuration

The main external config file is mounted at:

`/app/config/config.yaml`

Key settings:

| Key | Purpose |
| --- | --- |
| `face-recognition.streams.processing-fps` | Limits per-camera processing rate |
| `face-recognition.streams.frame-queue-capacity` | Caps buffered decoded frames |
| `face-recognition.streams.max-consecutive-null-frames` | Reconnect threshold for unstable streams |
| `face-recognition.detection.person-confidence-threshold` | Minimum confidence for person detections |
| `face-recognition.acceleration.*` | OpenCV backend and target selection |
| `face-recognition.telegram.*` | Telegram bot/chat and clip settings |

## Training data

Training data is organized as:

`train/<person-name>/<images...>`

Rules:

- one folder per identity
- folder name becomes the recognition label
- each accepted training image must contain exactly one detectable face

The runtime stores:

- serialized face embeddings in `trained_dataset`
- folder hashes in `training_metadata`

## Deployment layout

The application source lives in:

`/home/icaroerasmo/RTSP Face Recognition`

The Frigate deployment that runs the published image lives in:

`/run/media/games/frigate`

In the Frigate stack:

- `go2rtc` relays/transcodes RTSP streams
- `rtsp-face-recognition` consumes those relayed streams
- `telegram-live` builds the Telegram live layout from the same relayed streams

## Local development and validation

To build the application locally:

```bash
mvn test
docker build -t ghcr.io/icaroerasmo/java-face-recognition:local .
```

To run the local image in the Frigate stack:

```bash
cd /run/media/games/frigate
docker compose -f docker-compose.yaml -f docker-compose.local.yaml up -d --force-recreate rtsp-face-recognition
```

`docker-compose.local.yaml` overrides the image to:

`ghcr.io/icaroerasmo/java-face-recognition:local`

## Release workflow

The repository uses two GitHub Actions workflows.

### 1. Release branch version bump

Workflow: `.github/workflows/changes-version.yml`

Trigger:

- creation of a branch matching `release/*`

Expected branch name format:

- `release/<major>.<minor>.<patch>`
- example: `release/0.1.6`

Behavior:

1. The workflow extracts the version from the branch name.
2. It updates `pom.xml` to that version.
3. It commits the version bump back to the release branch.

### 2. Publish image after merge

Workflow: `.github/workflows/publish-image.yml`

Trigger:

- a pull request closed and merged into `main`

Behavior:

1. The workflow reads the version from `pom.xml`.
2. It creates a Git tag named `release/<version>`.
3. It builds and pushes the Docker image to GHCR.
4. It publishes two image tags:
   - `ghcr.io/icaroerasmo/java-face-recognition:<version>`
   - `ghcr.io/icaroerasmo/java-face-recognition:latest`

## Standard release procedure

1. Ensure the desired code is present locally.
2. Create a release branch **from `develop`** using the required format, for example:
   - `release/0.1.6`
3. Push that branch to GitHub.
4. Wait for the version-bump workflow to update `pom.xml` on that branch.
5. Open a pull request from the release branch to `main`.
6. Merge the pull request.
7. Wait for the publish workflow to:
   - create the release tag
   - publish the versioned image
   - update `latest`
8. Update the Frigate deployment to use `ghcr.io/icaroerasmo/java-face-recognition:latest`.
9. Recreate the containers in `/run/media/games/frigate`.

## Production deployment in Frigate

The production compose file is:

`/run/media/games/frigate/docker-compose.yaml`

After a successful release:

1. make sure `rtsp-face-recognition` points to `ghcr.io/icaroerasmo/java-face-recognition:latest`
2. recreate the stack containers

Example:

```bash
cd /run/media/games/frigate
docker compose up -d --force-recreate
```

## Troubleshooting

### `garagem1` freezes or repeats the same frame

Symptoms:

- identical snapshots across multiple seconds
- repeated-frame watchdog resets in `telegram-live`
- stale detections in face recognition

Likely cause:

- unstable upstream camera/go2rtc relay rather than application crash

First response:

1. restart `go2rtc`
2. restart `rtsp-face-recognition`
3. restart `telegram-live`

### Person detector sees cars as people

Increase:

- `face-recognition.detection.person-confidence-threshold`

Tradeoff:

- higher threshold reduces false positives
- too high a threshold can miss real people farther from the camera

### Stream connects but face recognition never triggers

Check:

- `go2rtc` stream health
- frame freshness
- person-detection logs
- training data availability
- per-camera FPS and resolution load

### Heavy camera behavior

For heavier cameras like `garagem1`, prefer:

- lower effective FPS
- short queues
- stable RTSP transport
- enough probe/analyze time for consumers
- `go2rtc` transcoding to a predictable H.264 output when the source is less stable
