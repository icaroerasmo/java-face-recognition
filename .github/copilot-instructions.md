# Copilot Instructions

## Build, test, and lint

| Task | Command | Notes |
| --- | --- | --- |
| Build jar | `mvn clean package` | Produces the runnable jar in `target/`. The Dockerfile uses this packaging flow. |
| Run test suite | `mvn test` | There are currently no `src/test` sources, so this completes with "No tests to run." |
| Run one test class | `mvn -Dtest=ClassName test` | Use Surefire's `-Dtest` selector when tests are added. |
| Run one test method | `mvn -Dtest=ClassName#methodName test` | Use this for focused test runs. |
| Build container image | `docker build -t rtsp-object-detection .` | Mirrors the image packaging used by `.github/workflows/publish-image.yml`. |

There is no dedicated lint command configured in `pom.xml` or in GitHub Actions. Verification in this repository currently goes through Maven compile/test/package steps.

## High-level architecture

- This is a **Spring Boot background/CLI application**, not a web app. `JavaObjectDetection` starts Spring and immediately hands execution to `RtspRecognitionRunner`, which keeps one long-lived processing loop per configured camera.
- Runtime configuration is split between `application.yaml` and an external `config/config.yaml`. Spring imports `optional:config/`, and the Docker image starts the app with `-Dspring.config.additional-location=/app/config/config.yaml`.
- Face recognition training is persisted in SQLite, not only on disk. On this branch, `FaceRecognitionService` builds a `FaceRecognizerSF` embedding gallery from the training folders, stores the serialized gallery in the `trained_dataset` table, and stores per-person folder hashes in `training_metadata` so scheduled retraining can detect changes.
- Frame processing is intentionally staged:
  1. `RtspFrameExtractorService` reads RTSP frames with `FFmpegFrameGrabber` and a bounded producer/consumer queue so `processing-fps` and `frame-queue-capacity` control freshness vs backlog.
  2. `PersonDetectionService` runs SSD MobileNet first and drops frames with no person detections.
  3. `FaceRecognitionService` then uses SCRFD face detection with 5 landmarks plus SFace (`FaceRecognizerSF`) embedding matching on the remaining frames.
  4. `PeopleTrackingService` aggregates detections across frames, waits for motion/track thresholds, chooses the best frame, and only then publishes via `TelegramPublisherService` (RabbitMQ).
- Notification de-duplication is stateful. `DetectionHistoryService` enforces short per-camera cooldowns, while `PeopleTrackingService` keeps in-memory tracks and can emit both a still image and an MP4 animation built by `GifCreationService`.
- Native OpenCV/JavaCV resources are central to the implementation. `MatUtil` is the shared helper for releasing `Mat` instances and for drawing annotations.
- `ScheduledTasks` is responsible for two maintenance loops: clearing old detection/tracking state and retraining the recognizer when the training folders change.

## Key conventions

- Keep all committed configuration sanitized. `README.MD` and `config.yaml` use placeholders only; do not commit real RTSP URLs or camera credentials.
- Repository-specific configuration lives under the `object-detection.*` property tree. New config should follow the existing nested properties structure (`streams`, `training`, `acceleration`, `clips`) instead of adding ad hoc top-level keys. In `training`, only `root-folder` is still live on this branch; `dataset-path` was removed.
- Treat OpenCV inference objects as **not concurrency-safe**. Existing code serializes DNN execution through `DnnInferenceCoordinator`, synchronizes shared detector methods, and updates the active `FaceRecognizer` through `FaceRecognizerHolder` instead of mutating it in place.
- Person tracking is the source of truth for notifications. Do not publish alerts directly from raw per-frame detections unless you are intentionally bypassing the duplicate-suppression and movement heuristics already implemented in `PeopleTrackingService` and `DetectionHistoryService`.
- Training data is organized as one folder per identity under the configured training root. The folder name becomes the recognition label, and each accepted training image must contain exactly one detectable face.
- The SFace runtime depends on the bundled `src/main/resources/opencv/face_recognition_sface_2021dec.onnx` model and on SCRFD detections carrying 5 landmarks so `alignCrop()` can normalize faces before embedding extraction.
- When adding OpenCV/JavaCV logic, follow the existing pattern of explicit cleanup for `Mat`, `BytePointer`, `MatVector`, and similar native objects; this codebase does not rely on GC to reclaim native memory safely.
- Release automation is tied to GitHub Actions: creating a `release/*` branch updates the Maven version in `pom.xml`, and merging a pull request into `main` publishes the container image to GHCR.
- For local container validation, build
  `ghcr.io/icaroerasmo/java-object-detection:local`. Keep environment-specific
  deployment instructions in the private deployment repository.
