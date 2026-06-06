# Copilot Instructions

## Build, test, and lint

| Task | Command | Notes |
| --- | --- | --- |
| Build jar | `mvn clean package` | Produces the runnable jar in `target/`. The Dockerfile uses this packaging flow. |
| Run test suite | `mvn test` | There are currently no `src/test` sources, so this completes with "No tests to run." |
| Run one test class | `mvn -Dtest=ClassName test` | Use Surefire's `-Dtest` selector when tests are added. |
| Run one test method | `mvn -Dtest=ClassName#methodName test` | Use this for focused test runs. |
| Build container image | `docker build -t rtsp-face-recognition .` | Mirrors the image packaging used by `.github/workflows/publish-image.yml`. |

There is no dedicated lint command configured in `pom.xml` or in GitHub Actions. Verification in this repository currently goes through Maven compile/test/package steps.

## High-level architecture

- This is a **Spring Boot background/CLI application**, not a web app. `JavaRtspFaceRecognition` starts Spring and immediately hands execution to `RtspRecognitionRunner`, which keeps one long-lived processing loop per configured camera.
- Runtime configuration is split between `application.yaml` and an external `config/config.yaml`. Spring imports `optional:config/`, and the Docker image starts the app with `-Dspring.config.additional-location=/app/config/config.yaml`.
- Startup requires valid Telegram settings. `BeansAndConfig` creates the `TelegramBot` bean eagerly and exits the process if `face-recognition.telegram.bot-token` or `chat-id` is missing.
- Face recognition training is persisted in SQLite, not only on disk. `FaceRecognitionService` trains an LBPH recognizer from the training folders, stores the serialized model in the `trained_dataset` table, and stores per-person folder hashes in `training_metadata` so scheduled retraining can detect changes.
- Frame processing is intentionally staged:
  1. `RtspFrameExtractorService` reads RTSP frames with `FFmpegFrameGrabber` and a bounded producer/consumer queue so `processing-fps` and `frame-queue-capacity` control freshness vs backlog.
  2. `PersonDetectionService` runs SSD MobileNet first and drops frames with no person detections.
  3. `FaceRecognitionService` then uses SCRFD face detection plus LBPH recognition on the remaining frames.
  4. `PeopleTrackingService` aggregates detections across frames, waits for motion/track thresholds, chooses the best frame, and only then calls `TelegramPublisherService`.
- Notification de-duplication is stateful. `DetectionHistoryService` enforces short per-camera cooldowns, while `PeopleTrackingService` keeps in-memory tracks and can emit both a still image and an MP4 animation built by `GifCreationService`.
- Native OpenCV/JavaCV resources are central to the implementation. `MatUtil` is the shared helper for releasing `Mat` instances and for drawing annotations.
- `ScheduledTasks` is responsible for two maintenance loops: clearing old detection/tracking state and retraining the recognizer when the training folders change.

## Key conventions

- Keep all committed configuration sanitized. `README.MD` and `config.yaml` use placeholders only; do not commit real RTSP URLs, Telegram tokens, chat IDs, or camera credentials.
- Repository-specific configuration lives under the `face-recognition.*` property tree. New config should follow the existing nested properties structure (`streams`, `training`, `acceleration`, `telegram`) instead of adding ad hoc top-level keys.
- Treat OpenCV inference objects as **not concurrency-safe**. Existing code serializes DNN execution through `DnnInferenceCoordinator`, synchronizes shared detector methods, and updates the active `FaceRecognizer` through `FaceRecognizerHolder` instead of mutating it in place.
- Person tracking is the source of truth for notifications. Do not send Telegram alerts directly from raw per-frame detections unless you are intentionally bypassing the duplicate-suppression and movement heuristics already implemented in `PeopleTrackingService` and `DetectionHistoryService`.
- Training data is organized as one folder per identity under the configured training root. The folder name becomes the recognition label.
- When adding OpenCV/JavaCV logic, follow the existing pattern of explicit cleanup for `Mat`, `BytePointer`, `MatVector`, and similar native objects; this codebase does not rely on GC to reclaim native memory safely.
- Release automation is tied to GitHub Actions: creating a `release/*` branch updates the Maven version in `pom.xml`, and merging a pull request into `main` publishes the container image to GHCR.
- After a release is merged and the publish workflow completes, update deployed environments by pulling `ghcr.io/icaroerasmo/java-face-recognition:latest` and recreating the `rtsp-face-recognition` container from `the private production compose file`.
- For local validation before release, build and run `ghcr.io/icaroerasmo/java-face-recognition:local` via the `a private local deployment override` override instead of editing the main compose file.
