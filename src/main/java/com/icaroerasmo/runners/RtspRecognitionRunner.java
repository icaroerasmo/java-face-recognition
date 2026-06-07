package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.properties.FaceRecognitionProperties;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.detectors.person.services.FaceRecognitionService;
import com.icaroerasmo.detectors.person.services.FaceRecognizerHolderService;
import com.icaroerasmo.detectors.person.services.PeopleTrackingService;
import com.icaroerasmo.service.TelegramPublisherService;
import com.icaroerasmo.detectors.person.services.RtspFrameExtractorService;
import com.icaroerasmo.detectors.person.PersonDetector;
import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.jetbrains.annotations.NotNull;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import static com.icaroerasmo.utils.Constants.DESIRED_SCORE;


@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private final FaceRecognitionService faceRecognitionService;
    private final FaceRecognizerHolderService faceRecognizerHolderService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final MatUtil matUtil;
    private final TelegramPublisherService telegramPublisherService;
    private final PeopleTrackingService peopleTrackingService;
    private final PersonDetector personDetector;
    private final FaceRecognitionProperties faceRecognitionProperties;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;

    public void start(String... args) throws Exception {

        try {
            File trainingRootDir = getTrainedFile();
            FaceRecognizer faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            // Initialize the holder with the trained recognizer
            faceRecognizerHolderService.updateRecognizer(faceRecognizer);

            List<CameraProperties> cameraProperties = streamsProperties.getCameras();
            if (cameraProperties == null || cameraProperties.isEmpty()) {
                throw new IllegalStateException("No cameras configured under face-recognition.streams.cameras");
            }

            // Create thread pool with one thread per camera
            ExecutorService executorService = Executors.newFixedThreadPool(cameraProperties.size());
            List<Future<?>> futures = new ArrayList<>();

            for (CameraProperties camera : cameraProperties) {
                if (camera == null || camera.getUrl() == null || camera.getUrl().isBlank()) {
                    continue;
                }

                // Submit camera processing task to executor
                Future<?> future = executorService.submit(() -> {
                    processCameraStream(camera);
                });
                futures.add(future);
            }

            // Wait for all camera streams to complete (they run indefinitely, so this won't return)
            for (Future<?> future : futures) {
                try {
                    future.get(); // This will block if the task is still running
                } catch (Exception e) {
                    log.error("Camera stream processing failed", e);
                }
            }

            executorService.shutdown();

        } catch (Exception e) {
            log.error("Error in RtspRecognitionRunner", e);
            throw e;
        }
    }

    private FaceRecognition getFaceRecognition(Mat img) {
        // Get the current recognizer from the holder (thread-safe)
        FaceRecognizer currentRecognizer = faceRecognizerHolderService.get();

        if (currentRecognizer == null) {
            log.warn("FaceRecognizer not initialized yet, skipping frame");
            return null;
        }

        // STEP 2: Try to recognize faces in the frame
        return faceRecognitionService.test(currentRecognizer, img);
    }

    /**
     * Process a single camera stream with automatic reconnection
     */
    private void processCameraStream(CameraProperties cameraProperties) {
        String cameraName = cameraProperties.getName() != null ? cameraProperties.getName() : "unknown";
        String rtspUrl = cameraProperties.getUrl();

        log.info("Starting recognition for camera '{}' with {} transport: {}", cameraName, cameraProperties.getProtocol(), rtspUrl);

        // Infinite reconnection loop with hibernate mechanism
        int reconnectAttempt = 0;
        int consecutiveFailures = 0;
        boolean connectionNotified = false; // Track if we've sent connection success notification
        final int HIBERNATE_AFTER_FAILURES = 3;
        final long HIBERNATE_DURATION_MS = 5 * 60 * 1000; // 5 minutes

        while (true) {
            try {
                if (reconnectAttempt > 0) {
                    log.info("Reconnection attempt #{} for camera '{}'", reconnectAttempt, cameraName);

                    // Send Telegram notification about reconnection attempt
                    try {
                        telegramPublisherService.sendTextMessage(
                            String.format("🔄 Camera '%s': Attempting to reconnect (attempt #%d)...",
                                cameraName, reconnectAttempt)
                        );
                    } catch (Exception e) {
                        log.warn("Failed to send reconnection notification to Telegram: {}", e.getMessage());
                    }
                }

                // Send initial connection notification before starting extraction
                if (!connectionNotified) {
                    try {
                        telegramPublisherService.sendTextMessage(
                            String.format("✅ Camera '%s': Connected successfully and streaming", cameraName)
                        );
                        log.info("Camera '{}': Connection established successfully", cameraName);
                        connectionNotified = true;
                    } catch (Exception e) {
                        log.warn("Failed to send connection notification to Telegram: {}", e.getMessage());
                    }
                }

                rtspFrameExtractorService.extract(rtspUrl, cameraProperties.getProtocol(), (img) -> {

                    FaceRecognition faceRecognition = null;

                    try {

                        if (img == null) {
                            return;
                        }

                        // STEP 1: First detect if there are any people in the frame
                        List<Rect> detectedPeople = personDetector.detect(img);

                        if (detectedPeople.isEmpty()) {
                            // No people detected at all - skip this frame
                            return;
                        }

                        log.debug("Camera '{}': Detected {} person(s) in frame", cameraName, detectedPeople.size());

                        if(faceRecognitionProperties.getEnabled()) {
                            // STEP 2: Try to recognize faces in the frame
                            faceRecognition = getFaceRecognition(img);

                            if (faceRecognition == null) {
                                return;
                            }
                        }

                        List<FaceRecognition.DetectedFaces> faces =
                                faceRecognition != null ? faceRecognition.getFaces() : null;

                        // STEP 3: Check if faces were detected
                        if (faces == null || faces.isEmpty()) {

                            drawRectanglesOnPeople(img, cameraName, detectedPeople);

                            return; // Don't continue processing if no faces detected
                        }

                    // Filter out faces with score > DESIRED_SCORE
                    faces = faces.stream()
                            .filter(face -> face.getDistance() <= DESIRED_SCORE)
                            .collect(Collectors.toList());

                    if (faces.isEmpty()) {
                        log.debug("All detected faces in frame from camera '{}' have score > {}, skipping frame", cameraName, DESIRED_SCORE);
                        return;
                    }

                    // Collect detected names with their distance scores
                    Map<String, Double> detectedPeopleWithScores = faces.stream()
                            .collect(Collectors.toMap(
                                    FaceRecognition.DetectedFaces::getPersonName,
                                    FaceRecognition.DetectedFaces::getDistance,
                                    (existing, replacement) -> Math.min(existing, replacement) // Keep lowest distance if duplicate
                            ));

                    // Find the lowest distance score across all detections
                    double lowestDistance = faces.stream()
                            .mapToDouble(FaceRecognition.DetectedFaces::getDistance)
                            .min()
                            .orElse(DESIRED_SCORE);

                    // Send notification for ALL detections with score <= DESIRED_SCORE (recognized or Unknown)
                    String namesStr = String.join(", ", detectedPeopleWithScores.keySet());
                    log.info("Pessoas detectadas em '{}': {} (lowest distance: {})",
                            cameraName, namesStr, String.format("%.2f", lowestDistance));

                    // Publish to Telegram with detected people information
                    try {
                            // Convert Mat to byte array WITHOUT drawing rectangles yet
                            // Drawing will happen in PeopleTrackingService after person is tracked across frames
                            org.bytedeco.javacpp.BytePointer buf = new org.bytedeco.javacpp.BytePointer();
                            org.bytedeco.javacpp.BytePointer jpgExt = new org.bytedeco.javacpp.BytePointer(".jpg");
                            org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExt, img, buf);
                            byte[] imageBytes = new byte[(int) buf.limit()];
                            buf.get(imageBytes);
                            buf.deallocate();
                            jpgExt.deallocate();

                            // Create PersonDetection list with detected names and rectangles
                            // Each frame's recognition results are preserved for tracking
                            List<PeopleTrackingService.PersonDetection> allPeopleDetections = faces.stream()
                                .filter(face -> face.getFaceRect() != null)
                                .map(face -> new PeopleTrackingService.PersonDetection(face.getPersonName(), face.getFaceRect()))
                                .collect(Collectors.toList());

                            // Track EACH detected face individually
                            for (int faceIdx = 0; faceIdx < faces.size(); faceIdx++) {
                                FaceRecognition.DetectedFaces face = faces.get(faceIdx);

                                // Extract this face's region for similarity comparison
                                byte[] faceHash = null;
                                if (face.getFaceRect() != null) {
                                    org.bytedeco.javacpp.BytePointer faceBuf = null;
                                    org.bytedeco.javacpp.BytePointer jpgExtFace = null;
                                    Mat faceRegion = null;
                                    try {
                                        // Extract the face region from original image
                                        faceRegion = new Mat(img, face.getFaceRect());

                                        // Convert face to byte array for hashing
                                        faceBuf = new org.bytedeco.javacpp.BytePointer();
                                        jpgExtFace = new org.bytedeco.javacpp.BytePointer(".jpg");
                                        org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExtFace, faceRegion, faceBuf);
                                        faceHash = new byte[(int) faceBuf.limit()];
                                        faceBuf.get(faceHash);
                                    } catch (Exception e) {
                                        log.warn("Failed to extract face region {} for camera '{}': {}",
                                            faceIdx + 1, cameraName, e.getMessage());
                                        continue; // Skip this face
                                    } finally {
                                        // Clean up resources
                                        if (faceBuf != null) faceBuf.deallocate();
                                        if (jpgExtFace != null) jpgExtFace.deallocate();
                                        if (faceRegion != null) matUtil.releaseResources(faceRegion);
                                    }
                                }

                                log.info("🔍 FRAME PROCESSING: camera='{}', face #{}, person='{}', distance={}, rect={}, faceHash={}, imageBytes={}",
                                    cameraName,
                                    faceIdx + 1,
                                    face.getPersonName(),
                                    String.format("%.2f", face.getDistance()),
                                    (face.getFaceRect() != null),
                                    (faceHash != null ? faceHash.length : 0),
                                    (imageBytes != null ? imageBytes.length : 0));

                                if (face.getFaceRect() == null) {
                                    log.warn("⚠️ SKIPPED: faceRect is null for face #{} in camera '{}'", faceIdx + 1, cameraName);
                                    continue;
                                }

                                if (faceHash == null) {
                                    log.warn("⚠️ SKIPPED: faceHash is null for face #{} in camera '{}'", faceIdx + 1, cameraName);
                                    continue;
                                }

                                log.info("📊 CALLING TRACKING SERVICE: camera='{}', face #{}, person='{}'",
                                    cameraName, faceIdx + 1, face.getPersonName());

                                // Track this face with all people detections for drawing
                                PeopleTrackingService.TrackingResult trackingResult = peopleTrackingService.trackFace(
                                    cameraName,
                                    face.getPersonName(), // This person's name
                                    face.getFaceRect(),   // This person's rect
                                    faceHash,             // This person's hash
                                    face.getDistance(),   // This person's distance
                                    imageBytes,           // Full frame image bytes
                                    allPeopleDetections,  // ALL detected people with names and rectangles for drawing
                                    true
                                );

                                log.info("📋 TRACKING RESULT: camera='{}', face #{}, shouldSend={}, personName={}, score={}",
                                    cameraName,
                                    faceIdx + 1,
                                    trackingResult.isShouldSend(),
                                    trackingResult.getPersonName(),
                                    trackingResult.isShouldSend() ? String.format("%.2f", trackingResult.getBestDistance()) : "N/A");

                                // When this person's tracking is ready, notification will have ALL faces highlighted
                                if (trackingResult.isShouldSend()) {
                                    log.info("✅ VERDICT REACHED! Face #{} tracked successfully - notification sent with {} faces highlighted for '{}'",
                                        faceIdx + 1, allPeopleDetections.size(), trackingResult.getPersonName());
                                } else {
                                    log.debug("⏳ STILL TRACKING: camera='{}', face #{} not ready to send yet",
                                        cameraName, faceIdx + 1);
                                }
                            }

                            if (faces.isEmpty()) {
                                log.debug("📭 NO FACES: Skipping frame from camera '{}'", cameraName);
                            }

                        } catch (Exception e) {
                            log.error("Failed to publish detection to Telegram for camera '{}': {}", cameraName, e.getMessage(), e);
                        }

                    } catch (Exception e) {
                        log.error("Error processing frame from camera '{}': {}", cameraName, e.getMessage(), e);
                    } finally {
                    try {
                        Mat detectionImg = null;
                        if (faceRecognition != null) {
                            detectionImg = faceRecognition.getDetectionImg();
                        }
                        matUtil.releaseResources(img, detectionImg);
                    } catch (Exception releaseEx) {
                        log.warn("Error releasing resources for camera '{}'", cameraName, releaseEx);
                    }
                }
                });

                // If extract() returns normally, connection was lost
                log.warn("Stream ended for camera '{}' - Connection may have been lost", cameraName);

                // Reset connection notification flag so we can notify on successful reconnection
                connectionNotified = false;

                reconnectAttempt++;
                consecutiveFailures++;

            } catch (Exception e) {
                connectionNotified = false; // Reset flag on error
                reconnectAttempt++;
                consecutiveFailures++;
                log.error("Error with camera '{}' (attempt #{}): {}", cameraName, reconnectAttempt, e.getMessage());
            }

            // Check if we need to hibernate after 3 consecutive failures
            if (consecutiveFailures >= HIBERNATE_AFTER_FAILURES) {
                log.warn("Camera '{}': {} consecutive failures detected. Entering hibernate mode for {} minutes...",
                    cameraName, HIBERNATE_AFTER_FAILURES, HIBERNATE_DURATION_MS / 60000);

                // Send hibernate notification to Telegram
                try {
                    telegramPublisherService.sendTextMessage(
                        String.format("😴 Camera '%s': Entering hibernate mode for 5 minutes after %d failed connection attempts. Will retry automatically.",
                            cameraName, HIBERNATE_AFTER_FAILURES)
                    );
                } catch (Exception e) {
                    log.warn("Failed to send hibernate notification to Telegram: {}", e.getMessage());
                }

                // Hibernate for 5 minutes
                try {
                    Thread.sleep(HIBERNATE_DURATION_MS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    log.error("Camera '{}' hibernate interrupted", cameraName);
                    return;
                }

                // Send wake-up notification
                try {
                    telegramPublisherService.sendTextMessage(
                        String.format("⏰ Camera '%s': Hibernate complete. Resuming connection attempts...",
                            cameraName)
                    );
                } catch (Exception e) {
                    log.warn("Failed to send wake-up notification to Telegram: {}", e.getMessage());
                }

                log.info("Camera '{}': Hibernate complete. Resuming connection attempts...", cameraName);

                // Reset consecutive failures counter after hibernate
                consecutiveFailures = 0;

            } else {
                // Normal exponential backoff (2s, 4s, 8s, 16s, max 30s)
                long delayMs = Math.min(30000, 2000 * (long) Math.pow(2, Math.min(reconnectAttempt - 1, 4)));
                log.info("Waiting {}ms before reconnecting camera '{}'...", delayMs, cameraName);

                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    log.error("Camera '{}' reconnection interrupted", cameraName);
                    return;
                }
            }
        }
    }

    private void drawRectanglesOnPeople(Mat img, String cameraName, List<Rect> detectedPeople) {

        if(faceRecognitionProperties.getEnabled()) {
            // People detected but NO FACES - track ALL people individually
            log.info("Camera '{}': {} people detected but no faces recognized - tracking all people",
                    cameraName, detectedPeople.size());
        } else {
            log.info("Camera '{}': {} people detected (face recognition disabled) - tracking all people",
                    cameraName, detectedPeople.size());
        }

        try {
            // Convert full frame to byte array once (reused for all people)
            byte[] fullFrameBytes = null;
            org.bytedeco.javacpp.BytePointer frameBuf = null;
            org.bytedeco.javacpp.BytePointer frameJpgExt = null;
            try {
                frameBuf = new org.bytedeco.javacpp.BytePointer();
                frameJpgExt = new org.bytedeco.javacpp.BytePointer(".jpg");
                org.bytedeco.opencv.global.opencv_imgcodecs.imencode(frameJpgExt, img, frameBuf);
                fullFrameBytes = new byte[(int) frameBuf.limit()];
                frameBuf.get(fullFrameBytes);
            } catch (Exception e) {
                log.warn("Failed to convert frame to bytes: {}", e.getMessage());
            } finally {
                if (frameBuf != null) frameBuf.deallocate();
                if (frameJpgExt != null) frameJpgExt.deallocate();
            }

            if (fullFrameBytes != null) {
                // Convert rectangles to PersonDetection objects (all unknown at this point)
                List<PeopleTrackingService.PersonDetection> allPeopleDetections = detectedPeople.stream()
                    .map(rect -> new PeopleTrackingService.PersonDetection("Unknown", rect))
                    .collect(Collectors.toList());

                // Track EACH person individually
                for (int i = 0; i < detectedPeople.size(); i++) {
                    Rect personRect = detectedPeople.get(i);

                    // Extract person region for tracking
                    byte[] personHash = null;
                    org.bytedeco.javacpp.BytePointer personBuf = null;
                    org.bytedeco.javacpp.BytePointer jpgExtPerson = null;
                    Mat personRegion = null;
                    try {
                        personRegion = new Mat(img, personRect);
                        personBuf = new org.bytedeco.javacpp.BytePointer();
                        jpgExtPerson = new org.bytedeco.javacpp.BytePointer(".jpg");
                        org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExtPerson, personRegion, personBuf);
                        personHash = new byte[(int) personBuf.limit()];
                        personBuf.get(personHash);
                    } catch (Exception e) {
                        log.warn("Failed to extract person region {}: {}", i + 1, e.getMessage());
                        continue; // Skip this person
                    } finally {
                        if (personBuf != null) personBuf.deallocate();
                        if (jpgExtPerson != null) jpgExtPerson.deallocate();
                        if (personRegion != null) matUtil.releaseResources(personRegion);
                    }

                    // Track this person, passing ALL people detections for drawing
                    if (personHash != null) {
                        PeopleTrackingService.TrackingResult trackingResult = peopleTrackingService.trackFace(
                                cameraName,
                            "Unknown",
                            personRect,
                            personHash,
                            DESIRED_SCORE, // High distance that it's unknown
                            fullFrameBytes,
                            allPeopleDetections, // Pass ALL detected people with names for drawing
                            false
                        );

                        // When this person's tracking is ready, notification will have ALL people highlighted
                        if (trackingResult.isShouldSend()) {
                            log.info("Camera '{}': Person #{} tracked successfully - notification sent with {} people highlighted",
                                    cameraName, i + 1, detectedPeople.size());
                        } else {
                            log.debug("Camera '{}': Still tracking person #{} (total {} people)",
                                    cameraName, i + 1, detectedPeople.size());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to process unknown people for camera '{}': {}", cameraName, e.getMessage(), e);
        }
    }

    @NotNull
    private File getTrainedFile() {
        String trainingRootFolder = trainingProperties.getRootFolder();

        // Try 1: Check if it's an absolute path or relative path on filesystem
        File filesystemFolder = new File(trainingRootFolder);
        if (filesystemFolder.exists() && filesystemFolder.isDirectory()) {
            log.debug("Found training folder on filesystem: {}", filesystemFolder.getAbsolutePath());
            return filesystemFolder;
        }

        // Try 2: Check in /app/train (Docker runtime path)
        File dockerFolder = new File("/app/train");
        if (dockerFolder.exists() && dockerFolder.isDirectory()) {
            log.debug("Found training folder in Docker path: {}", dockerFolder.getAbsolutePath());
            return dockerFolder;
        }

        // Try 3: Check classpath (development/JAR with embedded resources)
        ClassPathResource trainingResource = new ClassPathResource(trainingRootFolder);
        if (trainingResource.exists()) {
            try {
                File trainingRootDir = trainingResource.getFile();
                log.debug("Found training folder on classpath: {}", trainingRootDir.getAbsolutePath());
                return trainingRootDir;
            } catch (IOException ex) {
                log.warn("Unable to extract training folder from JAR classpath, will try to use JAR resources", ex);
                // If JAR extraction fails, we'll create a fallback or error
                throw new IllegalStateException("Training root folder '" + trainingRootFolder + "' found in JAR but cannot be extracted. " +
                        "For Docker, mount training data at /app/train or set TRAINING_ROOT_FOLDER environment variable.", ex);
            }
        }

        // No training folder found anywhere
        throw new IllegalStateException(
                "Training root folder '" + trainingRootFolder + "' not found in any of the following locations:\n" +
                "1. Filesystem path: " + filesystemFolder.getAbsolutePath() + "\n" +
                "2. Docker path: /app/train\n" +
                "3. Classpath resource: " + trainingRootFolder + "\n" +
                "Please ensure your training dataset is available in one of these locations.");
    }
}
