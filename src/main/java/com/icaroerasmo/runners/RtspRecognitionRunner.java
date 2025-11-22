package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionHistoryService;
import com.icaroerasmo.service.FaceRecognitionService;
import com.icaroerasmo.service.FaceRecognizerHolder;
import com.icaroerasmo.service.FaceTrackingService;
import com.icaroerasmo.service.TelegramPublisherService;
import com.icaroerasmo.service.RtspFrameExtractorService;
import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
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


@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private final FaceRecognitionService faceRecognitionService;
    private final FaceRecognizerHolder faceRecognizerHolder;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final MatUtil matUtil;
    private final TelegramPublisherService telegramPublisherService;
    private final DetectionHistoryService detectionHistoryService;
    private final FaceTrackingService faceTrackingService;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;

    public void start(String... args) throws Exception {

        try {
            File trainingRootDir = getTrainedFile();
            FaceRecognizer faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            // Initialize the holder with the trained recognizer
            faceRecognizerHolder.updateRecognizer(faceRecognizer);

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

                        // Get the current recognizer from the holder (thread-safe)
                        FaceRecognizer currentRecognizer = faceRecognizerHolder.get();
                        if (currentRecognizer == null) {
                            log.warn("FaceRecognizer not initialized yet, skipping frame");
                            return;
                        }

                        faceRecognition = faceRecognitionService.test(currentRecognizer, img);

                    if (faceRecognition == null) {
                        return;
                    }

                    List<FaceRecognition.DetectedFaces> faces = faceRecognition.getFaces();

                    if (faces == null || faces.isEmpty()) {
                        return;
                    }

                    // Filter out faces with score > 100
                    faces = faces.stream()
                            .filter(face -> face.getConfidence() <= 100)
                            .collect(Collectors.toList());

                    if (faces.isEmpty()) {
                        log.debug("All detected faces in frame from camera '{}' have score > 100, skipping frame", cameraName);
                        return;
                    }

                    // Collect detected names with their confidence scores
                    Map<String, Double> detectedPeopleWithScores = faces.stream()
                            .collect(Collectors.toMap(
                                    FaceRecognition.DetectedFaces::getPersonName,
                                    FaceRecognition.DetectedFaces::getConfidence,
                                    (existing, replacement) -> existing // Keep first if duplicate
                            ));

                    // Send notification for ALL detections with score <= 100 (recognized or Unknown)
                    String namesStr = String.join(", ", detectedPeopleWithScores.keySet());
                    log.info("Pessoas detectadas em '{}': {}", cameraName, namesStr);

                    // Publish to Telegram with detected people information
                    try {
                            Mat finalImg = img.clone();
                            faces.forEach(output -> {
                                if (output.getFaceRect() != null && output.getPersonName() != null) {
                                    matUtil.drawRectangleAndName(finalImg, output.getPersonName(), output.getFaceRect());
                                }
                            });

                            // Convert Mat to byte array using imencode
                            org.bytedeco.javacpp.BytePointer buf = new org.bytedeco.javacpp.BytePointer();
                            org.bytedeco.javacpp.BytePointer jpgExt = new org.bytedeco.javacpp.BytePointer(".jpg");
                            org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExt, finalImg, buf);
                            byte[] imageBytes = new byte[(int) buf.limit()];
                            buf.get(imageBytes);
                            buf.deallocate();
                            jpgExt.deallocate();

                            // Extract face region for similarity comparison
                            byte[] faceHash = null;
                            if (!faces.isEmpty()) {
                                FaceRecognition.DetectedFaces firstFace = faces.get(0);
                                if (firstFace.getFaceRect() != null) {
                                    org.bytedeco.javacpp.BytePointer faceBuf = null;
                                    org.bytedeco.javacpp.BytePointer jpgExtFace = null;
                                    Mat faceRegion = null;
                                    try {
                                        // Extract the face region from original image
                                        faceRegion = new Mat(img, firstFace.getFaceRect());

                                        // Convert face to byte array for hashing
                                        faceBuf = new org.bytedeco.javacpp.BytePointer();
                                        jpgExtFace = new org.bytedeco.javacpp.BytePointer(".jpg");
                                        org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExtFace, faceRegion, faceBuf);
                                        faceHash = new byte[(int) faceBuf.limit()];
                                        faceBuf.get(faceHash);
                                    } catch (Exception e) {
                                        log.warn("Failed to extract face region for camera '{}': {}", cameraName, e.getMessage());
                                    } finally {
                                        // Clean up resources
                                        if (faceBuf != null) {
                                            faceBuf.deallocate();
                                        }
                                        if (jpgExtFace != null) {
                                            jpgExtFace.deallocate();
                                        }
                                        if (faceRegion != null) {
                                            matUtil.releaseResources(faceRegion);
                                        }
                                    }
                                }
                            }

                            // Compute image hash and check if this detection should be sent
                            String imageHash = detectionHistoryService.computeImageHash(imageBytes);

                            // Use face tracking for ALL persons (both known and unknown)
                            // Track the face across multiple frames to determine true identity
                            if (!faces.isEmpty()) {
                                FaceRecognition.DetectedFaces firstFace = faces.get(0);
                                log.info("🔍 FRAME PROCESSING: camera='{}', person='{}', confidence={}, rect={}, faceHash={}, imageBytes={}",
                                    cameraName,
                                    firstFace.getPersonName(),
                                    String.format("%.2f", firstFace.getConfidence()),
                                    (firstFace.getFaceRect() != null),
                                    (faceHash != null ? faceHash.length : 0),
                                    (imageBytes != null ? imageBytes.length : 0));

                                if (firstFace.getFaceRect() == null) {
                                    log.warn("⚠️ SKIPPED: faceRect is null for camera '{}'", cameraName);
                                } else if (faceHash == null) {
                                    log.warn("⚠️ SKIPPED: faceHash is null for camera '{}'", cameraName);
                                } else {
                                    log.info("📊 CALLING TRACKING SERVICE: camera='{}', person='{}'", cameraName, firstFace.getPersonName());

                                    // Track face with current detection name and confidence score
                                    FaceTrackingService.TrackingResult trackingResult = faceTrackingService.trackFace(
                                            cameraName,
                                            firstFace.getPersonName(), // Pass current detection name
                                            firstFace.getFaceRect(),
                                            faceHash,
                                            firstFace.getConfidence(),
                                            imageBytes // Pass current frame image bytes
                                    );

                                    log.info("📋 TRACKING RESULT: shouldSend={}, personName={}, score={}",
                                        trackingResult.isShouldSend(),
                                        trackingResult.getPersonName(),
                                        trackingResult.isShouldSend() ? String.format("%.2f", trackingResult.getBestConfidenceScore()) : "N/A");

                                    if (trackingResult.isShouldSend()) {
                                        log.info("✅ VERDICT REACHED! Starting notification process...");
                                        // Face has been tracked through multiple frames with movement
                                        // Use the determined identity (most common) and best score
                                        String determinedIdentity = trackingResult.getPersonName();
                                        boolean isUnknown = "Unknown".equalsIgnoreCase(determinedIdentity);

                                        log.info("🎯 IDENTITY: '{}', isUnknown={}", determinedIdentity, isUnknown);

                                        // Create scores map with determined identity and best score
                                        Map<String, Double> bestScores = Map.of(determinedIdentity, trackingResult.getBestConfidenceScore());

                                        // Use the BEST frame's image bytes (not current frame)
                                        byte[] bestImageBytes = trackingResult.getBestImageBytes();

                                        log.info("📷 IMAGE DATA: bestImageBytes={} bytes",
                                            (bestImageBytes != null ? bestImageBytes.length : 0));

                                        if (bestImageBytes == null || bestImageBytes.length == 0) {
                                            log.error("❌ BLOCKED: Best image bytes is null or empty for camera '{}'", cameraName);
                                        } else {
                                            log.info("🔍 CHECKING DETECTION HISTORY...");

                                            // Check if we recently sent this person (using appropriate method)
                                            boolean shouldSend = isUnknown
                                                    ? detectionHistoryService.shouldSendUnknownDetection(imageHash, determinedIdentity, cameraName, trackingResult.getBestFaceHash())
                                                    : detectionHistoryService.shouldSendDetection(imageHash, determinedIdentity, cameraName, trackingResult.getBestFaceHash());

                                            log.info("📊 COOLDOWN CHECK: shouldSend={}, isUnknown={}, identity='{}'",
                                                shouldSend, isUnknown, determinedIdentity);

                                            if (shouldSend) {
                                                log.info("🚀 SENDING TO TELEGRAM: camera='{}', identity='{}', score={}, imageSize={} bytes",
                                                    cameraName, determinedIdentity,
                                                    String.format("%.2f", trackingResult.getBestConfidenceScore()),
                                                    bestImageBytes.length);

                                                try {
                                                    // Send notification with BEST frame image and determined identity
                                                    telegramPublisherService.publishDetection(bestImageBytes, bestScores, cameraName);

                                                    log.info("✅ SUCCESS: Telegram notification sent for '{}'", determinedIdentity);

                                                    // Mark as sent to prevent duplicates
                                                    if (isUnknown) {
                                                        detectionHistoryService.markUnknownDetectionAsSent(imageHash, determinedIdentity, cameraName, trackingResult.getBestFaceHash());
                                                        log.debug("Marked unknown '{}' as sent", determinedIdentity);
                                                    }
                                                } catch (Exception telegramEx) {
                                                    log.error("❌ TELEGRAM FAILED: {}", telegramEx.getMessage(), telegramEx);
                                                }
                                            } else {
                                                log.info("⏭️ COOLDOWN: Skipping '{}' for camera '{}' (sent recently)", determinedIdentity, cameraName);
                                            }
                                        }
                                    } else {
                                        log.debug("⏳ STILL TRACKING: camera='{}', not ready to send yet", cameraName);
                                    }
                                }
                            } else {
                                log.debug("📭 NO FACES: Skipping frame from camera '{}'", cameraName);
                            }

                            matUtil.releaseResources(finalImg);
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
