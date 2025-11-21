package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionHistoryService;
import com.icaroerasmo.service.FaceRecognitionService;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private static final AtomicInteger COUNT = new AtomicInteger(0);

    private final FaceRecognitionService faceRecognitionService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final MatUtil matUtil;
    private final TelegramPublisherService telegramPublisherService;
    private final DetectionHistoryService detectionHistoryService;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;

    public void start(String... args) throws Exception {

        FaceRecognizer faceRecognizer = null;

        try {
            File trainingRootDir = getTrainedFile();
            faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            final FaceRecognizer finalFaceRecognizer = faceRecognizer;

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
                    processCameraStream(camera, finalFaceRecognizer);
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

        } finally {
            if (faceRecognizer != null) {
                try {
                    faceRecognizer.close();
                } catch (Exception ignore) {}
            }
        }
    }

    /**
     * Process a single camera stream
     */
    private void processCameraStream(CameraProperties cameraProperties, FaceRecognizer finalFaceRecognizer) {
        String cameraName = cameraProperties.getName() != null ? cameraProperties.getName() : "unknown";
        String rtspUrl = cameraProperties.getUrl();

        log.info("Starting recognition for camera '{}' with {} transport: {}", cameraName, cameraProperties.getProtocol(), rtspUrl);

        try {
            rtspFrameExtractorService.extract(rtspUrl, cameraProperties.getProtocol(), (img) -> {

                FaceRecognition faceRecognition = null;

                try {

                    if (img == null) {
                        return;
                    }

                    faceRecognition = faceRecognitionService.test(finalFaceRecognizer, img);

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

                    // Check if we have recognized people (non-Unknown)
                    boolean hasRecognizedPeople = detectedPeopleWithScores.keySet().stream()
                            .anyMatch(name -> !"Unknown".equalsIgnoreCase(name));

                    // Publish if we have recognized people OR if unknown should be announced
                    if (hasRecognizedPeople) {
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
                            org.bytedeco.opencv.global.opencv_imgcodecs.imencode(new org.bytedeco.javacpp.BytePointer(".jpg"), finalImg, buf);
                            byte[] imageBytes = new byte[(int) buf.limit()];
                            buf.get(imageBytes);
                            buf.deallocate();

                            // Compute image hash and check if this detection should be sent
                            String imageHash = detectionHistoryService.computeImageHash(imageBytes);
                            String detectedPeopleKey = String.join(",", detectedPeopleWithScores.keySet());

                            if (detectionHistoryService.shouldSendDetection(imageHash, detectedPeopleKey, cameraName)) {
                                // Publish to Telegram with detected people information
                                telegramPublisherService.publishDetection(imageBytes, detectedPeopleWithScores, cameraName);
                            } else {
                                log.debug("Skipping duplicate detection for camera '{}' with people: {}", cameraName, detectedPeopleKey);
                            }

                            matUtil.releaseResources(finalImg);
                        } catch (Exception e) {
                            log.error("Failed to publish detection to Telegram for camera '{}'", cameraName, e);
                        }
                    }

                } catch (Exception e) {
                    log.error("Error processing frame from camera '{}'", cameraName, e);
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
        } catch (Exception e) {
            log.error("Error starting stream extraction for camera '{}': {}", cameraName, e.getMessage(), e);
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
