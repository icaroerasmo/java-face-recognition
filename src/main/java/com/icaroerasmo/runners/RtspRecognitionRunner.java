package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.Camera;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionService;
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
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private static final AtomicInteger COUNT = new AtomicInteger(0);

    private final FaceRecognitionService faceRecognitionService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final DetectionService detectionService;
    private final MatUtil matUtil;
    private final TelegramPublisherService telegramPublisherService;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;

    public void start(String... args) throws Exception {

        FaceRecognizer faceRecognizer = null;

        try {
            File trainingRootDir = getTrainedFile();
            faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            final FaceRecognizer finalFaceRecognizer = faceRecognizer;

            List<Camera> cameras = streamsProperties.getCameras();
            if (cameras == null || cameras.isEmpty()) {
                throw new IllegalStateException("No cameras configured under face-recognition.streams.cameras");
            }

            for (Camera camera : cameras) {
                if (camera == null || camera.getUrl() == null || camera.getUrl().isBlank()) {
                    continue;
                }

                String cameraName = camera.getName() != null ? camera.getName() : "unknown";
                String rtspUrl = camera.getUrl();

                log.info("Starting recognition for camera '{}': {}", cameraName, rtspUrl);

                rtspFrameExtractorService.extract(rtspUrl, (img) -> {

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

                        // Collect detected names with their confidence scores
                        Map<String, Double> detectedPeopleWithScores = faces.stream()
                                .collect(Collectors.toMap(
                                        FaceRecognition.DetectedFaces::getPersonName,
                                        FaceRecognition.DetectedFaces::getConfidence,
                                        (existing, replacement) -> existing // Keep first if duplicate
                                ));

                        // Check if we have recognized persons (non-Unknown)
                        boolean hasRecognizedPersons = detectedPeopleWithScores.keySet().stream()
                                .anyMatch(name -> !"Unknown".equalsIgnoreCase(name));

                        // Check if we should announce Unknown detections
                        boolean shouldAnnounceUnknown = faces.stream()
                                .filter(face -> "Unknown".equalsIgnoreCase(face.getPersonName()))
                                .anyMatch(face -> detectionService.shouldAnnounceDetection(
                                        face.getPersonName(),
                                        face.getConfidence()));

                        // Publish if we have recognized persons OR if unknown should be announced
                        if (hasRecognizedPersons || shouldAnnounceUnknown) {
                            String namesStr = String.join(", ", detectedPeopleWithScores.keySet());
                            log.info("Pessoas detectadas em '{}': {}", cameraName, namesStr);

                            // Publish to MQTT in double-take format
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

                                // Publish to Telegram with detected people information
                                telegramPublisherService.publishDetection(imageBytes, detectedPeopleWithScores, cameraName);

                                matUtil.releaseResources(finalImg);
                            } catch (Exception e) {
                                log.error("Failed to publish detection to MQTT for camera '{}'", cameraName, e);
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
            }

        } finally {
            if (faceRecognizer != null) {
                try {
                    faceRecognizer.close();
                } catch (Exception ignore) {}
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
