package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.Camera;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionService;
import com.icaroerasmo.service.FaceRecognitionService;
import com.icaroerasmo.service.MqttPublisherService;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imencode;

@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private static final AtomicInteger COUNT = new AtomicInteger(0);

    private final FaceRecognitionService faceRecognitionService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final DetectionService detectionService;
    private final MatUtil matUtil;
    private final MqttPublisherService mqttPublisherService;
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

                        final java.util.Map<String, Boolean> shouldAnnounceMap = faces.parallelStream().
                                map(output -> java.util.Map.entry(
                                        output.getPersonName(),
                                        detectionService.shouldAnnounceDetection(
                                                output.getPersonName(),
                                                output.getConfidence()))
                                ).collect(toMap(
                                        java.util.Map.Entry::getKey,
                                        java.util.Map.Entry::getValue,
                                        (e1, e2) -> e1,
                                        LinkedHashMap::new
                                ));

                        boolean shouldAnnounce = shouldAnnounceMap.values().stream().allMatch(Boolean::booleanValue);

                        if (shouldAnnounce) {
                            // Collect detected names with their confidence scores
                            Map<String, Double> detectedPeopleWithScores = faces.stream()
                                    .collect(Collectors.toMap(
                                            FaceRecognition.DetectedFaces::getPersonName,
                                            FaceRecognition.DetectedFaces::getConfidence,
                                            (existing, replacement) -> existing // Keep first if duplicate
                                    ));

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

                                // Publish to MQTT in double-take format using camera name and scores
                                mqttPublisherService.publishDetection(imageBytes, detectedPeopleWithScores, cameraName);

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
        ClassPathResource trainingResource = new ClassPathResource(trainingRootFolder);
        if (!trainingResource.exists()) {
            throw new IllegalStateException("Training root folder '" + trainingRootFolder + "' not found on classpath. " +
                    "Place your training dataset under src/main/resources/" + trainingRootFolder + "/");
        }

        File trainingRootDir;
        try {
            trainingRootDir = trainingResource.getFile();
        } catch (IOException ex) {
            throw new IllegalStateException("Unable to resolve training root folder from classpath resource '" +
                    trainingRootFolder + "'", ex);
        }
        return trainingRootDir;
    }
}

