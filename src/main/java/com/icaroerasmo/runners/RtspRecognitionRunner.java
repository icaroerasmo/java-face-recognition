package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionService;
import com.icaroerasmo.service.FaceRecognitionService;
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
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private static final AtomicInteger COUNT = new AtomicInteger(0);

    private final FaceRecognitionService faceRecognitionService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final DetectionService detectionService;
    private final MatUtil matUtil;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;

    public void start(String... args) throws Exception {

        FaceRecognizer faceRecognizer = null;

        try {
            File trainingRootDir = getTrainedFile();
            faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            final FaceRecognizer finalFaceRecognizer = faceRecognizer;

            List<String> urls = streamsProperties.getRtspUrls();
            if (urls == null || urls.isEmpty()) {
                throw new IllegalStateException("No RTSP URLs configured under face-recognition.streams.rtsp-urls");
            }

            for (String rtspUrl : urls) {
                if (rtspUrl == null || rtspUrl.isBlank()) {
                    continue;
                }

                log.info("Starting recognition for RTSP stream: {}", rtspUrl);

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

                        final Map<String, Boolean> shouldAnnounceMap = faces.parallelStream().
                                map(output -> Map.entry(
                                        output.getPersonName(),
                                        detectionService.shouldAnnounceDetection(
                                                output.getPersonName(),
                                                output.getConfidence()))
                                ).collect(toMap(
                                        Map.Entry::getKey,
                                        Map.Entry::getValue,
                                        (e1, e2) -> e1,
                                        LinkedHashMap::new
                                ));

                        boolean shouldAnnounce = shouldAnnounceMap.values().stream().allMatch(Boolean::booleanValue);

                        if (shouldAnnounce) {
                            String names = faces.stream()
                                    .map(FaceRecognition.DetectedFaces::getPersonName)
                                    .collect(Collectors.joining());
                            log.info("Pessoas detectadas: {}", names);

                            // Save images ONLY when people are recognized
                            try {
                                // Create recognized_faces folder if it doesn't exist
                                java.io.File recognizedDir = new java.io.File("recognized_faces");
                                if (!recognizedDir.exists()) {
                                    recognizedDir.mkdirs();
                                }

                                Mat finalImg = img.clone();
                                faces.forEach(output -> {
                                    if (output.getFaceRect() != null && output.getPersonName() != null) {
                                        matUtil.drawRectangleAndName(finalImg, output.getPersonName(), output.getFaceRect());
                                    }
                                });
                                String filename = String.format("recognized_faces/img_final_%d.jpg", COUNT.getAndIncrement());
                                imwrite(filename, finalImg);
                                log.info("Saved final image with all recognized faces: {}", filename);
                                matUtil.releaseResources(finalImg);
                            } catch (Exception e) {
                                // Silently ignore
                            }
                        }

                    } catch (Exception e) {
                        // Silently ignore
                    } finally {
                        try {
                            Mat detectionImg = null;
                            if (faceRecognition != null) {
                                detectionImg = faceRecognition.getDetectionImg();
                            }
                            matUtil.releaseResources(img, detectionImg);
                            faceRecognition = null;
                        } catch (Exception releaseEx) {
                            // Silently ignore
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

