package com.icaroerasmo.runners;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.service.DetectionService;
import com.icaroerasmo.service.FaceRecognitionService;
import com.icaroerasmo.service.RtspFrameExtractorService;
import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_core.Size;

@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private static final Path DATASET = Paths.get("trained_dataset.xml");
    private static final String TRAINING_ROOT_CLASSPATH = "training"; // folder inside classpath with training dataset
    private static final AtomicInteger COUNT = new AtomicInteger(0);

    private final FaceRecognitionService faceRecognitionService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final DetectionService detectionService;
    private final MatUtil matUtil;
    private final StreamsProperties streamsProperties;

    public void start(String... args) throws Exception {

        FaceRecognizer faceRecognizer = null;

        try {
            if (DATASET.toFile().exists()) {
                faceRecognizer = faceRecognitionService.load();
            } else {
                // Dataset does not exist: use a fixed training folder from the classpath
                ClassPathResource trainingResource = new ClassPathResource(TRAINING_ROOT_CLASSPATH);
                if (!trainingResource.exists()) {
                    throw new IllegalStateException("Training root folder '" + TRAINING_ROOT_CLASSPATH + "' not found on classpath. " +
                            "Place your training dataset under src/main/resources/" + TRAINING_ROOT_CLASSPATH + "/");
                }

                File trainingRootDir;
                try {
                    trainingRootDir = trainingResource.getFile();
                } catch (IOException ex) {
                    throw new IllegalStateException("Unable to resolve training root folder from classpath resource '" +
                            TRAINING_ROOT_CLASSPATH + "'", ex);
                }

                // FaceRecognitionService.train(Path root) expects a Path
                faceRecognizer = faceRecognitionService.train(trainingRootDir.toPath());
            }

            final FaceRecognizer finalFaceRecognizer = faceRecognizer; // for lambda capture

            // Use all RTSP URLs from configuration instead of a single hard-coded value
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
                            log.debug("No faces detected in the image.");
                            return;
                        }

                        final Map<String, Boolean> shouldAnnounceMap = faces.parallelStream().
                                peek(output -> {
                                    String label = output.getPersonName();
                                    if (label != null && output.getFaceRect() != null) {
                                        Mat roiClone = null;
                                        Mat resized = null;
                                        try {
                                            roiClone = img.clone();
                                            Size newSize = new Size(roiClone.cols() / 2, roiClone.rows() / 2);
                                            resized = new Mat();
                                            resize(roiClone, resized, newSize);

                                            matUtil.drawRectangleAndName(resized, label, output.getFaceRect());
                                            imwrite("img_%s_%d_.jpg".formatted(label, COUNT.getAndIncrement()), resized);
                                        } catch (Exception ex) {
                                            log.warn("Error while writing or drawing detection image", ex);
                                        } finally {
                                            matUtil.releaseResources(roiClone, resized);
                                        }
                                    }
                                }).
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
                            try {
                                Mat finalImg = new Mat();
                                Size finalSize = new Size(img.cols() / 2, img.rows() / 2);
                                resize(img, finalImg, finalSize);

                                faces.forEach(output -> {
                                    if (output.getFaceRect() != null && output.getPersonName() != null) {
                                        matUtil.drawRectangleAndName(finalImg, output.getPersonName(), output.getFaceRect());
                                    }
                                });
                                imwrite("img_%s_%d_.jpg".formatted("final", COUNT.getAndIncrement()), finalImg);

                                matUtil.releaseResources(finalImg);
                            } catch (Exception e) {
                                log.warn("Couldn't write final image", e);
                            }
                        }

                    } catch (Exception e) {
                        log.error("Error processing frame", e);
                    } finally {
                        try {
                            Mat detectionImg = null;
                            if (faceRecognition != null) {
                                detectionImg = faceRecognition.getDetectionImg();
                            }
                            matUtil.releaseResources(img, detectionImg);

                            faceRecognition = null;
                        } catch (Exception releaseEx) {
                            log.warn("Error releasing Mats", releaseEx);
                        }
                    }
                });
            }

        } finally {
            if (faceRecognizer != null) {
                try {
                    faceRecognizer.close();
                } catch (Exception ignore) {}
                faceRecognizer = null;
            }
        }
    }
}
