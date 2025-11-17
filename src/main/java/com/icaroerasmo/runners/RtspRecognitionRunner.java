package com.icaroerasmo;

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
import org.springframework.stereotype.Component;

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
                faceRecognizer = faceRecognitionService.train(args[0]);
            }

            final FaceRecognizer finalFaceRecognizer = faceRecognizer; // for lambda capture

            // Use RTSP URL from configuration instead of hard-coded value
            final String rtspUrl = streamsProperties.getRtspUrl();

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
                                Double confidence = output.getConfidence();
                                if (label != null && output.getFaceRect() != null) {
                                    // clone and downscale the frame to reduce memory and disk usage
                                    Mat roiClone = null;
                                    Mat resized = null;
                                    try {
                                        roiClone = img.clone();
                                        // Resize to half width/height (tune factors as needed)
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
                            map(
                                    output -> Map.entry(
                                            output.getPersonName(),
                                            detectionService.
                                                    shouldAnnounceDetection(
                                                            output.getPersonName(),
                                                            output.getConfidence()))
                            ).collect(toMap(
                                    Map.Entry::getKey,
                                    Map.Entry::getValue,
                                    (e1, e2) -> e1,
                                    LinkedHashMap::new
                            ));

                    Boolean shouldAnnounce = shouldAnnounceMap.values().stream().allMatch(Boolean::booleanValue);

                    if (shouldAnnounce) {
                        String names = faces.stream().map(output -> output.getPersonName()).collect(Collectors.joining());
                        log.info("Pessoas detectadas: {}", names);
                        try {
                            // Downscale final image before drawing rectangles/saving to reduce memory usage
                            Mat finalImg = new Mat();
                            Size finalSize = new Size(img.cols() / 2, img.rows() / 2);
                            resize(img, finalImg, finalSize);

                            // Draw rectangles and labels around all detected faces on the final image
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
                    // Safely release mats. faceRecognition might be null if an exception occurred earlier.
                    try {
                        Mat detectionImg = null;
                        if (faceRecognition != null) {
                            detectionImg = faceRecognition.getDetectionImg();
                        }
                        matUtil.releaseResources(img, detectionImg);

                        // Null out local references to help GC and native memory release
                        faceRecognition = null;

                        // Suggest GC/finalization for native resources (FFmpeg/OpenCV) - best-effort only
                        try {
                            System.runFinalization();
                            System.gc();
                        } catch (Exception ignore) {}

                    } catch (Exception releaseEx) {
                        log.warn("Error releasing Mats", releaseEx);
                    }
                }
            });

        } finally {
            // Ensure faceRecognizer closed and freed
            if (faceRecognizer != null) {
                try {
                    faceRecognizer.close();
                } catch (Exception ignore) {}
                faceRecognizer = null;

                try { System.runFinalization(); System.gc(); } catch (Exception ignore) {}
            }
        }
    }
}
