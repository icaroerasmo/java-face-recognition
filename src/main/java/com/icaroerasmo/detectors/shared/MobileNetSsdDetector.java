package com.icaroerasmo.detectors.shared;

import com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper;
import com.icaroerasmo.properties.AccelerationProperties;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL_FP16;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

/**
 * Shared MobileNet-SSD (COCO) detector. Loads {@code opencv/SSD_MobileNet.caffemodel}
 * exactly ONCE and exposes raw detections for all COCO classes. Both
 * {@code PersonDetector} and {@code PetDetector} consume this instance so only a
 * single model copy ever exists in memory.
 *
 * <p>Every DNN forward pass runs inside
 * {@link DnnInferenceCoordinatorHelper#runExclusive} (person, face and pet share the
 * inference coordinator). The returned {@link Rect}s are owned by the caller and must
 * be deallocated when consumed.
 */
@Log4j2
@Service
public class MobileNetSsdDetector {

    private static final String PROTO_FILE = "opencv/SSD_MobileNet_prototxt.txt";
    private static final String MODEL_FILE = "opencv/SSD_MobileNet.caffemodel";
    private static final int INPUT_SIZE = 300;
    private static final double SCALE_FACTOR = 0.007843; // 1/127.5

    // Diagnostic logging: at most one RAW_DET line every 2s, listing only detections
    // above this confidence floor. Purely observational - never changes detection.
    private static final long RAW_LOG_INTERVAL_MS = 2000;
    private static final float RAW_LOG_MIN_CONFIDENCE = 0.10f;

    private volatile long lastRawLogTime = 0L;

    private final Net net;
    private final MatUtil matUtil;
    private final DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper;

    public MobileNetSsdDetector(
            MatUtil matUtil,
            ObjectDetectionProperties objectDetectionProperties,
            DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper
    ) {
        this.matUtil = matUtil;
        this.dnnInferenceCoordinatorHelper = dnnInferenceCoordinatorHelper;
        try {
            String protoPath = OpenCvResourceHelper.getResourcePath(PROTO_FILE, MobileNetSsdDetector.class);
            String modelPath = OpenCvResourceHelper.getResourcePath(MODEL_FILE, MobileNetSsdDetector.class);

            log.info("Loading MobileNet-SSD model from: {} and {}", protoPath, modelPath);
            this.net = readNetFromCaffe(protoPath, modelPath);
            configureNet(
                    this.net,
                    objectDetectionProperties.getAcceleration().getBackend(),
                    objectDetectionProperties.getAcceleration().getPersonDetectionTarget(),
                    objectDetectionProperties.getAcceleration().isFallbackToCpu(),
                    "object detection"
            );

            if (this.net == null || this.net.empty()) {
                throw new IllegalStateException("Failed to load network - network is null or empty");
            }

            log.info("MobileNet-SSD model loaded successfully");
        } catch (Exception e) {
            log.error("Failed to load MobileNet-SSD model: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to initialize MobileNet-SSD model", e);
        }
    }

    /**
     * Runs the MobileNet-SSD forward pass and returns raw COCO detections with
     * pixel-scaled rectangles. The returned {@link Rect}s are owned by the caller.
     */
    public List<CocoDetection> detectRaw(Mat image) {
        List<CocoDetection> detections = new ArrayList<>();
        if (image == null || image.empty()) {
            log.warn("Cannot run detection on null or empty image");
            return detections;
        }

        int originalWidth = image.size().width();
        int originalHeight = image.size().height();

        Mat resizedImage = null, blob = null, output = null, detectionMat = null;
        FloatIndexer indexer = null;
        Size inputSize = null, blobSize = null, detectionSize = null;
        Scalar meanValues = null;

        try {
            resizedImage = new Mat();
            inputSize = new Size(INPUT_SIZE, INPUT_SIZE);
            resize(image, resizedImage, inputSize);

            if (resizedImage.empty()) {
                log.warn("Failed to resize image for MobileNet-SSD detection");
                return detections;
            }

            blobSize = new Size(INPUT_SIZE, INPUT_SIZE);
            meanValues = new Scalar(127.5, 127.5, 127.5, 0);
            blob = blobFromImage(resizedImage, SCALE_FACTOR, blobSize,
                    meanValues, false, false, CV_32F);

            if (blob == null || blob.empty()) {
                log.warn("Failed to create blob from image");
                return detections;
            }

            Mat inferenceBlob = blob;
            output = dnnInferenceCoordinatorHelper.runExclusive("object detection", () -> {
                // OpenCL-backed DNN inference is not stable when multiple networks run concurrently.
                net.setInput(inferenceBlob);
                return net.forward();
            });

            if (output == null || output.empty()) {
                log.warn("Neural network forward pass returned empty output");
                return detections;
            }

            // Output format: [1, 1, N, 7] where N is number of detections
            // Each detection: [image_id, class_id, confidence, x1, y1, x2, y2]
            detectionSize = new Size(output.size(3), output.size(2));
            detectionMat = new Mat(detectionSize, CV_32F, output.ptr(0, 0));

            if (detectionMat.empty()) {
                log.warn("Failed to extract detection matrix from network output");
                return detections;
            }

            indexer = detectionMat.createIndexer();

            for (int i = 0; i < output.size(3); i++) {
                int classId = (int) indexer.get(i, 1);
                float confidence = indexer.get(i, 2);

                // Convert normalized coordinates to pixel coordinates
                float x1 = indexer.get(i, 3);
                float y1 = indexer.get(i, 4);
                float x2 = indexer.get(i, 5);
                float y2 = indexer.get(i, 6);

                int left = clamp((int) (x1 * originalWidth), 0, originalWidth - 1);
                int top = clamp((int) (y1 * originalHeight), 0, originalHeight - 1);
                int right = clamp((int) (x2 * originalWidth), 0, originalWidth - 1);
                int bottom = clamp((int) (y2 * originalHeight), 0, originalHeight - 1);

                int width = right - left;
                int height = bottom - top;

                if (width <= 0 || height <= 0) {
                    continue;
                }

                detections.add(new CocoDetection(classId, confidence, new Rect(left, top, width, height)));
            }

            logRawDetections(detections);

            return detections;
        } catch (Exception e) {
            log.error("Error during MobileNet-SSD detection: {}", e.getMessage(), e);
            // Release any rects already produced so no native memory leaks.
            for (CocoDetection detection : detections) {
                if (detection.rect() != null) {
                    detection.rect().deallocate();
                }
            }
            return new ArrayList<>();
        } finally {
            if (indexer != null) {
                try {
                    indexer.release();
                } catch (Exception e) {
                    log.debug("Error releasing indexer: {}", e.getMessage());
                }
            }
            if (meanValues != null) {
                try {
                    meanValues.deallocate();
                } catch (Exception e) {
                    log.debug("Error deallocating meanValues: {}", e.getMessage());
                }
            }
            if (inputSize != null) {
                try {
                    inputSize.deallocate();
                } catch (Exception e) {
                    log.debug("Error deallocating inputSize: {}", e.getMessage());
                }
            }
            if (blobSize != null) {
                try {
                    blobSize.deallocate();
                } catch (Exception e) {
                    log.debug("Error deallocating blobSize: {}", e.getMessage());
                }
            }
            if (detectionSize != null) {
                try {
                    detectionSize.deallocate();
                } catch (Exception e) {
                    log.debug("Error deallocating detectionSize: {}", e.getMessage());
                }
            }
            matUtil.releaseResources(resizedImage, blob, output, detectionMat);
        }
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(value, max));
    }

    /**
     * Throttled INFO diagnostic: one line listing raw detections with class name +
     * confidence (e.g. {@code RAW_DET: person=0.82 dog=0.31 pottedplant=0.12 car=0.11}).
     * One entry per class (highest confidence kept), only detections above
     * {@value #RAW_LOG_MIN_CONFIDENCE}. At most one line per {@value #RAW_LOG_INTERVAL_MS} ms.
     */
    private void logRawDetections(List<CocoDetection> detections) {
        long now = System.currentTimeMillis();
        if (now - lastRawLogTime < RAW_LOG_INTERVAL_MS) {
            return;
        }
        lastRawLogTime = now;

        // Preserve first-seen (model output) order, keeping the best confidence per class.
        Map<Integer, Float> bestByClass = new LinkedHashMap<>();
        for (CocoDetection detection : detections) {
            if (detection.confidence() <= RAW_LOG_MIN_CONFIDENCE) {
                continue;
            }
            bestByClass.merge(detection.classId(), detection.confidence(), Math::max);
        }
        if (bestByClass.isEmpty()) {
            return;
        }

        StringBuilder sb = new StringBuilder("RAW_DET:");
        for (Map.Entry<Integer, Float> entry : bestByClass.entrySet()) {
            sb.append(' ')
              .append(DetectionClassFilter.classIdToName(entry.getKey()))
              .append('=')
              .append(String.format(Locale.ROOT, "%.2f", entry.getValue()));
        }
        log.info(sb.toString());
    }

    private static void configureNet(
            Net net,
            AccelerationProperties.Backend backend,
            AccelerationProperties.Target target,
            boolean fallbackToCpu,
            String modelName
    ) {
        try {
            if (backend != null && backend != AccelerationProperties.Backend.AUTO) {
                net.setPreferableBackend(mapBackend(backend));
            }
            if (target != null && target != AccelerationProperties.Target.AUTO) {
                net.setPreferableTarget(mapTarget(target));
            }
            log.info("Configured {} DNN backend={} target={}", modelName, backend, target);
        } catch (Exception e) {
            if (!fallbackToCpu) {
                throw e;
            }
            log.warn("Failed to configure {} acceleration (backend={}, target={}), falling back to CPU: {}",
                    modelName, backend, target, e.getMessage());
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }
    }

    private static int mapBackend(AccelerationProperties.Backend backend) {
        return switch (backend) {
            case AUTO, OPENCV -> DNN_BACKEND_OPENCV;
        };
    }

    private static int mapTarget(AccelerationProperties.Target target) {
        return switch (target) {
            case AUTO, CPU -> DNN_TARGET_CPU;
            case OPENCL -> DNN_TARGET_OPENCL;
            case OPENCL_FP16 -> DNN_TARGET_OPENCL_FP16;
        };
    }
}
