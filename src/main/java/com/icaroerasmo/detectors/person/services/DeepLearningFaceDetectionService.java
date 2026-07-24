package com.icaroerasmo.detectors.person.services;

import com.icaroerasmo.properties.AccelerationProperties;
import com.icaroerasmo.properties.FaceRecognitionProperties;
import com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.indexer.FloatIndexer;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Created on Jul 28, 2018
 *
 * @author Taha Emara
 * Email : taha@emaraic.com
 *
 * Face detection powered by SCRFD ONNX.
 */
@Log4j2
@Service
public class DeepLearningFaceDetectionService {

    public static final int MODEL_INPUT_SIZE = 640;
    private static final String SCRFD_MODEL_FILE = "opencv/scrfd_2.5g_bnkps.onnx";
    private static final float CONFIDENCE_THRESHOLD = 0.45f;
    private static final float NMS_THRESHOLD = 0.4f;
    private static final int MAX_DETECTIONS = 50;
    private static final int[] SCRFD_STRIDES = {8, 16, 32};
    private static final int SCRFD_ANCHORS_PER_LOCATION = 2;
    private final Net net;

    private final MatUtil matUtil;
    private final DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper;

    public DeepLearningFaceDetectionService(
            MatUtil matUtil,
            FaceRecognitionProperties faceRecognitionProperties,
            DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper
    ) {
        this.matUtil = matUtil;
        this.dnnInferenceCoordinatorHelper = dnnInferenceCoordinatorHelper;
        try {
            String modelPath = OpenCvResourceHelper.getResourcePath(SCRFD_MODEL_FILE, DeepLearningFaceDetectionService.class);

            log.info("Loading SCRFD face detection model from: {}", modelPath);
            this.net = readNetFromONNX(modelPath);
            configureNet(
                    this.net,
                    faceRecognitionProperties.getAcceleration().getBackend(),
                    resolveTarget(
                            faceRecognitionProperties.getAcceleration(),
                            faceRecognitionProperties.getAcceleration().getFaceDetectionTarget()
                    ),
                    faceRecognitionProperties.getAcceleration().isFallbackToCpu(),
                    "face detection"
            );
            log.info("SCRFD face detection model loaded successfully");
        } catch (Exception e) {
            log.error("Failed to load face detection model", e);
            throw new RuntimeException("Failed to initialize face detection model", e);
        }
    }

    /**
     * Detect faces in an image.
     * THREAD-SAFE: Synchronized to prevent concurrent access to the shared Net object,
     * which was causing false detections when multiple cameras were processing frames simultaneously.
     */
    public synchronized List<FaceDetection> detect(Mat testImage) {
        List<FaceDetection> faces = new ArrayList<>();

        // Validate input
        if (testImage == null || testImage.empty()) {
            log.warn("Cannot detect faces in null or empty image");
            return faces;
        }

        Mat blob = null, resizedImage = null, paddedImage = null;
        MatVector outputs = null;
        StringVector outputNames = null;
        Size inputSize = null;
        Size resizedSize = null;
        Scalar meanValues = null;
        Scalar paddingColor = null;
        Rect roiRect = null;
        Mat roi = null;

        try {
            // Store original dimensions
            int originalWidth = testImage.size().width();
            int originalHeight = testImage.size().height();

            float scale = Math.min(
                    MODEL_INPUT_SIZE / (float) originalWidth,
                    MODEL_INPUT_SIZE / (float) originalHeight
            );
            int resizedWidth = Math.round(originalWidth * scale);
            int resizedHeight = Math.round(originalHeight * scale);

            resizedImage = new Mat();
            inputSize = new Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
            resizedSize = new Size(resizedWidth, resizedHeight);
            resize(testImage, resizedImage, resizedSize);

            // Validate resized image
            if (resizedImage.empty()) {
                log.warn("Failed to resize image for face detection");
                return faces;
            }

            paddingColor = new Scalar(0, 0, 0, 0);
            paddedImage = new Mat(inputSize, testImage.type(), paddingColor);
            roiRect = new Rect(0, 0, resizedWidth, resizedHeight);
            roi = new Mat(paddedImage, roiRect);
            resizedImage.copyTo(roi);

            // Create blob from resized image
            // NCHW: Number of images, Channels, Height, Width
            meanValues = new Scalar(127.5, 127.5, 127.5, 0);
            blob = blobFromImage(paddedImage, 1.0 / 128.0, inputSize, meanValues, true, false, CV_32F);

            if (blob == null || blob.empty()) {
                log.warn("Failed to create blob from image");
                return faces;
            }

            Mat inferenceBlob = blob;
            outputNames = net.getUnconnectedOutLayersNames();
            StringVector finalOutputNames = outputNames;
            outputs = dnnInferenceCoordinatorHelper.runExclusive("face detection", () -> {
                // OpenCL-backed DNN inference is not stable when multiple networks run concurrently.
                net.setInput(inferenceBlob);
                MatVector out = new MatVector(finalOutputNames.size());
                net.forward(out, finalOutputNames);
                return out;
            });

            if (outputs == null || outputs.size() < SCRFD_STRIDES.length * 3L) {
                log.warn("Neural network forward pass returned empty output");
                return faces;
            }

            List<Detection> detections = decodeScrfdOutputs(outputs, outputNames, scale, originalWidth, originalHeight);
            detections.sort(Comparator.comparing(Detection::confidence).reversed());
            faces.addAll(applyNms(detections));

        } catch (Exception e) {
            log.error("Error during face detection", e);
            // Return empty list instead of throwing - more resilient
            return new ArrayList<>();
        } finally {
            if (meanValues != null) {
                meanValues.deallocate();
            }
            if (inputSize != null) {
                inputSize.deallocate();
            }
            if (resizedSize != null) {
                resizedSize.deallocate();
            }
            if (outputNames != null) {
                outputNames.deallocate();
            }
            if (outputs != null) {
                outputs.deallocate();
            }
            if (paddingColor != null) {
                paddingColor.deallocate();
            }
            if (roi != null) {
                roi.deallocate();
            }
            if (roiRect != null) {
                roiRect.deallocate();
            }
            // Release all resources in reverse order of creation
            matUtil.releaseResources(blob, paddedImage, resizedImage);
        }

        return faces;
    }

    private List<Detection> decodeScrfdOutputs(
            MatVector outputs,
            StringVector outputNames,
            float scale,
            int originalWidth,
            int originalHeight
    ) {
        List<Detection> detections = new ArrayList<>();
        Map<String, Mat> outputByName = new HashMap<>();

        for (long index = 0; index < outputs.size(); index++) {
            String outputName = sanitizeOutputName(outputNames.get(index).getString());
            outputByName.put(outputName, outputs.get(index));
        }

        for (int strideIndex = 0; strideIndex < SCRFD_STRIDES.length; strideIndex++) {
            int stride = SCRFD_STRIDES[strideIndex];
            Mat scores = outputByName.get("score_" + stride);
            Mat boxes = outputByName.get("bbox_" + stride);
            Mat keypoints = outputByName.get("kps_" + stride);

            if (scores == null || scores.empty() || boxes == null || boxes.empty() || keypoints == null || keypoints.empty()) {
                log.warn("SCRFD outputs missing for stride {} (score present: {}, bbox present: {}, kps present: {})",
                        stride,
                        scores != null && !scores.empty(),
                        boxes != null && !boxes.empty(),
                        keypoints != null && !keypoints.empty());
                continue;
            }

            FloatIndexer scoreIndexer = null;
            FloatIndexer boxIndexer = null;
            FloatIndexer keypointIndexer = null;
            try {
                scoreIndexer = scores.createIndexer();
                boxIndexer = boxes.createIndexer();
                keypointIndexer = keypoints.createIndexer();

                int featureWidth = MODEL_INPUT_SIZE / stride;
                int featureHeight = MODEL_INPUT_SIZE / stride;
                int anchorCount = featureWidth * featureHeight * SCRFD_ANCHORS_PER_LOCATION;
                long scoreEntries = getAnchoredEntryCount(scoreIndexer);
                long boxEntries = getAnchoredEntryCount(boxIndexer);
                long keypointEntries = getAnchoredEntryCount(keypointIndexer);

                if (scoreEntries < anchorCount || boxEntries < anchorCount || keypointEntries < anchorCount) {
                    log.warn("SCRFD output shape mismatch for stride {}: expected at least {} anchors, got scores={} boxes={} keypoints={}",
                            stride, anchorCount, scoreEntries, boxEntries, keypointEntries);
                    continue;
                }

                for (int anchorIndex = 0; anchorIndex < anchorCount; anchorIndex++) {
                    float confidence = getScore(scoreIndexer, anchorIndex);
                    if (confidence < CONFIDENCE_THRESHOLD) {
                        continue;
                    }

                    int locationIndex = anchorIndex / SCRFD_ANCHORS_PER_LOCATION;
                    int x = locationIndex % featureWidth;
                    int y = locationIndex / featureWidth;
                    float anchorCenterX = x * stride;
                    float anchorCenterY = y * stride;

                    float left = getBoxCoordinate(boxIndexer, anchorIndex, 0) * stride;
                    float top = getBoxCoordinate(boxIndexer, anchorIndex, 1) * stride;
                    float right = getBoxCoordinate(boxIndexer, anchorIndex, 2) * stride;
                    float bottom = getBoxCoordinate(boxIndexer, anchorIndex, 3) * stride;

                    int x1 = clamp(Math.round((anchorCenterX - left) / scale), 0, originalWidth - 1);
                    int y1 = clamp(Math.round((anchorCenterY - top) / scale), 0, originalHeight - 1);
                    int x2 = clamp(Math.round((anchorCenterX + right) / scale), 0, originalWidth - 1);
                    int y2 = clamp(Math.round((anchorCenterY + bottom) / scale), 0, originalHeight - 1);

                    if (x2 <= x1 || y2 <= y1) {
                        continue;
                    }

                    float[] landmarks = new float[10];
                    for (int landmarkIndex = 0; landmarkIndex < 5; landmarkIndex++) {
                        float landmarkX = (anchorCenterX + getKeypointCoordinate(keypointIndexer, anchorIndex, landmarkIndex * 2) * stride) / scale;
                        float landmarkY = (anchorCenterY + getKeypointCoordinate(keypointIndexer, anchorIndex, landmarkIndex * 2 + 1) * stride) / scale;
                        landmarks[landmarkIndex * 2] = clamp(landmarkX, 0, originalWidth - 1);
                        landmarks[landmarkIndex * 2 + 1] = clamp(landmarkY, 0, originalHeight - 1);
                    }

                    detections.add(new Detection(new Rect(x1, y1, x2 - x1, y2 - y1), confidence, landmarks));
                }
            } finally {
                if (scoreIndexer != null) {
                    scoreIndexer.release();
                }
                if (boxIndexer != null) {
                    boxIndexer.release();
                }
                if (keypointIndexer != null) {
                    keypointIndexer.release();
                }
            }
        }

        return detections;
    }

    private long getAnchoredEntryCount(FloatIndexer indexer) {
        if (indexer.rank() >= 2) {
            return indexer.size(1);
        }
        return indexer.size(0);
    }

    private String sanitizeOutputName(String outputName) {
        if (outputName == null) {
            return "";
        }
        return outputName.replace("\u0000", "").trim();
    }

    private float getScore(FloatIndexer indexer, int anchorIndex) {
        if (indexer.rank() >= 3) {
            return indexer.get(0, anchorIndex, 0);
        }
        if (indexer.rank() == 2) {
            return indexer.get(0, anchorIndex);
        }
        return indexer.get(anchorIndex);
    }

    private float getBoxCoordinate(FloatIndexer indexer, int anchorIndex, int coordinateIndex) {
        if (indexer.rank() >= 3) {
            return indexer.get(0, anchorIndex, coordinateIndex);
        }
        if (indexer.rank() == 2) {
            return indexer.get(anchorIndex, coordinateIndex);
        }
        return indexer.get(anchorIndex * 4L + coordinateIndex);
    }

    private float getKeypointCoordinate(FloatIndexer indexer, int anchorIndex, int coordinateIndex) {
        if (indexer.rank() >= 3) {
            return indexer.get(0, anchorIndex, coordinateIndex);
        }
        if (indexer.rank() == 2) {
            return indexer.get(anchorIndex, coordinateIndex);
        }
        return indexer.get(anchorIndex * 10L + coordinateIndex);
    }

    private List<FaceDetection> applyNms(List<Detection> detections) {
        List<FaceDetection> selected = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];

        try {
            for (int i = 0; i < detections.size() && selected.size() < MAX_DETECTIONS; i++) {
                if (suppressed[i]) {
                    continue;
                }

                Detection current = detections.get(i);
                selected.add(new FaceDetection(matUtil.cloneRect(current.rect()), current.confidence(), current.landmarks()));

                for (int j = i + 1; j < detections.size(); j++) {
                    if (!suppressed[j] && intersectionOverUnion(current.rect(), detections.get(j).rect()) > NMS_THRESHOLD) {
                        suppressed[j] = true;
                    }
                }
            }
        } finally {
            for (Detection detection : detections) {
                if (detection.rect() != null) {
                    detection.rect().deallocate();
                }
            }
        }

        return selected;
    }

    private float intersectionOverUnion(Rect a, Rect b) {
        int ax2 = a.x() + a.width();
        int ay2 = a.y() + a.height();
        int bx2 = b.x() + b.width();
        int by2 = b.y() + b.height();

        int intersectionX1 = Math.max(a.x(), b.x());
        int intersectionY1 = Math.max(a.y(), b.y());
        int intersectionX2 = Math.min(ax2, bx2);
        int intersectionY2 = Math.min(ay2, by2);

        int intersectionWidth = Math.max(0, intersectionX2 - intersectionX1);
        int intersectionHeight = Math.max(0, intersectionY2 - intersectionY1);
        int intersectionArea = intersectionWidth * intersectionHeight;
        int unionArea = a.width() * a.height() + b.width() * b.height() - intersectionArea;

        if (unionArea <= 0) {
            return 0;
        }

        return intersectionArea / (float) unionArea;
    }

    private int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(value, max));
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(value, max));
    }

    public record FaceDetection(Rect rect, float confidence, float[] landmarks) {}

    private record Detection(Rect rect, float confidence, float[] landmarks) {}

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

    private static AccelerationProperties.Target resolveTarget(
            AccelerationProperties accelerationProperties,
            AccelerationProperties.Target configuredTarget
    ) {
        if (configuredTarget != null && configuredTarget != AccelerationProperties.Target.AUTO) {
            return configuredTarget;
        }
        if (accelerationProperties.isEnableOpencl()) {
            return AccelerationProperties.Target.OPENCL;
        }
        return accelerationProperties.getTarget();
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
