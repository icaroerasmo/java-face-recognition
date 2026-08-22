package com.icaroerasmo.detectors.person;

import com.icaroerasmo.detectors.IDetector;
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
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Service for detecting people (and other objects) in images using SSD MobileNet.
 * This model detects various objects including people, and is used as a first-pass
 * detection before attempting face recognition.
 *
 * Based on MobileNet-SSD trained on COCO dataset.
 */
@Log4j2
@Service
public class PersonDetector implements IDetector {

    private static final String PROTO_FILE = "opencv/SSD_MobileNet_prototxt.txt";
    private static final String MODEL_FILE = "opencv/SSD_MobileNet.caffemodel";
    private static final int INPUT_SIZE = 300;

    // Class index for "person" in MobileNet-SSD (COCO dataset)
    private static final int PERSON_CLASS_ID = 15;

    // Class index for "car" in MobileNet-SSD (COCO dataset), used to suppress
    // false positives such as cars being misclassified as people
    private static final int CAR_CLASS_ID = 7;
    // IoU threshold above which a person box overlapping a car box is suppressed
    private static final double CAR_SUPPRESSION_IOU = 0.35;

    // Scale factor for MobileNet preprocessing
    private static final double SCALE_FACTOR = 0.007843; // 1/127.5
    // Mean values: 127.5 for each channel (created per use to avoid memory leak)

    private final Net net;
    private final MatUtil matUtil;
    private final DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper;
    private final double personConfidenceThreshold;
    private final double carConfidenceThreshold;
    private final double maxPersonAreaRatio;

    public PersonDetector(
            MatUtil matUtil,
            FaceRecognitionProperties faceRecognitionProperties,
            DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper
    ) {
        this.matUtil = matUtil;
        this.dnnInferenceCoordinatorHelper = dnnInferenceCoordinatorHelper;
        this.personConfidenceThreshold = normalizeConfidenceThreshold(
                faceRecognitionProperties.getDetection().getPersonConfidenceThreshold()
        );
        this.carConfidenceThreshold = normalizeConfidenceThreshold(
                faceRecognitionProperties.getDetection().getCarConfidenceThreshold());
        this.maxPersonAreaRatio = Math.max(0.0, Math.min(1.0,
                faceRecognitionProperties.getDetection().getMaxPersonAreaRatio()));
        try {
            String protoPath = OpenCvResourceHelper.getResourcePath(PROTO_FILE, PersonDetector.class);
            String modelPath = OpenCvResourceHelper.getResourcePath(MODEL_FILE, PersonDetector.class);

            log.info("Loading person detection model from: {} and {}", protoPath, modelPath);
            this.net = readNetFromCaffe(protoPath, modelPath);
            configureNet(
                    this.net,
                    faceRecognitionProperties.getAcceleration().getBackend(),
                    faceRecognitionProperties.getAcceleration().getPersonDetectionTarget(),
                    faceRecognitionProperties.getAcceleration().isFallbackToCpu(),
                    "person detection"
            );

            if (this.net == null || this.net.empty()) {
                throw new IllegalStateException("Failed to load network - network is null or empty");
            }

            log.info("Person detection model loaded successfully with confidence threshold {}",
                    String.format("%.2f", this.personConfidenceThreshold));
        } catch (Exception e) {
            log.error("Failed to load person detection model: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to initialize person detection model", e);
        }
    }

    public synchronized List<Rect> detect(Mat image) {
        List<Rect> people = new ArrayList<>();
        // Car boxes as plain int arrays {left, top, right, bottom} to avoid native
        // Rect allocations for a class we never return to the caller
        List<int[]> cars = new ArrayList<>();

        // Validate input
        if (image == null || image.empty()) {
            log.warn("Cannot detect people in null or empty image");
            return people;
        }

        // Store original dimensions BEFORE any operations
        int originalWidth = image.size().width();
        int originalHeight = image.size().height();

        Mat resizedImage = null, blob = null, output = null, detectionMat = null;
        FloatIndexer indexer = null;
        Size inputSize = null, blobSize = null;
        Scalar meanValues = null;

        try {
            // Resize image to model input size
            resizedImage = new Mat();
            inputSize = new Size(INPUT_SIZE, INPUT_SIZE);
            resize(image, resizedImage, inputSize);

            if (resizedImage.empty()) {
                log.warn("Failed to resize image for person detection");
                return people;
            }

            // Create blob from image with MobileNet preprocessing
            blobSize = new Size(INPUT_SIZE, INPUT_SIZE);
            meanValues = new Scalar(127.5, 127.5, 127.5, 0);
            blob = blobFromImage(resizedImage, SCALE_FACTOR, blobSize,
                                meanValues, false, false, CV_32F);

            if (blob == null || blob.empty()) {
                log.warn("Failed to create blob from image");
                return people;
            }

            Mat inferenceBlob = blob;
            output = dnnInferenceCoordinatorHelper.runExclusive("person detection", () -> {
                // OpenCL-backed DNN inference is not stable when multiple networks run concurrently.
                net.setInput(inferenceBlob);
                return net.forward();
            });

            if (output == null || output.empty()) {
                log.warn("Neural network forward pass returned empty output");
                return people;
            }

            // Output format: [1, 1, N, 7] where N is number of detections
            // Each detection: [image_id, class_id, confidence, x1, y1, x2, y2]
            Size detectionSize = new Size(output.size(3), output.size(2));
            detectionMat = new Mat(detectionSize, CV_32F, output.ptr(0, 0));
            detectionSize.deallocate(); // Deallocate immediately after use

            if (detectionMat.empty()) {
                log.warn("Failed to extract detection matrix from network output");
                return people;
            }

            indexer = detectionMat.createIndexer();

            // Iterate through detections
            for (int i = 0; i < output.size(3); i++) {
                int classId = (int) indexer.get(i, 1);
                float confidence = indexer.get(i, 2);

                // Convert normalized coordinates to pixel coordinates
                float x1 = indexer.get(i, 3);
                float y1 = indexer.get(i, 4);
                float x2 = indexer.get(i, 5);
                float y2 = indexer.get(i, 6);

                int left = (int) (x1 * originalWidth);
                int top = (int) (y1 * originalHeight);
                int right = (int) (x2 * originalWidth);
                int bottom = (int) (y2 * originalHeight);

                // Ensure coordinates are within image bounds
                left = Math.max(0, left);
                top = Math.max(0, top);
                right = Math.min(originalWidth - 1, right);
                bottom = Math.min(originalHeight - 1, bottom);

                int width = right - left;
                int height = bottom - top;

                // Validate rectangle dimensions
                if (width <= 0 || height <= 0) {
                    continue;
                }

                // Check if detection is a person with sufficient confidence
                if (classId == PERSON_CLASS_ID && confidence > personConfidenceThreshold) {
                    Rect personRect = new Rect(left, top, width, height);
                    people.add(personRect);
                    log.debug("Detected person at ({}, {}) with size {}x{} and confidence {}",
                            left, top, width, height, String.format("%.2f", confidence));
                } else if (classId == CAR_CLASS_ID && confidence > carConfidenceThreshold) {
                    // Collect car boxes to suppress person false positives (no native allocation)
                    cars.add(new int[]{left, top, right, bottom});
                    log.debug("Detected car at ({}, {}, {}, {}) with confidence {}",
                            left, top, right, bottom, String.format("%.2f", confidence));
                }
            }

            if (!people.isEmpty()) {
                log.debug("Detected {} person(s) in image", people.size());
            }

            // Suppress false-positive person boxes:
            // 1. Person box overlapping a detected car box with IoU > CAR_SUPPRESSION_IOU
            // 2. Person box whose area exceeds maxPersonAreaRatio of the frame area
            double frameArea = (double) originalWidth * originalHeight;
            List<Rect> survivors = new ArrayList<>(people.size());
            for (Rect person : people) {
                boolean suppressed = person.width() * person.height() > maxPersonAreaRatio * frameArea;
                if (!suppressed) {
                    for (int[] car : cars) {
                        if (iou(person, car) > CAR_SUPPRESSION_IOU) {
                            suppressed = true;
                            break;
                        }
                    }
                }
                if (suppressed) {
                    log.debug("Suppressing person box at ({}, {}) size {}x{} (car overlap or area exceeds {} of frame)",
                            person.x(), person.y(), person.width(), person.height(),
                            String.format("%.2f", maxPersonAreaRatio));
                    // Caller only deallocates the Rects in the returned list, so release suppressed ones here
                    person.deallocate();
                } else {
                    survivors.add(person);
                }
            }
            people = survivors;

        } catch (Exception e) {
            log.error("Error during person detection: {}", e.getMessage(), e);
        } finally {
            // Release indexer first
            if (indexer != null) {
                try {
                    indexer.release();
                } catch (Exception e) {
                    log.debug("Error releasing indexer: {}", e.getMessage());
                }
            }
            // Release Scalar
            if (meanValues != null) {
                try {
                    meanValues.deallocate();
                } catch (Exception e) {
                    log.debug("Error deallocating meanValues: {}", e.getMessage());
                }
            }
            // Release Size objects
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
            // Release Mat resources
            matUtil.releaseResources(resizedImage, blob, output, detectionMat);
        }

        return people;
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

    private static double normalizeConfidenceThreshold(double threshold) {
        return Math.max(0.0, Math.min(1.0, threshold));
    }

    private static double iou(Rect p, int[] car) {
        int ix1 = Math.max(p.x(), car[0]);
        int iy1 = Math.max(p.y(), car[1]);
        int ix2 = Math.min(p.x() + p.width(), car[2]);
        int iy2 = Math.min(p.y() + p.height(), car[3]);
        if (ix2 <= ix1 || iy2 <= iy1) {
            return 0.0;
        }
        double inter = (double) (ix2 - ix1) * (iy2 - iy1);
        double union = (double) p.width() * p.height()
                + (double) (car[2] - car[0]) * (car[3] - car[1]) - inter;
        return union <= 0 ? 0.0 : inter / union;
    }

    private static int mapTarget(AccelerationProperties.Target target) {
        return switch (target) {
            case AUTO, CPU -> DNN_TARGET_CPU;
            case OPENCL -> DNN_TARGET_OPENCL;
            case OPENCL_FP16 -> DNN_TARGET_OPENCL_FP16;
        };
    }
}
