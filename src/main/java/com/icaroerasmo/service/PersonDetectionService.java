package com.icaroerasmo.service;

import com.icaroerasmo.utils.MatUtil;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
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
public class PersonDetectionService {

    private static final String PROTO_FILE = "opencv/SSD_MobileNet_prototxt.txt";
    private static final String MODEL_FILE = "opencv/SSD_MobileNet.caffemodel";
    private static final int INPUT_SIZE = 300;
    private static final double CONFIDENCE_THRESHOLD = 0.7; // 70% confidence

    // Class index for "person" in MobileNet-SSD (COCO dataset)
    private static final int PERSON_CLASS_ID = 15;

    // Scale factor for MobileNet preprocessing
    private static final double SCALE_FACTOR = 0.007843; // 1/127.5
    // Mean values: 127.5 for each channel (created per use to avoid memory leak)

    private final MatUtil matUtil;
    private final String protoPath;
    private final String modelPath;
    private final ThreadLocal<Net> netHolder;

    public PersonDetectionService(
        MatUtil matUtil,
        OpenCvDnnAccelerationService openCvDnnAccelerationService
    ) {
        this.matUtil = matUtil;
        try {
            this.protoPath = getResourcePath(PROTO_FILE);
            this.modelPath = getResourcePath(MODEL_FILE);
        } catch (Exception e) {
            log.error("Failed to load person detection model: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to initialize person detection model", e);
        }

        this.netHolder = ThreadLocal.withInitial(() -> createNet(openCvDnnAccelerationService));
    }

    private Net createNet(OpenCvDnnAccelerationService openCvDnnAccelerationService) {
        log.info("Loading person detection model from: {} and {}", protoPath, modelPath);
        Net net = readNetFromCaffe(protoPath, modelPath);

        if (net == null || net.empty()) {
            throw new IllegalStateException("Failed to load network - network is null or empty");
        }

        openCvDnnAccelerationService.configure(net, "person detection");
        log.info("Person detection model loaded successfully");
        return net;
    }

    /**
     * Get resource path from classpath or filesystem.
     * First tries to load from /app/opencv/ (for Docker),
     * then from classpath resources (for development).
     */
    private static String getResourcePath(String resourceName) throws IOException, URISyntaxException {
        // Extract just the filename from the path
        String fileName = resourceName.contains("/") ?
            resourceName.substring(resourceName.lastIndexOf('/') + 1) : resourceName;

        // Try Docker deployment path first
        File dockerFile = new File("/app/opencv/" + fileName);
        if (dockerFile.exists()) {
            log.debug("Loading {} from Docker filesystem: {}", resourceName, dockerFile.getAbsolutePath());
            return dockerFile.getAbsolutePath();
        }

        // Try loading from classpath (development/testing)
        var resource = ClassLoader.getSystemResource(resourceName);
        if (resource != null) {
            log.debug("Loading {} from classpath", resourceName);
            return Path.of(resource.toURI()).toString();
        }

        // Try extracting from JAR to temp directory
        try (InputStream is = PersonDetectionService.class.getClassLoader().getResourceAsStream(resourceName)) {
            if (is != null) {
                Path tempFile = Files.createTempFile("opencv_", "_" + fileName);
                Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
                tempFile.toFile().deleteOnExit();
                log.debug("Extracted {} from JAR to temp file: {}", resourceName, tempFile);
                return tempFile.toString();
            }
        }

        throw new IOException("Resource not found: " + resourceName +
            ". Checked: /app/opencv/" + fileName + ", classpath:" + resourceName);
    }

    /**
     * Detect people in an image using a per-thread Net instance.
     *
     * @param image Input image
     * @return List of rectangles representing detected people
     */
    public List<Rect> detectPeople(Mat image) {
        List<Rect> people = new ArrayList<>();

        // Validate input
        if (image == null || image.empty()) {
            log.warn("Cannot detect people in null or empty image");
            return people;
        }

        // Store original dimensions BEFORE any operations
        int originalWidth = image.size().width();
        int originalHeight = image.size().height();

        Mat clonedImage = null, resizedImage = null, blob = null, output = null, detectionMat = null;
        FloatIndexer indexer = null;
        Size inputSize = null, blobSize = null;
        Scalar meanValues = null;

        try {
            // Clone the input image to avoid ANY modifications to original
            clonedImage = new Mat();
            image.copyTo(clonedImage);

            if (clonedImage.empty()) {
                log.warn("Failed to clone input image for person detection");
                return people;
            }

            // Resize image to model input size
            resizedImage = new Mat();
            inputSize = new Size(INPUT_SIZE, INPUT_SIZE);
            resize(clonedImage, resizedImage, inputSize);

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

            Net net = netHolder.get();
            net.setInput(blob);
            output = net.forward();

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

                // Check if detection is a person with sufficient confidence
                if (classId == PERSON_CLASS_ID && confidence > CONFIDENCE_THRESHOLD) {
                    float x1 = indexer.get(i, 3);
                    float y1 = indexer.get(i, 4);
                    float x2 = indexer.get(i, 5);
                    float y2 = indexer.get(i, 6);

                    // Convert normalized coordinates to pixel coordinates
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
                    if (width > 0 && height > 0) {
                        Rect personRect = new Rect(left, top, width, height);
                        people.add(personRect);
                        log.debug("Detected person at ({}, {}) with size {}x{} and confidence {}",
                                left, top, width, height, String.format("%.2f", confidence));
                    }
                }
            }

            if (!people.isEmpty()) {
                log.debug("Detected {} person(s) in image", people.size());
            }

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
            matUtil.releaseResources(clonedImage, resizedImage, blob, output, detectionMat);
        }

        return people;
    }

    /**
     * Check if there are any people detected in the image.
     * This is a convenience method that returns a boolean.
     *
     * @param image Input image
     * @return true if at least one person is detected, false otherwise
     */
    public boolean hasPeople(Mat image) {
        List<Rect> people = detectPeople(image);
        return !people.isEmpty();
    }
}
