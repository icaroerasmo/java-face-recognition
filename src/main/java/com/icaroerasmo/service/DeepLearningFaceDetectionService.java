package com.icaroerasmo.service;

import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
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
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Created on Jul 28, 2018
 *
 * @author Taha Emara
 * Email : taha@emaraic.com
 *
 * This example does face detection using deep learning model which provides a
 * great accuracy compared to OpenCV face detection using Haar cascades.
 *
 * This example is based on this code
 * https://github.com/opencv/opencv/blob/master/modules/dnn/misc/face_detector_accuracy.py
 *
 * To run this example you need two files: deploy.prototxt can be downloaded
 * from
 * https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
 *
 * and res10_300x300_ssd_iter_140000.caffemodel
 * https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
 *
 */
@Log4j2
@Service
@RequiredArgsConstructor
public class DeepLearningFaceDetectionService {

    public static final int MODEL_INPUT_SIZE = 300;
    private static final String PROTO_FILE = "opencv/deploy.prototxt";
    private static final String CAFFE_MODEL_FILE = "opencv/res10_300x300_ssd_iter_140000.caffemodel";
    private static Net net = null;

    private final MatUtil matUtil;

    static {
        try {
            String protoPath = getResourcePath(PROTO_FILE);
            String caffeModelPath = getResourcePath(CAFFE_MODEL_FILE);

            log.info("Loading face detection model from: {} and {}", protoPath, caffeModelPath);
            net = readNetFromCaffe(protoPath, caffeModelPath);
            log.info("Face detection model loaded successfully");
        } catch (Exception e) {
            log.error("Failed to load face detection model", e);
            throw new RuntimeException("Failed to initialize face detection model", e);
        }
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
        try (InputStream is = DeepLearningFaceDetectionService.class.getClassLoader().getResourceAsStream(resourceName)) {
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
     * Detect faces in an image.
     * THREAD-SAFE: Synchronized to prevent concurrent access to the shared Net object,
     * which was causing false detections when multiple cameras were processing frames simultaneously.
     */
    public synchronized List<Rect> detect(Mat testImage) {
        List<Rect> faces = new ArrayList<>();

        // Validate input
        if (testImage == null || testImage.empty()) {
            log.warn("Cannot detect faces in null or empty image");
            return faces;
        }

        Mat output = null, ne = null, blob = null, image = null, resizedImage = null;

        try {
            // Store original dimensions
            int originalWidth = testImage.size().width();
            int originalHeight = testImage.size().height();

            // CRITICAL: Clone the input image to avoid any shared state issues across threads
            // Even though method is synchronized, we clone to be extra safe
            image = testImage.clone();

            // Validate clone
            if (image == null || image.empty()) {
                log.warn("Failed to clone input image for face detection");
                return faces;
            }

            // Create a separate resized image instead of resizing in-place
            resizedImage = new Mat();
            resize(image, resizedImage, new Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));

            // Validate resized image
            if (resizedImage.empty()) {
                log.warn("Failed to resize image for face detection");
                return faces;
            }

            // Create blob from resized image
            // NCHW: Number of images, Channels, Height, Width
            blob = blobFromImage(resizedImage, 1.0, new Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

            if (blob == null || blob.empty()) {
                log.warn("Failed to create blob from image");
                return faces;
            }

            // CRITICAL: These operations on shared 'net' object must be atomic
            net.setInput(blob);
            output = net.forward();

            if (output == null || output.empty()) {
                log.warn("Neural network forward pass returned empty output");
                return faces;
            }

            // Extract 2D matrix from 4D output (detections x 7)
            ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));

            if (ne.empty()) {
                log.warn("Failed to extract detection matrix from network output");
                return faces;
            }

            FloatIndexer srcIndexer = ne.createIndexer();

            // Iterate through detections
            for (int i = 0; i < output.size(3); i++) {
                float confidence = srcIndexer.get(i, 2);

                if (confidence > 0.7) {
                    float f1 = srcIndexer.get(i, 3); // x1
                    float f2 = srcIndexer.get(i, 4); // y1
                    float f3 = srcIndexer.get(i, 5); // x2
                    float f4 = srcIndexer.get(i, 6); // y2

                    float tx = f1 * MODEL_INPUT_SIZE; // top left x
                    float ty = f2 * MODEL_INPUT_SIZE; // top left y
                    float bx = f3 * MODEL_INPUT_SIZE; // bottom right x
                    float by = f4 * MODEL_INPUT_SIZE; // bottom right y

                    Rect faceRect = createReact(tx, ty, bx, by, originalWidth, originalHeight);
                    faces.add(faceRect);
                }
            }

        } catch (Exception e) {
            log.error("Error during face detection", e);
            // Return empty list instead of throwing - more resilient
            return new ArrayList<>();
        } finally {
            // Release all resources in reverse order of creation
            matUtil.releaseResources(ne, output, blob, resizedImage, image);
        }

        return faces;
    }

    // Creates rect based on original image size that was resized due model input
    private Rect createReact(float tx, float ty, float bx, float by, int width, int height) {
        float newTx = (tx/ MODEL_INPUT_SIZE)*width;
        float newTy = (ty/ MODEL_INPUT_SIZE)*height;
        float newBx = (bx/ MODEL_INPUT_SIZE)*width;
        float newBy = (by/ MODEL_INPUT_SIZE)*height;
        return new Rect(new Point((int) newTx, (int) newTy), new Point((int) newBx, (int) newBy));
    }
}
