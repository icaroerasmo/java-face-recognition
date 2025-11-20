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

    public List<Rect> detect(Mat testImage) {//detect faces and draw a blue rectangle arroung each face

        List<Rect> faces = new ArrayList<>();

        Mat output = null, ne = null, blob = null, image = null;

        try {
            image = new Mat(testImage);

            resize(image, image, new Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));//resize the image to match the input size of the model

            //create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
            //for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
            blob = blobFromImage(image, 1.0, new Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

            net.setInput(blob);//set the input to network model
            output = net.forward();//feed forward the input to the netwrok to get the output matrix

            ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

            FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric

            for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
                float confidence = srcIndexer.get(i, 2);
                float f1 = srcIndexer.get(i, 3);
                float f2 = srcIndexer.get(i, 4);
                float f3 = srcIndexer.get(i, 5);
                float f4 = srcIndexer.get(i, 6);
                if (confidence > .6) {
                    float tx = f1 * MODEL_INPUT_SIZE;//top left point's x
                    float ty = f2 * MODEL_INPUT_SIZE;//top left point's y
                    float bx = f3 * MODEL_INPUT_SIZE;//bottom right point's x
                    float by = f4 * MODEL_INPUT_SIZE;//bottom right point's y
                    faces.add(createReact(tx, ty, bx, by, testImage.size().width(), testImage.size().height()));
                }
            }

        } catch (Exception e) {
            throw new RuntimeException("Error during face detection", e);
        } finally {
            matUtil.releaseResources(ne, output, blob, image);
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
