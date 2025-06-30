package com.icaroerasmo.service;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;
import org.bytedeco.opencv.opencv_videoio.*;
import org.springframework.stereotype.Service;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.*;

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
@Service
public class DeepLearningFaceDetectionService {

    private static final String PROTO_FILE = "opencv/deploy.prototxt";
    private static final String CAFFE_MODEL_FILE = "opencv/res10_300x300_ssd_iter_140000.caffemodel";
    private static Net net = null;

    static {
        try {
            net = readNetFromCaffe(Path.of(ClassLoader.getSystemResource(PROTO_FILE).toURI()).toString(),
                    Path.of(ClassLoader.getSystemResource(CAFFE_MODEL_FILE).toURI()).toString());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public List<Rect> detect(Mat testImage) {//detect faces and draw a blue rectangle arroung each face

        final Mat image = new Mat(testImage);

        List<Rect> faces = new ArrayList<>();

        resize(image, image, new Size(300, 300));//resize the image to match the input size of the model

        //create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
        //for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        net.setInput(blob);//set the input to network model
        Mat output = net.forward();//feed forward the input to the netwrok to get the output matrix

        Mat ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

        FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric

        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > .6) {
                float tx = f1 * 300;//top left point's x
                float ty = f2 * 300;//top left point's y
                float bx = f3 * 300;//bottom right point's x
                float by = f4 * 300;//bottom right point's y
                Rect rect = new Rect(new Point((int) tx, (int) ty), new Point((int) bx, (int) by));
                faces.add(transformRect(testImage, rect));
            }
        }
        return faces;
    }

    private Rect transformRect(Mat originalImg, Rect rect) {
        int[] tl = calculateNewPoints(originalImg.size().width(), originalImg.size().height(), rect.tl().x(), rect.tl().y());
        int[] br = calculateNewPoints(originalImg.size().width(), originalImg.size().height(), rect.br().x(), rect.br().y());
        return new Rect(new Point(tl[0], tl[1]), new Point(br[0], br[1]));
    }

    private int[] calculateNewPoints(double originalWidth, double originalHeight, double x, double y) {
        double newX = (x/300)*originalWidth;
        double newY = (y/300)*originalHeight;
        return new int[]{(int) newX, (int) newY};
    }
}
