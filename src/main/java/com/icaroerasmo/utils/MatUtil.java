package com.icaroerasmo.utils;

import org.bytedeco.opencv.opencv_core.*;
import org.springframework.stereotype.Component;

import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

@Component
public class MatUtil {
    public void releaseResources(Mat... matArr) {
        Arrays.asList(matArr).stream().filter(mat -> mat != null).forEach(Mat::release);
    }

    public Mat convertToGray(Mat testImage) {
        Mat target = new Mat();
        cvtColor(testImage, target, COLOR_RGB2GRAY);
        return target;
    }

    public void drawRectangleAndName(Mat img, String text, Rect rect) {
        // Calculate proportional values based on image resolution
        int imageWidth = img.cols();
        int imageHeight = img.rows();
        double scaleFactor = Math.sqrt(imageWidth * imageHeight) / 1000.0; // Base on 1000x1000 reference

        // Line thickness proportional to image size (minimum 1, scales with resolution)
        int thickness = Math.max(1, (int) Math.round(2 * scaleFactor));

        // Font scale proportional to image size
        double fontScale = Math.max(0.4, 0.8 * scaleFactor);

        // Text offset proportional to image size
        int textOffset = Math.max(15, (int) Math.round(25 * scaleFactor));

        int textX = rect.x();
        int textY = rect.y() + rect.height() + textOffset;
        int fontFace = FONT_HERSHEY_SIMPLEX;
        Scalar color = new Scalar(76, 175, 80, 1);
        int lineType = LINE_8;

        rectangle(img, rect, color, thickness, lineType, 0);
        putText(img, text, new Point(textX, textY), fontFace, fontScale, color, thickness, lineType, false);
    }

    public void clearMatVector(MatVector images) {
        try (MatVector.Iterator iterator = images.begin()) {
            while (!iterator.equals(images.end())) {
                Mat mat = iterator.get();
                releaseResources(mat);
                iterator.increment();
            }
        } finally {
            images.deallocate();
        }

    }
}
