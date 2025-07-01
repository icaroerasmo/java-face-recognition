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
        int textX = rect.x(); // or adjust for centering
        int textY = rect.y()+rect.height()+25; // offset to create space below rectangle.
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        Scalar color = new Scalar(76, 175, 80, 1);
        int thicknessText = 2;
        int lineType = LINE_8;
        rectangle(img, rect, color, thicknessText, lineType, 0);
        putText(img, text, new Point(textX, textY), fontFace, fontScale, color, thicknessText, lineType, false);
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
