package com.icaroerasmo.utils;

import org.bytedeco.opencv.opencv_core.*;
import org.springframework.stereotype.Component;

import java.awt.image.DataBufferByte;
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
        Point textPoint = new Point(textX, textY);
        int lineType = LINE_8;

        try {
            rectangle(img, rect, color, thickness, lineType, 0);
            putText(img, text, textPoint, fontFace, fontScale, color, thickness, lineType, false);
        } finally {
            textPoint.deallocate();
            color.deallocate();
        }
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

    /**
     * Convert OpenCV Mat to Java BufferedImage
     * Used for GIF creation
     */
    public java.awt.image.BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();

        byte[] sourcePixels = new byte[width * height * channels];
        mat.data().get(sourcePixels);

        java.awt.image.BufferedImage image;
        if (channels == 3) {
            // BGR to RGB conversion
            image = new java.awt.image.BufferedImage(width, height, java.awt.image.BufferedImage.TYPE_3BYTE_BGR);
            final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);
        } else if (channels == 1) {
            // Grayscale
            image = new java.awt.image.BufferedImage(width, height, java.awt.image.BufferedImage.TYPE_BYTE_GRAY);
            final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);
        } else {
            return null;
        }

        return image;
    }
}
