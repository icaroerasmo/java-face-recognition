package com.icaroerasmo.detectors;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.List;

public interface IDetector {
    /**
     * Detect people in an image.
     * THREAD-SAFE: Synchronized to prevent concurrent access to the shared Net object.
     *
     * @param image Input image
     * @return List of rectangles representing detected people
     */
    List<Rect> detect(Mat image);
}
