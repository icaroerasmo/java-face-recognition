package com.icaroerasmo.detectors.movement;

import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Per-camera frame-differencing movement detection.
 */
public interface MovementDetector {

    /**
     * @return {@code true} when movement is detected in the frame. The first frame
     *         per camera (or the first frame after a dimension change / reset) only
     *         initializes the reference state and returns {@code false}.
     */
    boolean detect(String cameraName, Mat frame);

    /**
     * Clears the stored reference state for a camera (used on stream reconnect).
     */
    void reset(String cameraName);

    void resetAll();
}
