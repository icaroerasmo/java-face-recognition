package com.icaroerasmo.detectors.movement;

import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.utils.MatUtil;
import org.bytedeco.opencv.opencv_core.Mat;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Optional in-memory OpenCV tests for {@link OpenCvMovementDetector} using tiny
 * CPU-only Mats (no GPU/OpenCL). If OpenCV native init fails in the test JVM these
 * tests are skipped via JUnit assumptions.
 */
class OpenCvMovementDetectorTest {

    private static boolean openCvAvailable;

    @BeforeAll
    static void checkOpenCvNativeAvailability() {
        try {
            Mat probe = new Mat(4, 4, CV_8UC3);
            probe.release();
            openCvAvailable = true;
        } catch (Throwable t) {
            openCvAvailable = false;
        }
    }

    private static Mat solidFrame(int width, int height, byte value) {
        Mat frame = new Mat(height, width, CV_8UC3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                frame.ptr(y, x).put(value, value, value);
            }
        }
        return frame;
    }

    private static Mat frameWithBlock(int width, int height, byte background, byte blockValue, int blockSize) {
        Mat frame = solidFrame(width, height, background);
        for (int y = 0; y < Math.min(blockSize, height); y++) {
            for (int x = 0; x < Math.min(blockSize, width); x++) {
                frame.ptr(y, x).put(blockValue, blockValue, blockValue);
            }
        }
        return frame;
    }

    private static OpenCvMovementDetector newDetector() {
        ObjectDetectionProperties props = new ObjectDetectionProperties();
        props.getDetection().getMovement().setProcessingWidth(320);
        props.getDetection().getMovement().setGaussianKernelSize(5);
        props.getDetection().getMovement().setDilationIterations(2);
        props.getDetection().getMovement().setDifferenceThreshold(25);
        props.getDetection().getMovement().setMinMotionRatio(0.01);
        return new OpenCvMovementDetector(new MovementStateStore(), new MatUtil(), props);
    }

    @Test
    void identicalFramesProduceNoMovement() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat frameA = solidFrame(64, 64, (byte) 128);
        Mat frameB = solidFrame(64, 64, (byte) 128);
        try {
            assertFalse(detector.detect("cam", frameA), "first frame initializes state only");
            assertFalse(detector.detect("cam", frameB), "identical frames must not report movement");
        } finally {
            frameA.release();
            frameB.release();
        }
    }

    @Test
    void changedBlockProducesMovement() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat frameA = solidFrame(64, 64, (byte) 0);
        Mat frameB = frameWithBlock(64, 64, (byte) 0, (byte) 255, 16);
        try {
            assertFalse(detector.detect("cam", frameA));
            assertTrue(detector.detect("cam", frameB), "expected movement when a block changes");
        } finally {
            frameA.release();
            frameB.release();
        }
    }

    @Test
    void dimensionChangeResetsStateAndTreatsFrameAsInit() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat smallA = solidFrame(64, 32, (byte) 0);                    // 2:1 aspect
        Mat smallB = frameWithBlock(64, 32, (byte) 0, (byte) 255, 8);
        Mat bigA = solidFrame(64, 64, (byte) 0);                      // 1:1 aspect -> different processed size
        Mat bigB = frameWithBlock(64, 64, (byte) 0, (byte) 255, 8);
        try {
            assertFalse(detector.detect("cam", smallA));
            assertTrue(detector.detect("cam", smallB));
            // Dimension change: fresh init, no movement on the first big frame.
            assertFalse(detector.detect("cam", bigA));
            // Second big frame with a change -> movement.
            assertTrue(detector.detect("cam", bigB));
        } finally {
            smallA.release();
            smallB.release();
            bigA.release();
            bigB.release();
        }
    }

    @Test
    void camerasAreIsolated() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat frameA = solidFrame(64, 64, (byte) 0);
        Mat frameB = frameWithBlock(64, 64, (byte) 0, (byte) 255, 16);
        try {
            assertFalse(detector.detect("camA", frameA));
            assertFalse(detector.detect("camB", frameA), "camB first frame initializes only");
            assertTrue(detector.detect("camA", frameB));
            // camB still on its second frame and identical -> no movement.
            assertFalse(detector.detect("camB", frameA));
        } finally {
            frameA.release();
            frameB.release();
        }
    }

    @Test
    void resetClearsStateForCamera() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat frameA = solidFrame(64, 64, (byte) 0);
        Mat frameB = frameWithBlock(64, 64, (byte) 0, (byte) 255, 16);
        try {
            assertFalse(detector.detect("cam", frameA));
            assertTrue(detector.detect("cam", frameB));
            detector.reset("cam");
            assertFalse(detector.detect("cam", frameA), "reset means next frame initializes again");
            assertFalse(detector.detect("cam", frameA), "identical frames after re-init: no movement");
        } finally {
            frameA.release();
            frameB.release();
        }
    }

    @Test
    void resetAllClearsAllCameras() {
        Assumptions.assumeTrue(openCvAvailable, "OpenCV native library unavailable; skipping");
        OpenCvMovementDetector detector = newDetector();
        Mat frameA = solidFrame(64, 64, (byte) 0);
        Mat frameB = frameWithBlock(64, 64, (byte) 0, (byte) 255, 16);
        try {
            assertFalse(detector.detect("camA", frameA));
            assertTrue(detector.detect("camA", frameB));
            detector.resetAll();
            assertFalse(detector.detect("camA", frameA), "resetAll means next frame initializes again");
            assertFalse(detector.detect("camA", frameA), "identical frames after re-init: no movement");
        } finally {
            frameA.release();
            frameB.release();
        }
    }
}
