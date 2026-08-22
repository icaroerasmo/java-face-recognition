package com.icaroerasmo.detectors.movement;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Hermetic tests for the pure movement decision ({@link MotionScoreCalculator}).
 * First-frame behavior lives in {@link OpenCvMovementDetector} and is covered by
 * {@link OpenCvMovementDetectorTest}.
 */
class MotionScoreCalculatorTest {

    @Test
    void ratioBelowThresholdIsNoMovement() {
        assertFalse(MotionScoreCalculator.isMovementDetected(50, 1000, 25, 0.1)); // 5% < 10%
    }

    @Test
    void ratioAboveThresholdIsMovement() {
        assertTrue(MotionScoreCalculator.isMovementDetected(200, 1000, 25, 0.1)); // 20% >= 10%
    }

    @Test
    void ratioEqualToThresholdIsMovement() {
        assertTrue(MotionScoreCalculator.isMovementDetected(100, 1000, 25, 0.1)); // exactly 10%
    }

    @Test
    void zeroTotalPixelsIsNoMovement() {
        assertFalse(MotionScoreCalculator.isMovementDetected(0, 0, 25, 0.01));
    }

    @Test
    void zeroChangedPixelsIsNoMovement() {
        assertFalse(MotionScoreCalculator.isMovementDetected(0, 1000, 25, 0.01));
    }

    @Test
    void negativeTotalPixelsIsNoMovement() {
        assertFalse(MotionScoreCalculator.isMovementDetected(10, -1, 25, 0.01));
    }

    @Test
    void changedPixelsLargerThanTotalIsNoMovement() {
        assertFalse(MotionScoreCalculator.isMovementDetected(1000, 100, 25, 0.01));
    }

    @Test
    void zeroRatioThresholdStillRequiresChangedPixels() {
        // Even a zero min-motion-ratio must not report movement on an unchanged frame.
        assertFalse(MotionScoreCalculator.isMovementDetected(0, 1000, 25, 0.0));
        assertTrue(MotionScoreCalculator.isMovementDetected(1, 1000, 25, 0.0));
    }

    @Test
    void deterministicForSameInputs() {
        boolean a = MotionScoreCalculator.isMovementDetected(333, 1000, 25, 0.3);
        boolean b = MotionScoreCalculator.isMovementDetected(333, 1000, 25, 0.3);
        assertEquals(a, b);
        assertTrue(a);
    }

    @Test
    void differenceThresholdDoesNotInfluencePureDecision() {
        // differenceThreshold is consumed by the OpenCV thresholding layer; the pure
        // decision depends only on the changed-pixel ratio.
        assertEquals(
                MotionScoreCalculator.isMovementDetected(50, 1000, 25, 0.05),
                MotionScoreCalculator.isMovementDetected(50, 1000, 80, 0.05));
    }
}
