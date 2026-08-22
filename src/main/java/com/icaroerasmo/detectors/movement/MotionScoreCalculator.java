package com.icaroerasmo.detectors.movement;

/**
 * Pure, OpenCV-free movement decision: compares the fraction of changed pixels
 * of a binary difference image against a minimum motion ratio.
 *
 * <p>{@code differenceThreshold} is consumed by the OpenCV thresholding layer in
 * {@link OpenCvMovementDetector}; it is accepted here for signature symmetry with
 * the pipeline configuration but does not influence the pure ratio decision.
 */
public final class MotionScoreCalculator {

    private MotionScoreCalculator() {
    }

    /**
     * Deterministic movement decision. Returns {@code false} for zero or invalid
     * totals and for any input that could not represent real change.
     *
     * @param changedPixels      non-zero pixels in the binary difference image
     * @param totalPixels        total pixels of the processed frame
     * @param differenceThreshold pixel-intensity threshold used upstream (not used here)
     * @param minMotionRatio      minimum changed-pixel ratio that counts as movement
     */
    public static boolean isMovementDetected(long changedPixels, long totalPixels, int differenceThreshold, double minMotionRatio) {
        if (totalPixels <= 0 || changedPixels <= 0 || changedPixels > totalPixels) {
            return false;
        }
        return (double) changedPixels / totalPixels >= minMotionRatio;
    }
}
