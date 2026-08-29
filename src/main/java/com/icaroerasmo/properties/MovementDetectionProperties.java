package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;

/**
 * Configuration for OpenCV frame-differencing movement detection.
 */
@Getter
@Setter
public class MovementDetectionProperties {

    /**
     * Master flag for movement detection (independent of {@code object-detection.enabled}).
     */
    private boolean enabled = true;

    /**
     * Whether movement events should also publish a notification (throttled).
     */
    private boolean notify = true;

    /**
     * Pixel-intensity threshold for the binary difference image.
     */
    private int differenceThreshold = 25;

    /**
     * Minimum changed-pixel ratio (nonzero/total) that counts as movement.
     */
    private double minMotionRatio = 0.01;

    /**
     * Debounce window (ms) for overlay {@code MOVEMENT_DETECTED} events per camera.
     */
    private long debounceMs = 5000;

    /**
     * Minimum interval (ms) between movement notification messages per camera.
     */
    private long throttleMs = 30000;

    /**
     * Width used to downscale frames for differencing (aspect ratio preserved).
     */
    private int processingWidth = 320;

    /**
     * Gaussian blur kernel size (forced to be odd at use time).
     */
    private int gaussianKernelSize = 5;

    /**
     * Number of dilation iterations applied to the binary difference image.
     */
    private int dilationIterations = 2;
}
