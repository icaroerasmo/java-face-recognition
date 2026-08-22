package com.icaroerasmo.detectors.shared;

import org.bytedeco.opencv.opencv_core.Rect;

/**
 * A raw MobileNet-SSD detection.
 *
 * <p>The {@link Rect} is owned by the caller of
 * {@link YoloDetector#detectRaw} and must be deallocated once consumed.
 */
public record CocoDetection(int classId, float confidence, Rect rect) {
}
