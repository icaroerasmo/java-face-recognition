package com.icaroerasmo.detectors.movement;

import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-camera store of the previous processed grayscale frame used by
 * {@link OpenCvMovementDetector}.
 *
 * <p>Owns exactly one cloned Mat per camera. {@link #replacePrevious} clones the
 * incoming Mat, stores the clone, and releases the previously stored Mat - the
 * callback-owned input frame is never retained.
 */
@Log4j2
@Component
public class MovementStateStore {

    private final ConcurrentHashMap<String, Mat> previousFrames = new ConcurrentHashMap<>();

    /**
     * Read-only access to the currently stored reference frame, or {@code null}
     * when this camera has not produced a frame yet.
     */
    public Mat getPrevious(String cameraName) {
        return previousFrames.get(cameraName);
    }

    /**
     * Stores a clone of {@code current} as the new reference for the camera and
     * releases the previously stored Mat (if any). The input Mat is never retained.
     */
    public void replacePrevious(String cameraName, Mat current) {
        if (current == null || current.empty()) {
            return;
        }
        Mat clone = current.clone();
        Mat previous = previousFrames.put(cameraName, clone);
        if (previous != null) {
            try {
                previous.release();
            } catch (Exception e) {
                log.debug("Error releasing previous movement state for camera '{}': {}", cameraName, e.getMessage());
            }
        }
    }

    public void reset(String cameraName) {
        Mat previous = previousFrames.remove(cameraName);
        if (previous != null) {
            try {
                previous.release();
            } catch (Exception e) {
                log.debug("Error releasing movement state for camera '{}': {}", cameraName, e.getMessage());
            }
        }
    }

    public void resetAll() {
        previousFrames.values().forEach(mat -> {
            try {
                mat.release();
            } catch (Exception e) {
                log.debug("Error releasing movement state Mat: {}", e.getMessage());
            }
        });
        previousFrames.clear();
    }
}
