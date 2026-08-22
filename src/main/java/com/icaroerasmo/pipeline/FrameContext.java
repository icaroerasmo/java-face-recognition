package com.icaroerasmo.pipeline;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.utils.MatUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.ArrayList;
import java.util.List;

/**
 * Mutable value object that flows through the pipeline stages for a single frame.
 *
 * <p>{@code FrameContext} owns every {@link Rect} transferred into it (detected
 * people rects plus face rects) and is the SINGLE owner that deallocates them,
 * exactly once, on {@link #close()}. Stages MUST NOT deallocate Rects they
 * transfer into the context. The frame {@link Mat} is NOT owned by the context
 * - the frame extractor releases it after the callback returns.
 */
@Getter
@Setter
@Log4j2
public class FrameContext implements AutoCloseable {

    private final String cameraName;
    private final Mat frame;

    private List<Rect> detectedPeople = List.of();
    private boolean movementDetected = false; // set by the pipeline's movement-detection step
    private FaceRecognition faceRecognition = null;
    private boolean processingComplete = false;

    public FrameContext(String cameraName, Mat frame) {
        this.cameraName = cameraName;
        this.frame = frame;
    }

    /**
     * Signal the pipeline to stop running the remaining stages for this frame.
     */
    public void markProcessingComplete() {
        this.processingComplete = true;
    }

    @Override
    public void close() {
        try {
            List<Rect> ownedRects = new ArrayList<>();
            if (detectedPeople != null) {
                ownedRects.addAll(detectedPeople);
            }
            if (faceRecognition != null && faceRecognition.getFaces() != null) {
                faceRecognition.getFaces().stream()
                    .map(FaceRecognition.DetectedFaces::getFaceRect)
                    .forEach(ownedRects::add);
            }
            // Identity-deduped deallocation guarantees every rect is released exactly once.
            MatUtil.deallocateRects(ownedRects);
        } catch (Exception releaseEx) {
            log.warn("Error releasing native rectangles for camera '{}'", cameraName, releaseEx);
        }
    }
}
