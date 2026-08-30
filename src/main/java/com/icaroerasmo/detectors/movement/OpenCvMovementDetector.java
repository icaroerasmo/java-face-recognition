package com.icaroerasmo.detectors.movement;

import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.utils.MatUtil;
import jakarta.annotation.PreDestroy;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.absdiff;
import static org.bytedeco.opencv.global.opencv_core.countNonZero;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.dilate;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

/**
 * Deterministic OpenCV frame-differencing movement detector.
 *
 * <p>Pipeline per camera: resize to {@code processingWidth} (aspect preserved,
 * BGR) {@code ->} grayscale {@code ->} GaussianBlur {@code ->} {@code absdiff}
 * vs the previous processed grayscale frame {@code ->} binary threshold
 * ({@code differenceThreshold}) {@code ->} dilate ({@code dilationIterations})
 * {@code ->} changed-pixel ratio = nonzero / total. Movement is declared when
 * {@code ratio >= minMotionRatio}.
 *
 * <p>Resizing before the grayscale conversion keeps the expensive
 * {@code cvtColor} on the small downscaled image instead of the full-resolution
 * frame, which is the dominant CPU cost of the differencing pipeline.</p>
 *
 * <p>The first frame per camera only initializes state and returns {@code false}.
 * When the processed frame dimensions change, the camera state is reset and the
 * frame is treated as an initialization frame. All temporary Mats are released in
 * {@code finally}; the previous-frame Mat is owned by {@link MovementStateStore}.
 */
@Log4j2
@Service
public class OpenCvMovementDetector implements MovementDetector {

    private final MovementStateStore stateStore;
    private final MatUtil matUtil;
    private final int processingWidth;
    private final int gaussianKernelSize;
    private final int dilationIterations;
    private final int differenceThreshold;
    private final double minMotionRatio;

    public OpenCvMovementDetector(
            MovementStateStore stateStore,
            MatUtil matUtil,
            ObjectDetectionProperties objectDetectionProperties
    ) {
        this.stateStore = stateStore;
        this.matUtil = matUtil;
        var movement = objectDetectionProperties.getDetection().getMovement();
        this.processingWidth = Math.max(1, movement.getProcessingWidth());
        // Gaussian kernels must be odd and positive.
        this.gaussianKernelSize = Math.max(1, movement.getGaussianKernelSize() | 1);
        this.dilationIterations = Math.max(0, movement.getDilationIterations());
        this.differenceThreshold = movement.getDifferenceThreshold();
        this.minMotionRatio = Math.max(0.0, movement.getMinMotionRatio());
    }

    @Override
    public boolean detect(String cameraName, Mat frame) {
        if (cameraName == null || frame == null || frame.empty()) {
            return false;
        }

        Mat gray = null, resized = null, blurred = null, diff = null, thresholded = null, dilated = null;
        Mat kernel = null;
        Size resizedSize = null, kernelSize = null;
        Point anchor = null;
        Scalar borderValue = null;

        try {
            int frameWidth = frame.cols();
            int frameHeight = frame.rows();
            if (frameWidth <= 0 || frameHeight <= 0) {
                return false;
            }

            // 1. Resize to processingWidth preserving aspect ratio (BGR, BEFORE
            //    grayscale) so the expensive cvtColor runs on the small image
            //    instead of the full-resolution frame.
            double scale = (double) processingWidth / frameWidth;
            int targetHeight = Math.max(1, (int) Math.round(frameHeight * scale));
            resized = new Mat();
            resizedSize = new Size(processingWidth, targetHeight);
            resize(frame, resized, resizedSize);

            // 2. Grayscale on the downscaled image
            gray = new Mat();
            cvtColor(resized, gray, COLOR_BGR2GRAY);

            // 3. GaussianBlur to suppress sensor noise
            kernelSize = new Size(gaussianKernelSize, gaussianKernelSize);
            blurred = new Mat();
            GaussianBlur(gray, blurred, kernelSize, 0);

            // 4. Compare against the previous processed frame (owned by the state store)
            Mat previous = stateStore.getPrevious(cameraName);
            if (previous == null
                    || previous.cols() != blurred.cols()
                    || previous.rows() != blurred.rows()) {
                // First frame per camera (or dimension change): initialize state only.
                stateStore.replacePrevious(cameraName, blurred);
                return false;
            }

            // 5. absdiff
            diff = new Mat();
            absdiff(blurred, previous, diff);

            // 6. Binary threshold
            thresholded = new Mat();
            threshold(diff, thresholded, differenceThreshold, 255, THRESH_BINARY);

            // 7. Dilate to merge small blobs
            dilated = new Mat();
            kernel = new Mat();
            anchor = new Point(-1, -1);
            borderValue = new Scalar(0, 0, 0, 0);
            dilate(thresholded, dilated, kernel, anchor, dilationIterations, BORDER_CONSTANT, borderValue);

            long totalPixels = (long) dilated.cols() * dilated.rows();
            long changedPixels = countNonZero(dilated);

            // Persist the current processed frame as the new reference BEFORE releasing temporaries.
            stateStore.replacePrevious(cameraName, blurred);

            return MotionScoreCalculator.isMovementDetected(
                    changedPixels, totalPixels, differenceThreshold, minMotionRatio);
        } catch (Exception e) {
            log.warn("Error computing movement for camera '{}': {}", cameraName, e.getMessage(), e);
            return false;
        } finally {
            matUtil.releaseResources(gray, resized, blurred, diff, thresholded, dilated, kernel);
            if (resizedSize != null) {
                resizedSize.deallocate();
            }
            if (kernelSize != null) {
                kernelSize.deallocate();
            }
            if (anchor != null) {
                anchor.deallocate();
            }
            if (borderValue != null) {
                borderValue.deallocate();
            }
        }
    }

    @Override
    public void reset(String cameraName) {
        stateStore.reset(cameraName);
    }

    @Override
    public void resetAll() {
        stateStore.resetAll();
    }

    @PreDestroy
    public void shutdown() {
        resetAll();
        log.info("Released all movement detection state");
    }
}
