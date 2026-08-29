package com.icaroerasmo.detectors.shared.engine;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;
import java.util.Map;

/**
 * Abstraction over a DNN inference engine (OpenCV DNN or ONNX Runtime).
 *
 * <p>Implementations own their native resources and must be {@link #close() closed}
 * when no longer needed. Every forward pass returns plain {@code float[]} tensors so
 * the detection post-processing (YOLO parsing, SCRFD decoding, NMS) is engine-agnostic.
 */
public interface DnnEngine extends AutoCloseable {

    /**
     * Runs a single-output forward pass (e.g. YOLOv8 {@code [1, 84, 8400]}).
     *
     * @param blob NCHW float blob as produced by {@code blobFromImage}
     * @return the first model output as a flat tensor
     */
    TensorData forward(Mat blob);

    /**
     * Runs a multi-output forward pass (e.g. SCRFD {@code score_8/bbox_8/kps_8/...}).
     *
     * @param blob        NCHW float blob as produced by {@code blobFromImage}
     * @param outputNames requested output names, in model order
     * @return outputs keyed by (sanitized) output name
     */
    Map<String, TensorData> forward(Mat blob, List<String> outputNames);

    /**
     * Output names in model order (used to request multi-output forward passes).
     */
    List<String> getOutputNames();

    /**
     * Human-readable description for startup logging,
     * e.g. {@code "OpenCV DNN (backend=CUDA, target=CUDA)"} or {@code "ONNX Runtime (EP=CPU)"}.
     */
    String describe();

    @Override
    void close();
}